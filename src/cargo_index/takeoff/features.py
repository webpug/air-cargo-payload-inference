"""Per-departure takeoff feature table: ``data/processed/takeoffs/{date}.parquet``.

Rejections (with reason) go to ``data/processed/qc/takeoff_rejects/{date}.parquet``.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict

import numpy as np
import polars as pl

from cargo_index.config import Config
from cargo_index.identify import flights as flights_mod
from cargo_index.model import physics
from cargo_index.takeoff import segment, weather

log = logging.getLogger(__name__)

PRE_S = 20 * 60     # points before first airborne sample to include
POST_S = 8 * 60     # ... and after
METAR_TOL = "90m"


def _departures(cfg: Config, legs: pl.DataFrame, date: dt.date) -> pl.DataFrame:
    # Only departures whose ground roll was observed can yield a takeoff measurement
    return legs.filter(
        (pl.col("date") == date) & pl.col("is_freighter") & pl.col("origin").is_in(cfg.target_airports)
        & (pl.col("origin_how") == "ground")
    )


def takeoffs_for_day(cfg: Config, date: dt.date, legs: pl.DataFrame, airports: pl.DataFrame,
                     runways: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    deps = _departures(cfg, legs, date)
    raw = cfg.path("raw") / "adsb" / "adsblol"
    files = [raw / f"date={d.isoformat()}" / "points.parquet" for d in (date - dt.timedelta(days=1), date)]
    pts = pl.concat([pl.scan_parquet(f) for f in files if f.exists()]).filter(
        pl.col("hex").is_in(deps["hex"].unique().implode())).collect().sort("hex", "ts")
    elev = dict(zip(airports["icao"], airports["elev_ft"]))
    p = cfg["takeoff"]
    rows, rejects = [], []
    by_hex = {h: g for (h,), g in pts.group_by(["hex"])}
    for d in deps.iter_rows(named=True):
        key = {"hex": d["hex"], "dep_ts": d["dep_ts"], "origin": d["origin"]}
        g = by_hex.get(d["hex"])
        if g is None:
            rejects.append({**key, "reason": "no_points"})
            continue
        w = g.filter(pl.col("ts").is_between(d["first_air_ts"] - PRE_S, d["first_air_ts"] + POST_S))
        try:
            t = segment.extract(w, elev[d["origin"]], runways.filter(pl.col("icao") == d["origin"]), p)
        except segment.Reject as e:
            rejects.append({**key, "reason": str(e)})
            continue
        row = {**key, **{k: v for k, v in asdict(t).items() if k != "rwy"},
               "rwy": t.rwy["rwy"], "rwy_heading": t.rwy["heading"], "runway_length_m": t.rwy["length_m"],
               "runway_slope": t.rwy["slope"], "rwy_median_cross_m": t.rwy["median_cross_m"],
               "quality": segment.quality(t)}
        rows.append(row)
    rej = pl.DataFrame(rejects, schema={"hex": pl.Utf8, "dep_ts": pl.Float64, "origin": pl.Utf8, "reason": pl.Utf8})
    if not rows:
        return pl.DataFrame(), rej
    to = pl.DataFrame(rows, infer_schema_length=None).join(
        deps.select("hex", "dep_ts", "dest", "through_dest", "origin_region", "through_dest_region",
                    "registration", "typecode", "operator", "variant", "callsign", "great_circle_nm", "tech_stop"),
        on=["hex", "dep_ts"], how="left")
    return to, rej


def add_weather_and_physics(to: pl.DataFrame, metar: pl.DataFrame, airports: pl.DataFrame) -> pl.DataFrame:
    to = to.with_columns(lof_dt=pl.from_epoch(pl.col("t_lof"), time_unit="s").dt.replace_time_zone("UTC")).sort("lof_dt")
    m = metar.rename({"icao": "origin", "ts": "metar_ts"}).sort("metar_ts")
    to = to.join_asof(m, left_on="lof_dt", right_on="metar_ts", by="origin", strategy="nearest",
                      tolerance=METAR_TOL).join(airports.select(pl.col("icao").alias("origin"), "elev_ft", "lon"),
                                                on="origin", how="left")
    rho = weather.air_density(to["qnh_hpa"].to_numpy(), to["temp_c"].to_numpy(), to["elev_ft"].to_numpy(),
                              to["dewpoint_c"].to_numpy())
    calm = to["wind_kt"].fill_null(0).to_numpy() == 0
    hw = np.where(calm, 0.0, weather.headwind_kt(to["wind_dir"].to_numpy(), to["wind_kt"].to_numpy(),
                                                 to["rwy_heading"].to_numpy()))
    to = to.with_columns(rho=pl.Series(rho), headwind_kt=pl.Series(hw))
    to = to.with_columns(
        tas_lof=pl.col("gs_lof") + pl.col("headwind_kt"),
        roll_time_s=pl.col("t_lof") - pl.col("t_roll_start"),
        roll_dist_m=pl.col("along_lof_m") - pl.col("along_roll_start_m"),
        remaining_runway_m=pl.col("runway_length_m") - pl.col("along_lof_m"),
        hour_local=((pl.col("lof_dt").dt.hour() + (pl.col("lon") / 15).round()) % 24).cast(pl.Int8),
        dow=pl.col("lof_dt").dt.weekday(),
        week=pl.col("lof_dt").dt.truncate("1w").dt.date(),
    ).with_columns(
        mean_accel_kt_s=(pl.col("gs_lof") - pl.col("gs_roll_start")) / pl.col("roll_time_s"),
    )
    rho, tas = to["rho"].to_numpy(), to["tas_lof"].to_numpy()
    lo, hi = physics.m_rel_band(rho, tas, to["mean_accel_kt_s"].to_numpy(), to["sample_interval_s"].to_numpy())
    return to.with_columns(m_rel=pl.Series(physics.m_rel(rho, tas)).fill_nan(None),
                           m_rel_lo=pl.Series(lo).fill_nan(None), m_rel_hi=pl.Series(hi).fill_nan(None))


def build_all(cfg: Config, force: bool = False) -> pl.DataFrame:
    ref, proc = cfg.path("ref"), cfg.path("processed")
    airports = pl.read_parquet(ref / "airports.parquet")
    runways = pl.read_parquet(ref / "runways.parquet").filter(pl.col("icao").is_in(cfg.target_airports))
    metar = pl.read_parquet(proc / "metar.parquet")
    legs = flights_mod.load(cfg)
    out, qc = proc / "takeoffs", proc / "qc" / "takeoff_rejects"
    out.mkdir(exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)
    for date in sorted(legs["date"].unique()):
        f = out / f"{date.isoformat()}.parquet"
        if f.exists() and not force:
            continue
        to, rej = takeoffs_for_day(cfg, date, legs, airports, runways)
        rej.write_parquet(qc / f"{date.isoformat()}.parquet")
        if to.height:
            to = add_weather_and_physics(to, metar, airports)
        to.write_parquet(f)
        log.info("%s: %d takeoffs, %d rejected %s", date, to.height, rej.height,
                 dict(rej["reason"].value_counts().iter_rows()) if rej.height else {})
    files = [f for f in sorted(out.glob("*.parquet")) if f.stat().st_size > 0]
    allto = pl.concat([pl.read_parquet(f) for f in files if pl.read_parquet_schema(f)], how="diagonal_relaxed")
    allto = physics.robust_index(allto.filter(pl.col("m_rel").is_not_null()))
    allto.write_parquet(proc / "takeoffs_all.parquet")
    return allto
