"""Coverage audit: per target airport per day, how many freighter departures we
see and how many yield a usable takeoff measurement. Feeds notebooks/01."""

from __future__ import annotations

import logging

import polars as pl

from cargo_index.config import Config

log = logging.getLogger(__name__)


def build(cfg: Config) -> pl.DataFrame:
    proc = cfg.path("processed")
    legs = pl.read_parquet(proc / "flights_all.parquet").filter(pl.col("is_freighter"))
    to = pl.read_parquet(proc / "takeoffs_all.parquet") if (proc / "takeoffs_all.parquet").exists() else None
    deps = legs.filter(pl.col("origin").is_in(cfg.target_airports)).group_by("origin", "date").agg(
        n_departures=pl.len(),
        n_dep_seen_on_ground=(pl.col("origin_how") == "ground").sum(),
        n_dep_seen_airborne=pl.col("origin_how").is_in(["air", "air_far"]).sum(),
        n_dep_route_db=(pl.col("origin_how") == "route_db").sum(),
    )
    if to is not None and to.height:
        t = to.with_columns(date=pl.from_epoch("dep_ts", time_unit="s").dt.date()).group_by("origin", "date").agg(
            n_takeoffs=pl.len(), median_sample_interval_s=pl.col("sample_interval_s").median(),
            mean_quality=pl.col("quality").mean())
        deps = deps.join(t, on=["origin", "date"], how="left").with_columns(pl.col("n_takeoffs").fill_null(0))
    out = deps.sort("origin", "date")
    qc = proc / "qc"
    qc.mkdir(exist_ok=True)
    out.write_parquet(qc / "coverage_airport_day.parquet")
    out.write_csv(qc / "coverage_airport_day.csv")
    log.info("coverage audit: %d airport-days", out.height)
    return out
