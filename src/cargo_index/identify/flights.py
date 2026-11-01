"""Segment per-hex point streams into flight legs and assign origin/destination.

A point is *low* if it is on the ground or within ``ground_alt_agl_ft`` of the
nearest airport's elevation (and within ``endpoint_max_dist_nm`` of it). A leg
is a maximal run of non-low points, split further at data gaps that look like
an unobserved stop: gap > ``leg_gap_min`` with both bracketing points below
10,000 ft and an implied ground speed under 250 kt, or readsb's own new-leg flag.

Origin/destination come from the low points bracketing the run (or the first /
last airborne point if it is itself close to an airport and low, ``air`` /
``air_far``). Endpoints coverage never saw are filled from the callsign route
database when it agrees with the endpoint we did see (``route_db``). Legs are
owned by the UTC day of their departure; day D is processed with day D+1's
points so overnight legs close correctly.

Tech stops are linked across all days in :func:`link_tech_stops`.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from cargo_index.config import Config
from cargo_index.ingest import routes as routes_mod
from cargo_index.util.geo import EARTH_R_M, M_PER_NM, haversine_m

log = logging.getLogger(__name__)

BRACKET_MAX_S = 30 * 60      # low point must be this close to the run to count as its endpoint
ENDPOINT_MAX_AGL_FT = 3000   # first/last airborne point accepted as endpoint below this
FAR_ENDPOINT_MAX_NM = 30     # ... or, as "air_far", below FAR_ENDPOINT_MAX_FT within this radius
FAR_ENDPOINT_MAX_FT = 10000


class AirportIndex:
    def __init__(self, airports: pl.DataFrame):
        self.icao = airports["icao"].to_numpy()
        self.elev = airports["elev_ft"].to_numpy()
        lat, lon = np.radians(airports["lat"].to_numpy()), np.radians(airports["lon"].to_numpy())
        self.tree = cKDTree(np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    def nearest(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (index, distance_m) of nearest airport for each point."""
        la, lo = np.radians(lat), np.radians(lon)
        d, i = self.tree.query(np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        return i, 2 * EARTH_R_M * np.arcsin(np.clip(d / 2, 0, 1))


def segment_hex(pts: pl.DataFrame, ap: AirportIndex, cfg_id: dict) -> list[dict]:
    """Legs for one aircraft's (sorted) points."""
    ts = pts["ts"].to_numpy()
    lat, lon = pts["lat"].to_numpy(), pts["lon"].to_numpy()
    alt = pts["baro_alt"].to_numpy().astype(float)  # nulls -> nan
    gnd = pts["on_ground"].to_numpy()
    flags = pts["flags"].fill_null(0).to_numpy()
    callsign = pts["callsign"].to_list()

    ai, dist = ap.nearest(lat, lon)
    near = dist < cfg_id["endpoint_max_dist_nm"] * M_PER_NM
    agl = alt - ap.elev[ai]
    low = gnd | (near & (agl < cfg_id["ground_alt_agl_ft"]))
    air = ~low

    # Break airborne runs at suspicious gaps and readsb new-leg flags
    gap = np.diff(ts, prepend=ts[0])
    seg_d = np.r_[0.0, haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])]
    with np.errstate(divide="ignore", invalid="ignore"):
        implied_kt = seg_d / M_PER_NM / (gap / 3600)
    prev_alt = np.r_[np.nan, alt[:-1]]
    stop_gap = (gap > cfg_id["leg_gap_min"] * 60) & (alt < 10000) & (prev_alt < 10000) & (implied_kt < 250)
    breaks = stop_gap | ((flags & 2) > 0)

    legs = []
    n = len(ts)
    i = 0
    while i < n:
        if not air[i]:
            i += 1
            continue
        j = i + 1
        while j < n and air[j] and not breaks[j]:
            j += 1
        # run = [i, j)
        legs.append(_leg(i, j, ts, lat, lon, alt, low, near, dist, agl, ai, ap, callsign, n))
        i = j
    return [leg for leg in legs if leg is not None]


def _endpoint(k_low: int | None, k_air: int, ts, near, dist, agl, ai, ap, dt_ok: bool):
    if k_low is not None and dt_ok and near[k_low]:
        return ap.icao[ai[k_low]], "ground"
    if near[k_air] and agl[k_air] < ENDPOINT_MAX_AGL_FT:
        return ap.icao[ai[k_air]], "air"
    if dist[k_air] < FAR_ENDPOINT_MAX_NM * M_PER_NM and agl[k_air] < FAR_ENDPOINT_MAX_FT:
        return ap.icao[ai[k_air]], "air_far"
    return None, None


def _leg(i, j, ts, lat, lon, alt, low, near, dist, agl, ai, ap, callsign, n):
    if j - i < 5:
        return None
    k_before = i - 1 if i > 0 and low[i - 1] else None
    k_after = j if j < n and low[j] else None
    orig, orig_how = _endpoint(k_before, i, ts, near, dist, agl, ai, ap,
                               k_before is not None and ts[i] - ts[k_before] < BRACKET_MAX_S)
    dest, dest_how = _endpoint(k_after, j - 1, ts, near, dist, agl, ai, ap,
                               k_after is not None and ts[k_after] - ts[j - 1] < BRACKET_MAX_S)
    cs = [c for c in callsign[i:j] if c]
    cs_mode = max(set(cs), key=cs.count) if cs else None
    return {
        "dep_ts": ts[k_before] if orig_how == "ground" else ts[i],
        "arr_ts": ts[k_after] if dest_how == "ground" else ts[j - 1],
        "first_air_ts": ts[i], "last_air_ts": ts[j - 1],
        "origin": orig, "origin_how": orig_how, "dest": dest, "dest_how": dest_how,
        "first_lat": lat[i], "first_lon": lon[i], "last_lat": lat[j - 1], "last_lon": lon[j - 1],
        "callsign": cs_mode, "n_points": j - i, "max_alt": float(np.nanmax(alt[i:j])) if np.isfinite(alt[i:j]).any() else None,
    }


def legs_for_day(cfg: Config, date: dt.date, ap: AirportIndex) -> pl.DataFrame:
    raw = cfg.path("raw") / "adsb" / "adsblol"
    files = [raw / f"date={d.isoformat()}" / "points.parquet" for d in (date, date + dt.timedelta(days=1))]
    files = [f for f in files if f.exists()]
    pts = pl.concat([pl.read_parquet(f) for f in files]).sort("hex", "ts")
    day_start = dt.datetime.combine(date, dt.time(), dt.UTC).timestamp()
    day_end = day_start + 86400
    cfg_id = cfg["identify"]
    rows = []
    for (hex_,), g in pts.group_by(["hex"], maintain_order=True):
        for leg in segment_hex(g, ap, cfg_id):
            if day_start <= leg["dep_ts"] < day_end:
                rows.append({"hex": hex_, **leg})
    schema = {"hex": pl.Utf8, "dep_ts": pl.Float64, "arr_ts": pl.Float64, "first_air_ts": pl.Float64,
              "last_air_ts": pl.Float64, "origin": pl.Utf8, "origin_how": pl.Utf8, "dest": pl.Utf8,
              "dest_how": pl.Utf8, "first_lat": pl.Float64, "first_lon": pl.Float64, "last_lat": pl.Float64,
              "last_lon": pl.Float64, "callsign": pl.Utf8, "n_points": pl.Int64, "max_alt": pl.Float64}
    return pl.DataFrame(rows, schema=schema).with_columns(date=pl.lit(date))


def enrich(legs: pl.DataFrame, airports: pl.DataFrame, registry: pl.DataFrame,
           routes: pl.DataFrame | None = None) -> pl.DataFrame:
    if routes is not None:
        legs = routes_mod.infer_endpoints(legs, routes)
    a = airports.select("icao", "lat", "lon", "region")
    legs = (
        legs.join(a.rename({"icao": "origin", "lat": "o_lat", "lon": "o_lon", "region": "origin_region"}), on="origin", how="left")
        .join(a.rename({"icao": "dest", "lat": "d_lat", "lon": "d_lon", "region": "dest_region"}), on="dest", how="left")
    )
    gc = haversine_m(legs["o_lat"].to_numpy(), legs["o_lon"].to_numpy(), legs["d_lat"].to_numpy(), legs["d_lon"].to_numpy()) / M_PER_NM
    legs = legs.with_columns(
        great_circle_nm=pl.Series(gc).fill_nan(None),
        block_time_min=(pl.col("arr_ts") - pl.col("dep_ts")) / 60,
        operator_cs=pl.col("callsign").str.extract(r"^([A-Z]{3})\d", 1),
    ).drop("o_lat", "o_lon", "d_lat", "d_lon")
    reg = registry.select("hex", "registration", "typecode", "operator_icao", "is_freighter", "rule", "variant", "lessor")
    return legs.join(reg, on="hex", how="left").with_columns(
        operator=pl.coalesce("operator_cs", "operator_icao"), is_freighter=pl.col("is_freighter").fill_null(False))


def link_tech_stops(legs: pl.DataFrame, tech_airports: list[str], max_h: float) -> pl.DataFrame:
    """Mark legs that end at a tech-stop airport and continue within ``max_h``; add through-destination."""
    legs = legs.sort("hex", "dep_ts").with_columns(
        next_origin=pl.col("origin").shift(-1).over("hex"),
        next_dest=pl.col("dest").shift(-1).over("hex"),
        next_dest_region=pl.col("dest_region").shift(-1).over("hex"),
        next_dep=pl.col("dep_ts").shift(-1).over("hex"),
        prev_dest=pl.col("dest").shift(1).over("hex"),
        prev_origin=pl.col("origin").shift(1).over("hex"),
        prev_origin_region=pl.col("origin_region").shift(1).over("hex"),
        prev_arr=pl.col("arr_ts").shift(1).over("hex"),
    )
    ends_tech = (pl.col("dest").is_in(tech_airports) & (pl.col("next_origin") == pl.col("dest"))
                 & ((pl.col("next_dep") - pl.col("arr_ts")) < max_h * 3600))
    starts_tech = (pl.col("origin").is_in(tech_airports) & (pl.col("prev_dest") == pl.col("origin"))
                   & ((pl.col("dep_ts") - pl.col("prev_arr")) < max_h * 3600))
    return legs.with_columns(
        tech_stop=ends_tech,
        after_tech_stop=starts_tech,
        through_dest=pl.when(ends_tech).then("next_dest").otherwise("dest"),
        through_dest_region=pl.when(ends_tech).then("next_dest_region").otherwise("dest_region"),
        through_origin=pl.when(starts_tech).then("prev_origin").otherwise("origin"),
        through_origin_region=pl.when(starts_tech).then("prev_origin_region").otherwise("origin_region"),
    ).drop("next_origin", "next_dest", "next_dest_region", "next_dep", "prev_dest", "prev_origin",
           "prev_origin_region", "prev_arr")


def build_all(cfg: Config, force: bool = False) -> pl.DataFrame:
    ref = cfg.path("ref")
    airports = pl.read_parquet(ref / "airports.parquet")
    registry = pl.read_parquet(ref / "aircraft_registry.parquet")
    ap = AirportIndex(airports)
    out = cfg.path("processed") / "flights"
    out.mkdir(exist_ok=True)
    raw = cfg.path("raw") / "adsb" / "adsblol"
    dates = sorted(dt.date.fromisoformat(p.parent.name.split("=")[1]) for p in raw.glob("date=*/points.parquet"))
    for d in dates:
        f = out / f"{d.isoformat()}.parquet"
        if f.exists() and not force:
            continue
        legs = legs_for_day(cfg, d, ap)
        legs.write_parquet(f)
        log.info("%s: %d legs (%d with origin and dest)", d, legs.height,
                 legs.filter(pl.col("origin").is_not_null() & pl.col("dest").is_not_null()).height)
    files = sorted(out.glob("*.parquet"))
    rp = ref / "routes.parquet"
    routes = pl.read_parquet(rp) if rp.exists() else None
    legs = enrich(pl.concat([pl.read_parquet(f) for f in files]), airports, registry, routes)
    idc = cfg["identify"]
    legs = link_tech_stops(legs, idc["tech_stop_airports"], idc["tech_stop_max_h"])
    legs.write_parquet(cfg.path("processed") / "flights_all.parquet")
    return legs


def load(cfg: Config) -> pl.DataFrame:
    return pl.read_parquet(Path(cfg.path("processed")) / "flights_all.parquet")
