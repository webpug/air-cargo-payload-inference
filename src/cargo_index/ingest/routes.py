"""Callsign -> route lookup from vradarserver/standing-data (routes/schema-01).

Used only to fill an origin (or destination) that ADS-B coverage did not see,
and only when the route contains the endpoint we *did* observe. The data is a
current snapshot; historical versions live in that repo's git history (TODO:
pin per-date versions for 2024 legs).
"""

from __future__ import annotations

import logging

import polars as pl

from cargo_index.config import Config
from cargo_index.util.fetch import cached_download

log = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/vradarserver/standing-data/main/routes/schema-01/{l}/{op}-all.csv"


def fetch_all(cfg: Config) -> pl.DataFrame:
    src = cfg.path("ref") / "src" / "routes"
    ops = sorted(set(cfg["operators"]["all_cargo"]) | set(cfg["operators"]["mixed"]))
    dfs = []
    for op in ops:
        try:
            f = cached_download(URL.format(l=op[0], op=op), src / f"{op}-all.csv")
        except Exception as e:  # noqa: BLE001 - operator missing upstream
            log.warning("routes: %s unavailable (%s)", op, e)
            continue
        dfs.append(pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy"))
    df = pl.concat(dfs).rename(lambda c: c.lstrip("﻿")).select(
        "Callsign", pl.col("AirportCodes").str.split("-").alias("route")).rename({"Callsign": "callsign"})
    df.write_parquet(cfg.path("ref") / "routes.parquet")
    log.info("routes: %d callsigns", df.height)
    return df


def infer_endpoints(legs: pl.DataFrame, routes: pl.DataFrame) -> pl.DataFrame:
    """Fill null origin/dest from the route where the observed other endpoint sits on it."""
    r = dict(zip(routes["callsign"], routes["route"].to_list()))

    def fill(cs, origin, dest):
        route = r.get(cs)
        if not route:
            return None, None
        o = d = None
        if origin is None and dest in route[1:]:
            o = route[route.index(dest, 1) - 1]
        if dest is None and origin in route[:-1]:
            d = route[route.index(origin) + 1]
        return o, d

    filled = [fill(c, o, d) for c, o, d in zip(legs["callsign"], legs["origin"], legs["dest"])]
    o_inf = pl.Series([f[0] for f in filled], dtype=pl.Utf8)
    d_inf = pl.Series([f[1] for f in filled], dtype=pl.Utf8)
    return legs.with_columns(
        origin_how=pl.when(pl.col("origin").is_null() & o_inf.is_not_null()).then(pl.lit("route_db")).otherwise("origin_how"),
        dest_how=pl.when(pl.col("dest").is_null() & d_inf.is_not_null()).then(pl.lit("route_db")).otherwise("dest_how"),
    ).with_columns(origin=pl.coalesce("origin", o_inf), dest=pl.coalesce("dest", d_inf))
