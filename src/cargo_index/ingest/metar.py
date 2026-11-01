"""METAR history from the Iowa Environmental Mesonet ASOS archive (global, ICAO ids).

Stored raw as ``data/raw/metar/{icao}/{yyyymm}.csv`` (cached per month; the
current month is refetched) and parsed to ``data/processed/metar.parquet``.
"""

from __future__ import annotations

import datetime as dt
import logging

import polars as pl

from cargo_index.config import Config
from cargo_index.util.fetch import cached_download

log = logging.getLogger(__name__)

FIELDS = ["drct", "sknt", "gust", "tmpf", "dwpf", "alti", "mslp"]
INHG_TO_HPA = 33.8639


def _months(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_all(cfg: Config) -> pl.DataFrame:
    raw = cfg.path("raw") / "metar"
    url = cfg["refdata"]["metar_url"]
    today = dt.date.today()
    for icao in cfg.target_airports:
        for y, m in _months(cfg.start, cfg.end):
            nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
            params = {"station": icao, "data": FIELDS, "tz": "Etc/UTC", "format": "onlycomma",
                      "latlon": "no", "missing": "empty", "trace": "empty", "report_type": [3, 4],
                      "year1": y, "month1": m, "day1": 1, "year2": nxt.year, "month2": nxt.month, "day2": 1}
            current = (y, m) == (today.year, today.month)
            cached_download(url, raw / icao / f"{y}{m:02d}.csv", params=params, force=current)
    return parse_all(cfg)


def parse_all(cfg: Config) -> pl.DataFrame:
    files = sorted((cfg.path("raw") / "metar").glob("*/*.csv"))
    # IEM drops the K prefix for US stations, so take the ICAO id from the directory name
    dfs = [pl.read_csv(f, infer_schema_length=0).with_columns(station=pl.lit(f.parent.name))
           for f in files if f.stat().st_size > 60]
    df = pl.concat([d for d in dfs if d.height], how="diagonal")
    f = lambda c: pl.col(c).cast(pl.Float64, strict=False)
    df = df.select(
        pl.col("station").alias("icao"),
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("ts"),
        f("drct").alias("wind_dir"), f("sknt").alias("wind_kt"), f("gust").alias("gust_kt"),
        ((f("tmpf") - 32) * 5 / 9).alias("temp_c"), ((f("dwpf") - 32) * 5 / 9).alias("dewpoint_c"),
        f("alti").alias("altimeter_inhg"),
    ).with_columns(qnh_hpa=pl.col("altimeter_inhg") * INHG_TO_HPA).unique(["icao", "ts"]).sort("icao", "ts")
    df.write_parquet(cfg.path("processed") / "metar.parquet")
    log.info("metar: %d obs for %d stations", df.height, df["icao"].n_unique())
    return df
