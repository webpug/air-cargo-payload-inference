"""Reference data: OurAirports airports/runways and the OpenSky aircraft DB."""

from __future__ import annotations

import logging

import polars as pl

from cargo_index.config import Config
from cargo_index.util.fetch import cached_download
from cargo_index.util.geo import M_PER_FT, bearing_deg, haversine_m

log = logging.getLogger(__name__)


def fetch_all(cfg: Config) -> None:
    src = cfg.path("ref") / "src"
    rd = cfg["refdata"]
    cached_download(rd["ourairports_airports"], src / "airports.csv")
    cached_download(rd["ourairports_runways"], src / "runways.csv")
    cached_download(rd["opensky_aircraft"], src / "opensky_aircraft.csv")
    build_airports(cfg)
    build_runways(cfg)
    build_opensky(cfg)


def build_airports(cfg: Config) -> pl.DataFrame:
    ref = cfg.path("ref")
    df = (
        pl.read_csv(ref / "src" / "airports.csv", infer_schema_length=0)
        .with_columns(icao=pl.coalesce(pl.col("icao_code"), pl.col("gps_code"), pl.col("ident")))
        # Scheduled-service only: otherwise military fields next to hubs (e.g. VHSK by VHHH)
        # capture approaches as "nearest airport".
        .filter((pl.col("type").is_in(cfg["airports"]["assignable_types"]) & (pl.col("scheduled_service") == "yes"))
                | pl.col("icao").is_in(cfg.target_airports))
        .select(
            "icao", pl.col("iata_code").alias("iata"), "name", "type", pl.col("iso_country").alias("country"),
            pl.col("latitude_deg").cast(pl.Float64).alias("lat"),
            pl.col("longitude_deg").cast(pl.Float64).alias("lon"),
            pl.col("elevation_ft").cast(pl.Float64).fill_null(0).alias("elev_ft"),
        )
        .unique("icao")
    )
    df = df.with_columns(region=pl.col("icao").map_elements(cfg.region_of, return_dtype=pl.Utf8))
    df.write_parquet(ref / "airports.parquet")
    return df


def build_runways(cfg: Config) -> pl.DataFrame:
    """One row per runway *end* (direction of takeoff), with threshold and far end."""
    ref = cfg.path("ref")
    rw = pl.read_csv(ref / "src" / "runways.csv", infer_schema_length=0).filter(
        (pl.col("closed") == "0") & pl.col("le_latitude_deg").is_not_null() & pl.col("he_latitude_deg").is_not_null()
    )
    f = lambda c: pl.col(c).cast(pl.Float64)
    ends = []
    for a, b in (("le", "he"), ("he", "le")):
        ends.append(rw.select(
            pl.col("airport_ident").alias("icao"),
            pl.col(f"{a}_ident").alias("rwy"),
            f(f"{a}_latitude_deg").alias("thr_lat"), f(f"{a}_longitude_deg").alias("thr_lon"),
            f(f"{b}_latitude_deg").alias("end_lat"), f(f"{b}_longitude_deg").alias("end_lon"),
            f(f"{a}_elevation_ft").alias("thr_elev_ft"), f(f"{b}_elevation_ft").alias("end_elev_ft"),
            f(f"{a}_displaced_threshold_ft").fill_null(0).alias("displaced_ft"),
            f("length_ft").alias("length_ft_listed"), "surface",
        ))
    df = pl.concat(ends)
    # Geometry from endpoints is more trustworthy than the listed heading
    hdg = bearing_deg(df["thr_lat"].to_numpy(), df["thr_lon"].to_numpy(), df["end_lat"].to_numpy(), df["end_lon"].to_numpy())
    length = haversine_m(df["thr_lat"].to_numpy(), df["thr_lon"].to_numpy(), df["end_lat"].to_numpy(), df["end_lon"].to_numpy())
    df = df.with_columns(heading=pl.Series(hdg), length_m=pl.Series(length)).with_columns(
        slope=((pl.col("end_elev_ft") - pl.col("thr_elev_ft")) * M_PER_FT / pl.col("length_m")).fill_nan(None),
    ).filter(pl.col("length_m") > 1500)
    df.write_parquet(ref / "runways.parquet")
    return df


def build_opensky(cfg: Config) -> pl.DataFrame:
    ref = cfg.path("ref")
    df = pl.read_csv(ref / "src" / "opensky_aircraft.csv", quote_char="'", infer_schema_length=0).select(
        pl.col("icao24").str.to_lowercase().alias("hex"), "registration", "typecode", "model",
        pl.col("operatorIcao").alias("operator_icao"), "operator", "owner", "built",
    ).filter(pl.col("hex").str.len_chars() == 6)
    df = df.with_columns([pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c) for c in df.columns])
    df.write_parquet(ref / "opensky_aircraft.parquet")
    return df
