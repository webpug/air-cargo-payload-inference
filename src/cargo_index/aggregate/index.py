"""Weekly lane-level indices.

* Tier 1 (``weekly_lane_counts``): freighter departures by week x lane x operator x type.
  Lanes use *through* origin/destination so CN -> ANC -> US counts as CN_HK -> US.
* Tier 2 (``weekly_lane_weight``): median / IQR of ``m_idx`` by week x lane.
"""

from __future__ import annotations

import logging

import polars as pl

from cargo_index.config import Config

log = logging.getLogger(__name__)


def lane_expr() -> pl.Expr:
    return pl.concat_str([pl.col("through_origin_region"), pl.col("through_dest_region")], separator="->")


def build(cfg: Config) -> None:
    proc = cfg.path("processed")
    out = proc / "index"
    out.mkdir(exist_ok=True)
    mtow = _mtow_by_type(cfg)
    legs = (
        pl.read_parquet(proc / "flights_all.parquet")
        .filter(pl.col("is_freighter") & ~pl.col("after_tech_stop"))   # count each through-journey once
        .with_columns(week=pl.col("date").dt.truncate("1w"), lane=lane_expr())
        .join(mtow, on="typecode", how="left")
    )
    counts = legs.group_by("week", "lane", "through_origin_region", "through_dest_region", "operator", "typecode").agg(
        n_departures=pl.len(), n_airframes=pl.col("hex").n_unique(), mtow_t=pl.col("mtow_kg").sum() / 1000,
    ).sort("week", "lane")
    counts.write_parquet(out / "weekly_lane_counts.parquet")

    tf = proc / "takeoffs_all.parquet"
    if tf.exists():
        to = pl.read_parquet(tf).with_columns(
            week=pl.from_epoch("dep_ts", time_unit="s").dt.date().dt.truncate("1w"),
            lane=pl.concat_str([pl.col("origin_region"), pl.col("through_dest_region")], separator="->"))
        w = to.group_by("week", "lane", "origin").agg(
            n=pl.len(), m_idx_median=pl.col("m_idx").median(),
            m_idx_p25=pl.col("m_idx").quantile(0.25), m_idx_p75=pl.col("m_idx").quantile(0.75),
            n_airframes=pl.col("hex").n_unique(), mean_quality=pl.col("quality").mean(),
        ).sort("week", "lane")
        w.write_parquet(out / "weekly_lane_weight.parquet")
    log.info("index: %d count rows", counts.height)


def _mtow_by_type(cfg: Config) -> pl.DataFrame:
    import openap
    rows = []
    for t in cfg["types"]:
        try:
            rows.append({"typecode": t, "mtow_kg": float(openap.prop.aircraft(t.lower())["mtow"])})
        except Exception:  # noqa: BLE001 - type missing from OpenAP
            rows.append({"typecode": t, "mtow_kg": None})
    return pl.DataFrame(rows, schema={"typecode": pl.Utf8, "mtow_kg": pl.Float64})
