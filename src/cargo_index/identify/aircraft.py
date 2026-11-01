"""hex -> registration/type/operator/freighter registry.

Layers, highest priority first:

1. ``data/ref/freighter_registrations.csv`` — hand-curated overrides
   (``is_freighter`` yes/no per registration). Maintained by a human.
2. Model / description string says freighter (``747-4R7F``, ``A330-243F``, ``BCF``...).
3. Type is freighter-only in practice (``types.<T>.freighter_only``), or is a
   freighter unless flown by a listed passenger operator
   (``types.<T>.freighter_unless_operator``).
4. Operator (OpenSky / adsb.lol ``ownOp`` / dominant callsign prefix) is an
   all-cargo operator (``operators.all_cargo``).

Aircraft of a ``review_when_mixed`` type whose operator is in ``operators.mixed`` and that no
rule classifies are written to ``data/ref/review/freighter_review.csv`` for a
human to label.
"""

from __future__ import annotations

import logging
import re

import polars as pl

from cargo_index.config import Config

log = logging.getLogger(__name__)

FREIGHTER_RE = re.compile(r"(\dF\b|\dF$|F$|-F|BCF|BDSF|P2F|SF\b|ERF|SCD|FREIGHT|\(F\)|\bF\b)", re.IGNORECASE)
CALLSIGN_RE = re.compile(r"^([A-Z]{3})\d")


def registry_path(cfg: Config):
    return cfg.path("ref") / "aircraft_registry.parquet"


def load_whitelist_hexes(cfg: Config) -> set[str]:
    p = registry_path(cfg)
    if not p.exists():
        return set()
    return set(pl.read_parquet(p).filter(pl.col("is_freighter"))["hex"].to_list())


def model_says_freighter(s: str | None) -> bool:
    return bool(s) and bool(FREIGHTER_RE.search(s))


def dominant_callsign_prefix(cfg: Config) -> pl.DataFrame:
    """Most common 3-letter ICAO callsign prefix per hex across all ingested days."""
    pts = cfg.path("raw") / "adsb" / "adsblol"
    files = sorted(pts.glob("date=*/points.parquet"))
    if not files:
        return pl.DataFrame(schema={"hex": pl.Utf8, "cs_operator": pl.Utf8})
    cs = (
        pl.scan_parquet(files).select("hex", "callsign").unique()
        .with_columns(cs_operator=pl.col("callsign").str.extract(CALLSIGN_RE.pattern, 1))
        .drop_nulls("cs_operator").group_by("hex", "cs_operator").len()
        .sort("len", descending=True).group_by("hex").first().select("hex", "cs_operator")
    )
    return cs.collect()


def build_registry(cfg: Config) -> pl.DataFrame:
    ref = cfg.path("ref")
    types = cfg["types"]
    ops = cfg["operators"]
    all_cargo, mixed = set(ops["all_cargo"]), set(ops["mixed"])

    seen_files = sorted((cfg.path("raw") / "adsb" / "adsblol").glob("date=*/aircraft.parquet"))
    seen = (pl.scan_parquet(seen_files).unique("hex", keep="last").collect() if seen_files
            else pl.DataFrame(schema={"hex": pl.Utf8, "registration": pl.Utf8, "typecode": pl.Utf8,
                                      "desc": pl.Utf8, "ownOp": pl.Utf8}))
    osky = pl.read_parquet(ref / "opensky_aircraft.parquet")
    reg = (
        seen.select("hex", "registration", "typecode", "desc", "ownOp")
        .join(osky.select("hex", pl.col("registration").alias("os_reg"), pl.col("typecode").alias("os_type"),
                          "model", "operator_icao", "owner"), on="hex", how="full", coalesce=True)
        .with_columns(registration=pl.coalesce("registration", "os_reg"), typecode=pl.coalesce("typecode", "os_type"))
        .drop("os_reg", "os_type")
        .filter(pl.col("typecode").is_in(list(types)))
        .join(dominant_callsign_prefix(cfg), on="hex", how="left")
        .with_columns(operator_icao=pl.coalesce("cs_operator", "operator_icao"))
    )

    freighter_only = {t for t, v in types.items() if v.get("freighter_only")}
    unless = pl.lit(False)
    for t, v in types.items():
        if "freighter_unless_operator" in v:
            unless = unless | ((pl.col("typecode") == t) & pl.col("operator_icao").is_not_null()
                               & ~pl.col("operator_icao").is_in(v["freighter_unless_operator"]))
    reg = reg.with_columns(
        model_freighter=pl.struct("model", "desc").map_elements(
            lambda s: model_says_freighter(s["model"]) or model_says_freighter(s["desc"]), return_dtype=pl.Boolean),
    ).with_columns(
        rule=pl.when(pl.col("model_freighter")).then(pl.lit("model"))
        .when(pl.col("typecode").is_in(list(freighter_only))).then(pl.lit("type"))
        .when(unless).then(pl.lit("type_default"))
        .when(pl.col("operator_icao").is_in(list(all_cargo))).then(pl.lit("operator"))
        .otherwise(None),
    )

    curated_path = ref / "freighter_registrations.csv"
    if curated_path.exists():
        cur = pl.read_csv(curated_path, infer_schema_length=0).select(
            "registration", pl.col("is_freighter").str.to_lowercase().is_in(["1", "true", "yes", "y"]).alias("cur_f"),
            "variant", "lessor")
        reg = reg.join(cur, on="registration", how="left")
    else:
        reg = reg.with_columns(cur_f=pl.lit(None, pl.Boolean), variant=pl.lit(None, pl.Utf8), lessor=pl.lit(None, pl.Utf8))

    reg = reg.with_columns(
        is_freighter=pl.coalesce("cur_f", pl.col("rule").is_not_null()),
        rule=pl.when(pl.col("cur_f").is_not_null()).then(pl.lit("curated")).otherwise("rule"),
    ).drop("cur_f")
    reg.write_parquet(registry_path(cfg))

    # Mixed-operator A33x/B763 are overwhelmingly passenger aircraft (P2F conversions go to
    # all-cargo operators), so only the types that operators fly in both variants are reviewed.
    review_types = [t for t, v in types.items() if v.get("review_when_mixed")]
    review = reg.filter(pl.col("rule").is_null() & pl.col("operator_icao").is_in(list(mixed))
                        & pl.col("typecode").is_in(review_types))
    (ref / "review").mkdir(exist_ok=True)
    review.select("registration", "hex", "typecode", "operator_icao", "model", "desc", "ownOp", "owner").sort(
        "operator_icao", "typecode", "registration").write_csv(ref / "review" / "freighter_review.csv")
    log.info("registry: %d target-type aircraft, %d freighters, %d need review",
             reg.height, reg["is_freighter"].sum(), review.height)
    return reg
