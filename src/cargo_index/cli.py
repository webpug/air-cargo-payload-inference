"""Command-line entry point: ``python -m cargo_index <command> --config configs/mvp.yaml``."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cargo_index.config import Config

log = logging.getLogger("cargo_index")


def cmd_refdata(cfg: Config, args) -> None:
    from cargo_index.ingest import refdata, routes
    refdata.fetch_all(cfg)
    routes.fetch_all(cfg)


def _make_filter(cfg: Config):
    import polars as pl

    from cargo_index.identify.aircraft import load_whitelist_hexes
    from cargo_index.ingest.adsblol import TraceFilter
    aps = pl.read_parquet(cfg.path("ref") / "airports.parquet").filter(pl.col("icao").is_in(cfg.target_airports))
    cap = cfg["adsb"]["capture"]
    return TraceFilter(aps, set(cap["candidate_types"]), load_whitelist_hexes(cfg),
                       cap["radius_nm"], cap["max_alt_ft"])


def _ingest_one(cfg_path: str, date: dt.date, local: list[str] | None, force: bool) -> dict:
    from cargo_index.ingest import adsblol
    _setup_logging()
    cfg = Config.load(cfg_path)
    a = cfg["adsb"]["adsblol"]
    return adsblol.ingest_day(date, cfg.path("raw"), _make_filter(cfg), a["repo_template"], a["tag_template"],
                              local_parts=[Path(p) for p in local] if local else None, force=force)


def cmd_ingest(cfg: Config, args) -> None:
    dates = [dt.date.fromisoformat(d) for d in args.dates] if args.dates else cfg.dates()
    if args.local:
        for d in dates:
            print(_ingest_one(args.config, d, args.local, args.force))
        return
    workers = args.workers or cfg["adsb"]["adsblol"]["workers"]
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_ingest_one, args.config, d, None, args.force): d for d in dates}
        for f in as_completed(futs):
            try:
                m = f.result()
                log.info("%s %s", futs[f], m.get("status"))
            except Exception:
                log.exception("ingest failed for %s", futs[f])


def cmd_metar(cfg: Config, args) -> None:
    from cargo_index.ingest import metar
    metar.fetch_all(cfg)


def cmd_identify(cfg: Config, args) -> None:
    from cargo_index.identify import aircraft, flights
    aircraft.build_registry(cfg)
    flights.build_all(cfg, force=args.force)


def cmd_takeoffs(cfg: Config, args) -> None:
    from cargo_index.takeoff import features
    features.build_all(cfg, force=args.force)


def cmd_index(cfg: Config, args) -> None:
    from cargo_index.aggregate import index
    index.build(cfg)


def cmd_audit(cfg: Config, args) -> None:
    from cargo_index.aggregate import coverage
    coverage.build(cfg)


def cmd_charts(cfg: Config, args) -> None:
    from cargo_index.viz import charts
    charts.build_all(cfg)


def cmd_run(cfg: Config, args) -> None:
    for step in (cmd_refdata, cmd_metar, cmd_ingest, cmd_identify, cmd_takeoffs, cmd_audit, cmd_index, cmd_charts):
        log.info("== %s", step.__name__)
        step(cfg, args)


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv=None) -> None:
    _setup_logging()
    p = argparse.ArgumentParser(prog="cargo_index")
    p.add_argument("command", choices=["refdata", "ingest", "metar", "identify", "takeoffs", "audit",
                                       "index", "charts", "run"])
    p.add_argument("--config", default="configs/mvp.yaml")
    p.add_argument("--dates", nargs="*", help="ISO dates (default: config window)")
    p.add_argument("--local", nargs="*", help="local tar parts to ingest instead of downloading")
    p.add_argument("--workers", type=int)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = Config.load(args.config)
    globals()[f"cmd_{args.command}"](cfg, args)
