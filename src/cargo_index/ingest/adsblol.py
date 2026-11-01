"""Stream-ingest adsb.lol ``globe_history`` daily tarballs.

Each day is a ~2-4 GB tar (split into ``.aa``/``.ab`` release assets) of readsb
``traces/xx/trace_full_<hex>.json`` files. We stream the parts straight from
GitHub into ``tarfile`` in pipe mode, so the global set is never written to
disk, and keep only:

* aircraft whose readsb type is in ``adsb.capture.candidate_types`` and that
  come within ``radius_nm`` of a target airport below ``max_alt_ft``, or
* aircraft on the freighter hex whitelist (if one exists yet).

For kept aircraft we store every point below ``max_alt_ft`` (takeoff/landing
detail) and one point per ``cruise_decimate_s`` above it (enough for leg
segmentation). Output per day::

    data/raw/adsb/adsblol/date=YYYY-MM-DD/points.parquet
    data/raw/adsb/adsblol/date=YYYY-MM-DD/aircraft.parquet
    data/raw/adsb/adsblol/date=YYYY-MM-DD/manifest.json   # part sha256s, counts

readsb trace row layout: ``[dt, lat, lon, alt|"ground", gs, track, flags,
vrate, details|null, source, geom_alt, geom_rate, ias, roll]``; flags bit 1 =
stale, bit 2 = new leg.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import logging
import os
import tarfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np
import orjson
import polars as pl

from cargo_index.util.fetch import SESSION
from cargo_index.util.geo import M_PER_NM, haversine_m

log = logging.getLogger(__name__)

CRUISE_DECIMATE_S = 60.0

POINT_SCHEMA = {
    "hex": pl.Utf8, "ts": pl.Float64, "lat": pl.Float64, "lon": pl.Float64,
    "baro_alt": pl.Int32, "geom_alt": pl.Int32, "gs": pl.Float32, "track": pl.Float32,
    "vrate": pl.Int32, "on_ground": pl.Boolean, "callsign": pl.Utf8, "squawk": pl.Utf8,
    "source_type": pl.Utf8, "flags": pl.Int8,
}


class _ChainedStream(io.RawIOBase):
    """Read several byte iterators as one stream, hashing each part."""

    def __init__(self, parts: list[tuple[str, Iterator[bytes]]]):
        self._parts = iter(parts)
        self._cur: Iterator[bytes] | None = None
        self._buf = b""
        self.digests: dict[str, dict] = {}
        self._name = None
        self._hash = None
        self._n = 0

    def readable(self) -> bool:
        return True

    def _next_part(self) -> bool:
        self._finish_part()
        try:
            self._name, self._cur = next(self._parts)
        except StopIteration:
            self._cur = None
            return False
        self._hash, self._n = hashlib.sha256(), 0
        return True

    def _finish_part(self):
        if self._name is not None and self._hash is not None:
            self.digests[self._name] = {"sha256": self._hash.hexdigest(), "bytes": self._n}
            self._name = None

    def readinto(self, b) -> int:
        while not self._buf:
            if self._cur is None and not self._next_part():
                return 0
            try:
                chunk = next(self._cur)
            except StopIteration:
                self._cur = None
                continue
            self._hash.update(chunk)
            self._n += len(chunk)
            self._buf = chunk
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n

    def close(self):
        self._finish_part()
        super().close()


def release_assets(date: dt.date, repo_template: str, tag_template: str) -> list[tuple[str, str, int]]:
    """Return ``[(name, url, size)]`` for the day's tar parts, sorted. Empty if missing."""
    repo = repo_template.format(year=date.year)
    tag = tag_template.format(date=date.strftime("%Y.%m.%d"))
    # Unauthenticated GitHub API calls are capped at 60/h; a token raises that to 5,000/h.
    # Sent only to api.github.com, never to the asset CDN or other hosts.
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = SESSION.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=headers, timeout=60)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    assets = [(a["name"], a["browser_download_url"], a["size"]) for a in r.json()["assets"]
              if ".tar" in a["name"]]
    return sorted(assets)


def _http_parts(assets) -> list[tuple[str, Iterator[bytes]]]:
    def gen(url):
        with SESSION.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            yield from r.iter_content(1 << 20)
    return [(name, gen(url)) for name, url, _ in assets]


def _file_parts(paths: Iterable[Path]) -> list[tuple[str, Iterator[bytes]]]:
    def gen(p):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                yield chunk
    return [(p.name, gen(p)) for p in sorted(paths)]


def _load_trace(raw: bytes) -> dict:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return orjson.loads(raw)


class TraceFilter:
    def __init__(self, airports: pl.DataFrame, candidate_types: set[str], whitelist: set[str],
                 radius_nm: float, max_alt_ft: float):
        self.ap_lat = airports["lat"].to_numpy()
        self.ap_lon = airports["lon"].to_numpy()
        self.ap_elev = airports["elev_ft"].to_numpy()
        self.radius_m = radius_nm * M_PER_NM
        self.max_alt_ft = max_alt_ft
        self.types = candidate_types
        self.whitelist = whitelist
        # Coarse bounding boxes to reject most points before haversine
        pad = radius_nm / 60.0
        self.lat_min, self.lat_max = self.ap_lat - pad, self.ap_lat + pad
        self.lon_pad = pad / np.maximum(np.cos(np.radians(self.ap_lat)), 0.2)

    def wanted_header(self, d: dict) -> bool:
        return d.get("icao") in self.whitelist or d.get("t") in self.types

    def near_airport_low(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> bool:
        low = alt < self.max_alt_ft
        if not low.any():
            return False
        lat, lon = lat[low], lon[low]
        for i in range(len(self.ap_lat)):
            box = ((lat > self.lat_min[i]) & (lat < self.lat_max[i])
                   & (np.abs(lon - self.ap_lon[i]) < self.lon_pad[i]))
            if box.any() and (haversine_m(lat[box], lon[box], self.ap_lat[i], self.ap_lon[i])
                              < self.radius_m).any():
                return True
        return False


def parse_trace(d: dict, filt: TraceFilter) -> dict | None:
    """Turn one readsb trace dict into column lists, or None if filtered out."""
    if not filt.wanted_header(d):
        return None
    rows = d.get("trace") or []
    if not rows:
        return None
    t0 = float(d["timestamp"])
    n = len(rows)
    ts = np.fromiter((t0 + r[0] for r in rows), float, n)
    lat = np.fromiter((r[1] for r in rows), float, n)
    lon = np.fromiter((r[2] for r in rows), float, n)
    alt_raw = [r[3] for r in rows]
    on_ground = np.fromiter((a == "ground" for a in alt_raw), bool, n)
    baro = np.array([a if isinstance(a, (int, float)) else np.nan for a in alt_raw], float)
    # ground points count as low; unknown altitude counts as high
    alt_for_filter = np.where(on_ground, 0.0, np.nan_to_num(baro, nan=1e9))
    if d.get("icao") not in filt.whitelist and not filt.near_airport_low(lat, lon, alt_for_filter):
        return None

    # Decimate cruise: keep all low points, one per CRUISE_DECIMATE_S above.
    keep = alt_for_filter < filt.max_alt_ft
    bucket = np.floor(ts / CRUISE_DECIMATE_S)
    first_in_bucket = np.r_[True, bucket[1:] != bucket[:-1]]
    flags = np.fromiter(((r[6] or 0) for r in rows), int, n)
    keep |= first_in_bucket | ((flags & 2) > 0)
    keep[[0, -1]] = True

    callsign, squawk = [], []
    cs = sq = None
    for r in rows:
        det = r[8]
        if det:
            if det.get("flight"):
                cs = det["flight"].strip() or cs
            if det.get("squawk"):
                sq = det["squawk"]
        callsign.append(cs)
        squawk.append(sq)

    idx = np.nonzero(keep)[0]

    def col(i, cast=None):
        vals = [rows[j][i] if len(rows[j]) > i else None for j in idx]
        return vals

    return {
        "hex": [d["icao"]] * len(idx),
        "ts": ts[idx].tolist(),
        "lat": lat[idx].tolist(),
        "lon": lon[idx].tolist(),
        "baro_alt": [None if np.isnan(baro[j]) else int(baro[j]) for j in idx],
        "geom_alt": [int(v) if isinstance(v, (int, float)) else None for v in col(10)],
        "gs": col(4),
        "track": col(5),
        "vrate": [int(v) if isinstance(v, (int, float)) else None for v in col(7)],
        "on_ground": on_ground[idx].tolist(),
        "callsign": [callsign[j] for j in idx],
        "squawk": [squawk[j] for j in idx],
        "source_type": col(9),
        "flags": flags[idx].tolist(),
    }


def ingest_stream(stream: BinaryIO, filt: TraceFilter) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    cols = {k: [] for k in POINT_SCHEMA}
    aircraft = []
    stats = {"n_traces": 0, "n_kept": 0, "n_bad": 0}
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for m in tar:
            if not (m.isfile() and "trace_full_" in m.name):
                continue
            stats["n_traces"] += 1
            f = tar.extractfile(m)
            try:
                d = _load_trace(f.read())
            except Exception:  # noqa: BLE001 - corrupt member; count and move on
                stats["n_bad"] += 1
                continue
            out = parse_trace(d, filt)
            if out is None:
                continue
            stats["n_kept"] += 1
            for k, v in cols.items():
                v.extend(out[k])
            aircraft.append({k: d.get(k) for k in ("icao", "r", "t", "desc", "ownOp", "year", "dbFlags")})
    points = pl.DataFrame(cols, schema=POINT_SCHEMA, strict=False)
    ac = pl.DataFrame(aircraft, schema={"icao": pl.Utf8, "r": pl.Utf8, "t": pl.Utf8, "desc": pl.Utf8,
                                        "ownOp": pl.Utf8, "year": pl.Utf8, "dbFlags": pl.Int64},
                      strict=False).rename({"icao": "hex", "r": "registration", "t": "typecode"})
    return points, ac, stats


def day_dir(raw_root: Path, date: dt.date) -> Path:
    return raw_root / "adsb" / "adsblol" / f"date={date.isoformat()}"


def ingest_day(date: dt.date, raw_root: Path, filt: TraceFilter, repo_template: str,
               tag_template: str, local_parts: list[Path] | None = None, force: bool = False) -> dict:
    out = day_dir(raw_root, date)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())
    t_start = time.time()
    if local_parts:
        parts, source = _file_parts(local_parts), "local"
    else:
        assets = release_assets(date, repo_template, tag_template)
        if not assets:
            log.warning("%s: no adsb.lol release", date)
            return {"date": date.isoformat(), "status": "missing"}
        parts, source = _http_parts(assets), "github"
    stream = _ChainedStream(parts)
    points, ac, stats = ingest_stream(io.BufferedReader(stream, 1 << 22), filt)
    stream.close()
    out.mkdir(parents=True, exist_ok=True)
    points.sort(["hex", "ts"]).write_parquet(out / "points.parquet", compression="zstd")
    ac.write_parquet(out / "aircraft.parquet")
    manifest = {"date": date.isoformat(), "status": "ok", "source": source, "parts": stream.digests,
                **stats, "n_points": points.height, "runtime_s": round(time.time() - t_start, 1),
                "bytes_out": sum(p.stat().st_size for p in out.glob("*.parquet"))}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("%s: kept %d/%d aircraft, %d points in %.0fs", date, stats["n_kept"],
             stats["n_traces"], points.height, manifest["runtime_s"])
    return manifest
