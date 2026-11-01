"""Cached, checksummed HTTP downloads. A file is never fetched twice."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import requests

log = logging.getLogger(__name__)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "cargo-index/0.1 (research)"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def cached_download(url: str, dest: Path, params: dict | None = None, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless it already exists; record sha256 in a sidecar."""
    sidecar = dest.with_name(dest.name + ".sha256.json")
    if dest.exists() and not force:
        if not sidecar.exists():  # fetched by hand earlier; adopt it
            sidecar.write_text(json.dumps({"url": url, "params": params, "sha256": sha256_file(dest),
                                           "bytes": dest.stat().st_size}))
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.info("download %s -> %s", url, dest)
    with SESSION.get(url, params=params, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            f.writelines(r.iter_content(1 << 20))
    tmp.rename(dest)
    sidecar.write_text(json.dumps({"url": url, "params": params, "sha256": sha256_file(dest),
                                   "bytes": dest.stat().st_size}))
    return dest
