"""Runway assignment for a takeoff roll."""

from __future__ import annotations

import numpy as np
import polars as pl

from cargo_index.util.geo import angle_diff_deg, local_xy_m


def along_cross_m(lat, lon, rwy: dict) -> tuple[np.ndarray, np.ndarray]:
    """Along-track distance from threshold and signed cross-track distance (m) for points vs a runway end."""
    x, y = local_xy_m(lat, lon, rwy["thr_lat"], rwy["thr_lon"])
    h = np.radians(rwy["heading"])
    ux, uy = np.sin(h), np.cos(h)
    return x * ux + y * uy, x * uy - y * ux


def assign_runway(lat, lon, track, runways: pl.DataFrame, align_tol_deg: float, centerline_tol_m: float) -> dict | None:
    """Pick the runway end whose heading matches the roll track and whose centerline the roll follows.

    ``runways`` holds the candidate ends at the origin airport. Returns the runway
    row (as dict) plus ``median_cross_m``, or None if nothing fits.
    """
    if len(lat) == 0:
        return None
    trk = np.nanmedian(track)
    best, best_cross = None, np.inf
    for rwy in runways.iter_rows(named=True):
        if abs(angle_diff_deg(trk, rwy["heading"])) > align_tol_deg:
            continue
        along, cross = along_cross_m(lat, lon, rwy)
        med = float(np.median(np.abs(cross)))
        inside = (along > -200) & (along < rwy["length_m"] + 200)
        if med < best_cross and inside.mean() > 0.5:
            best, best_cross = rwy, med
    if best is None or best_cross > centerline_tol_m:
        return None
    return {**best, "median_cross_m": best_cross}
