"""Vectorised geodesy helpers (spherical earth; adequate at runway scale)."""

from __future__ import annotations

import numpy as np

EARTH_R_M = 6_371_008.8
M_PER_NM = 1852.0
M_PER_FT = 0.3048
MS_PER_KT = M_PER_NM / 3600.0


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(y, x)) % 360


def angle_diff_deg(a, b):
    """Smallest signed difference a - b in degrees, in [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def local_xy_m(lat, lon, lat0, lon0):
    """Equirectangular projection to metres east/north of (lat0, lon0)."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R_M * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R_M
    return x, y
