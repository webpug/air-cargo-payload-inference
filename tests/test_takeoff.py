import numpy as np
import polars as pl
import pytest

from cargo_index.takeoff import segment
from cargo_index.takeoff.runway import along_cross_m, assign_runway
from cargo_index.util.geo import bearing_deg, haversine_m

THR = (61.16, -150.0)


def _offset(lat, lon, bearing, dist_m):
    """Point dist_m from (lat, lon) along bearing (flat-earth, fine at runway scale)."""
    dlat = dist_m * np.cos(np.radians(bearing)) / 111_195
    dlon = dist_m * np.sin(np.radians(bearing)) / (111_195 * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def _runways(heading=345.0, length=3500.0):
    end = _offset(*THR, heading, length)
    rows = []
    for (a, b, ident) in ((THR, end, "33"), (end, THR, "15")):
        rows.append({"icao": "TEST", "rwy": ident, "thr_lat": a[0], "thr_lon": a[1], "end_lat": b[0], "end_lon": b[1],
                     "heading": float(bearing_deg(a[0], a[1], b[0], b[1])), "length_m": float(haversine_m(a[0], a[1], b[0], b[1])),
                     "slope": 0.0})
    return pl.DataFrame(rows)


def test_assign_runway_picks_aligned_end():
    rw = _runways()
    pts = [_offset(*THR, 345.0, d) for d in np.linspace(200, 2000, 10)]
    lat, lon = np.array(pts).T
    r = assign_runway(lat, lon, np.full(10, 346.0), rw, 10, 60)
    assert r["rwy"] == "33"
    r2 = assign_runway(lat[::-1], lon[::-1], np.full(10, 165.0), rw, 10, 60)
    assert r2["rwy"] == "15"


def test_assign_runway_rejects_parallel_offset():
    rw = _runways()
    pts = [_offset(*_offset(*THR, 75.0, 300), 345.0, d) for d in np.linspace(200, 2000, 10)]  # 300 m to the side
    lat, lon = np.array(pts).T
    assert assign_runway(lat, lon, np.full(10, 345.0), rw, 10, 60) is None


def test_along_cross():
    rw = _runways().row(0, named=True)
    p = _offset(*THR, 345.0, 1000)
    along, cross = along_cross_m(np.array([p[0]]), np.array([p[1]]), rw)
    assert along[0] == pytest.approx(1000, abs=2)
    assert abs(cross[0]) < 2


def test_fit_hinge_recovers_liftoff():
    t = np.arange(0, 20, 1.0)
    t0_true = 8.3
    v = 120 * np.maximum(0, t - t0_true) + np.random.default_rng(1).normal(0, 20, t.size)
    t0, _b, k, _rmse = segment.fit_hinge(t, v, 0, 18)
    assert t0 == pytest.approx(t0_true, abs=0.4)
    assert k == pytest.approx(120, rel=0.1)


def _synthetic_takeoff(v_lof=160.0, t_lof=40.0, accel=3.0, dt=1.0, switch_kt=90.0):
    """Roll from 30 kt at constant accel; liftoff at t_lof; vrate ramps 150 fpm/s after."""
    rows = []
    t = 0.0
    v0 = v_lof - accel * t_lof
    dist = 300.0
    while t < t_lof + 40:
        v = min(v0 + accel * t, v_lof + 15)
        gnd = v < switch_kt
        climbing = t > t_lof
        vr = 150 * (t - t_lof) if climbing else 0
        vr = min(vr, 2500)
        alt = 150 + (0.5 * 150 / 60 * (t - t_lof) ** 2 if climbing else 0)
        lat, lon = _offset(*THR, 345.0, dist)
        rows.append({"ts": 1e9 + t, "lat": lat, "lon": lon, "gs": v, "track": 345.0,
                     "baro_alt": None if gnd else int(round(alt / 25) * 25), "vrate": None if gnd else int(round(vr / 64) * 64),
                     "on_ground": gnd})
        dist += v * 0.5144 * dt
        t += dt
    return pl.DataFrame(rows, schema_overrides={"baro_alt": pl.Int32, "vrate": pl.Int32})


TAKEOFF_PARAMS = {"roll_min_gs_kt": 30, "align_tol_deg": 10, "lof_alt_ft": 50, "lof_vrate_fpm": 300,
                  "climb_top_agl_ft": 1500, "min_roll_points": 4, "max_gap_near_lof_s": 15,
                  "max_lof_from_threshold_ft": 14000, "runway_centerline_tol_m": 60}


def test_extract_liftoff_speed_from_synthetic_roll():
    pts = _synthetic_takeoff()
    to = segment.extract(pts, 150.0, _runways(), TAKEOFF_PARAMS)
    assert to.lof_method == "hinge"
    assert to.rwy["rwy"] == "33"
    # on_ground flips at 90 kt but liftoff is at 160 kt
    assert to.gs_switch == pytest.approx(90, abs=3)
    assert to.gs_lof == pytest.approx(160, abs=3)
    assert to.t_lof - 1e9 == pytest.approx(40, abs=1.0)


def test_extract_rejects_gap_at_liftoff():
    pts = _synthetic_takeoff()
    pts = pts.filter(~pl.col("ts").is_between(1e9 + 30, 1e9 + 50))
    with pytest.raises(segment.Reject):
        segment.extract(pts, 150.0, _runways(), TAKEOFF_PARAMS)
