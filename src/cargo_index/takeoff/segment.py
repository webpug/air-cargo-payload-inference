"""Extract the ground roll, liftoff and initial climb from a departure's points.

Liftoff detection. readsb's ``on_ground`` comes from the transponder switching
from surface to airborne position messages, which on many freighters happens on
a speed rule (~80-100 kt) well before rotation (see docs/DECISIONS.md, D3). So
``on_ground`` only bounds the roll start. Liftoff time ``t_lof`` is instead
found by fitting a hinge to the vertical rate after the air/ground switch::

    vrate(t) = b + k * max(0, t - t_lof)

over the samples up to the first with ``vrate >= HINGE_TOP_FPM``, grid-searching
``t_lof`` at 0.1 s. Liftoff speed is ground speed interpolated at ``t_lof``
from airborne velocity messages (1 kt resolution). If the hinge can't be fit,
we fall back to the first sample with ``vrate > lof_vrate_fpm`` or altitude
``> baseline + lof_alt_ft`` (``lof_method = "threshold"``, lower quality).

Barometric vertical rate lags the true value by an air-data-computer-dependent
second or two, which biases ``t_lof`` late (and ``V_LOF`` high) by a roughly
constant amount per airframe type; it largely cancels in a within-type index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from cargo_index.takeoff.runway import along_cross_m, assign_runway
from cargo_index.util.geo import M_PER_FT, angle_diff_deg, haversine_m

HINGE_TOP_FPM = 1500
HINGE_GRID_S = 0.1


class Reject(Exception):
    pass


@dataclass
class Takeoff:
    rwy: dict
    t_roll_start: float
    t_switch: float        # first airborne-message sample (transponder air/ground switch)
    t_lof: float           # estimated liftoff time
    gs_switch: float
    gs_lof: float
    gs_roll_start: float
    along_roll_start_m: float
    along_lof_m: float
    n_points_roll: int
    n_points_hinge: int
    sample_interval_s: float
    lof_gap_s: float       # gap between samples bracketing t_lof
    lof_method: str
    hinge_rmse_fpm: float | None
    climb_grad_pct: float | None
    mean_vrate_fpm: float | None
    climb_gs_kt: float | None
    climb_time_s: float | None


def fit_hinge(t: np.ndarray, v: np.ndarray, t_lo: float, t_hi: float, step: float = HINGE_GRID_S):
    """Least-squares hinge v = b + k*max(0, t - t0) over t0 in [t_lo, t_hi]. Returns (t0, b, k, rmse)."""
    best = (None, None, None, np.inf)
    for t0 in np.arange(t_lo, t_hi + step / 2, step):
        x = np.maximum(0.0, t - t0)
        A = np.c_[np.ones_like(x), x]
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        if coef[1] <= 0:
            continue
        rmse = float(np.sqrt(np.mean((A @ coef - v) ** 2)))
        if rmse < best[3]:
            best = (float(t0), float(coef[0]), float(coef[1]), rmse)
    return best


def extract(pts: pl.DataFrame, field_elev_ft: float, runways: pl.DataFrame, p: dict) -> Takeoff:
    """``pts``: one departure's points (sorted), from before the roll to past the initial climb."""
    ts = pts["ts"].to_numpy()
    lat, lon = pts["lat"].to_numpy(), pts["lon"].to_numpy()
    gs = pts["gs"].to_numpy().astype(float)
    trk = pts["track"].to_numpy().astype(float)
    alt = pts["baro_alt"].to_numpy().astype(float)
    vr = pts["vrate"].to_numpy().astype(float)
    gnd = pts["on_ground"].to_numpy()
    n = len(ts)

    # --- first airborne-message sample after the last ground run (or first fast sample)
    if gnd.any():
        sw = int(np.nonzero(gnd)[0][-1]) + 1
        if sw >= n:
            raise Reject("no_airborne_after_ground")
    else:
        fast = np.nonzero(np.nan_to_num(gs) >= p["roll_min_gs_kt"])[0]
        if not len(fast):
            raise Reject("no_roll")
        sw = int(fast[0])

    # baseline pressure altitude on the runway (airborne-mode samples before any climb)
    pre = np.arange(sw, n)
    base_alt = np.nanmin(alt[pre][: max(3, len(pre) // 10)]) if np.isfinite(alt[pre]).any() else np.nan
    if not np.isfinite(base_alt):
        raise Reject("no_altitude")
    if base_alt > field_elev_ft + 3000 or base_alt < field_elev_ft - 3000:
        raise Reject("not_departure_window")

    # --- hinge window: from switch until vrate >= HINGE_TOP_FPM (or alt well above baseline)
    top = np.nonzero((np.arange(n) >= sw) & ((np.nan_to_num(vr) >= HINGE_TOP_FPM) | (alt > base_alt + 400)))[0]
    if not len(top):
        raise Reject("no_climb")
    e = int(top[0])
    win = np.arange(max(sw - 0, 0), e + 1)
    win = win[np.isfinite(vr[win])]
    lof_method = "hinge"
    t0 = rmse = None
    if len(win) >= 4 and ts[win[-1]] - ts[win[0]] > 3:
        t0, _b, _k, rmse = fit_hinge(ts[win], vr[win], ts[win[0]], ts[win[-1]] - 1.0)
    if t0 is None:
        lof_method = "threshold"
        c = np.nonzero((np.arange(n) >= sw) & ((np.nan_to_num(vr) > p["lof_vrate_fpm"])
                                                | (alt > base_alt + p["lof_alt_ft"])))[0]
        if not len(c):
            raise Reject("no_liftoff")
        t0 = float(ts[c[0]])

    # bracket samples around t0 and gap check
    j = int(np.searchsorted(ts, t0))
    j = min(max(j, 1), n - 1)
    lof_gap = ts[j] - ts[j - 1]
    if lof_gap > p["max_gap_near_lof_s"]:
        raise Reject("gap_at_liftoff")
    ok = np.isfinite(gs)
    if ok.sum() < 2:
        raise Reject("missing_speed")
    gs_lof = float(np.interp(t0, ts[ok], gs[ok]))

    # --- roll: contiguous fast samples ending at liftoff
    k_ = j - 1
    roll = []
    while k_ >= 0 and np.isfinite(gs[k_]) and gs[k_] >= p["roll_min_gs_kt"]:
        if roll and ts[roll[-1]] - ts[k_] > p["max_gap_near_lof_s"]:
            break
        roll.append(k_)
        k_ -= 1
    roll = np.array(roll[::-1])
    if len(roll) < p["min_roll_points"]:
        raise Reject("few_roll_points")

    rwy = assign_runway(lat[roll], lon[roll], trk[roll], runways, p["align_tol_deg"], p["runway_centerline_tol_m"])
    if rwy is None:
        raise Reject("runway_unidentified")
    tr = trk[roll][np.isfinite(trk[roll])]
    if np.any(np.abs(angle_diff_deg(tr, rwy["heading"])) > p["align_tol_deg"]):
        raise Reject("track_deviates_on_roll")

    lat_lof = np.interp(t0, ts, lat)
    lon_lof = np.interp(t0, ts, lon)
    along, _ = along_cross_m(np.r_[lat[roll[0]], lat_lof], np.r_[lon[roll[0]], lon_lof], rwy)
    if along[1] / M_PER_FT > p["max_lof_from_threshold_ft"] or along[1] < 0:
        raise Reject("implausible_liftoff_position")

    # --- initial climb to field + climb_top
    top2 = np.nonzero((ts > t0) & (alt >= base_alt + p["climb_top_agl_ft"]))[0]
    climb_grad = vrate = climb_gs = climb_t = None
    if len(top2):
        e2 = int(top2[0])
        dist = haversine_m(lat_lof, lon_lof, lat[e2], lon[e2])
        dalt = (alt[e2] - base_alt) * M_PER_FT
        climb_t = float(ts[e2] - t0)
        if dist > 0 and climb_t > 0:
            climb_grad = float(100 * dalt / dist)
            vrate = float(dalt / M_PER_FT / (climb_t / 60))
            climb_gs = float(np.nanmean(gs[j:e2 + 1]))

    dts = np.diff(ts[roll])
    return Takeoff(
        rwy=rwy, t_roll_start=float(ts[roll[0]]), t_switch=float(ts[sw]), t_lof=float(t0),
        gs_switch=float(gs[sw]) if np.isfinite(gs[sw]) else float("nan"), gs_lof=gs_lof,
        gs_roll_start=float(gs[roll[0]]), along_roll_start_m=float(along[0]), along_lof_m=float(along[1]),
        n_points_roll=len(roll), n_points_hinge=len(win), sample_interval_s=float(np.median(dts)) if len(dts) else float("nan"),
        lof_gap_s=float(lof_gap), lof_method=lof_method, hinge_rmse_fpm=rmse, climb_grad_pct=climb_grad,
        mean_vrate_fpm=vrate, climb_gs_kt=climb_gs, climb_time_s=climb_t,
    )


def quality(t: Takeoff) -> float:
    """0-1 score: penalise wide liftoff brackets, sparse rolls/hinge windows and threshold fallback."""
    q = np.clip(1 - (t.lof_gap_s - 2) / 13, 0, 1) * min(1.0, t.n_points_roll / 8) * min(1.0, t.n_points_hinge / 6)
    if t.lof_method != "hinge":
        q *= 0.5
    return float(q)
