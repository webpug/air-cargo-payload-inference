"""Physics-informed relative takeoff-weight proxy.

Lift at liftoff balances weight::

    W = 1/2 * rho * V_LOF^2 * S * C_L,LOF

For a fixed airframe type S is fixed, and if operators use a consistent flap
setting C_L,LOF is approximately constant, so::

    W  ∝  rho * V_LOF^2          (V_LOF = true airspeed at liftoff)

This is the primary index, ``m_rel``. It is the least confounded by thrust
derate: derate lengthens the roll but does not change the speed at which the
wing lifts the aircraft. (Rotation-speed schedules do scale with weight, which
is exactly the signal we want.)

Secondary relations, contaminated by derate and used only diagnostically:

* ground roll ``s ∝ W^2 / (rho S C_L T_net)``  =>  ``W ∝ sqrt(s rho T_net)``
* initial climb gradient ``gamma ≈ (T - D) / W`` (lower when heavier *or* more derated)

Within each type, ``m_rel`` is expressed as a robust z-score against the type's
long-run distribution: ``m_idx = (m_rel - median_type) / IQR_type``.

Uncertainty: liftoff is only located to within the ADS-B sample interval, so
:func:`m_rel_band` reports m_rel with liftoff shifted by ±half the interval.
"""

from __future__ import annotations

import numpy as np
import polars as pl

KT_TO_MS = 1852.0 / 3600.0


def m_rel(rho, tas_kt):
    """Relative weight proxy rho * V^2 (kg/m^3 * (m/s)^2 = Pa x2, i.e. twice dynamic pressure)."""
    v = np.asarray(tas_kt) * KT_TO_MS
    return np.asarray(rho) * v**2


def robust_index(df: pl.DataFrame, value: str = "m_rel", by: str = "typecode") -> pl.DataFrame:
    """Add ``m_idx`` = (value - median) / IQR within each group."""
    return df.with_columns(
        m_idx=((pl.col(value) - pl.col(value).median().over(by))
               / (pl.col(value).quantile(0.75).over(by) - pl.col(value).quantile(0.25).over(by)))
    )


def m_rel_band(rho, tas_kt, accel_kt_s, sample_interval_s):
    """(lo, hi) of m_rel if liftoff is off by ±half a sample interval.

    During the roll speed rises at ``accel_kt_s``, so a timing error dt maps to a
    speed error accel*dt. Hinge fitting usually does better than this bound.
    """
    dv = np.abs(np.asarray(accel_kt_s)) * np.asarray(sample_interval_s) / 2
    return m_rel(rho, np.asarray(tas_kt) - dv), m_rel(rho, np.asarray(tas_kt) + dv)
