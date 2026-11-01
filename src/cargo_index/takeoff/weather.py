"""Atmosphere helpers: ISA density from METAR, headwind projection."""

from __future__ import annotations

import numpy as np

R_DRY = 287.05287          # J/(kg K)
G0 = 9.80665
ISA_T0 = 288.15            # K
ISA_P0 = 1013.25           # hPa
ISA_LAPSE = 0.0065         # K/m
ISA_RHO0 = 1.225           # kg/m^3
M_PER_FT = 0.3048


def station_pressure_hpa(qnh_hpa, field_elev_ft):
    """Pressure at the field from QNH (inverse of the ISA altimeter setting reduction)."""
    h = np.asarray(field_elev_ft) * M_PER_FT
    return np.asarray(qnh_hpa) * (1 - ISA_LAPSE * h / ISA_T0) ** (G0 / (R_DRY * ISA_LAPSE))


def air_density(qnh_hpa, temp_c, field_elev_ft, dewpoint_c=None):
    """Air density (kg/m^3) at the field. Includes humidity (virtual temperature) when dewpoint is given."""
    p_pa = station_pressure_hpa(qnh_hpa, field_elev_ft) * 100.0
    t_k = np.asarray(temp_c) + 273.15
    if dewpoint_c is not None:
        dp = np.asarray(dewpoint_c)
        e_hpa = 6.1078 * 10 ** (7.5 * dp / (237.3 + dp))   # vapour pressure (Tetens)
        e_pa = np.where(np.isfinite(e_hpa), e_hpa * 100.0, 0.0)
        t_k = t_k / (1 - (e_pa / p_pa) * (1 - 0.622))
    return p_pa / (R_DRY * t_k)


def headwind_kt(wind_dir_deg, wind_kt, runway_heading_deg):
    """Headwind component (positive = headwind) for wind FROM ``wind_dir_deg``."""
    return np.asarray(wind_kt) * np.cos(np.radians(np.asarray(wind_dir_deg) - np.asarray(runway_heading_deg)))
