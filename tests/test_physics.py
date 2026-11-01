import numpy as np
import polars as pl
import pytest

from cargo_index.model import physics
from cargo_index.takeoff import weather


def test_isa_sea_level_density():
    assert weather.air_density(1013.25, 15.0, 0.0) == pytest.approx(1.225, rel=1e-3)


def test_density_drops_with_elevation_and_heat():
    base = weather.air_density(1013.25, 15.0, 0.0)
    assert weather.air_density(1013.25, 15.0, 5000.0) < base
    assert weather.air_density(1013.25, 35.0, 0.0) < base


def test_station_pressure_at_altitude_matches_isa():
    # ISA pressure at 5,000 ft is ~843.1 hPa
    assert weather.station_pressure_hpa(1013.25, 5000.0) == pytest.approx(843.1, abs=0.5)


def test_humidity_lowers_density():
    dry = weather.air_density(1013.25, 30.0, 0.0)
    humid = weather.air_density(1013.25, 30.0, 0.0, dewpoint_c=25.0)
    assert humid < dry
    assert (dry - humid) / dry == pytest.approx(0.012, abs=0.004)


@pytest.mark.parametrize("wdir,hdg,expected", [(330, 330, 20), (150, 330, -20), (60, 330, 0), (0, 330, 20 * np.cos(np.radians(30)))])
def test_headwind_projection(wdir, hdg, expected):
    assert weather.headwind_kt(wdir, 20, hdg) == pytest.approx(expected, abs=1e-9)


def test_m_rel_formula():
    # rho * V^2 with V in m/s
    assert physics.m_rel(1.2, 100 / physics.KT_TO_MS) == pytest.approx(1.2 * 100**2)


def test_m_rel_scales_with_weight():
    # W ∝ rho V^2: 10% more weight at the same rho needs sqrt(1.1)x the speed
    v = 160.0
    assert physics.m_rel(1.2, v * np.sqrt(1.1)) / physics.m_rel(1.2, v) == pytest.approx(1.1)


def test_robust_index_per_type():
    df = pl.DataFrame({"typecode": ["A"] * 5 + ["B"] * 5, "m_rel": [1.0, 2, 3, 4, 5, 10, 20, 30, 40, 50]})
    out = physics.robust_index(df)
    assert out.filter(pl.col("m_rel") == 3)["m_idx"][0] == 0
    assert out.filter(pl.col("m_rel") == 30)["m_idx"][0] == 0
    a, b = out.filter(pl.col("m_rel") == 5)["m_idx"][0], out.filter(pl.col("m_rel") == 50)["m_idx"][0]
    assert a == pytest.approx(b)


def test_m_rel_band_brackets_value():
    lo, hi = physics.m_rel_band(1.2, 160.0, 3.0, 1.0)
    mid = physics.m_rel(1.2, 160.0)
    assert lo < mid < hi
