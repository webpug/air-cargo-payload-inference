import datetime as dt

import numpy as np
import polars as pl
import pytest

from cargo_index.aggregate import event_study

EVENT = dt.date(2025, 5, 5)


def _synthetic_departures(effect=-0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    weeks = [EVENT + dt.timedelta(weeks=k) for k in range(-12, 12)]
    season = {w: rng.normal(0, 0.2) for w in weeks}           # common shock -> absorbed by week FE
    for group, treated in (("CN->US", True), ("CN->EU", False)):
        for a in range(40):
            airframe = f"{group}-{a}"
            af = rng.normal(0, 0.3)                            # airframe effect
            for w in weeks:
                for _ in range(2):
                    y = af + season[w] + (effect if treated and w >= EVENT else 0) + rng.normal(0, 0.3)
                    rows.append({"group": group, "treated": treated, "week": w, "hex": airframe, "m_idx": y})
    return pl.DataFrame(rows)


def test_did_beta_recovers_effect():
    deps = _synthetic_departures(effect=-0.3)
    panel = event_study.build_panel(deps, pl.col("m_idx").mean(), EVENT, 12)
    assert event_study.did_beta(panel) == pytest.approx(-0.3, abs=0.07)


def test_did_null_effect_ci_covers_zero():
    deps = _synthetic_departures(effect=0.0, seed=3)
    r = event_study.did_bootstrap(deps, pl.col("m_idx").median(), EVENT, 12, reps=100)
    assert r["ci_lo"] < 0 < r["ci_hi"]


def test_did_bootstrap_ci_excludes_zero_for_real_effect():
    deps = _synthetic_departures(effect=-0.5, seed=5)
    r = event_study.did_bootstrap(deps, pl.col("m_idx").median(), EVENT, 12, reps=100)
    assert r["ci_hi"] < 0
