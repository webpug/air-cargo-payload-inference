"""Difference-in-differences around event dates, with airframe-cluster bootstrap.

Weekly panel ``y[g, w]`` for groups g in {treatment lanes, control lanes}:

    y[g, w] = alpha_g + delta_w + beta * treated_g * post_w + e[g, w]

Week fixed effects ``delta_w`` absorb shocks common to all lanes (incl. the
seasonality shared by treatment and control). ``beta`` is the DiD estimate.
Uncertainty: resample *airframes* with replacement (their departures are
correlated), rebuild the weekly panel, re-estimate beta.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import numpy as np
import polars as pl


def did_beta(panel: pl.DataFrame, y: str = "y") -> float:
    """OLS beta from a (group, week, treated, post, y) panel via two-way demeaning-free dummies."""
    p = panel.drop_nulls(y)
    groups = sorted(p["group"].unique())
    weeks = sorted(p["week"].unique())
    gi = {g: i for i, g in enumerate(groups)}
    wi = {w: i for i, w in enumerate(weeks)}
    n = p.height
    X = np.zeros((n, 1 + len(groups) + len(weeks) - 1))
    X[:, 0] = (p["treated"] & p["post"]).cast(pl.Float64).to_numpy()
    for r, (g, w) in enumerate(zip(p["group"], p["week"])):
        X[r, 1 + gi[g]] = 1.0
        if wi[w] > 0:
            X[r, len(groups) + wi[w]] = 1.0
    coef, *_ = np.linalg.lstsq(X, p[y].to_numpy(), rcond=None)
    return float(coef[0])


def build_panel(deps: pl.DataFrame, agg: pl.Expr, event: dt.date, window_weeks: int) -> pl.DataFrame:
    """``deps``: one row per departure with ``group``, ``treated``, ``week`` (date) columns."""
    lo, hi = event - dt.timedelta(weeks=window_weeks), event + dt.timedelta(weeks=window_weeks)
    d = deps.filter(pl.col("week").is_between(lo, hi))
    return d.group_by("group", "week").agg(y=agg, treated=pl.col("treated").first()).with_columns(
        post=pl.col("week") >= event)


def did_bootstrap(deps: pl.DataFrame, agg: pl.Expr, event: dt.date, window_weeks: int = 12,
                  reps: int = 1000, seed: int = 0, cluster: str = "hex") -> dict:
    beta = did_beta(build_panel(deps, agg, event, window_weeks))
    rng = np.random.default_rng(seed)
    ids = deps[cluster].unique().to_numpy()
    draws = []
    for _ in range(reps):
        pick = pl.DataFrame({cluster: rng.choice(ids, len(ids), replace=True)}).with_row_index("_draw")
        bs = pick.join(deps, on=cluster, how="inner")
        try:
            draws.append(did_beta(build_panel(bs, agg, event, window_weeks)))
        except Exception:  # noqa: BLE001, S112 - degenerate resample (e.g. empty group)
            continue
    draws = np.array(draws)
    return {"event": event, "beta": beta, "ci_lo": float(np.quantile(draws, 0.025)),
            "ci_hi": float(np.quantile(draws, 0.975)), "se": float(draws.std(ddof=1)), "reps": len(draws)}


AGGS: dict[str, Callable[[], pl.Expr]] = {
    "weight_index": lambda: pl.col("m_idx").median(),
    "departures": lambda: pl.len().cast(pl.Float64),
    "capacity_mtow_t": lambda: pl.col("mtow_kg").sum() / 1000,
}
