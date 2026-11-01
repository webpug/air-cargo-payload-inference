"""Figures for the writeup. Each chart is exported as PNG (matplotlib) and HTML (plotly).

Colour follows the entity: each lane gets a fixed slot from ``LANE_ORDER`` so a
lane keeps its colour across every chart, whatever subset is shown.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import polars as pl

from cargo_index.config import Config

log = logging.getLogger(__name__)

# Reference categorical palette (light mode), fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Lanes shown on charts, in colour-slot order (treatment first, then controls / redeployment targets).
LANE_ORDER = ["CN_HK->US", "CN_HK->EUROPE", "CN_HK->ASIA_OTHER", "CN_HK->LATAM", "CN_HK->MIDEAST",
              "ANC->US", "EUROPE->US", "US->US"]
LANE_COLOR = dict(zip(LANE_ORDER, PALETTE))


def _style(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _events(cfg: Config, ax=None, fig=None):
    for e in cfg["events"]:
        if ax is not None:
            ax.axvline(e["date"], color=TEXT_2, linewidth=1, linestyle="--")
            ax.annotate(e["label"], (e["date"], 1), xycoords=("data", "axes fraction"), rotation=90,
                        va="top", ha="right", fontsize=8, color=TEXT_2)
        if fig is not None:
            fig.add_vline(x=str(e["date"]), line_dash="dash", line_color=TEXT_2, line_width=1)
            fig.add_annotation(x=str(e["date"]), y=1, yref="paper", text=e["label"], showarrow=False,
                               textangle=-90, xanchor="right", yanchor="top", font={"size": 10, "color": TEXT_2})


def _save(fig_mpl, fig_ply, out: Path, name: str):
    fig_mpl.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig_mpl)
    fig_ply.update_layout(template="simple_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                          font={"color": TEXT}, hovermode="x unified")
    fig_ply.write_html(out / f"{name}.html", include_plotlyjs="cdn")


def weekly_departures(cfg: Config, out: Path) -> None:
    c = pl.read_parquet(cfg.path("processed") / "index" / "weekly_lane_counts.parquet")
    w = c.filter(pl.col("lane").is_in(LANE_ORDER)).group_by("week", "lane").agg(pl.col("n_departures").sum()).sort("week")
    fig, ax = plt.subplots(figsize=(10, 5))
    ply = go.Figure()
    for lane in LANE_ORDER:
        d = w.filter(pl.col("lane") == lane)
        if not d.height:
            continue
        ax.plot(d["week"], d["n_departures"], color=LANE_COLOR[lane], linewidth=2, label=lane)
        ply.add_trace(go.Scatter(x=d["week"], y=d["n_departures"], name=lane, mode="lines",
                                 line={"color": LANE_COLOR[lane], "width": 2}))
    _style(ax, "Weekly freighter departures by lane (observed in ADS-B)", "departures / week")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2)
    _events(cfg, ax=ax, fig=ply)
    ply.update_layout(title="Weekly freighter departures by lane", yaxis_title="departures / week")
    _save(fig, ply, out, "02_weekly_departures_by_lane")


def weight_index(cfg: Config, out: Path) -> None:
    f = cfg.path("processed") / "index" / "weekly_lane_weight.parquet"
    if not f.exists():
        return
    w = pl.read_parquet(f).group_by("week", "lane").agg(
        pl.col("m_idx_median").median(), pl.col("m_idx_p25").median(), pl.col("m_idx_p75").median(), pl.col("n").sum()
    ).filter(pl.col("lane").is_in(LANE_ORDER)).sort("week")
    fig, ax = plt.subplots(figsize=(10, 5))
    ply = go.Figure()
    for lane in LANE_ORDER:
        d = w.filter(pl.col("lane") == lane)
        if not d.height:
            continue
        col = LANE_COLOR[lane]
        ax.fill_between(d["week"], d["m_idx_p25"], d["m_idx_p75"], color=col, alpha=0.15, linewidth=0)
        ax.plot(d["week"], d["m_idx_median"], color=col, linewidth=2, label=lane)
        ply.add_trace(go.Scatter(x=d["week"], y=d["m_idx_median"], name=lane, mode="lines",
                                 line={"color": col, "width": 2}, customdata=d["n"],
                                 hovertemplate="%{y:.2f} (n=%{customdata})"))
    _style(ax, "Takeoff-weight index by departure lane (median, IQR band)", "m_idx (within-type robust z)")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2)
    _events(cfg, ax=ax, fig=ply)
    ply.update_layout(title="Takeoff-weight index by lane", yaxis_title="m_idx")
    _save(fig, ply, out, "04_weight_index_by_lane")


def type_distributions(cfg: Config, out: Path) -> None:
    f = cfg.path("processed") / "takeoffs_all.parquet"
    if not f.exists():
        return
    to = pl.read_parquet(f)
    types = to.group_by("typecode").len().filter(pl.col("len") >= 10).sort("len", descending=True)["typecode"].to_list()
    fig, ax = plt.subplots(figsize=(10, 5))
    ply = go.Figure()
    for i, t in enumerate(types[: len(PALETTE)]):
        v = to.filter(pl.col("typecode") == t)["tas_lof"].drop_nulls()
        ax.hist(v, bins=30, histtype="step", linewidth=2, color=PALETTE[i], label=f"{t} (n={len(v)})")
        ply.add_trace(go.Histogram(x=v, name=t, marker_color=PALETTE[i], opacity=0.6))
    _style(ax, "Liftoff true airspeed by type", "departures")
    ax.set_xlabel("TAS at liftoff (kt)", color=TEXT_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2)
    ply.update_layout(barmode="overlay", title="Liftoff TAS by type", xaxis_title="kt")
    _save(fig, ply, out, "07_liftoff_tas_by_type")


def build_all(cfg: Config) -> None:
    out = cfg.path("figures")
    weekly_departures(cfg, out)
    weight_index(cfg, out)
    type_distributions(cfg, out)
    log.info("charts written to %s", out)
