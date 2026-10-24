"""Static figures for reports and the paper (matplotlib, Agg backend; PNG and PDF).

Colours follow one rule: grey for context, one accent for what the figure is about.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from ..eval.calibration import ece_binary, reliability_diagram

ACCENT, GREY, LIGHT = "#1f6fb4", "#8c8c8c", "#d9d9d9"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _despine(ax) -> None:
    for side in ("top", "right"):      # matplotlib 3.3: no list indexing of spines
        ax.spines[side].set_visible(False)


def save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    _plt().close(fig)


def reliability_figure(path: Path, curves: Mapping[str, Tuple[Sequence[float], Sequence[bool]]], *,
                       n_bins: int = 10, title: str = "") -> None:
    """Reliability diagram (fraction of positives against mean predicted probability per bin) for each curve,
    with the bin counts underneath; the last curve gets the accent colour."""
    plt = _plt()
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(4.2, 4.8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=1, zorder=0)
    names = list(curves)
    width = 1.0 / n_bins / max(1, len(names))
    for i, name in enumerate(names):
        p, y = curves[name]
        rd = reliability_diagram(p, y, n_bins)
        mask = rd.counts > 0
        colour = ACCENT if i == len(names) - 1 else GREY
        ax.plot(rd.mean_pred[mask], rd.frac_pos[mask], "o-", color=colour, ms=4, lw=1.2,
                label=f"{name} (ECE {ece_binary(p, y, n_bins):.3f})")
        centres = (rd.edges[:-1] + rd.edges[1:]) / 2 - 0.5 / n_bins + (i + 0.5) * width
        axn.bar(centres, rd.counts, width=width, color=colour)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction that are the goal")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    axn.set_xlabel("predicted P(goal)")
    axn.set_ylabel("calls")
    axn.set_yscale("log")
    for a in (ax, axn):
        _despine(a)
    save(fig, path)


def interval_bars(path: Path, rows: Sequence[Mapping[str, float]], *, label: str = "arm", value: str = "estimate",
                  lo: str = "low", hi: str = "high", accent: Sequence[str] = (), xlabel: str = "", title: str = "") -> None:
    """Horizontal point-and-whisker chart (one row per arm): the estimate with its 95% interval."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.0, 0.35 * len(rows) + 0.9))
    ys = np.arange(len(rows))[::-1]
    for yv, r in zip(ys, rows):
        c = ACCENT if r[label] in accent else GREY
        ax.plot([r[lo], r[hi]], [yv, yv], color=c, lw=1.5)
        ax.plot([r[value]], [yv], "o", color=c, ms=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([str(r[label]) for r in rows], fontsize=8)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    _despine(ax)
    save(fig, path)


def funnel_figure(path: Path, funnels: Mapping[str, Sequence[float]], stages: Sequence[str], *, accent: Sequence[str] = (),
                  title: str = "") -> None:
    """Share of episodes reaching each stage (goal in view -> detected -> verified -> success), one line per arm."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = np.arange(len(stages))
    for name, vals in funnels.items():
        c = ACCENT if name in accent else GREY
        ax.plot(x, vals, "o-", color=c, lw=1.4 if name in accent else 1.0, ms=4, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of episodes")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    _despine(ax)
    save(fig, path)


def stage_figure(path: Path, shares: Mapping[str, Mapping[str, float]], stages: Sequence[str],
                 focus: Sequence[str] = (), *, title: str = "") -> None:
    """One stacked horizontal bar per arm: the share of episodes that succeeded and that failed at each stage
    (stages partition the episodes, D-045). Stages in ``focus`` (the language->vision interface) take shades of the
    accent; the others shades of grey; success is the lightest grey."""
    plt = _plt()
    import matplotlib.colors as mcolors

    arms = list(shares)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(arms) + 1.6))
    ys = np.arange(len(arms))[::-1]
    focus_list = [s for s in stages if s in focus]
    other_list = [s for s in stages if s not in focus and s != "success"]

    def shade(base: str, i: int, n: int) -> Tuple[float, float, float]:
        rgb = np.array(mcolors.to_rgb(base))
        return tuple(rgb + (1.0 - rgb) * (0.55 * i / max(1, n - 1) if n > 1 else 0.0))

    colours = {"success": "#f0f0f0"}
    colours.update({s: shade(ACCENT, i, len(focus_list)) for i, s in enumerate(focus_list)})
    colours.update({s: shade("#5a5a5a", i, len(other_list)) for i, s in enumerate(other_list)})
    left = np.zeros(len(arms))
    for s in ["success", *stages]:
        if s == "success" and s in stages:
            continue
        vals = np.array([float(shares[a].get(s, 0.0)) for a in arms])
        if not vals.any():
            continue
        ax.barh(ys, vals, left=left, color=colours.get(s, LIGHT), edgecolor="white", lw=0.5, label=s)
        left += vals
    ax.set_yticks(ys)
    ax.set_yticklabels(arms, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of episodes")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    _despine(ax)
    save(fig, path)
