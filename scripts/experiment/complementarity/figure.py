from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from scripts.experiment.complementarity import PLUGINS
from scripts.experiment.complementarity.analyze import (
    GROUP_HEADERS,
    GROUP_MAP,
    METRICS,
    ROLE_ALONE,
    ROLE_CAMP,
    ROLE_PLUGIN,
    TABLE_STEM,
)
from scripts.experiment.lib import plotting, tables
from scripts.experiment.lib.items import Context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

FIGURE_STEM = "complementarity"
IMPROVED = plotting.CAMP
WORSENED = plotting.FAR
PANEL_HEIGHT = 2.3
BAND_HALF_HEIGHT = 0.17


def figure_stem(
    metric: str,
) -> str:
    return FIGURE_STEM if metric == "FPR@95" else f"{FIGURE_STEM}_{metric.lower().replace('@', '')}"


def load_table(
    ctx: Context,
) -> pd.DataFrame:

    path = ctx.out / "tables" / f"{TABLE_STEM}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `python -m scripts.experiment complementarity analyze` first")

    frame = pd.read_csv(path)
    frame["label"] = frame["label"].astype(object).where(frame["label"].notna(), None)
    return frame


def value(
    frame: pd.DataFrame,
    detector: str,
    role: str,
    group: str,
    metric: str,
) -> tuple[float, float, int]:

    hit = frame[
        (frame["detector"] == detector) &
        (frame["role"] == role) &
        (frame["group"] == group) &
        (frame["metric"] == metric)
    ]
    if hit.empty:
        return math.nan, math.nan, 0

    record = hit.iloc[0]
    return float(record["mean"]), float(record["std"]), int(record["n_seeds"])


def std_band(
    ax: Axes,
    center: float,
    std: float,
    n_seeds: int,
    y: float,
    color: str,
    *,
    zorder: int,
) -> bool:

    if n_seeds <= 1 or std is None or not math.isfinite(std) or std <= 0:
        return False

    ax.fill_betweenx(
        [y - BAND_HALF_HEIGHT, y + BAND_HALF_HEIGHT],
        center - std,
        center + std,
        color=color,
        alpha=plotting.BAND_OPACITY,
        linewidth=0,
        zorder=zorder,
    )

    return True


def draw_panel(
    ax: Axes,
    frame: pd.DataFrame,
    group: str,
    metric: str,
    detectors: list[str],
) -> None:

    lower_better = metric in tables.LOWER_IS_BETTER
    positions = list(range(len(detectors)))[::-1]
    for y, detector in zip(positions, detectors):
        alone, alone_std, alone_n = value(frame=frame, detector=detector, role=ROLE_ALONE, group=group, metric=metric)
        plugin, plugin_std, plugin_n = value(frame=frame, detector=detector, role=ROLE_PLUGIN, group=group, metric=metric)
        if not math.isnan(alone) and not math.isnan(plugin):
            improved = plugin < alone if lower_better else plugin > alone
            ax.plot([alone, plugin], [y, y], color=IMPROVED if improved else WORSENED, linewidth=1.6, zorder=1)

        if not math.isnan(alone):
            std_band(ax, alone, alone_std, alone_n, y, plotting.NEUTRAL, zorder=2)
            ax.plot([alone], [y], "o", color=plotting.NEUTRAL, markersize=4.5, zorder=2)

        if not math.isnan(plugin):
            std_band(ax, plugin, plugin_std, plugin_n, y, plotting.CAMP, zorder=3)
            ax.plot([plugin], [y], "o", color=plotting.CAMP, markersize=4.5, zorder=3)

    camp, _, _ = value(frame=frame, detector="camp", role=ROLE_CAMP, group=group, metric=metric)
    if not math.isnan(camp):
        ax.axvline(camp, color=plotting.CAMP, linestyle="--", linewidth=1.0, zorder=0)

    ax.set_yticks(positions)
    ax.set_yticklabels([tables.method_name(d) for d in detectors])
    ax.set_ylim(-0.6, len(detectors) - 0.4)
    ax.margins(x=0.08)
    ax.set_title(GROUP_HEADERS[group])
    ax.set_xlabel(f"{metric} (%)" + (" ↓" if lower_better else " ↑"))
    ax.grid(True, axis="x", color=plotting.GRID, linewidth=0.6)
    ax.grid(False, axis="y")


def figure(
    ctx: Context,
) -> list[Path]:

    metric = ctx.option("metric", "FPR@95") or "FPR@95"
    if metric not in METRICS:
        raise ValueError(f"--opt metric must be one of {', '.join(METRICS)}, got {metric!r}")

    frame = load_table(ctx)
    groups = [group for group in GROUP_MAP if group in set(frame["group"])]
    if not groups:
        raise ValueError(f"{TABLE_STEM}.csv holds no column group of the complementarity table; nothing to draw")

    detectors = tables.method_order(PLUGINS)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plotting.figure_style():
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(groups),
            figsize=(plotting.FULL_WIDTH, PANEL_HEIGHT),
            sharey=True,
            squeeze=False,
        )
        for ax, group in zip(axes[0], groups):
            draw_panel(ax=ax, frame=frame, group=group, metric=metric, detectors=detectors)

        for index, ax in enumerate(axes[0]):
            plotting.panel_label(ax, f"({'abc'[index]})", x=-0.08 if index else -0.55)

        handles = [
            Line2D([], [], marker="o", linestyle="", color=plotting.NEUTRAL, markersize=4.5),
            Line2D([], [], marker="o", linestyle="", color=plotting.CAMP, markersize=4.5),
            Line2D([], [], linestyle="--", color=plotting.CAMP, linewidth=1.0),
            Line2D([], [], linestyle="-", color=IMPROVED, linewidth=1.6),
            Line2D([], [], linestyle="-", color=WORSENED, linewidth=1.6),
        ]
        labels = ["Method alone", "+ CAMP", "CAMP alone", "+ CAMP improves", "+ CAMP worsens"]
        plotting.legend_above(fig, handles, labels, ncol=len(labels), y=1.04)
        fig.tight_layout()
        return plotting.save_figure(fig, ctx.figures / figure_stem(metric))


__all__ = [
    "BAND_HALF_HEIGHT",
    "FIGURE_STEM",
    "draw_panel",
    "figure",
    "figure_stem",
    "load_table",
    "std_band",
    "value",
]
