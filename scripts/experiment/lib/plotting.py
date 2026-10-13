"""
Shared matplotlib figure style (fonts, palette, figure widths) and the SVG and PNG writer.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np

from scripts.experiment.lib.svgfig import (BAND_OPACITY, CAMP, FAR, GRID, ID, IN1K, LOWER_BOUND, NEAR, NEUTRAL,
    OOD, TANL)

GROUP_COLORS: dict[str, str] = {
    "IN-1K": IN1K,
    "near": NEAR,
    "far": FAR,
}
GROUP_LABELS: dict[str, str] = {
    "IN-1K": "ImageNet-1K",
    "near": "Near-OOD",
    "far": "Far-OOD",
}

SINGLE_COLUMN = 3.3
FULL_WIDTH = 6.27

PNG_DPI = 300

RC_PARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "svg.fonttype": "none",
}


def rc_params() -> dict[str, object]:
    return dict(RC_PARAMS)


@contextmanager
def figure_style() -> Iterator[None]:

    logger = logging.getLogger("matplotlib.font_manager")
    level = logger.level
    logger.setLevel(logging.ERROR)

    try:
        with matplotlib.rc_context(rc_params()):
            yield
    finally:
        logger.setLevel(level)


def save_figure(
    fig: matplotlib.figure.Figure,
    stem: Path,
) -> list[Path]:

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=PNG_DPI)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def error_bars(
    std: Sequence[float],
    n_seeds: Sequence[int] | int,
) -> np.ndarray | None:

    counts = np.broadcast_to(array=np.asarray(n_seeds), shape=np.shape(std))
    if not np.any(counts > 1):
        return None

    err = np.where(counts > 1, np.nan_to_num(np.asarray(std, dtype=float), nan=0.0), 0.0)
    return err


def legend_above(
    fig: matplotlib.figure.Figure,
    handles: Sequence[object],
    labels: Sequence[str],
    ncol: int | None = None,
    y: float = 1.02,
) -> matplotlib.legend.Legend:

    return fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or len(labels),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )


def panel_label(
    ax: matplotlib.axes.Axes,
    text: str,
    x: float = -0.12,
    y: float = 1.04,
) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


__all__ = [
    "BAND_OPACITY",
    "CAMP",
    "FAR",
    "FULL_WIDTH",
    "GRID",
    "GROUP_COLORS",
    "GROUP_LABELS",
    "ID",
    "IN1K",
    "LOWER_BOUND",
    "NEAR",
    "NEUTRAL",
    "OOD",
    "PNG_DPI",
    "RC_PARAMS",
    "SINGLE_COLUMN",
    "TANL",
    "error_bars",
    "legend_above",
    "panel_label",
    "figure_style",
    "rc_params",
    "save_figure",
]
