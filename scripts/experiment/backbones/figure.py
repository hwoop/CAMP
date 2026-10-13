from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment.backbones.analyze import TABLE_FILES
from scripts.experiment.lib import layout, plotting, tables
from scripts.experiment.lib.items import Context

STEM = "backbones"
BAR_WIDTH = 0.26
PANEL_HEIGHT = 2.3
SHORT_NAMES: dict[str, str] = {
    "camp": "CAMP",
}


def short_name(
    method: str,
) -> str:
    return SHORT_NAMES.get(method, tables.method_name(method))


def load_table(
    ctx: Context,
) -> pd.DataFrame:

    path = ctx.tables / f"{TABLE_FILES['FPR@95']}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{layout.relative(path)} is missing: run `backbones analyze` first")

    return pd.read_csv(path, keep_default_na=True)


def draw(
    frame: pd.DataFrame,
    stem: Path,
) -> list[Path]:

    import matplotlib.pyplot as plt

    backbones = list(dict.fromkeys(frame["backbone"]))
    methods = tables.method_order(dict.fromkeys(frame["method"]))
    groups = [group for group in plotting.GROUP_COLORS if group in set(frame["group"])]
    x = np.arange(len(methods), dtype=float)
    offsets = np.linspace(start=-BAR_WIDTH * (len(groups) - 1) / 2, stop=BAR_WIDTH * (len(groups) - 1) / 2, num=len(groups))

    with plotting.figure_style():
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(backbones),
            figsize=(plotting.FULL_WIDTH, PANEL_HEIGHT),
            sharey=True,
            squeeze=False,
        )

        handles: list[object] = []
        labels: list[str] = []

        for ax, backbone in zip(axes[0], backbones):
            panel = frame[frame["backbone"] == backbone]
            recorded = panel[panel["n_seeds"] > 0]
            ax.set_title(layout.BACKBONE_NAMES.get(backbone, backbone))
            ax.set_xticks(x)
            ax.set_xticklabels([short_name(m) for m in methods], rotation=45, ha="right", rotation_mode="anchor")
            ax.set_axisbelow(True)

            if recorded.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no run recorded",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=plotting.NEUTRAL,
                )

                continue

            for offset, group in zip(offsets, groups):
                part = recorded[recorded["group"] == group].set_index("method").reindex(methods)
                means = part["mean"].to_numpy(dtype=float)
                keep = np.isfinite(means)
                if not keep.any():
                    continue

                yerr = plotting.error_bars(
                    part["std"].to_numpy(dtype=float)[keep],
                    part["n_seeds"].fillna(0).to_numpy(dtype=int)[keep],
                )

                bars = ax.bar(
                    x[keep] + offset,
                    means[keep],
                    BAR_WIDTH,
                    color=plotting.GROUP_COLORS[group],
                    yerr=yerr,
                    error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": plotting.NEUTRAL},
                    label=plotting.GROUP_LABELS[group],
                )

                if plotting.GROUP_LABELS[group] not in labels:
                    handles.append(bars)
                    labels.append(plotting.GROUP_LABELS[group])

        axes[0][0].set_ylabel("FPR@95 (%)")

        for ax in axes[0]:
            ax.set_ylim(bottom=0)

        if handles:
            plotting.legend_above(fig, handles, labels, y=1.0)

        fig.tight_layout()
        return plotting.save_figure(fig, stem)


def figure(
    ctx: Context,
) -> list[Path]:

    frame = load_table(ctx)
    if frame.empty or not (frame["n_seeds"] > 0).any():
        print("backbones: no recorded run in the tables, nothing to draw")
        return []

    return draw(frame, ctx.figures / STEM)
