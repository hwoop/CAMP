from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment.lib import plotting
from scripts.experiment.lib.items import Context

TABLE_NAME = "ablation.csv"
STEM = "ablation_bars"
GROUP_ORDER: tuple[str, ...] = ("IN-1K", "near", "far")
GROUP_LABELS: dict[str, str] = {
    "IN-1K": "ImageNet-1K",
    "near": "Near-OOD",
    "far": "Far-OOD",
}
VARIANT_COLORS: tuple[str, ...] = (plotting.CAMP, "#3d3d3d", "#7a7a7a", "#b5b5b5")
HATCHES: tuple[str, ...] = ("", "", "//", "..")
AXIS_LABELS: dict[str, str] = {
    "FPR@95": "FPR@95 (%)",
    "AUROC": "AUROC (%)",
}


def read_table(
    path: Path,
    metric: str,
) -> pd.DataFrame:

    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `python -m scripts.experiment ablation analyze` first")

    frame = pd.read_csv(path)
    part = frame[frame["metric"] == metric]
    if part.empty:
        raise ValueError(f"{path} holds no {metric!r} rows (metrics: {sorted(set(frame['metric']))})")

    return part.reset_index(drop=True)


def draw(
    ctx: Context,
) -> list[Path]:

    import matplotlib.pyplot as plt

    metric = ctx.option("metric", "FPR@95") or "FPR@95"
    part = read_table(ctx.tables / TABLE_NAME, metric)
    groups = [g for g in GROUP_ORDER if g in set(part["group"])]
    variants = list(dict.fromkeys(part["variant"]))
    width = 0.8 / max(len(variants), 1)
    x = np.arange(len(groups))

    with plotting.figure_style():
        fig, ax = plt.subplots(figsize=(plotting.SINGLE_COLUMN, 2.3))
        handles = []

        for index, variant in enumerate(variants):
            rows = part[part["variant"] == variant].set_index("group")
            means = np.array([rows["mean"].get(g, np.nan) for g in groups], dtype=float)
            stds = [float(rows["std"].get(g, np.nan)) for g in groups]
            counts = np.array([int(rows["n_seeds"].get(g, 0)) for g in groups])
            yerr = plotting.error_bars(stds, counts)
            if yerr is not None:
                yerr = np.where(counts > 1, yerr, np.nan)

            offset = (index - (len(variants) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                np.nan_to_num(means, nan=0.0),
                width * 0.95,
                yerr=yerr,
                capsize=1.5,
                color=VARIANT_COLORS[index % len(VARIANT_COLORS)],
                hatch=HATCHES[index % len(HATCHES)],
                edgecolor="white",
                linewidth=0.4,
                error_kw={"elinewidth": 0.7, "ecolor": plotting.NEUTRAL},
                label=variant,
            )
            for bar, mean in zip(bars, means):
                bar.set_visible(not np.isnan(mean))

            handles.append(bars)

        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS[g] for g in groups])
        ax.set_ylabel(AXIS_LABELS.get(metric, metric))
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        plotting.legend_above(fig, handles, variants, ncol=2, y=0.98)
        stem = STEM if metric == "FPR@95" else f"{STEM}_{metric.lower().replace('@', '')}"
        return plotting.save_figure(fig, ctx.figures / stem)
