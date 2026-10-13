"""Robustness of CAMP to the OOD ratio of the stream."""
from __future__ import annotations

from pathlib import Path

from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec
from scripts.experiment.ood_ratio.analyze import analyze as _write_tables
from scripts.experiment.ood_ratio.analyze import expand_runs
from scripts.experiment.ood_ratio.figure import figure as _draw_figure

ITEM = Item(
    name="ood_ratio",
    title="Robustness to the OOD ratio of the stream (CAMP vs MCM, 1 % to 99 % OOD)",
    benchmarks=("imagenet_standard", "imagenet_openood"),
    seeds=(0, 1, 42),
)


def runs(
    ctx: Context,
) -> list[RunSpec]:
    return expand_runs(ctx)


def analyze(
    ctx: Context,
) -> list[Path]:
    return _write_tables(ctx)


def figure(
    ctx: Context,
) -> list[Path]:
    return _draw_figure(ctx)
