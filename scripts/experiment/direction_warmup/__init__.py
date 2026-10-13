"""
Whole-stream detection of the direction estimated from the first t samples,
compared with MCM, for the warm-up analysis.
"""
from __future__ import annotations

from pathlib import Path

from scripts.experiment.direction_warmup.analyze import analyze as _analyze
from scripts.experiment.direction_warmup.analyze import item_runs
from scripts.experiment.direction_warmup.figure import figure as _figure
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

ITEM = Item(
    name="direction_warmup",
    title="Window fill against detection: the direction from the first t samples, scored on the whole stream, vs MCM",
    benchmarks=("imagenet_standard", "imagenet_openood"),
    seeds=(0, 1, 42),
)


def runs(
    ctx: Context,
) -> list[RunSpec]:
    return item_runs(ctx)


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analyze(ctx)


def figure(
    ctx: Context,
) -> list[Path]:
    return _figure(ctx)
