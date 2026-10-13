"""
Cost of every method on the same streams, measured as end-to-end time,
postprocessor time and GPU memory.
"""
from __future__ import annotations

from pathlib import Path

from scripts.experiment.cost_methods.analyze import analyze as _analyze
from scripts.experiment.cost_methods.analyze import item_runs
from scripts.experiment.cost_methods.figure import figure as _figure
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

ITEM = Item(
    name="cost_methods",
    title="Cost of every method on the same streams: end-to-end time, postprocessor time and GPU memory",
    benchmarks=("imagenet_standard",),
    seeds=(0,),
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
