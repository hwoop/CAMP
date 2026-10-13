"""
ID and OOD sample counts and the OOD ratio of every test stream.
"""
from __future__ import annotations

from pathlib import Path

from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec
from scripts.experiment.stream_composition import analyze as _analyze
from scripts.experiment.stream_composition import figure as _figure
from scripts.experiment.stream_composition.analyze import cross_check_runs

ITEM = Item(
    name="stream_composition",
    title="Composition of the test streams: ID samples, OOD samples and OOD ratio per OOD dataset",
    benchmarks=("imagenet_standard", "imagenet_openood", "cifar10_openood", "cifar100_openood"),
    seeds=(0, 1, 42),
)


def runs(
    ctx: Context,
) -> list[RunSpec]:

    del ctx
    return []


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analyze.analyze(ctx)


def figure(
    ctx: Context,
) -> list[Path]:
    return _figure.figure(ctx)


__all__ = [
    "ITEM",
    "analyze",
    "cross_check_runs",
    "figure",
    "runs",
]
