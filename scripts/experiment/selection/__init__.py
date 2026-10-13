"""
Detection performance when CAMP's window keeps only the samples admitted by the TANL selection rule.
"""
from __future__ import annotations

from pathlib import Path

from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec
from . import analyze as _analyze
from . import figure as _figure

ITEM = Item(
    name="selection",
    title="Effect of selection-free estimation: CAMP's window restricted to the samples the TANL rule admits",
    benchmarks=("imagenet_openood",),
    seeds=(0, 1, 42),
)


def runs(
    ctx: Context,
) -> list[RunSpec]:
    return _analyze.runs(ctx)


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analyze.analyze(ctx)


def figure(
    ctx: Context,
) -> list[Path]:
    return _figure.figure(ctx)
