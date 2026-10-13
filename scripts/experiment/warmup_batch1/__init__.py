"""
Warm-up length at batch size 1, 
measured on the whole test stream against MCM. 
"""
from __future__ import annotations

from pathlib import Path

from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec
from scripts.experiment.warmup_batch1.analyze import analyze as _analyze
from scripts.experiment.warmup_batch1.analyze import item_runs
from scripts.experiment.warmup_batch1.figure import figure as _figure

ITEM = Item(
    name="warmup_batch1",
    title="Warm-up length with one sample per batch: whole-stream FPR@95 and AUROC against MCM",
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
