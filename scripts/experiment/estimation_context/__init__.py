"""
Confidence-weight concentration and the CIFAR-10 near-OOD shortfall,
measured from existing runs.
"""
from __future__ import annotations

from pathlib import Path

from scripts.experiment.ablation import ARMS
from scripts.experiment.estimation_context import analyze as _analysis
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

ITEM = Item(
    name="estimation_context",
    title="Confidence weight concentration and the CIFAR-10 near-OOD shortfall",
    benchmarks=("imagenet_standard", "imagenet_openood", "cifar10_openood"),
)


def runs(
    ctx: Context,
) -> list[RunSpec]:

    run_specs = [
        RunSpec(
            arm.method,
            benchmark,
            seed,
            backbone=ctx.backbone,
            label=arm.label,
            overrides=dict(arm.overrides),
            overlays=arm.overlays,
        )
        for benchmark in ctx.benchmarks
        for arm in ARMS
        if arm.label != "ABL_cov_identity" and (benchmark.startswith("imagenet_") or arm.label is None)
        for seed in ctx.seeds
    ]

    return run_specs


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analysis.analyze(ctx, runs(ctx))


def figure(
    ctx: Context,
) -> list[Path]:
    return []
