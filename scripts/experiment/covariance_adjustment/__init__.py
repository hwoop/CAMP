"""Distributions along the mean difference before and after the covariance adjustment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.experiment.covariance_adjustment import analyze as _analyze
from scripts.experiment.covariance_adjustment import figure as _figure
from scripts.experiment.lib.config_io import load_yaml
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

SPEC_PATH: Path = Path(__file__).with_name("spec.yml")
SPEC: dict[str, Any] = load_yaml(SPEC_PATH)

ITEM = Item(
    name="covariance_adjustment",
    title="ID and OOD distributions along the mean difference before and after the covariance adjustment",
    benchmarks=tuple(str(b) for b in SPEC["benchmarks"]),
    seeds=tuple(int(s) for s in SPEC["seeds"]),
)


def seeds_of(
    ctx: Context,
) -> tuple[int, ...]:

    chosen = ctx.option_list("seeds")
    return tuple(int(seed) for seed in chosen) if chosen else tuple(ctx.seeds)


def runs(
    ctx: Context,
) -> list[RunSpec]:

    specs: list[RunSpec] = []

    for entry in SPEC["runs"]:
        for benchmark in ctx.benchmarks:
            for seed in seeds_of(ctx):
                specs.append(
                    RunSpec(
                        method=str(entry["method"]),
                        benchmark=benchmark,
                        seed=seed,
                        backbone=ctx.backbone,
                        label=entry.get("label"),
                        overrides=dict(entry.get("overrides") or {}),
                        overlays=tuple(entry.get("overlays") or ()),
                    ),
                )

    return specs


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analyze.analyze(ctx, runs(ctx))


def figure(
    ctx: Context,
) -> list[Path]:
    return _figure.figure(ctx)
