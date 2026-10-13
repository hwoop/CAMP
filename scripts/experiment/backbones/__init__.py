"""CAMP and the compared methods with other CLIP backbones."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.experiment.lib import layout
from scripts.experiment.lib.config_io import load_yaml
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

SPEC_PATH: Path = Path(__file__).resolve().parent / "spec.yml"


def load_spec(
    path: Path = SPEC_PATH,
) -> dict[str, Any]:

    spec = load_yaml(path)
    if not isinstance(spec, Mapping):
        raise ValueError(f"{path} does not hold a mapping")

    return dict(spec)


SPEC: dict[str, Any] = load_spec()

ITEM = Item(
    name="backbones",
    title="CAMP and the compared methods with other CLIP backbones (ResNet-50, ViT-B/32, ViT-L/14)",
    benchmarks=tuple(SPEC["benchmarks"]),
    seeds=tuple(int(seed) for seed in SPEC["seeds"]),
)

DEFAULT_BACKBONES: tuple[str, ...] = tuple(SPEC["backbones"])


def backbones_of(
    ctx: Context,
) -> tuple[str, ...]:

    chosen = tuple(ctx.option_list("backbones", DEFAULT_BACKBONES))
    unknown = [name for name in chosen if name not in layout.BACKBONES]
    if unknown:
        raise ValueError(f"unknown backbone(s) {', '.join(unknown)}; one of {', '.join(layout.BACKBONES)}")

    return chosen


def methods_of(
    ctx: Context,
) -> tuple[str, ...]:

    spec_methods = [str(run["method"]) for run in SPEC["runs"]]
    chosen = ctx.option_list("methods", spec_methods)
    unknown = [name for name in chosen if name not in layout.METHODS]
    if unknown:
        raise ValueError(f"unknown method(s) {', '.join(unknown)}; one of {', '.join(layout.METHODS)}")

    return tuple(method for method in layout.METHODS if method in chosen)


def expand(
    methods: Sequence[str],
    backbones: Sequence[str],
    benchmarks: Sequence[str],
    seeds: Sequence[int],
) -> list[RunSpec]:

    labels = {str(run["method"]): run.get("label") for run in SPEC["runs"]}
    run_specs = [
        RunSpec(method, benchmark, int(seed), backbone=backbone, label=labels.get(method))
        for backbone in backbones
        for method in methods
        for benchmark in benchmarks
        for seed in seeds
    ]

    return run_specs


def runs(
    ctx: Context,
) -> list[RunSpec]:
    return expand(methods_of(ctx), backbones_of(ctx), ctx.benchmarks, ctx.seeds)


# imported here, under the definitions: the submodules share their names with the stage functions below
from scripts.experiment.backbones import analyze as _analyze_module  # noqa: E402
from scripts.experiment.backbones import figure as _figure_module  # noqa: E402


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analyze_module.analyze(ctx)


def figure(
    ctx: Context,
) -> list[Path]:
    return _figure_module.figure(ctx)
