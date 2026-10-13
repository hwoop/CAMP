"""
Main result tables for every method on every benchmark.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.experiment.lib import layout
from scripts.experiment.lib.config_io import load_yaml
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

# imported here, under the definitions: the submodules share their names with the stage functions below
from scripts.experiment.main_results import analyze as _analyze_impl  # noqa: E402  (isort: skip)
from scripts.experiment.main_results import figure as _figure_impl  # noqa: E402  (isort: skip)

ITEM_DIR: Path = Path(__file__).resolve().parent
SPEC_PATH: Path = ITEM_DIR / "spec.yml"

ITEM = Item(
    name="main_results",
    title="Main results: every method on every benchmark (FPR@95, AUROC; mean and sample std over seeds)",
    benchmarks=("imagenet_standard", "imagenet_openood", "cifar10_openood", "cifar100_openood"),
    seeds=(0, 1, 42),
)


def load_spec(
    path: Path = SPEC_PATH,
) -> dict[str, Any]:

    spec = load_yaml(path)
    if not isinstance(spec, Mapping) or not isinstance(spec.get("runs"), Sequence):
        raise ValueError(f"{path} must hold a mapping with a `runs` list")

    for entry in spec["runs"]:
        if not isinstance(entry, Mapping) or "method" not in entry:
            raise ValueError(f"{path}: every run needs a `method` key, got {entry!r}")

        if entry["method"] not in layout.METHODS:
            raise ValueError(f"{path}: unknown method {entry['method']!r}; one of {', '.join(layout.METHODS)}")

    return dict(spec)


def spec_runs(
    spec: Mapping[str, Any],
    benchmarks: Sequence[str],
    seeds: Sequence[int],
    backbone: str = "vitb16",
) -> list[RunSpec]:

    specs: list[RunSpec] = []

    for entry in spec["runs"]:
        overrides = dict(entry.get("overrides") or {})
        overlays = tuple(entry.get("overlays") or ())

        for benchmark in benchmarks:
            for seed in seeds:
                specs.append(
                    RunSpec(
                        entry["method"],
                        benchmark,
                        int(seed),
                        backbone=backbone,
                        label=entry.get("label"),
                        overrides=overrides,
                        overlays=overlays,
                    ),
                )

    return specs


def methods_of(
    ctx: Context,
) -> list[str]:

    spec_methods = [entry["method"] for entry in load_spec()["runs"]]
    wanted = ctx.option_list("methods", spec_methods)
    unknown = [m for m in wanted if m not in spec_methods]
    if unknown:
        raise ValueError(f"--opt methods names method(s) not in spec.yml: {', '.join(unknown)}")

    names = [
        m
        for m in layout.METHODS
        if m in wanted
    ]

    return names


def runs(
    ctx: Context,
) -> list[RunSpec]:

    spec = load_spec()
    methods = methods_of(ctx)
    spec = {
        **spec,
        "runs": [entry for entry in spec["runs"] if entry["method"] in methods],
    }
    return spec_runs(spec=spec, benchmarks=ctx.benchmarks, seeds=ctx.seeds, backbone=ctx.backbone)


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analyze_impl.analyze(ctx)


def figure(
    ctx: Context,
) -> list[Path]:
    return _figure_impl.figure(ctx)
