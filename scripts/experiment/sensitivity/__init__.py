"""Sensitivity of CAMP to the window size N, the warm-up length N_min and the ridge constant eps."""
from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from scripts.experiment.lib import config_io
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

ITEM = Item(
    name="sensitivity",
    title="Sensitivity to the window size N, the warm-up length N_min and the ridge constant eps",
    benchmarks=("imagenet_standard", "imagenet_openood"),
    seeds=(0, 1, 42),
)

SPEC_PATH: Path = Path(__file__).resolve().parent / "spec.yml"
TABLE_LONG = "sensitivity.csv"
TABLE_RAW = "sensitivity_long.csv"
FIGURE_STEM = "sensitivity"
MAIN_LABEL = ""


@dataclass(frozen=True)
class Axis:

    name: str
    key: str
    title: str
    symbol: str
    default: float
    scale: str
    table: str
    panel: str

    @property
    def base(self) -> int:
        return 2 if self.scale == "log2" else 10


    def format(
        self,
        value: float,
    ) -> str:

        if self.scale == "log2":
            return f"{int(round(value)):,}"

        exponent = math.log10(value)
        if abs(exponent - round(exponent)) < 1e-9:
            return f"10^{int(round(exponent))}"

        return f"{value:g}"


@dataclass(frozen=True)
class Arm:

    method: str
    label: str | None
    axis: str | None
    value: float | None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    overlays: tuple[str, ...] = ()

    @property
    def is_main(self) -> bool:
        return self.axis is None


@dataclass(frozen=True)
class Spec:

    method: str
    benchmarks: tuple[str, ...]
    seeds: tuple[int, ...]
    groups: dict[str, tuple[str, str]]
    axes: dict[str, Axis]
    arms: tuple[Arm, ...]

    @property
    def main(self) -> Arm:
        return next(arm for arm in self.arms if arm.is_main)


    def arms_of(
        self,
        axis: str,
    ) -> list[Arm]:
        return sorted((arm for arm in self.arms if arm.axis == axis), key=lambda arm: float(arm.value or 0.0))


    def values_of(
        self,
        axis: str,
    ) -> list[float]:

        return sorted(
            {float(arm.value) for arm in self.arms_of(axis) if arm.value is not None}
            | {float(self.axes[axis].default)},
        )


    def arm_at(
        self,
        axis: str,
        value: float,
    ) -> Arm:

        if math.isclose(a=value, b=float(self.axes[axis].default), rel_tol=1e-9):
            return self.main

        for arm in self.arms_of(axis):
            if arm.value is not None and math.isclose(a=float(arm.value), b=value, rel_tol=1e-9):
                return arm

        raise KeyError(f"no arm sets {axis} to {value!r}")


def _axis(
    name: str,
    node: Mapping[str, Any],
) -> Axis:

    scale = str(node["scale"])
    if scale not in ("log2", "log10"):
        raise ValueError(f"spec.yml: axis {name!r} has scale {scale!r}; expected log2 or log10")

    return Axis(
        name=name,
        key=str(node["key"]),
        title=str(node["title"]),
        symbol=str(node["symbol"]),
        default=float(node["default"]),
        scale=scale,
        table=str(node["table"]),
        panel=str(node["panel"]),
    )


def _arm(
    node: Mapping[str, Any],
    method: str,
    axes: Mapping[str, Axis],
) -> Arm:

    label = node.get("label")
    axis = node.get("axis")
    value = node.get("value")

    if axis is not None and axis not in axes:
        raise ValueError(f"spec.yml: arm {label!r} names the unknown axis {axis!r}")

    if (axis is None) != (value is None) or (label is None) != (axis is None):
        raise ValueError(f"spec.yml: arm {label!r} must set axis and value together, and only for a labelled arm")

    overrides = dict(node.get("overrides") or {})
    if axis is not None and overrides.get(axes[axis].key) != value:
        raise ValueError(f"spec.yml: arm {label!r} must override {axes[axis].key} with its value {value!r}")

    return Arm(
        method=str(node.get("method", method)),
        label=None if label is None else str(label),
        axis=axis,
        value=None if value is None else float(value),
        overrides=overrides,
        overlays=tuple(node.get("overlays") or ()),
    )


def load_spec(
    path: Path | None = None,
) -> Spec:

    tree = config_io.load_yaml(path or SPEC_PATH)
    axes = {name: _axis(name=name, node=node) for name, node in tree["axes"].items()}
    method = str(tree.get("method", "camp"))
    arms = tuple(_arm(node=node, method=method, axes=axes) for node in tree["runs"])
    labels = [arm.label for arm in arms]
    if len(set(labels)) != len(labels):
        raise ValueError("spec.yml: duplicate arm labels")

    if sum(arm.is_main for arm in arms) != 1:
        raise ValueError("spec.yml: exactly one arm must be the main run (label null)")

    groups = {str(name): (str(pair[0]), str(pair[1])) for name, pair in tree["groups"].items()}
    return Spec(
        method=method,
        benchmarks=tuple(tree["benchmarks"]),
        seeds=tuple(int(s) for s in tree["seeds"]),
        groups=groups,
        axes=axes,
        arms=arms,
    )


def selected_axes(
    ctx: Context,
    spec: Spec,
) -> list[str]:

    chosen = ctx.option_list("axes", list(spec.axes))
    unknown = [axis for axis in chosen if axis not in spec.axes]
    if unknown:
        raise ValueError(f"unknown axes {unknown}; one of {', '.join(spec.axes)}")

    return chosen


def spec_for(
    ctx: Context,
) -> Spec:

    spec = load_spec()
    value = ctx.option("batch_size")
    if value is None:
        return spec

    try:
        batch_size = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("batch_size must be a positive integer") from error

    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    ctx.notes["batch_size"] = batch_size
    overrides = {
        "dataset.test.batch_size": batch_size,
        "ood_dataset.batch_size": batch_size,
    }
    arms = tuple(replace(
            arm,
            label=f"{arm.label or 'SEN_main'}_batch{batch_size}",
            overrides={**arm.overrides, **overrides},
        ) for arm in spec.arms)

    return replace(spec, arms=arms)


def arms_for(
    ctx: Context,
    spec: Spec | None = None,
) -> list[Arm]:

    spec = spec or spec_for(ctx)
    axes = set(selected_axes(ctx=ctx, spec=spec))
    arms = [
        arm
        for arm in spec.arms
        if arm.is_main or arm.axis in axes
    ]

    return arms


def expand(
    arms: Sequence[Arm],
    benchmarks: Sequence[str],
    seeds: Sequence[int],
    backbone: str = "vitb16",
) -> list[RunSpec]:

    run_specs = [
        RunSpec(
            arm.method,
            benchmark,
            seed,
            backbone=backbone,
            label=arm.label,
            overrides=dict(arm.overrides),
            overlays=arm.overlays,
        )
        for arm in arms
        for benchmark in benchmarks
        for seed in seeds
    ]

    return run_specs


def runs(
    ctx: Context,
) -> list[RunSpec]:
    return expand(arms_for(ctx), ctx.benchmarks, ctx.seeds, ctx.backbone)


def _stage_module(
    name: str,
) -> ModuleType:

    function = globals()[name]
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = function
    return module


def analyze(
    ctx: Context,
) -> list[Path]:
    return _stage_module("analyze").analyze(ctx)


def figure(
    ctx: Context,
) -> list[Path]:
    return _stage_module("figure").figure(ctx)
