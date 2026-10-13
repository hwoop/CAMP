"""
Naming of runs, construction of their ``main.py`` commands and launching of missing runs.
"""
from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from scripts.experiment.lib import layout

if TYPE_CHECKING:
    from scripts.experiment.lib.items import Context

PIPELINE_CONFIG = "configs/pipelines/test/test_ood_clip_tta.yml"
CACHE_FLAGS: tuple[str, ...] = (
    "--cache.image.mode",
    "read_write",
    "--cache.image.load_validation",
    "fast",
    "--cache.image.source_validation",
    "auto",
)
DEFAULT_NUM_WORKERS = 8

DATASET_CONFIGS: dict[str, tuple[str, ...]] = {
    "imagenet_openood": (
        "configs/datasets/imagenet/imagenet_traditional_four_ood.yml",
        "configs/datasets/imagenet/imagenet_ood.yml",
    ),
    "imagenet_standard": ("configs/datasets/imagenet/imagenet_traditional_four_ood.yml",),
    "cifar10_openood": ("configs/datasets/cifar10/cifar10_clip.yml", "configs/datasets/cifar10/cifar10_ood.yml"),
    "cifar100_openood": ("configs/datasets/cifar100/cifar100_clip.yml", "configs/datasets/cifar100/cifar100_ood.yml"),
}


class UnknownOverrideError(KeyError):
    """Raised when an override names a key that the config stack does not define."""


@dataclass(frozen=True)
class RunSpec:

    method: str
    benchmark: str
    seed: int
    backbone: str = "vitb16"
    label: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    overlays: tuple[str, ...] = ()
    pipeline: str | None = None

    def __hash__(self) -> int:

        return hash(
            (
                self.method,
                self.benchmark,
                self.seed,
                self.backbone,
                self.label,
                tuple(sorted((k, str(v)) for k, v in self.overrides.items())),
                self.overlays,
                self.pipeline,
            ),
        )


    @property
    def mark(self) -> str:
        return layout.mark(self.benchmark, self.backbone, self.label)


    def dir(
        self,
        results: Path | None = None,
    ) -> Path:
        return layout.run_dir(self.method, self.benchmark, self.seed, self.backbone, self.label, results=results)


    def recorded(
        self,
        results: Path | None = None,
    ) -> bool:
        return (self.dir(results) / "ood.csv").is_file()


    def ood_csv(
        self,
        results: Path | None = None,
    ) -> Path:
        return self.dir(results) / "ood.csv"


def dataset_configs(
    benchmark: str,
) -> tuple[str, ...]:

    try:
        return DATASET_CONFIGS[benchmark]
    except KeyError:
        raise KeyError(f"unknown benchmark {benchmark!r}; one of {', '.join(DATASET_CONFIGS)}") from None


def network_config(
    backbone: str = "vitb16",
) -> str:

    if backbone not in layout.BACKBONES:
        raise KeyError(f"unknown backbone {backbone!r}; one of {', '.join(layout.BACKBONES)}")

    if backbone == "vitb16":
        return "configs/networks/vanilla_clip.yml"

    return f"configs/networks/vanilla_clip_{backbone}.yml"


def method_config(
    method: str,
) -> str:

    if method not in layout.METHODS:
        raise KeyError(f"unknown method {method!r}; one of {', '.join(layout.METHODS)}")

    return f"configs/postprocessors/{method}/default.yml"


def config_stack(
    spec: RunSpec,
) -> tuple[str, ...]:

    pipeline = spec.pipeline or PIPELINE_CONFIG
    return (
        *dataset_configs(spec.benchmark),
        network_config(spec.backbone),
        pipeline,
        method_config(spec.method),
        *spec.overlays,
    )


def format_value(
    value: Any,
) -> str:

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, float):
        text = format(Decimal(repr(value)), "f")
        return text.rstrip("0").rstrip(".") if "." in text else text

    return str(value)


def override_args(
    overrides: Mapping[str, Any],
) -> list[str]:

    args: list[str] = []

    for key, value in overrides.items():
        args += [f"--{key}", format_value(value)]

    return args


def command(
    spec: RunSpec,
    num_workers: int = DEFAULT_NUM_WORKERS,
    python: str = "python",
    validate: bool = True,
) -> list[str]:

    if validate and spec.overrides:
        validate_overrides(spec)

    return [
        python,
        "main.py",
        "--config",
        *config_stack(spec),
        "--num_workers",
        str(num_workers),
        *CACHE_FLAGS,
        "--mark",
        spec.mark,
        "--seed",
        str(spec.seed),
        *override_args(spec.overrides),
    ]


def _openood_config_module() -> ModuleType:

    name = "scripts.experiment.lib._openood_config"
    if name in sys.modules:
        return sys.modules[name]

    path = layout.ROOT / "openood" / "utils" / "config.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def merged_config(
    spec: RunSpec,
) -> Any:

    module = _openood_config_module()
    paths = [str(layout.ROOT / rel) for rel in config_stack(spec)]
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("config file(s) not found: " + ", ".join(layout.relative(Path(p)) for p in missing))

    return module.merge_configs(*[module.Config(path) for path in paths])


def validate_overrides(
    spec: RunSpec,
) -> None:

    config = merged_config(spec)

    for key, value in spec.overrides.items():
        if key not in config:
            raise UnknownOverrideError(
                f"override key {key!r} does not exist in the config stack of {spec.method}/{spec.benchmark} "
                f"({', '.join(config_stack(spec))}); OpenOOD accepts --dotted.key only for existing keys, "
                "a new key must come from an overlay YAML appended to --config",
            )

        current = config[key]
        if current is None or isinstance(current, bool):
            continue

        try:
            type(current)(format_value(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"override {key}={value!r} cannot be read as {type(current).__name__} "
                f"(the config's current value is {current!r})",
            ) from error


def recorded(
    spec: RunSpec,
) -> bool:
    return spec.recorded()


def cuda_visible_devices(
    device: str,
) -> str | None:

    if device == "cuda":
        return None

    if device.startswith("cuda:"):
        return device.split(":", 1)[1]

    if device == "cpu":
        return ""

    raise ValueError(f"unknown device {device!r}; expected cuda, cuda:<index> or cpu")


def launch_env(
    device: str,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:

    env = dict(os.environ if base is None else base)
    visible = cuda_visible_devices(device)
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible

    return env


def run_process(
    cmd: Sequence[str],
    env: Mapping[str, str],
) -> int:
    return subprocess.run(list(cmd), cwd=str(layout.ROOT), env=dict(env)).returncode


def launch(
    specs: Iterable[RunSpec],
    ctx: Context,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> list[Path]:

    launched: list[Path] = []
    env = launch_env(ctx.device)

    for spec in specs:
        if spec.recorded():
            print(f"[skip] {layout.relative(spec.ood_csv())} exists")
            continue

        cmd = command(spec, num_workers=num_workers, python=ctx.python)
        print(f"[run] {shlex.join(cmd)}")
        launched.append(spec.dir())

        if ctx.dry_run:
            continue

        code = run_process(cmd=cmd, env=env)
        if code != 0:
            raise RuntimeError(f"run failed with exit code {code}: {layout.relative(spec.dir())}")

    return launched


def missing(
    specs: Iterable[RunSpec],
) -> list[RunSpec]:

    run_specs = [
        spec
        for spec in specs
        if not spec.recorded()
    ]

    return run_specs

