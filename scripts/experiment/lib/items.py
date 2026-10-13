"""
Item interface, discovery of the ``scripts/experiment/<item>/`` packages,
stage dispatch and status.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from scripts.experiment.lib import layout, manifest, runs
from scripts.experiment.lib.runs import RunSpec

PACKAGE_NAME = "scripts.experiment"
PACKAGE_DIR: Path = layout.PACKAGE_DIR
STAGES: tuple[str, ...] = ("plan", "run", "analyze", "figure", "all", "status")
EXECUTING_STAGES: tuple[str, ...] = ("run", "analyze", "figure", "all")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 42)
ITEM_FUNCTIONS: tuple[str, ...] = ("runs", "analyze", "figure")


@dataclass(frozen=True)
class Item:

    name: str
    title: str
    benchmarks: tuple[str, ...]
    seeds: tuple[int, ...] = DEFAULT_SEEDS


@dataclass
class Context:

    root: Path
    item: Item
    out: Path
    benchmarks: tuple[str, ...]
    seeds: tuple[int, ...]
    backbone: str = "vitb16"
    device: str = "cpu"
    dry_run: bool = False
    options: dict[str, str] = field(default_factory=dict)
    python: str = sys.executable
    stage: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def _sub(
        self,
        name: str,
    ) -> Path:

        path = self.out / name
        path.mkdir(parents=True, exist_ok=True)
        return path


    @property
    def tables(self) -> Path:
        return self._sub("tables")


    @property
    def figures(self) -> Path:
        return self._sub("figures")


    @property
    def cache(self) -> Path:
        return self._sub("cache")


    def option(
        self,
        key: str,
        default: str | None = None,
    ) -> str | None:
        return self.options.get(key, default)


    def option_list(
        self,
        key: str,
        default: Sequence[str] = (),
    ) -> list[str]:

        value = self.options.get(key)
        if value is None:
            return list(default)

        names = [
            part
            for part in value.split(",")
            if part
        ]

        return names


def parse_options(
    pairs: Iterable[str] | None,
) -> dict[str, str]:

    options: dict[str, str] = {}

    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --opt {pair!r}; expected key=value")

        options[key] = value.strip() if sep else "1"

    return options


def default_device() -> str:

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def make_context(
    item: Item,
    *,
    benchmarks: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    backbone: str = "vitb16",
    device: str | None = None,
    dry_run: bool = False,
    options: Mapping[str, str] | None = None,
    out: Path | None = None,
    stage: str | None = None,
    root: Path | None = None,
) -> Context:

    for benchmark in benchmarks or ():
        if benchmark not in layout.BENCHMARKS:
            raise ValueError(f"unknown benchmark {benchmark!r}; one of {', '.join(layout.BENCHMARKS)}")

    if backbone not in layout.BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; one of {', '.join(layout.BACKBONES)}")

    return Context(
        root=root or layout.ROOT,
        item=item,
        out=Path(out) if out is not None else layout.item_dir(item.name),
        benchmarks=tuple(benchmarks) if benchmarks else tuple(item.benchmarks),
        seeds=tuple(int(s) for s in seeds) if seeds else tuple(item.seeds),
        backbone=backbone,
        device=device if device is not None else default_device(),
        dry_run=dry_run,
        options=dict(options or {}),
        stage=stage,
    )


def validate_item_module(
    module: ModuleType,
    name: str,
) -> Item:

    item = getattr(module, "ITEM", None)
    if not isinstance(item, Item):
        raise TypeError(f"{module.__name__}.ITEM is not an Item")

    if item.name != name:
        raise ValueError(f"{module.__name__}.ITEM.name is {item.name!r}, expected the directory name {name!r}")

    for function in ITEM_FUNCTIONS:
        if not callable(getattr(module, function, None)):
            raise TypeError(f"{module.__name__} lacks the function {function}(ctx)")

    return item


def item_modules() -> dict[str, ModuleType]:

    modules: dict[str, ModuleType] = {}

    for path in sorted(PACKAGE_DIR.iterdir()):
        if not path.is_dir() or path.name.startswith(("_", ".")) or not (path / "__init__.py").is_file():
            continue

        module = importlib.import_module(f"{PACKAGE_NAME}.{path.name}")
        if not hasattr(module, "ITEM"):
            continue

        validate_item_module(module=module, name=path.name)
        modules[path.name] = module

    return modules


def discover_items() -> list[Item]:

    items = [
        module.ITEM
        for module in item_modules().values()
    ]

    return items


def load_item(
    name: str,
) -> ModuleType:

    path = PACKAGE_DIR / name
    if not (path / "__init__.py").is_file():
        known = ", ".join(
            sorted(p.name for p in PACKAGE_DIR.iterdir()
                if p.is_dir() and (p / "__init__.py").is_file() and not p.name.startswith("_")
                and p.name not in ("lib", "tests")),
        )

        raise KeyError(f"unknown item {name!r}; items: {known or 'none'}")

    module = importlib.import_module(f"{PACKAGE_NAME}.{name}")
    if not hasattr(module, "ITEM"):
        raise KeyError(f"{name!r} is not an item ({PACKAGE_NAME}.{name} defines no ITEM)")

    validate_item_module(module=module, name=name)
    return module


def resolve(
    item: ModuleType | str,
) -> ModuleType:
    return load_item(item) if isinstance(item, str) else item


@dataclass(frozen=True)
class PlanRow:

    spec: RunSpec
    recorded: bool
    command: list[str]

    @property
    def dir(self) -> Path:
        return self.spec.dir()


def plan(
    item: ModuleType | str,
    ctx: Context,
) -> list[PlanRow]:

    module = resolve(item)
    rows: list[PlanRow] = []

    for spec in module.runs(ctx):
        rows.append(PlanRow(spec, spec.recorded(), runs.command(spec, python=ctx.python)))

    return rows


def run_stage(
    item: ModuleType | str,
    stage: str,
    ctx: Context,
) -> list[Path]:

    module = resolve(item)

    if stage not in EXECUTING_STAGES:
        raise ValueError(f"unknown stage {stage!r}; one of {', '.join(EXECUTING_STAGES)}")

    stages = ("run", "analyze", "figure") if stage == "all" else (stage,)
    written: list[Path] = []

    for current in stages:
        ctx.stage = current

        if current == "run":
            written += runs.launch(module.runs(ctx), ctx)
            continue

        function: Callable[[Context], list[Path]] = getattr(module, current)
        outputs = [Path(p) for p in function(ctx)]
        written += outputs

        if ctx.dry_run:
            continue

        inputs = [spec.dir() for spec in module.runs(ctx)]
        written.append(manifest.write_manifest(ctx, inputs=inputs, outputs=outputs))

    return written


@dataclass
class Status:

    item: Item
    out: Path
    recorded: list[RunSpec]
    missing: list[RunSpec]
    tables: list[Path]
    figures: list[Path]
    manifest: dict[str, Any] | None
    stale_inputs: list[Path]

    @property
    def stale(self) -> bool:
        return bool(self.tables) and (self.manifest is None or bool(self.stale_inputs))


    def lines(self) -> list[str]:

        total = len(self.recorded) + len(self.missing)
        text = [
            f"{self.item.name}: {self.item.title}",
            f"  runs: {len(self.recorded)}/{total} recorded",
        ]
        text += [f"    missing {layout.relative(spec.dir())}" for spec in self.missing]
        text.append(
            f"  tables: {len(self.tables)} in {layout.relative(self.out / 'tables')}"
            + (" (STALE)" if self.stale else ""),
        )

        text += [f"    newer than manifest: {layout.relative(p)}" for p in self.stale_inputs]
        text.append(f"  figures: {len(self.figures)} in {layout.relative(self.out / 'figures')}")
        text.append("  manifest: " + (self.manifest.get("time", "?") if self.manifest else "none"))
        return text


def _files(
    folder: Path,
) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.is_file()) if folder.is_dir() else []


def status(
    item: ModuleType | str,
    ctx: Context,
) -> Status:

    module = resolve(item)
    specs = list(module.runs(ctx))
    recorded = [spec for spec in specs if spec.recorded()]
    missing = [spec for spec in specs if not spec.recorded()]
    tables = _files(ctx.out / "tables")
    figures = _files(ctx.out / "figures")
    record = manifest.read_manifest(ctx.out)
    reference = manifest.stage_time(record, "analyze")
    if reference is None and (ctx.out / manifest.MANIFEST_NAME).is_file():
        reference = datetime.fromtimestamp((ctx.out / manifest.MANIFEST_NAME).stat().st_mtime, tz=timezone.utc)

    stale_inputs: list[Path] = []

    if reference is not None:
        for spec in recorded:
            path = spec.ood_csv()
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) > reference:
                stale_inputs.append(path)

    return Status(module.ITEM, ctx.out, recorded, missing, tables, figures, record, stale_inputs)
