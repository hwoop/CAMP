"""
Read and write ``results/experiment/<item>/manifest.json``, 
which records what an item's outputs were computed from and when.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.experiment.lib import layout

if TYPE_CHECKING:
    from scripts.experiment.lib.items import Context

MANIFEST_NAME = "manifest.json"


def sha256_of(
    path: Path,
    chunk: int = 1 << 20,
) -> str:

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)

    return digest.hexdigest()


def iso_time(
    timestamp: float,
) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="microseconds")


def describe_input(
    path: Path,
) -> dict[str, Any]:

    path = Path(path)
    target = path / "ood.csv" if path.is_dir() else path
    entry: dict[str, Any] = {
        "path": layout.relative(path),
        "exists": target.is_file(),
    }

    if target.is_file():
        stat = target.stat()
        entry["mtime"] = iso_time(stat.st_mtime)
        entry["sha256"] = sha256_of(target)

    return entry


def git_state(
    root: Path = layout.ROOT,
) -> dict[str, Any]:

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=20,
        )

        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "dirty": None}

    if head.returncode != 0:
        return {"head": None, "dirty": None}

    return {"head": head.stdout.strip(), "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}


def read_manifest(
    out: Path,
) -> dict[str, Any] | None:

    path = Path(out) / MANIFEST_NAME
    if not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_manifest(
    ctx: Context,
    inputs: Sequence[Path],
    outputs: Sequence[Path],
    **extra: Any,
) -> Path:

    ctx.out.mkdir(parents=True, exist_ok=True)
    path = ctx.out / MANIFEST_NAME
    existing = read_manifest(ctx.out) or {}
    stage = ctx.stage or "analyze"
    notes = {
        **ctx.notes,
        **extra,
    }
    section: dict[str, Any] = {
        "stage": stage,
        "time": iso_time(datetime.now(tz=timezone.utc).timestamp()),
        "argv": list(sys.argv),
        "python": sys.executable,
        "benchmarks": list(ctx.benchmarks),
        "seeds": list(ctx.seeds),
        "backbone": ctx.backbone,
        "device": ctx.device,
        "options": dict(ctx.options),
        "inputs": [describe_input(Path(p)) for p in inputs],
        "outputs": [layout.relative(Path(p)) for p in outputs],
        "git": git_state(ctx.root),
        **notes,
    }
    stages = dict(existing.get("stages", {}))
    stages[stage] = section
    manifest: dict[str, Any] = {
        "item": ctx.item.name,
        **section,
        "stages": stages,
    }
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def stage_time(
    manifest: dict[str, Any] | None,
    stage: str,
) -> datetime | None:

    if not manifest:
        return None

    section = manifest.get("stages", {}).get(stage)
    if section is None and manifest.get("stage") == stage:
        section = manifest

    if section is None or "time" not in section:
        return None

    return datetime.fromisoformat(section["time"])
