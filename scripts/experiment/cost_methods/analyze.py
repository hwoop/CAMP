from __future__ import annotations

import functools
import gc
import io
import os
import platform
import re
import subprocess
import sys
import time
import types
import weakref
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scripts.experiment.lib import layout, runs, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

if TYPE_CHECKING:
    import torch

    from scripts.experiment.lib.replay import RunFeatures, Stream

MEBIBYTE = float(1 << 20)


def to_mb(
    size_bytes: float,
) -> float:
    return float(size_bytes) / MEBIBYTE


def synchronize(
    device: torch.device,
) -> None:

    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_info(
    device: str | torch.device,
) -> dict[str, Any]:

    import torch

    device = torch.device(device)
    info: dict[str, Any] = {
        "device": platform.processor() or platform.machine(),
        "device_type": device.type,
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "cudnn": "",
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
    }

    if device.type == "cuda":
        info["device"] = torch.cuda.get_device_name(device)
        version = torch.backends.cudnn.version()
        info["cudnn"] = str(version) if version else ""

    return info


BASE_METHOD = "camp"
PROPOSED = "camp"
FEATURES, IMAGES = "features", "images"
MODES: tuple[str, ...] = (IMAGES, FEATURES)
DEFAULT_MODES: tuple[str, ...] = (FEATURES,)
SHARED = "encoder"
DEFAULT_PASSES = 3
DEFAULT_WORKERS = 8
ENCODER_WARMUP = 10
ENCODER_BATCHES = 4
COMPARABLE_FACTOR = 2.0
BUSY_UTILIZATION = 30.0
HEADROOM = 4 << 30
TOLERANCE: dict[str, float] = {
    "FPR@95": 1.0,
    "AUROC": 0.1,
}
STATE, FIXED = "state", "fixed"
TIME_PARTS: tuple[str, ...] = (
    "ms",
    "transfer_ms",
    "image_ms",
    "image_calls",
    "text_ms",
    "text_calls",
    "wait_ms",
)
PEAK_COLUMNS: tuple[str, ...] = (
    "gpu_held_mb",
    "gpu_peak_mb",
    "gpu_peak_method_mb",
)
INDEX = re.compile(r"\[\d+\]")
NOT_HELD = (types.FunctionType, types.MethodType, types.ModuleType, type)

TABLE_SUMMARY = "cost_methods.csv"
TABLE_MD = "cost_methods.md"
TABLE_TEX = "cost_methods.tex"
TABLE_STREAMS = "cost_methods_streams.csv"
TABLE_BATCHES = "cost_methods_batches.csv"
TABLE_TENSORS = "cost_methods_tensors.csv"
TABLE_CHECK = "cost_methods_check.csv"
TABLE_ENVIRONMENT = "cost_methods_environment.csv"
TABLE_ENCODER = "cost_methods_encoder.csv"

BATCH_COLUMNS: tuple[str, ...] = (
    "method",
    "mode",
    "benchmark",
    "seed",
    "leaf",
    "pass",
    "batch",
    "start",
    "size",
    "timed",
    "ms",
    "transfer_ms",
    "image_ms",
    "image_calls",
    "text_ms",
    "text_calls",
    "wait_ms",
)
ENCODER_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "seed",
    "pass",
    "batch",
    "leaf",
    "start",
    "size",
    "timed",
    "ms",
    "transfer_ms",
    "image_ms",
    "image_calls",
)
STREAM_COLUMNS: tuple[str, ...] = (
    "method",
    "mode",
    "benchmark",
    "seed",
    "leaf",
    "n_samples",
    "n_ood",
    "batch_size",
    "window_size",
    "n_batches",
    "n_timed",
    "passes",
    "ms",
    "ms_std",
    "ms_p10",
    "ms_p90",
    "ms_pass_min",
    "ms_pass_max",
    "transfer_ms",
    "image_ms",
    "image_calls",
    "text_ms",
    "text_calls",
    "wait_ms",
    "state_mb",
    "fixed_mb",
    "host_mb",
    "gpu_held_mb",
    "gpu_baseline_mb",
    "gpu_peak_mb",
    "gpu_peak_method_mb",
    "cosine_min",
    "cosine_below_999_pct",
    "FPR@95",
    "AUROC",
)
TENSOR_COLUMNS: tuple[str, ...] = (
    "method",
    "mode",
    "benchmark",
    "seed",
    "leaf",
    "tensor",
    "dtype",
    "device",
    "kind",
    "shape",
    "count",
    "bytes",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "method",
    "mode",
    "name",
    "kind",
    "ms",
    "ms_std",
    "ms_p10",
    "ms_p90",
    "ms_pass_min",
    "ms_pass_max",
    "ms_stream_min",
    "ms_stream_max",
    "transfer_ms",
    "image_ms",
    "image_calls",
    "text_ms",
    "text_calls",
    "wait_ms",
    "state_mb",
    "fixed_mb",
    "gpu_peak_mb",
    "gpu_peak_method_mb",
    "n_timed",
    "n_streams",
    "passes",
    "batch_size",
    "window_size",
    "note",
)
PAPER_COLUMNS: tuple[str, ...] = (
    "method",
    "name",
    "kind",
    "total_ms",
    "total_std",
    "shared_ms",
    "shared_std",
    "transfer_ms",
    "encoder_ms",
    "detect_ms",
    "detect_std",
    "detect_p10",
    "detect_p90",
    "detect_run_min",
    "detect_run_max",
    "passes",
    "post_gpu_mb",
    "peak_gpu_mb",
    "peak_gpu_method_mb",
    "state_mb",
    "fixed_mb",
    "image_calls",
    "text_calls",
    "recorded_dfpr",
    "recorded_dauroc",
    "within_tolerance",
    "runs_max_dconf",
    "e2e_dfpr",
    "e2e_dauroc",
    "note",
)
CHECK_COLUMNS: tuple[str, ...] = (
    "method",
    "benchmark",
    "seed",
    "leaf",
    "variant",
    "compared_to",
    "FPR@95",
    "AUROC",
    "dFPR@95",
    "dAUROC",
    "max_abs_dconf",
    "differing_scores",
    "within_tolerance",
)


@dataclass
class Records:
    batches: list[dict[str, Any]] = field(default_factory=list)
    streams: list[dict[str, Any]] = field(default_factory=list)
    tensors: list[dict[str, Any]] = field(default_factory=list)
    encoder: list[dict[str, Any]] = field(default_factory=list)
    runs: dict[tuple[str, str, str], list[np.ndarray]] = field(default_factory=dict)
    notes: dict[tuple[str, str], str] = field(default_factory=dict)


@dataclass
class Pass:

    ms: list[float] = field(default_factory=list)
    transfer_ms: list[float] = field(default_factory=list)
    image_ms: list[float] = field(default_factory=list)
    image_calls: list[int] = field(default_factory=list)
    text_ms: list[float] = field(default_factory=list)
    text_calls: list[int] = field(default_factory=list)
    wait_ms: list[float] = field(default_factory=list)
    cosine: list[np.ndarray] = field(default_factory=list)
    scores: list[np.ndarray] = field(default_factory=list)
    snapshot: Any = None
    peak: int = 0


@dataclass(frozen=True)
class ImageSource:

    dataset: Any
    order: np.ndarray


@dataclass(frozen=True, eq=False)
class ImageBatch:

    leaf: str
    start: int
    data: torch.Tensor
    cached: torch.Tensor


def item_runs(
    ctx: Context,
) -> list[RunSpec]:

    run_specs = [
        RunSpec(method=BASE_METHOD, benchmark=benchmark, seed=seed, backbone=ctx.backbone)
        for benchmark in ctx.benchmarks
        for seed in ctx.seeds
    ]

    return run_specs


def wanted_methods(
    ctx: Context,
) -> list[str]:

    selected = ctx.option_list("methods", list(layout.METHODS))
    unknown = [method for method in selected if method not in layout.METHODS]
    if unknown:
        raise KeyError(f"unknown method(s) {unknown}; one of {list(layout.METHODS)}")

    return tables.method_order(selected)


def wanted_streams(
    ctx: Context,
    benchmark: str,
) -> list[str]:

    leaves = [leaf for members in layout.LEAVES[benchmark].values() for leaf in members]
    selected = ctx.option_list("leaves", leaves)
    unknown = [leaf for leaf in selected if leaf not in leaves]
    if unknown:
        raise KeyError(f"unknown stream(s) {unknown} for {benchmark}; one of {leaves}")

    return selected


def wanted_modes(
    ctx: Context,
) -> list[str]:

    selected = ctx.option_list("modes", list(DEFAULT_MODES))
    unknown = [mode for mode in selected if mode not in MODES]
    if unknown:
        raise KeyError(f"unknown mode(s) {unknown}; one of {list(MODES)}")

    return [mode for mode in MODES if mode in selected]


def positive_option(
    ctx: Context,
    key: str,
    default: int,
) -> int:

    value = int(ctx.option(key, str(default)))
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")

    return value


def switched_on(
    ctx: Context,
    key: str,
    default: str = "0",
) -> bool:
    return str(ctx.option(key, default)).lower() in ("1", "true", "yes")


def run_config(
    spec: RunSpec,
    *,
    image_cache: bool = True,
) -> Any:

    from openood.utils.config import Config, merge_configs

    config = merge_configs(*[Config(str(layout.ROOT / path)) for path in runs.config_stack(spec)])
    config.parse_args(
        [
            "--num_workers",
            str(runs.DEFAULT_NUM_WORKERS),
            *runs.CACHE_FLAGS,
            "--mark",
            spec.mark,
            "--seed",
            str(spec.seed),
        ],
    )

    config.parse_refs()
    config.output_dir = os.path.join(config.output_dir, config.exp_name)

    if not image_cache:
        config["cache.image.mode"] = "disabled"

    return config


class EncoderClock:

    def __init__(
        self,
        net: Any,
    ) -> None:

        self.events: dict[str, list[tuple[Any, Any]]] = {
            "image": [],
            "text": [],
        }
        self.open: dict[str, list[Any]] = {
            "image": [],
            "text": [],
        }
        self.outputs: list[Any] = []
        self.handles = []

        for kind, module in (("image", net.model.visual), ("text", net.model.transformer)):
            self.handles.append(module.register_forward_pre_hook(functools.partial(self.start, kind)))
            self.handles.append(module.register_forward_hook(functools.partial(self.stop, kind)))


    def start(
        self,
        kind: str,
        module: Any,
        args: Any,
    ) -> None:

        import torch

        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.open[kind].append(event)


    def stop(
        self,
        kind: str,
        module: Any,
        args: Any,
        output: Any,
    ) -> None:

        import torch

        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[kind].append((self.open[kind].pop(), event))

        if kind == "image":
            self.outputs.append(output.detach())


    def take(self) -> dict[str, float | int]:

        parts: dict[str, float | int] = {}

        for kind, pairs in self.events.items():
            parts[f"{kind}_ms"] = float(sum(begin.elapsed_time(end) for begin, end in pairs))
            parts[f"{kind}_calls"] = len(pairs)
            pairs.clear()

        parts["outputs"] = list(self.outputs)
        self.outputs.clear()
        return parts


    def remove(self) -> None:

        for handle in self.handles:
            handle.remove()


def start_pass(
    postprocessor: Any,
    features: RunFeatures,
    clock: EncoderClock,
) -> Any:

    import torch

    with redirect_stdout(io.StringIO()):
        postprocessor.reset_memory()
        postprocessor.setup(features.net, None, None)

    snapshot = Snapshot(postprocessor)
    synchronize(features.device)
    clock.take()
    torch.cuda.reset_peak_memory_stats(features.device)
    return snapshot


def features_pass(
    postprocessor: Any,
    features: RunFeatures,
    stream: Stream,
    clock: EncoderClock,
) -> Pass:

    import torch

    from openood.utils.vlm.image.types import CachedImageFeatures

    result = Pass(snapshot=start_pass(postprocessor=postprocessor, features=features, clock=clock))

    with torch.no_grad():
        for start, stop in stream.batches():
            batch = CachedImageFeatures(values=stream.raw[start:stop].to(features.device), normalized=True)
            synchronize(features.device)
            began = time.perf_counter()
            _, conf = postprocessor.postprocess(features.net, batch)
            synchronize(features.device)
            result.ms.append((time.perf_counter() - began) * 1e3)
            result.transfer_ms.append(0.0)
            result.wait_ms.append(0.0)
            record_parts(result=result, parts=clock.take())
            result.scores.append(conf.detach().float().cpu().numpy())

    result.peak = int(torch.cuda.max_memory_allocated(features.device))
    return result


def images_pass(
    postprocessor: Any,
    features: RunFeatures,
    stream: Stream,
    source: ImageSource,
    workers: int,
    clock: EncoderClock,
) -> Pass:

    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    from openood.utils.reproducibility import make_torch_generator, seed_worker
    from scripts.experiment.lib.replay import StreamMismatch

    loader = DataLoader(
        source.dataset,
        batch_size=stream.batch_size,
        sampler=source.order.tolist(),
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(features.run.seed),
    )

    result = Pass(snapshot=start_pass(postprocessor=postprocessor, features=features, clock=clock))
    batches = iter(loader)

    with torch.no_grad():
        for start, stop in stream.batches():
            began = time.perf_counter()
            batch = next(batches)
            result.wait_ms.append((time.perf_counter() - began) * 1e3)

            if not np.array_equal(a1=batch["label"].numpy(), a2=stream.labels[start:stop]):
                raise StreamMismatch(
                    f"stream {stream.name!r}: the image batch at {start} carries other labels than "
                    "the recorded stream",
                )

            synchronize(features.device)
            began = time.perf_counter()
            data = batch["data"].to(features.device)
            synchronize(features.device)
            copied = time.perf_counter()
            _, conf = postprocessor.postprocess(features.net, data)
            synchronize(features.device)
            result.ms.append((time.perf_counter() - began) * 1e3)
            result.transfer_ms.append((copied - began) * 1e3)
            parts = clock.take()
            record_parts(result=result, parts=parts)
            result.scores.append(conf.detach().float().cpu().numpy())

            if parts["outputs"]:
                fresh = functional.normalize(parts["outputs"][0].float(), dim=-1)
                cached = functional.normalize(stream.raw[start:stop].to(features.device).float(), dim=-1)
                result.cosine.append((fresh * cached).sum(dim=-1).cpu().numpy())

    result.peak = int(torch.cuda.max_memory_allocated(features.device))
    del batches, loader
    return result


def record_parts(
    result: Pass,
    parts: dict[str, float | int],
) -> None:

    result.image_ms.append(float(parts["image_ms"]))
    result.image_calls.append(int(parts["image_calls"]))
    result.text_ms.append(float(parts["text_ms"]))
    result.text_calls.append(int(parts["text_calls"]))


def detection(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:

    from openood.evaluators.metrics import auc_and_fpr_recall

    auroc, _, _, fpr = auc_and_fpr_recall(conf=np.asarray(scores, dtype=np.float64), label=np.asarray(labels), tpr_th=0.95)
    return float(fpr) * 100.0, float(auroc) * 100.0


def held_tensors(
    owner: Any,
) -> tuple[list[tuple[str, torch.Tensor]], int]:

    import torch

    from openood.utils.config import Config

    not_held = (torch.nn.Module, Config, *NOT_HELD)
    found: list[tuple[str, torch.Tensor]] = []
    visited: set[int] = set()
    buffers: set[int] = set()
    host = 0
    pending = [("", owner)]
    while pending:
        path, value = pending.pop()
        if id(value) in visited or isinstance(value, not_held):
            continue

        visited.add(id(value))

        if torch.is_tensor(value):
            found.append((path, value))
            continue

        if isinstance(value, np.ndarray):
            buffer = value
            while isinstance(buffer.base, np.ndarray):
                buffer = buffer.base

            if id(buffer) not in buffers:
                buffers.add(id(buffer))
                host += int(buffer.nbytes)

            continue

        host += sys.getsizeof(value)

        if isinstance(value, dict):
            children = [(f"{path}[{key!r}]", child) for key, child in value.items()]
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = [(f"{path}[{index}]", child) for index, child in enumerate(value)]
        else:
            names = [*getattr(value, "__dict__", ()), *getattr(type(value), "__slots__", ())]
            children = [(f"{path}.{name}".lstrip("."), getattr(value, name, None)) for name in names]

        pending += reversed(children)

    return found, host


class Snapshot:

    def __init__(
        self,
        owner: Any,
    ) -> None:

        tensors, _ = held_tensors(owner)
        self.before = {id(tensor): (weakref.ref(tensor), tensor._version) for _, tensor in tensors}


    def fixed(
        self,
        tensor: torch.Tensor,
    ) -> bool:

        reference, version = self.before.get(id(tensor), (None, None))
        return reference is not None and reference() is tensor and tensor._version == version


def tensor_rows(
    owner: Any,
    snapshot: Snapshot,
) -> tuple[list[dict[str, Any]], int]:

    tensors, host = held_tensors(owner)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for path, tensor in tensors:
        storage = tensor.untyped_storage()
        address = (str(tensor.device), storage.data_ptr())
        size = 0 if address in seen else int(storage.nbytes())
        seen.add(address)

        if size == 0:
            continue

        rows.append(
            {
                "tensor": INDEX.sub("", path),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "device": tensor.device.type,
                "kind": FIXED if snapshot.fixed(tensor) else STATE,
                "shape": "x".join(str(length) for length in tensor.shape) or "scalar",
                "bytes": size,
            },
        )

    if not rows:
        return [], host

    grouped = pd.DataFrame(rows).groupby(["tensor", "dtype", "device", "kind"], sort=False).agg(
        shape=("shape", "first"),
        count=("bytes", "size"),
        bytes=("bytes", "sum"),
    )

    return grouped.reset_index().to_dict("records"), host


def warm_up(
    device: torch.device,
) -> None:

    import torch

    eye = torch.eye(512, dtype=torch.float64, device=device)
    torch.linalg.solve(eye @ eye, eye[0])
    synchronize(device)


def warm_encoder(
    net: Any,
    images: torch.Tensor,
    device: torch.device,
) -> None:

    import torch

    with torch.no_grad():
        batch = images.to(device)

        for _ in range(ENCODER_WARMUP):
            net.encode_image(batch, normalize=True)

    synchronize(device)


def limit_memory(
    ctx: Context,
    device: torch.device,
) -> int:

    import torch

    index = torch.cuda.current_device() if device.index is None else device.index
    free, total = torch.cuda.mem_get_info(index)
    budget = free - HEADROOM
    wanted = ctx.option("memory_gb")
    if wanted is not None:
        budget = min(budget, int(float(wanted) * (1 << 30)))

    if budget <= 0:
        raise RuntimeError(f"no GPU memory to measure in: {to_mb(free):.0f} MB free on {device}")

    torch.cuda.set_per_process_memory_fraction(budget / total, index)
    return budget


def nvidia_smi(
    device: torch.device,
    fields: Sequence[str],
) -> dict[str, str]:

    command = [
        "nvidia-smi",
        f"--id={device.index or 0}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]

    try:
        answer = subprocess.run(command, capture_output=True, text=True, timeout=10.0, check=True)
        values = [value.strip() for value in answer.stdout.splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError, IndexError):
        return {}

    return dict(zip(fields, values)) if len(values) == len(fields) else {}


def gpu_load(
    device: torch.device,
) -> dict[str, float]:

    answer = nvidia_smi(device=device, fields=("utilization.gpu", "memory.used"))

    try:
        return {"utilization_pct": float(answer["utilization.gpu"]), "memory_used_mib": float(answer["memory.used"])}
    except (KeyError, ValueError):
        return {}


def environment(
    device: torch.device,
    net: Any,
    backbone: str,
    workers: int,
    deterministic: bool,
) -> dict[str, Any]:

    import PIL
    import torch
    import torchvision

    info = dict(device_info(device))
    gpu = nvidia_smi(
        device=device,
        fields=(
            "driver_version",
            "memory.total",
            "clocks.max.sm",
            "power.limit",
            "pcie.link.gen.current",
            "pcie.link.width.current",
        ),
    )

    cpu = ""

    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name"))
    except (OSError, StopIteration):
        pass

    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    info.update({
            "driver": gpu.get("driver_version", ""), "gpu_memory_mib": gpu.get("memory.total", ""),
            "sm_clock_max_mhz": gpu.get("clocks.max.sm", ""), "power_limit_w": gpu.get("power.limit", ""),
            "pcie": f"gen {gpu.get('pcie.link.gen.current', '?')} x{gpu.get('pcie.link.width.current', '?')}",
            "cpu": cpu or info.get("cpu", ""), "cpu_threads": os.cpu_count(), "ram_gb": round(ram / (1 << 30), 1),
            "python": platform.python_version(), "torchvision": torchvision.__version__, "pillow": PIL.__version__,
            "kernel": platform.release(), "backbone": f"CLIP {backbone}",
            "encoder_dtype": str(getattr(net.model, "dtype", "")).replace("torch.", ""),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "deterministic": deterministic,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
            "loader_workers": workers,
        })

    return info


def cache_pipeline(
    features: RunFeatures,
) -> dict[str, str]:

    from openood.utils.vlm.cache import load_cache

    if not features.cache_paths:
        return {}

    metadata = load_cache(str(features.cache_paths[0]), weights_only=True).get("metadata") or {}
    identity = metadata.get("identity") or {}
    preprocess = identity.get("preprocess") or {}
    encoder = identity.get("encoder") or {}
    return {
        "pillow": str(preprocess.get("pillow_version", "")),
        "torchvision": str(preprocess.get("torchvision_version", "")),
        "torch": str(encoder.get("torch_version", "")),
        "machine": str((encoder.get("compute_device") or {}).get("platform_machine", "")),
    }


def image_sources(
    run: Any,
    streams: Sequence[Stream],
) -> dict[str, ImageSource]:

    from torch.utils.data import ConcatDataset

    from openood.datasets import get_dataloader, get_ood_dataloader
    from scripts.experiment.lib import replay

    config = replay.replay_config(run)
    config["cache.image.mode"] = "disabled"
    id_dataset = get_dataloader(config)["test"].dataset
    ood = get_ood_dataloader(config)
    sources: dict[str, ImageSource] = {}

    for stream in streams:
        dataset = ConcatDataset([ood[stream.group][stream.leaf].dataset, id_dataset])
        if len(dataset) != len(stream):
            raise replay.StreamMismatch(
                f"stream {stream.name!r}: {len(dataset)} images for {len(stream)} recorded "
                "samples",
            )

        sources[stream.name] = ImageSource(dataset=dataset, order=stream.order)

    return sources


def first_images(
    source: ImageSource,
    batch_size: int,
    workers: int,
) -> torch.Tensor:

    from torch.utils.data import DataLoader, Subset

    loader = DataLoader(
        Subset(source.dataset, source.order[:batch_size].tolist()),
        batch_size=batch_size,
        num_workers=min(workers, 8),
    )

    return next(iter(loader))["data"]


def timed_spans(
    stream: Stream,
    window_size: int,
) -> list[tuple[int, int]]:
    return [
        (start, stop) for start, stop in stream.batches() if start >= window_size and stop - start == stream.batch_size
    ]


def preload(
    sources: dict[str, ImageSource],
    streams: Sequence[Stream],
    window_size: int,
    per_stream: int,
    workers: int,
) -> list[ImageBatch]:

    from torch.utils.data import DataLoader, Subset

    from scripts.experiment.lib.replay import StreamMismatch

    batches: list[ImageBatch] = []

    for stream in streams:
        spans = timed_spans(stream=stream, window_size=window_size)[:per_stream]
        source = sources[stream.name]
        rows = np.concatenate([source.order[start:stop] for start, stop in spans]).tolist()
        loader = DataLoader(Subset(source.dataset, rows), batch_size=stream.batch_size, num_workers=min(workers, 8))

        for (start, stop), batch in zip(spans, loader):
            if not np.array_equal(batch["label"].numpy(), stream.labels[start:stop]):
                raise StreamMismatch(
                    f"stream {stream.name!r}: the image batch at {start} carries other labels than "
                    "the recorded stream"
                )

            batches.append(
                ImageBatch(leaf=str(stream.leaf), start=start, data=batch["data"], cached=stream.raw[start:stop])
            )

    return batches


def reencoded(
    features: RunFeatures,
    batches: Sequence[ImageBatch],
) -> np.ndarray:

    import torch
    import torch.nn.functional as functional

    cosine = []

    with torch.no_grad():
        for batch in batches:
            fresh = functional.normalize(features.net.encode_image(batch.data.to(features.device)).float(), dim=-1)
            cached = functional.normalize(batch.cached.to(features.device).float(), dim=-1)
            cosine.append((fresh * cached).sum(dim=-1).cpu().numpy())

    return np.concatenate(cosine)


def run_order(
    items: Sequence[Any],
    run: int,
) -> list[Any]:

    if run == 0:
        return list(items)

    if run == 1:
        return list(reversed(items))

    return [items[index] for index in np.random.default_rng(run).permutation(len(items))]


def measure_encoder(
    run: int,
    features: RunFeatures,
    batches: Sequence[ImageBatch],
    timed: int,
    records: Records,
) -> float:

    import torch

    device = features.device
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    clock = EncoderClock(features.net)
    rows: list[dict[str, Any]] = []
    identity = {"benchmark": features.run.benchmark, "seed": features.run.seed, "pass": run}

    try:
        synchronize(device)
        clock.take()
        torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            for index in range(ENCODER_WARMUP + timed):
                batch = batches[index % len(batches)]
                synchronize(device)
                began = time.perf_counter()
                data = batch.data.to(device)
                synchronize(device)
                copied = time.perf_counter()
                features.net.encode_image(data, normalize=True)
                synchronize(device)
                ended = time.perf_counter()
                parts = clock.take()
                del data

                if index >= ENCODER_WARMUP:
                    rows.append(
                        {
                            **identity,
                            "batch": index - ENCODER_WARMUP,
                            "leaf": batch.leaf,
                            "start": batch.start,
                            "size": int(batch.data.shape[0]),
                            "timed": True,
                            "ms": (ended - began) * 1e3,
                            "transfer_ms": (copied - began) * 1e3,
                            "image_ms": float(parts["image_ms"]),
                            "image_calls": int(parts["image_calls"]),
                        }
                    )

        peak = int(torch.cuda.max_memory_allocated(device))
    finally:
        clock.remove()

    gc.collect()
    torch.cuda.empty_cache()
    records.encoder += rows
    timing = pass_timing(pd.DataFrame.from_records(rows))
    print(
        f"[cost_methods] run {run + 1}: copy + image encoder {timing['ms']:.2f} ms per batch (copy "
        f"{timing['transfer_ms']:.2f} ms, encoder {timing['image_ms']:.2f} ms GPU time), {len(rows)} timed passes over "
        f"{len(batches)} preloaded batches",
        flush=True,
    )
    return to_mb(peak - baseline)


def seed_rngs(
    seed: int,
) -> None:

    import random

    import torch

    from openood.utils.reproducibility import normalize_seed

    normalized = normalize_seed(seed)
    random.seed(normalized)
    np.random.seed(normalized % (1 << 32))
    torch.manual_seed(normalized)


def measure(
    method: str,
    mode: str,
    run: int,
    features: RunFeatures,
    streams: Sequence[Stream],
    sources: dict[str, ImageSource],
    window_size: int,
    workers: int,
    warm_images: Any,
    records: Records,
    tensors: bool = True,
) -> bool:

    import torch

    from openood.postprocessors import get_postprocessor

    recorded = features.run
    device = features.device
    identity = {"method": method, "mode": mode, "benchmark": recorded.benchmark, "seed": recorded.seed}
    config = run_config(
        RunSpec(method=method, benchmark=recorded.benchmark, seed=recorded.seed, backbone=recorded.backbone),
        image_cache=mode == FEATURES,
    )
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    own = Records()
    postprocessor = None
    clock = None
    fits = True
    seed_rngs(recorded.seed)

    try:
        postprocessor = get_postprocessor(config)
        postprocessor.setup(features.net, None, None)
        clock = EncoderClock(features.net)

        if mode == IMAGES:
            warm_encoder(net=features.net, images=warm_images, device=device)

        for stream in streams:
            row = time_stream(
                postprocessor=postprocessor,
                identity=identity,
                baseline=baseline,
                features=features,
                stream=stream,
                source=sources.get(stream.name),
                mode=mode,
                run=run,
                window_size=window_size,
                workers=workers,
                clock=clock,
                records=own,
                tensors=tensors,
            )
            print(
                f"[cost_methods] run {run + 1}: {method} ({mode}) on {stream.name}: {row['ms']:.2f} ms per batch "
                f"(median of the timed batches; image encoder {row['image_ms']:.2f} ms in {row['image_calls']:g} "
                f"pass(es)), state {row['state_mb']:.1f} MB, peak GPU {row['gpu_peak_mb']:.0f} MB, FPR@95 "
                f"{row['FPR@95']:.2f}, AUROC {row['AUROC']:.2f}",
                flush=True,
            )
    except torch.cuda.OutOfMemoryError:
        fits = False
    finally:
        if clock is not None:
            clock.remove()

    del postprocessor
    gc.collect()
    torch.cuda.empty_cache()

    if fits:
        records.batches += own.batches
        records.streams += own.streams
        records.tensors += own.tensors

        for key, scores in own.runs.items():
            records.runs.setdefault(key, []).extend(scores)

    return fits


def time_stream(
    postprocessor: Any,
    identity: dict[str, Any],
    baseline: int,
    features: RunFeatures,
    stream: Stream,
    source: ImageSource | None,
    mode: str,
    run: int,
    window_size: int,
    workers: int,
    clock: EncoderClock,
    records: Records,
    tensors: bool = True,
) -> dict[str, Any]:

    import torch

    device = features.device
    spans = list(stream.batches())
    timed = np.array([start >= window_size and stop - start == stream.batch_size for start, stop in spans])
    if not timed.any():
        raise RuntimeError(f"stream {stream.name}: no full batch is scored after {window_size} samples")

    if mode == IMAGES:
        result = images_pass(
            postprocessor=postprocessor, features=features, stream=stream, source=source, workers=workers, clock=clock
        )
    else:
        result = features_pass(postprocessor=postprocessor, features=features, stream=stream, clock=clock)

    batch_rows = [
        {
            **identity,
            "leaf": stream.leaf,
            "pass": run,
            "batch": batch,
            "start": start,
            "size": stop - start,
            "timed": bool(timed[batch]),
            "ms": result.ms[batch],
            "transfer_ms": result.transfer_ms[batch],
            "image_ms": result.image_ms[batch],
            "image_calls": result.image_calls[batch],
            "text_ms": result.text_ms[batch],
            "text_calls": result.text_calls[batch],
            "wait_ms": result.wait_ms[batch],
        }
        for batch, (start, stop) in enumerate(spans)
    ]
    records.batches += batch_rows
    gc.collect()
    held = torch.cuda.memory_allocated(device)
    attributes, host = tensor_rows(owner=postprocessor, snapshot=result.snapshot)

    if tensors:
        records.tensors += [{**identity, "leaf": stream.leaf, **row} for row in attributes]

    scores = np.concatenate(result.scores)
    records.runs.setdefault((identity["method"], identity["mode"], str(stream.leaf)), []).append(scores)

    fpr, auroc = detection(scores=scores, labels=stream.labels)
    size = {kind: sum(row["bytes"] for row in attributes if row["kind"] == kind) for kind in (STATE, FIXED)}
    cosine = np.concatenate(result.cosine) if result.cosine else np.array([np.nan])
    row = {
        **identity,
        "leaf": stream.leaf,
        "n_samples": len(stream),
        "n_ood": int(stream.is_ood.sum()),
        "batch_size": stream.batch_size,
        "window_size": window_size,
        "n_batches": len(spans),
        "n_timed": int(timed.sum()),
        **pass_timing(pd.DataFrame.from_records(batch_rows)),
        "state_mb": to_mb(size[STATE]),
        "fixed_mb": to_mb(size[FIXED]),
        "host_mb": to_mb(host),
        "gpu_held_mb": to_mb(held - baseline),
        "gpu_baseline_mb": to_mb(baseline),
        "gpu_peak_mb": to_mb(result.peak),
        "gpu_peak_method_mb": to_mb(result.peak - baseline),
        "cosine_min": float(np.min(cosine)),
        "cosine_below_999_pct": float(np.mean(cosine < 0.999) * 100.0) if result.cosine else float("nan"),
        "FPR@95": fpr,
        "AUROC": auroc,
    }
    records.streams.append(row)
    return row


def summarize(
    batches: pd.DataFrame,
    streams: pd.DataFrame,
) -> pd.DataFrame:

    records: list[dict[str, Any]] = []

    for method in tables.method_order(streams["method"]):
        for mode in MODES:
            part = streams[(streams["method"] == method) & (streams["mode"] == mode)]
            if part.empty:
                continue

            rows = batches[(batches["method"] == method) & (batches["mode"] == mode)]
            timing = pass_timing(rows)
            first_run = rows[rows["pass"] == rows["pass"].min()]
            records.append(
                {
                    "method": method,
                    "mode": mode,
                    "name": tables.method_name(method),
                    "kind": layout.METHODS[method].kind,
                    "ms": timing["ms"],
                    "ms_std": timing["ms_std"],
                    "ms_p10": timing["ms_p10"],
                    "ms_p90": timing["ms_p90"],
                    "ms_pass_min": timing["ms_pass_min"],
                    "ms_pass_max": timing["ms_pass_max"],
                    "ms_stream_min": float(part["ms"].min()),
                    "ms_stream_max": float(part["ms"].max()),
                    **{name: timing[name] for name in TIME_PARTS[1:]},
                    "state_mb": float(part["state_mb"].max()),
                    "fixed_mb": float(part["fixed_mb"].max()),
                    "gpu_peak_mb": float(part["gpu_peak_mb"].max()),
                    "gpu_peak_method_mb": float(part["gpu_peak_method_mb"].max()),
                    "n_timed": int(first_run["timed"].sum()),
                    "n_streams": int(len(part)),
                    "passes": timing["passes"],
                    "batch_size": int(part["batch_size"].iloc[0]),
                    "window_size": int(part["window_size"].iloc[0]),
                }
            )

    return pd.DataFrame.from_records(records)


def recorded_conf(
    method: str,
    benchmark: str,
    seed: int,
    backbone: str,
    leaf: str,
    labels: np.ndarray,
) -> np.ndarray | None:

    from scripts.experiment.lib import replay

    spec = RunSpec(method=method, benchmark=benchmark, seed=seed, backbone=backbone)
    if not spec.recorded():
        return None

    recorded = replay.RecordedRun(spec.dir()).recorded_scores(leaf)
    if recorded is None or not np.array_equal(a1=np.asarray(recorded["label"]).astype(np.int64), a2=labels):
        return None

    return np.asarray(recorded["conf"], dtype=np.float64)


PREPROCESS = re.compile(r"preprocess=([0-9a-f]+)")


def recorded_pipeline(
    method: str,
    benchmark: str,
    seed: int,
    backbone: str,
    replay_cache: str,
) -> str:

    import PIL
    import torch
    import torchvision

    from openood.utils.vlm.cache import load_cache
    from scripts.experiment.lib import replay

    spec = RunSpec(method=method, benchmark=benchmark, seed=seed, backbone=backbone)
    if not spec.recorded():
        return ""

    paths = list(replay.logged_cache_paths(replay.RecordedRun(spec.dir())).values())
    hashes = {match.group(1) for path in paths if (match := PREPROCESS.search(path.name))}
    wanted = PREPROCESS.search(Path(replay_cache).name)
    if wanted is not None and hashes == {wanted.group(1)}:
        return "replay"

    if not paths or not paths[0].is_file():
        return "other"

    identity = (load_cache(str(paths[0]), weights_only=True).get("metadata") or {}).get("identity") or {}
    versions = (
        str((identity.get("preprocess") or {}).get("pillow_version")),
        str((identity.get("preprocess") or {}).get("torchvision_version")),
        str((identity.get("encoder") or {}).get("torch_version")),
    )
    return "here" if versions == (PIL.__version__, torchvision.__version__, torch.__version__) else "other"


def check_rows(
    records: Records,
    streams: Sequence[Stream],
    identity: dict[str, Any],
) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    for method in tables.method_order({key[0] for key in records.runs}):
        pipeline = recorded_pipeline(
            method=method,
            benchmark=identity["benchmark"],
            seed=identity["seed"],
            backbone=identity["backbone"],
            replay_cache=identity["replay_cache"],
        )

        for stream in streams:
            leaf = str(stream.leaf)
            on_features = (records.runs.get((method, FEATURES, leaf)) or [None])[0]
            end_to_end = (records.runs.get((method, IMAGES, leaf)) or [None])[0]
            comparisons: list[tuple[str, np.ndarray, str, np.ndarray | None]] = []

            if on_features is not None:
                comparisons.append((FEATURES, on_features, "", None))

            if end_to_end is not None and on_features is not None:
                comparisons.append((IMAGES, end_to_end, FEATURES, on_features))

            recorded = recorded_conf(
                method=method,
                benchmark=identity["benchmark"],
                seed=identity["seed"],
                backbone=identity["backbone"],
                leaf=leaf,
                labels=stream.labels,
            )
            if recorded is not None and pipeline == "replay" and on_features is not None:
                comparisons.append(("recorded_run", recorded, FEATURES, on_features))
            elif recorded is not None and pipeline == "here" and end_to_end is not None:
                comparisons.append(("recorded_run", recorded, IMAGES, end_to_end))

            for variant, conf, against, base in comparisons:
                fpr, auroc = detection(scores=conf, labels=stream.labels)
                row = {
                    "method": method,
                    "benchmark": identity["benchmark"],
                    "seed": identity["seed"],
                    "leaf": leaf,
                    "variant": variant,
                    "compared_to": against,
                    "FPR@95": fpr,
                    "AUROC": auroc,
                }

                if base is not None:
                    fpr0, auroc0 = detection(scores=base, labels=stream.labels)
                    delta = np.abs(np.asarray(conf, dtype=np.float64) - np.asarray(base, dtype=np.float64))
                    held = variant == "recorded_run"
                    within = abs(fpr - fpr0) <= TOLERANCE["FPR@95"] and abs(auroc - auroc0) <= TOLERANCE["AUROC"]
                    row.update(
                        {
                            "dFPR@95": fpr - fpr0,
                            "dAUROC": auroc - auroc0,
                            "max_abs_dconf": float(delta.max()),
                            "differing_scores": int((delta > 0).sum()),
                            "within_tolerance": within if held else None,
                        }
                    )

                rows.append(row)

            for mode in MODES:
                scores = records.runs.get((method, mode, leaf), [])
                if len(scores) < 2:
                    continue

                fpr0, auroc0 = detection(scores=scores[0], labels=stream.labels)
                first = np.asarray(scores[0], dtype=np.float64)
                later = [
                    (detection(scores=conf, labels=stream.labels), np.abs(np.asarray(conf, dtype=np.float64) - first))
                    for conf in scores[1:]
                ]
                rows.append(
                    {
                        "method": method,
                        "benchmark": identity["benchmark"],
                        "seed": identity["seed"],
                        "leaf": leaf,
                        "variant": "runs",
                        "compared_to": mode,
                        "FPR@95": fpr0,
                        "AUROC": auroc0,
                        "dFPR@95": max((fpr - fpr0 for (fpr, _), _ in later), key=abs),
                        "dAUROC": max((auroc - auroc0 for (_, auroc), _ in later), key=abs),
                        "max_abs_dconf": max(float(delta.max()) for _, delta in later),
                        "differing_scores": max(int((delta > 0).sum()) for _, delta in later),
                        "within_tolerance": None,
                    }
                )

    return rows


def method_rows(
    summary: pd.DataFrame,
    streams: pd.DataFrame,
    check: pd.DataFrame,
    notes: dict[tuple[str, str], str],
    shared: dict[str, float] | None = None,
) -> pd.DataFrame:

    records: list[dict[str, Any]] = []
    nan = float("nan")

    for method in tables.method_order(summary["method"]):
        part = summary[summary["method"] == method]

        def pick(
            mode: str,
        ) -> pd.Series | None:

            hit = part[part["mode"] == mode]
            return None if hit.empty else hit.iloc[0]

        images, features = pick(IMAGES), pick(FEATURES)
        recorded = check[(check["method"] == method) & (check["variant"] == "recorded_run")]
        encoded = check[(check["method"] == method) & (check["variant"] == IMAGES)]
        repeated = check[(check["method"] == method) & (check["variant"] == "runs")]
        first = features if features is not None else images
        if first is None:
            continue

        if shared is not None and features is not None:
            total = {
                "total_ms": shared["ms"] + features["ms"],
                "total_std": float(np.hypot(shared["ms_std"], features["ms_std"])),
                "shared_ms": shared["ms"],
                "shared_std": shared["ms_std"],
                "transfer_ms": shared["transfer_ms"],
                "encoder_ms": shared["image_ms"],
                "image_calls": shared["image_calls"],
            }
        elif images is not None:
            total = {
                "total_ms": images["ms"],
                "total_std": images["ms_std"],
                "shared_ms": images["transfer_ms"] + images["image_ms"],
                "shared_std": nan,
                "transfer_ms": images["transfer_ms"],
                "encoder_ms": images["image_ms"],
                "image_calls": images["image_calls"],
            }
        else:
            total = dict.fromkeys(
                ("total_ms", "total_std", "shared_ms", "shared_std", "transfer_ms", "encoder_ms", "image_calls"), nan
            )

        records.append(
            {
                "method": method,
                "name": first["name"],
                "kind": first["kind"],
                **total,
                "detect_ms": features["ms"] if features is not None else nan,
                "detect_std": features["ms_std"] if features is not None else nan,
                "detect_p10": features["ms_p10"] if features is not None else nan,
                "detect_p90": features["ms_p90"] if features is not None else nan,
                "detect_run_min": features["ms_pass_min"] if features is not None else nan,
                "detect_run_max": features["ms_pass_max"] if features is not None else nan,
                "passes": int(first["passes"]),
                "post_gpu_mb": features["gpu_peak_method_mb"] if features is not None else nan,
                "peak_gpu_mb": images["gpu_peak_mb"] if images is not None else nan,
                "peak_gpu_method_mb": images["gpu_peak_method_mb"] if images is not None else nan,
                "state_mb": first["state_mb"],
                "fixed_mb": first["fixed_mb"],
                "text_calls": first["text_calls"],
                "recorded_dfpr": float(recorded["dFPR@95"].abs().max()) if len(recorded) else nan,
                "recorded_dauroc": float(recorded["dAUROC"].abs().max()) if len(recorded) else nan,
                "within_tolerance": bool(recorded["within_tolerance"].astype(bool).all()) if len(recorded) else None,
                "runs_max_dconf": float(repeated["max_abs_dconf"].max()) if len(repeated) else nan,
                "e2e_dfpr": float(encoded["dFPR@95"].abs().max()) if len(encoded) else nan,
                "e2e_dauroc": float(encoded["dAUROC"].abs().max()) if len(encoded) else nan,
                "note": "; ".join(f"{mode}: {text}" for (name, mode), text in notes.items() if name == method),
            }
        )

    return pd.DataFrame.from_records(records, columns=PAPER_COLUMNS)


def pass_timing(
    rows: pd.DataFrame,
) -> dict[str, float]:

    timed = rows[rows["timed"].astype(bool)]
    parts = [name for name in TIME_PARTS if name in timed.columns]
    runs = timed.groupby("pass")[parts].median()
    return {
        "passes": int(rows["pass"].nunique()),
        "ms": float(runs["ms"].mean()),
        "ms_std": float(runs["ms"].std(ddof=1)) if len(runs) > 1 else float("nan"),
        "ms_p10": float(timed["ms"].quantile(0.10)),
        "ms_p90": float(timed["ms"].quantile(0.90)),
        "ms_pass_min": float(runs["ms"].min()),
        "ms_pass_max": float(runs["ms"].max()),
        **{name: float(runs[name].mean()) for name in parts[1:]},
    }


def restated(
    streams: pd.DataFrame,
    batches: pd.DataFrame,
) -> pd.DataFrame:

    keys = ["method", "mode", "benchmark", "seed", "leaf"]
    grouped = {key: frame for key, frame in batches.groupby(keys)}
    rows = [{**row.to_dict(), **pass_timing(grouped[tuple(row[keys])])} for _, row in streams.iterrows()]
    return pd.DataFrame.from_records(rows, columns=STREAM_COLUMNS)


def merged_runs(
    streams: pd.DataFrame,
    batches: pd.DataFrame,
) -> pd.DataFrame:

    keys = ["method", "mode", "benchmark", "seed", "leaf"]
    first = streams.drop_duplicates(keys).set_index(keys)
    first.update(streams.groupby(keys, sort=False)[list(PEAK_COLUMNS)].max())
    return restated(streams=first.reset_index(), batches=batches)


def appended(
    ctx: Context,
    batches: pd.DataFrame,
    streams: pd.DataFrame,
    tensors: pd.DataFrame,
    check: pd.DataFrame,
    encoder: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:

    keys = ["method", "mode", "benchmark", "seed", "leaf"]
    paths = [ctx.tables / name for name in (TABLE_BATCHES, TABLE_STREAMS, TABLE_TENSORS, TABLE_CHECK)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"--opt append=1 adds runs to the recorded tables, but {layout.relative(missing[0])} is missing",
        )

    old_batches, old_streams, old_tensors, old_check = (pd.read_csv(path) for path in paths)
    refuse_earlier(streams=old_streams, path=paths[1])
    before = int(old_batches["pass"].max()) + 1

    if (ctx.tables / TABLE_ENCODER).is_file():
        old_encoder = pd.read_csv(ctx.tables / TABLE_ENCODER)
        encoder = encoder.copy()
        encoder["pass"] += int(old_encoder["pass"].max()) + 1
        encoder = pd.concat([old_encoder, encoder], ignore_index=True)[list(ENCODER_COLUMNS)]

    offset = old_batches.groupby(keys)["pass"].max().add(1).rename("offset").reset_index()
    batches = batches.merge(offset, on=keys, how="left")
    batches["pass"] = batches["pass"] + batches["offset"].fillna(0).astype(int)
    batches = pd.concat([old_batches, batches.drop(columns="offset")], ignore_index=True)[list(BATCH_COLUMNS)]
    grouped = {key: frame for key, frame in batches.groupby(keys)}
    old_rows = {tuple(row[keys]): row for _, row in old_streams.iterrows()}
    rows = []

    for _, row in streams.iterrows():
        key = tuple(row[keys])
        merged = {**row.to_dict(), **pass_timing(grouped[key])}

        if key in old_rows:
            for column in ("gpu_peak_mb", "gpu_peak_method_mb"):
                merged[column] = max(float(merged[column]), float(old_rows[key][column]))

        rows.append(merged)

    fresh = set(tuple(row[keys]) for _, row in streams.iterrows())
    rows += [row.to_dict() for key, row in old_rows.items() if key not in fresh]
    streams = pd.DataFrame.from_records(rows, columns=STREAM_COLUMNS)
    kept = ~old_tensors[keys].apply(tuple, axis=1).isin(fresh)
    tensors = pd.concat([tensors, old_tensors[kept]], ignore_index=True)[list(TENSOR_COLUMNS)]
    check_keys = ["method", "benchmark", "seed", "leaf", "variant"]
    redone = set(check[check_keys].apply(tuple, axis=1)) if len(check) else set()
    kept = ~old_check[check_keys].apply(tuple, axis=1).isin(redone)
    check = pd.concat([check, old_check[kept]], ignore_index=True)[list(CHECK_COLUMNS)]
    return batches, streams, tensors, check, encoder, before


def refuse_earlier(
    streams: pd.DataFrame,
    path: Path,
) -> None:

    if "basis" in streams.columns:
        raise ValueError(
            f"{layout.relative(path)} was written by the earlier version of this item, which timed this "
            "repository's ports with the official computation patched in; time every method again "
            "(`python -m scripts.experiment cost_methods analyze`)"
        )


def listed(
    names: Sequence[str],
) -> str:

    if len(names) < 2:
        return "".join(names)

    return ", ".join(names[:-1]) + " and " + names[-1]


def compared(
    table: pd.DataFrame,
    column: str,
    unit: str,
    digits: int = 1,
) -> str:

    rows = table.set_index("method")
    camp = rows.loc[PROPOSED, column]
    others = rows.drop(index=PROPOSED)
    others = others[np.isfinite(others[column])]
    ratio = others[column] / camp

    def named(
        part: pd.DataFrame,
    ) -> str:
        return listed([f"{row['name']} ({row[column]:,.{digits}f} {unit})" for _, row in part.iterrows()])

    def spread(
        values: pd.Series,
    ) -> str:

        low, high = f"{values.min():.1f}", f"{values.max():.1f}"
        return low if low == high else f"{low} to {high}"

    near = others[(ratio >= 1.0 / COMPARABLE_FACTOR) & (ratio <= COMPARABLE_FACTOR)]
    slower = others[ratio > COMPARABLE_FACTOR]
    faster = others[ratio < 1.0 / COMPARABLE_FACTOR]
    clauses = []

    if len(near):
        clauses.append(f"comparable to {named(near)}")

    if len(slower):
        clauses.append(f"{spread(ratio[slower.index])} times less than {named(slower)}")

    if len(faster):
        clauses.append(f"{spread(1.0 / ratio[faster.index])} times more than {named(faster)}")

    return listed(clauses)


def conclusion(
    table: pd.DataFrame,
) -> str:

    rows = table.set_index("method")
    if PROPOSED not in rows.index or len(rows) < 2:
        return ""

    camp = rows.loc[PROPOSED]
    sentences = []

    if np.isfinite(camp["total_ms"]):
        share = (camp["total_ms"] - camp["shared_ms"]) / camp["total_ms"] * 100.0
        sentences.append(
            f"End to end, CAMP processes a batch of 256 images in {camp['total_ms']:.1f} ms, of which "
            f"{camp['encoder_ms']:.1f} ms are the CLIP image encoder and {share:.0f} % its postprocessor "
            f"(detection and adaptation); {compared(table=table, column='total_ms', unit='ms')}.",
        )

    if np.isfinite(camp["detect_ms"]):
        sentences.append(
            f"Its postprocessor takes {camp['detect_ms']:.1f} ms per batch, "
            f"{compared(table=table, column='detect_ms', unit='ms')}.",
        )

    if np.isfinite(camp["post_gpu_mb"]):
        sentences.append(
            f"Its postprocessor peaks at {camp['post_gpu_mb']:,.0f} MB of GPU memory above the CLIP "
            "model, "
            f"{compared(table=table, column='post_gpu_mb', unit='MB', digits=0)}.",
        )

    return " ".join(sentences)


def number(
    value: float,
    digits: int = 1,
) -> str:
    return tables.MISSING_CELL if value is None or not np.isfinite(value) else f"{value:,.{digits}f}"


def plural(
    word: str,
    items: Sequence[Any],
) -> str:
    return word if len(items) == 1 else f"{word}s"


def megabytes(
    value: float,
) -> str:
    return "none" if value == 0 else f"{value:.1f}"


def kind_parts(
    table: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:

    parts = [
        ("Zero-shot methods", table[table["kind"] == layout.ZERO_SHOT]),
        ("Test-time methods", table[table["kind"] == layout.TEST_TIME]),
    ]
    rows = [
        (title, part)
        for title, part in parts
        if not part.empty
    ]

    return rows


def mean_std(
    mean: float,
    std: float,
    digits: int = 1,
    plus_minus: str = "±",
) -> str:

    if std is None or not np.isfinite(std):
        return number(value=mean, digits=digits)

    return f"{number(value=mean, digits=digits)} {plus_minus} {number(value=std, digits=digits)}"


def shared_timing(
    encoder: pd.DataFrame,
) -> dict[str, float] | None:

    if encoder.empty:
        return None

    first = encoder[encoder["pass"] == encoder["pass"].min()]
    return {
        **pass_timing(encoder),
        "batches": int(len(encoder[["benchmark", "seed", "leaf", "start"]].drop_duplicates())),
        "timed": int(first["timed"].astype(bool).sum()),
    }


def markdown(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    streams: pd.DataFrame,
    check: pd.DataFrame,
    info: dict[str, Any],
    scope: dict[str, Any],
    shared: dict[str, float] | None = None,
) -> str:

    first = summary.iloc[0]
    context = {**info, **scope}
    leaves = list(dict.fromkeys(streams["leaf"]))
    names = [tables.leaf_name(leaf) for leaf in leaves]
    seeds = [str(seed) for seed in dict.fromkeys(streams["seed"])]
    benchmarks = listed([layout.BENCHMARK_NAMES[name] for name in dict.fromkeys(streams["benchmark"])])
    where = f"{benchmarks}, {plural(word='seed', items=seeds)} {listed(seeds)}, {plural(word='stream', items=names)} {listed(names)}"
    runs = int(table["passes"].max())
    timed = int(first["n_timed"])

    if shared is not None:
        end_to_end = (
            "*Copy + image encoder*, the same for every method: the host-to-device copy of a preprocessed image batch "
            "and CLIP's `encode_image` (the call that wrote the image-feature cache), timed on "
            f"{int(shared['batches'])} image batches of the streams decoded and preprocessed beforehand and kept in "
            "host memory, so that no loader runs while the GPU is timed; per run as many timed passes over them in "
            f"turn as a method has timed batches ({int(shared['timed']):,}). *End-to-end*: copy + image encoder + "
            "postprocessor; its ± combines their standard deviations as independent measurements. A run times the "
            "copy and the image encoder once and every method once over every stream, one after another in an order "
            "that changes between runs (as listed, reversed, then permutations seeded with the run's number), so that "
            "no method is always timed early or late in the session or after the same method; the runs follow one "
            "another in one session."
        )
    else:
        end_to_end = (
            "*End-to-end*: the image-feature cache disabled, from the host-to-device copy of the preprocessed image "
            "batch through the image encoder and the method's scoring and memory update to the scores (image decoding "
            "runs in the loader's worker processes, outside the timed span). A run scores every stream once with "
            "every method, in an order that changes between runs."
        )

    statistic = (
        "the table gives that single run's time"
        if runs == 1
        else f"the table gives the mean ± the sample standard deviation of the {runs} runs' medians"
    )
    lines = [
        "### Cost of every method on the same streams: end-to-end time, postprocessor time and GPU memory",
        "",
        f"{info.get('device', '')}, {info.get('backbone', 'CLIP')} (image encoder in "
        f"{info.get('encoder_dtype', '')}), batches of {int(first['batch_size'])}; {where}; one method on the GPU at "
        "a time; every method in the official code of its authors (run in this repository's pipeline) with the "
        "configuration of its Table 1 run. *Postprocessor*: the method's `postprocess` (detection and adaptation: "
        "its scoring and memory update) from the image features of the batch, already on the GPU, to the scores. "
        f"{end_to_end} The device is synchronised before and after every timed span. A run's time is the median over "
        f"its {timed:,} full batches scored after {int(first['window_size']):,} samples (CAMP's window is full; the "
        f"batches before are the warm-up); {statistic}. Setup (text features, negative labels) is excluded from the "
        "time; the tensors it prepares stay on the GPU and count in the memory. *Postprocessor peak GPU memory*: the "
        "largest allocation of PyTorch above the CLIP model while the postprocessor scores a stream, from the end of "
        "setup to the end of the stream (the prepared tensors, the method's state and the working memory of a batch; "
        "the CUDA context is not included), the largest over the streams and runs.",
        "",
        "| Method | End-to-end (ms/batch) | Postprocessor (ms/batch) | Postprocessor peak GPU memory (MB) |",
        "|---|---|---|---|",
    ]

    for title, part in kind_parts(table):
        lines.append(f"| *{title}* | | | |")

        for _, row in part.iterrows():
            name = f"**{row['name']}**" if row["method"] == PROPOSED else row["name"]
            lines.append(
                f"| {name}{'*' if row['note'] else ''} | {mean_std(mean=row['total_ms'], std=row['total_std'])} | "
                f"{mean_std(mean=row['detect_ms'], std=row['detect_std'])} | "
                f"{number(value=row['post_gpu_mb'], digits=0)} |",
            )

    sentence = conclusion(table)
    if sentence:
        lines += [
            "",
            sentence,
            "",
            f'("Comparable" lists the methods within a factor of {COMPARABLE_FACTOR:g} of CAMP\'s value; the '
            "sentences are generated from the table.)",
        ]

    for _, row in table[table["note"] != ""].iterrows():
        lines += ["", f"\\* {row['name']}: {row['note']}; not timed."]

    lines += ["", "#### Details", ""]

    if shared is not None:
        lines += [
            "Copy + image encoder (every method): "
            f"{mean_std(mean=shared['ms'], std=shared['ms_std'], digits=2)} ms per batch (the runs' medians "
            f"{number(value=shared['ms_pass_min'], digits=2)} to {number(value=shared['ms_pass_max'], digits=2)}): "
            f"the copy {number(value=shared['transfer_ms'], digits=2)} ms and the image encoder "
            f"{number(value=shared['image_ms'], digits=2)} ms of GPU time (CUDA events); peak GPU memory above the "
            "CLIP model "
            f"{context.get('copy + image encoder peak GPU memory (MB)', tables.MISSING_CELL)} MB.",
            "",
        ]

    lines += [
        "Postprocessor runs: the lowest and the highest of the runs' medians; p10-p90: the spread of the single timed "
        "batches of all runs. State: what a method writes while it scores a stream, at the end of the stream; fixed: "
        "the tensors `setup` prepares. Text-encoder passes: per batch, inside the postprocessor.",
        "",
        "| Method | Postprocessor runs (ms) | Postprocessor p10-p90 (ms) | State (MB) | Fixed (MB) | Text-encoder "
        "passes |",
        "|---|---|---|---|---|---|",
    ]

    for _, row in table.iterrows():
        lines.append(
            f"| {row['name']} | {number(value=row['detect_run_min'], digits=2)} to "
            f"{number(value=row['detect_run_max'], digits=2)} | "
            f"{number(row['detect_p10'])}-{number(row['detect_p90'])} | {megabytes(row['state_mb'])} | "
            f"{row['fixed_mb']:.1f} | {number(value=row['text_calls'], digits=0)} |",
        )

    measured = summary[summary["mode"] == IMAGES]
    if len(measured):
        lines += [
            "",
            "End to end measured as one span (images mode: the loader decoding in its worker processes, the image-"
            "feature cache disabled): the time, the method's peak above the CLIP model with the image encoder's "
            "working memory, and everything PyTorch allocated (the CLIP model included).",
            "",
            "| Method | End to end, one span (ms/batch) | Method memory, end to end (MB) | Peak GPU memory, end to end "
            "(GB) |",
            "|---|---|---|---|",
        ]

        for _, row in measured.iterrows():
            lines.append(
                f"| {row['name']} | {mean_std(mean=row['ms'], std=row['ms_std'])} | "
                f"{number(value=row['gpu_peak_method_mb'], digits=0)} | "
                f"{number(value=row['gpu_peak_mb'] / 1024.0, digits=2)} |",
            )

    end_to_end_check = bool((check["variant"] == IMAGES).any())
    lines += [
        "",
        "#### Score check",
        "",
        "Cached features: the detection performance of the scores computed here from the cached image features (the "
        "mean over the streams). Recorded run (held to the tolerance): the scores the method's own Table 1 run "
        "recorded, where it is in `results/`, against the scores computed here from the same image pipeline (the "
        "cached features when the run read the replay's cache, the end-to-end scores when it read a cache extracted "
        "in this environment); the largest deviation over the streams. Tolerance: |dFPR@95| <= "
        f"{TOLERANCE['FPR@95']:g} and |dAUROC| <= {TOLERANCE['AUROC']:g} percentage points on every stream. Runs "
        "(reported): the largest difference of a score of a later run from the first run's, over the streams. The "
        "official TANL and CoEvo code draws from the global RNG (seeded with the run's seed here), so their scores "
        "differ from a run's by the randomness of those draws as well. The image encoder of this environment "
        f"({context.get('image pipeline', '')}) changes {context.get('re-encoded features that changed', '')} of "
        "the image features against the cached ones (cosine < 0.999).",
        "",
        "| Method | Cached features: FPR@95 / AUROC (mean of streams) | Recorded run: max \\|dFPR@95\\| / "
        "\\|dAUROC\\| | Within tolerance | Runs: max \\|dconf\\| |"
        + (" End to end vs cached features: max \\|dFPR@95\\| / \\|dAUROC\\| |" if end_to_end_check else ""),
        "|---|---|---|---|---|" + ("---|" if end_to_end_check else ""),
    ]

    for method in tables.method_order(dict.fromkeys(check["method"])):
        part = check[check["method"] == method]
        reference = part[part["variant"] == FEATURES]

        def deviation(
            variant: str,
        ) -> str:

            hit = part[part["variant"] == variant]
            if hit.empty:
                return tables.MISSING_CELL

            text = f"{hit['dFPR@95'].abs().max():.3f} / {hit['dAUROC'].abs().max():.3f}"

            if variant == "recorded_run":
                text += " (cached)" if hit["compared_to"].iloc[0] == FEATURES else " (end to end)"

            return text

        recorded = part[part["variant"] == "recorded_run"]
        repeated = part[part["variant"] == "runs"]
        verdict = (
            tables.MISSING_CELL
            if recorded.empty
            else ("yes" if recorded["within_tolerance"].astype(bool).all() else "**no**")
        )
        scores = (
            tables.MISSING_CELL
            if reference.empty
            else f"{reference['FPR@95'].mean():.2f} / {reference['AUROC'].mean():.2f}"
        )
        agree = tables.MISSING_CELL if repeated.empty else f"{repeated['max_abs_dconf'].max():.3g}"
        lines.append(
            f"| {tables.method_name(method)} | {scores} | {deviation('recorded_run')} | {verdict} | "
            f"{agree} |" + (f" {deviation(IMAGES)} |" if end_to_end_check else "")
        )

    modes = [mode for mode in MODES if mode in set(streams["mode"])]
    titles = {IMAGES: "end to end, one span", FEATURES: "postprocessor"}
    lines += [
        "",
        f"#### Per stream: ms per batch ({' / '.join(titles[mode] for mode in modes)}; the mean of the runs' medians)",
        "",
        "| Method | " + " | ".join(names) + " |",
        "|" + "---|" * (1 + len(leaves)),
    ]

    for _, row in table.iterrows():
        cells = [row["name"]]

        for leaf in leaves:
            values = []

            for mode in modes:
                hit = streams[
                    (streams["method"] == row["method"]) & (streams["leaf"] == leaf) & (streams["mode"] == mode)
                ]
                values.append(number(float(hit["ms"].mean())) if len(hit) else tables.MISSING_CELL)

            cells.append(" / ".join(values))

        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "#### Environment and scope", ""]
    lines += [f"- {key}: {value}" for key, value in context.items()]
    return "\n".join(lines) + "\n"


def latex(
    table: pd.DataFrame,
) -> str:

    lines = [
        "% generated by `python -m scripts.experiment cost_methods analyze`; times: the mean over the runs of a run's "
        "median over the timed batches $\\pm$ the sample std of those medians; end-to-end: copy + image encoder + "
        "postprocessor",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & End-to-end (ms) & Postprocessor (ms) & Postprocessor peak GPU memory (MB) \\\\",
        "\\midrule",
    ]

    for index, (title, part) in enumerate(kind_parts(table)):
        if index:
            lines.append("\\midrule")

        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}} \\\\")

        for _, row in part.iterrows():
            cells = [
                row["name"],
                mean_std(mean=row["total_ms"], std=row["total_std"], plus_minus="$\\pm$"),
                mean_std(mean=row["detect_ms"], std=row["detect_std"], plus_minus="$\\pm$"),
                number(value=row["post_gpu_mb"], digits=0),
            ]

            if row["method"] == PROPOSED:
                cells = [f"\\textbf{{{cell}}}" for cell in cells]

            lines.append(" & ".join(cell.replace(",", "{,}") for cell in cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def recorded_notes(
    path: Path,
) -> dict[tuple[str, str], str]:

    if not path.is_file():
        return {}

    notes: dict[tuple[str, str], str] = {}
    table = pd.read_csv(path, keep_default_na=False)
    for method, note in zip(table["method"], table["note"]):
        for part in filter(None, str(note).split("; ")):
            mode, _, text = part.partition(": ")
            notes[(str(method), mode)] = text

    return notes


def rebuild(
    ctx: Context,
) -> list[Path]:

    paths = {name: ctx.tables / name for name in (TABLE_BATCHES, TABLE_STREAMS, TABLE_CHECK, TABLE_ENVIRONMENT)}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"--opt rebuild=1 reads the recorded tables, but {layout.relative(missing[0])} is "
            "missing; run `python -m scripts.experiment cost_methods analyze` first"
        )

    batches, streams, check = (pd.read_csv(paths[name]) for name in (TABLE_BATCHES, TABLE_STREAMS, TABLE_CHECK))
    refuse_earlier(streams=streams, path=paths[TABLE_STREAMS])
    streams = restated(streams=streams, batches=batches)
    encoder_path = ctx.tables / TABLE_ENCODER
    shared = shared_timing(pd.read_csv(encoder_path)) if encoder_path.is_file() else None
    frame = pd.read_csv(paths[TABLE_ENVIRONMENT], keep_default_na=False)
    recorded = dict(zip(frame["key"].astype(str), frame["value"].astype(str)))
    notes = recorded_notes(ctx.tables / TABLE_SUMMARY)
    summary = summarize(batches=batches, streams=streams)
    summary["note"] = [notes.get((method, mode), "") for method, mode in zip(summary["method"], summary["mode"])]
    summary = summary[list(SUMMARY_COLUMNS)]
    table = method_rows(summary=summary, streams=streams, check=check, notes=notes, shared=shared)
    ctx.notes.update(
        {
            "rebuilt": "from the recorded tables, without measuring",
            "measured": recorded.get("time", ""),
            "conclusion": conclusion(table),
        }
    )
    return [
        tables.write_csv(table, ctx.tables / TABLE_SUMMARY),
        tables.write_markdown(
            markdown(
                table=table, summary=summary, streams=streams, check=check, info=recorded, scope={}, shared=shared
            ),
            ctx.tables / TABLE_MD,
        ),
        tables.write_markdown(latex(table), ctx.tables / TABLE_TEX),
    ]


def analyze(
    ctx: Context,
) -> list[Path]:

    if switched_on(ctx=ctx, key="rebuild"):
        return rebuild(ctx)

    specs = item_runs(ctx)
    recorded = [spec for spec in specs if spec.recorded()]
    missing = [spec for spec in specs if not spec.recorded()]

    if not recorded:
        example = " ".join(runs.command(specs[0], python=ctx.python))
        raise FileNotFoundError(
            f"none of the {len(specs)} CAMP main runs is recorded, e.g. {layout.relative(specs[0].dir())}; the item "
            f"times every method on their streams, run them first (`python -m scripts.experiment cost_methods run`), e.g.\n"
            f"  {example}",
        )

    methods = wanted_methods(ctx)
    modes = wanted_modes(ctx)
    passes = positive_option(ctx=ctx, key="passes", default=DEFAULT_PASSES)
    workers = positive_option(ctx=ctx, key="workers", default=DEFAULT_WORKERS)
    encoder = switched_on(ctx=ctx, key="encoder", default="1")
    append = switched_on(ctx=ctx, key="append")
    deterministic = switched_on(ctx=ctx, key="deterministic")
    first_run = (
        int(pd.read_csv(ctx.tables / TABLE_BATCHES, usecols=["pass"])["pass"].max()) + 1
        if append and (ctx.tables / TABLE_BATCHES).is_file()
        else 0
    )
    items = ([(SHARED, "")] if encoder else []) + [(method, mode) for method in methods for mode in modes]
    orders = [run_order(items=items, run=first_run + run) for run in range(passes)]
    order_text = {
        f"order of run {first_run + run + 1}": ", ".join(item_name(item) for item in order)
        for run, order in enumerate(orders)
    }

    for spec in missing:
        print(f"  missing {layout.relative(spec.dir())}")

    if ctx.dry_run:
        for spec in recorded:
            print(
                f"[dry-run] would time {', '.join(methods)} in their official code ({', '.join(modes)}"
                f"{'; copy + image encoder' if encoder else ''}; {passes} run(s)) on {layout.relative(spec.dir())}, "
                f"streams {wanted_streams(ctx=ctx, benchmark=spec.benchmark)}, on {ctx.device}"
            )

        for key, value in order_text.items():
            print(f"[dry-run] {key}: {value}")

        return []

    if str(ctx.device).startswith("cpu"):
        raise RuntimeError(
            "the timing needs a CUDA device (the CLIP network of the runs is on CUDA); pass --device cuda",
        )

    import torch

    from openood.utils import seed_everything
    from scripts.experiment.lib import replay

    device = torch.device(ctx.device)
    if device.index is not None:
        torch.cuda.set_device(device)

    budget = limit_memory(ctx=ctx, device=device)
    load = gpu_load(device)
    if load.get("utilization_pct", 0.0) >= BUSY_UTILIZATION:
        print(
            f"[cost_methods] the GPU shows {load['utilization_pct']:.0f} % utilisation before the measurement: "
            "another job inflates the timings (one job on the GPU at a time)"
        )

    records = Records()
    checks: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    pipeline: dict[str, str] = {}
    cosine: list[np.ndarray] = []
    preloaded = 0
    encoder_peak = float("nan")

    for spec in recorded:
        if deterministic:
            seed_everything(spec.seed)

        run = replay.RecordedRun(spec.dir())
        features = replay.features_of(run, ctx.device)
        streams = replay.streams(run, features, leaves=wanted_streams(ctx=ctx, benchmark=spec.benchmark))

        for stream in streams:
            replay.verify_stream(stream)

        window_size = int(run.policy().camp.window_size)
        sources = image_sources(run=run, streams=streams) if IMAGES in modes or encoder else {}
        warm_images = (
            first_images(source=sources[streams[0].name], batch_size=streams[0].batch_size, workers=workers)
            if IMAGES in modes
            else None
        )
        batches = (
            preload(
                sources=sources, streams=streams, window_size=window_size, per_stream=ENCODER_BATCHES, workers=workers
            )
            if encoder
            else []
        )
        preloaded += len(batches)
        cosine += [reencoded(features=features, batches=batches)] if batches else []
        timed = sum(len(timed_spans(stream=stream, window_size=window_size)) for stream in streams)
        info = environment(
            device=device,
            net=features.net,
            backbone=str(run.config.network.backbone.name),
            workers=workers,
            deterministic=deterministic,
        )
        pipeline = cache_pipeline(features)
        warm_up(features.device)

        for run_index, order in enumerate(orders):
            print(
                f"[cost_methods] run {run_index + 1} of {passes}: {', '.join(item_name(item) for item in order)}",
                flush=True,
            )

            for method, mode in order:
                if method == SHARED:
                    encoder_peak = np.nanmax(
                        [
                            encoder_peak,
                            measure_encoder(
                                run=run_index, features=features, batches=batches, timed=timed, records=records
                            ),
                        ]
                    )
                    continue

                if (method, mode) in records.notes:
                    continue

                if measure(
                    method=method,
                    mode=mode,
                    run=run_index,
                    features=features,
                    streams=streams,
                    sources=sources,
                    window_size=window_size,
                    workers=workers,
                    warm_images=warm_images,
                    records=records,
                    tensors=run_index == 0,
                ):
                    continue

                records.notes[(method, mode)] = f"does not fit the memory budget of {budget / (1 << 30):.1f} GB"
                print(f"[cost_methods] {method} ({mode}) {records.notes[(method, mode)]}; not timed", flush=True)

        checks += check_rows(
            records=records,
            streams=streams,
            identity={
                "benchmark": run.benchmark,
                "seed": run.seed,
                "backbone": spec.backbone,
                "replay_cache": str(features.cache_paths[0]),
            },
        )
        records.runs.clear()
        del features, streams, sources, batches
        gc.collect()
        torch.cuda.empty_cache()

    batches = pd.DataFrame.from_records(records.batches, columns=BATCH_COLUMNS)
    stream_table = merged_runs(
        streams=pd.DataFrame.from_records(records.streams, columns=STREAM_COLUMNS), batches=batches
    )
    tensor_table = pd.DataFrame.from_records(records.tensors, columns=TENSOR_COLUMNS)
    check = pd.DataFrame.from_records(checks, columns=CHECK_COLUMNS)
    encoder_table = pd.DataFrame.from_records(records.encoder, columns=ENCODER_COLUMNS)
    finished = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    written: list[Path] = []
    before = 0

    if append:
        run_dir = ctx.out / "runs" / finished.replace(":", "")
        written += [
            tables.write_csv(frame, run_dir / name)
            for frame, name in (
                (batches, TABLE_BATCHES),
                (stream_table, TABLE_STREAMS),
                (tensor_table, TABLE_TENSORS),
                (check, TABLE_CHECK),
                (encoder_table, TABLE_ENCODER),
            )
        ]
        batches, stream_table, tensor_table, check, encoder_table, before = appended(
            ctx=ctx, batches=batches, streams=stream_table, tensors=tensor_table, check=check, encoder=encoder_table
        )

    shared = shared_timing(encoder_table)
    summary = summarize(batches=batches, streams=stream_table)
    summary["note"] = [
        records.notes.get((method, mode), "") for method, mode in zip(summary["method"], summary["mode"])
    ]
    summary = summary[list(SUMMARY_COLUMNS)]
    table = method_rows(summary=summary, streams=stream_table, check=check, notes=records.notes, shared=shared)
    agreement = np.concatenate(cosine) if cosine else np.array([], dtype=np.float64)
    if not len(agreement):
        reencoded_rows = stream_table[stream_table["mode"] == IMAGES]
        changed = (
            float(np.average(reencoded_rows["cosine_below_999_pct"], weights=reencoded_rows["n_samples"]))
            if len(reencoded_rows)
            else float("nan")
        )
        changed_text = f"{changed:.1f} %"
    else:
        changed_text = (
            f"{np.mean(agreement < 0.999) * 100.0:.1f} % (of {len(agreement):,} images of the preloaded batches)"
        )

    scope = {
        "time": finished,
        "batch size": int(summary["batch_size"].iloc[0]),
        "runs": int(stream_table["passes"].max()),
        "appended": f"{passes} run(s) of this measurement after {before} recorded" if append else "no",
        **order_text,
        "timed batches": f"full batches starting at or after sample {int(summary['window_size'].iloc[0]):,}",
        "statistic": "a run's median over its timed batches; the mean and the sample standard deviation of the runs' "
        "medians",
        "postprocessor span": "postprocess on the cached image features already on the GPU",
        "copy + image encoder": (
            f"{preloaded} preloaded image batches ({ENCODER_BATCHES} per stream, the first timed ones), "
            f"{int(shared['timed']):,} timed passes per run over them in turn after {ENCODER_WARMUP} not timed; "
            "the span from the host-to-device copy through encode_image"
            if shared is not None
            else "not timed"
        ),
        "copy + image encoder peak GPU memory (MB)": round(float(encoder_peak), 1),
        "end to end": (
            "copy + image encoder + postprocessor; ± combines their standard deviations as independent"
            if shared is not None
            else "one span in the images mode"
        ),
        "excluded": "loader (decoding, preprocessing), setup (text features, negative labels), the copy of the "
        "scores to the host",
        "memory": "torch.cuda.max_memory_allocated from the end of setup to the end of the stream; the largest over "
        "the streams and runs",
        "image pipeline": f"here Pillow {info.get('pillow', '')}, torchvision {info.get('torchvision', '')}, torch "
        f"{info.get('torch', '')} on {platform.machine()}; the cached features: Pillow "
        f"{pipeline.get('pillow', '?')}, torchvision {pipeline.get('torchvision', '?')}, torch "
        f"{pipeline.get('torch', '?')} on {pipeline.get('machine', '?')}",
        "re-encoded features that changed": changed_text,
        "gpu budget (GB)": round(budget / (1 << 30), 1),
        "gpu before": load,
    }
    environment_table = pd.DataFrame(
        {"key": list({**info, **scope}), "value": [str(value) for value in {**info, **scope}.values()]}
    )
    written += [
        tables.write_csv(table, ctx.tables / TABLE_SUMMARY),
        tables.write_markdown(
            markdown(
                table=table, summary=summary, streams=stream_table, check=check, info=info, scope=scope, shared=shared
            ),
            ctx.tables / TABLE_MD,
        ),
        tables.write_markdown(latex(table), ctx.tables / TABLE_TEX),
        tables.write_csv(stream_table, ctx.tables / TABLE_STREAMS),
        tables.write_csv(batches, ctx.tables / TABLE_BATCHES),
        tables.write_csv(tensor_table, ctx.tables / TABLE_TENSORS),
        tables.write_csv(check, ctx.tables / TABLE_CHECK),
        tables.write_csv(encoder_table, ctx.tables / TABLE_ENCODER),
        tables.write_csv(environment_table, ctx.tables / TABLE_ENVIRONMENT),
    ]
    ctx.notes.update(
        {
            "methods": methods,
            "modes": modes,
            "encoder": encoder,
            "runs": scope["runs"],
            "runs_this_measurement": passes,
            "appended_to_runs": before,
            "orders": order_text,
            "workers": workers,
            "not_timed": {f"{method} ({mode})": text for (method, mode), text in records.notes.items()},
            "gpu_budget_mb": round(to_mb(budget), 1),
            "deterministic": deterministic,
            "environment": info,
            "gpu_before": load,
            "runs_recorded": len(recorded),
            "missing_runs": [layout.relative(spec.dir()) for spec in missing],
            "shared_ms": None if shared is None else round(float(shared["ms"]), 3),
            "end_to_end_ms": {row["method"]: round(float(row["total_ms"]), 3) for _, row in table.iterrows()},
            "detection_ms": {row["method"]: round(float(row["detect_ms"]), 3) for _, row in table.iterrows()},
            "post_gpu_mb": {row["method"]: round(float(row["post_gpu_mb"]), 1) for _, row in table.iterrows()},
            "recorded_run_within_tolerance": {
                row["method"]: row["within_tolerance"]
                for _, row in table.iterrows()
                if row["within_tolerance"] is not None
            },
            "conclusion": conclusion(table),
        }
    )
    return written


def item_name(
    item: tuple[str, str],
) -> str:

    method, mode = item

    if method == SHARED:
        return "copy + image encoder"

    return tables.method_name(method) if mode == FEATURES else f"{tables.method_name(method)} ({mode})"


__all__ = [
    "BASE_METHOD",
    "DEFAULT_MODES",
    "FEATURES",
    "IMAGES",
    "MODES",
    "PROPOSED",
    "SHARED",
    "TABLE_ENCODER",
    "TABLE_ENVIRONMENT",
    "TABLE_SUMMARY",
    "TOLERANCE",
    "EncoderClock",
    "ImageBatch",
    "ImageSource",
    "Pass",
    "Records",
    "Snapshot",
    "analyze",
    "check_rows",
    "conclusion",
    "detection",
    "environment",
    "features_pass",
    "held_tensors",
    "image_sources",
    "images_pass",
    "item_runs",
    "latex",
    "markdown",
    "mean_std",
    "measure",
    "measure_encoder",
    "merged_runs",
    "method_rows",
    "pass_timing",
    "preload",
    "reencoded",
    "restated",
    "run_config",
    "run_order",
    "seed_rngs",
    "shared_timing",
    "summarize",
    "tensor_rows",
    "time_stream",
    "timed_spans",
    "wanted_methods",
    "wanted_modes",
    "wanted_streams",
]
