"""
Replay the test streams of a recorded run from the cached image features.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .config_io import load_recorded_config, load_yaml
from .geometry import camp_geometry, normalize_features

if TYPE_CHECKING:
    import torch

    from openood.evaluators.stream_protocol import StreamProtocol
    from openood.postprocessors.camp.config import CAMPPolicy
    from openood.utils.config import Config

    from .geometry import Geometry

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = ("imagenet_openood", "imagenet_standard", "cifar10_openood", "cifar100_openood")
BACKBONE_NAMES = {
    "ViT-B/16": "vitb16",
    "ViT-B/32": "vitb32",
    "ViT-L/14": "vitl14",
    "RN50": "rn50",
}
DEFAULT_BACKBONE = "vitb16"
RECORDED_SCORE_KEYS = ("pred", "conf", "label")
FEATURE_SOURCES = ("auto", "log", "manager")
_CACHE_LINE = re.compile(r"\[ImageFeatures\] (?:Loaded|Saved) cache: (?P<path>.+?)\s*$")
_CACHE_NAME = re.compile(r"source=(?P<source>.+?)_split=(?P<split>[A-Za-z0-9]+)_data=")


class StreamMismatch(ValueError):
    """Raised when a replayed stream does not match the recorded run in order, length or scores."""


def _split_mark(
    mark: str,
    backbone: str,
) -> tuple[str, str | None]:

    for benchmark in BENCHMARKS:
        if mark != benchmark and not mark.startswith(benchmark + "_"):
            continue

        rest = "" if mark == benchmark else mark[len(benchmark) + 1:]
        if backbone != DEFAULT_BACKBONE and (rest == backbone or rest.startswith(backbone + "_")):
            rest = "" if rest == backbone else rest[len(backbone) + 1:]

        return benchmark, (rest or None)

    raise ValueError(f"mark {mark!r} does not start with one of the benchmarks {list(BENCHMARKS)}.")


class RecordedRun:

    def __init__(
        self,
        dir: Path | str,
    ) -> None:

        self.dir = Path(dir)
        self.config: Config = load_recorded_config(self.dir)
        self.seed = int(self.config.seed)
        self.method = str(self.config.postprocessor.name)
        backbone_name = str(self.config.network.backbone.name)
        self.backbone = BACKBONE_NAMES.get(backbone_name, backbone_name)
        self.mark = str(self.config.mark)
        self.benchmark, self.label = _split_mark(mark=self.mark, backbone=self.backbone)


    def __repr__(self) -> str:

        return (f"RecordedRun({str(self.dir)!r}, method={self.method!r}, benchmark={self.benchmark!r}, "
            f"backbone={self.backbone!r}, label={self.label!r}, seed={self.seed})")


    @property
    def protocol(self) -> StreamProtocol:

        from openood.evaluators.stream_protocol import StreamProtocol
        from openood.utils.config import get_config

        return StreamProtocol.from_config(get_config(obj=self.config.evaluator, name="stream", default=None))


    @property
    def batch_size(self) -> int:
        return int(self.config.dataset.test.batch_size)


    @property
    def groups(self) -> dict[str, tuple[str, ...]]:

        ood = self.config.ood_dataset
        table = {
            str(split): tuple(str(leaf) for leaf in ood[split].datasets)
            for split in ood.split_names
        }

        return table


    def scores_path(
        self,
        leaf: str,
    ) -> Path:
        return self.dir / "scores" / f"{leaf}.npz"


    def recorded_scores(
        self,
        leaf: str,
    ) -> dict[str, np.ndarray] | None:

        path = self.scores_path(leaf)
        if not path.is_file():
            return None

        with np.load(path) as data:
            missing = [key for key in RECORDED_SCORE_KEYS if key not in data.files]
            if missing:
                raise ValueError(f"{path} lacks {missing}; expected {list(RECORDED_SCORE_KEYS)}.")

            table = {
                key: np.asarray(data[key])
                for key in RECORDED_SCORE_KEYS
            }

            return table


    def policy(self) -> CAMPPolicy:

        from openood.postprocessors.camp.config import CAMPPolicy

        if self.method != "camp":
            raise ValueError(f"{self.dir} is a {self.method!r} run; a CAMP policy must be given explicitly.")

        return CAMPPolicy.from_config(self.config.postprocessor.postprocessor_args)


@dataclass(frozen=True, eq=False)
class Features:

    name: str
    data: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.data.shape[0])


    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        return {"data": self.data[index], "label": self.labels[index]}


@dataclass(frozen=True, eq=False)
class RunFeatures:

    run: RecordedRun
    config: Config
    id: Features
    groups: dict[str, dict[str, Features]]
    net: torch.nn.Module
    device: torch.device
    logit_scale: torch.Tensor
    id_text_features: torch.Tensor
    prototypes: torch.Tensor
    class_num: int
    batch_size: int
    cache_paths: tuple[str, ...]

    def leaves(self) -> list[tuple[str, str]]:

        rows = [
            (group, leaf)
            for group, members in self.groups.items()
            for leaf in members
        ]

        return rows


def replay_config(
    run: RecordedRun,
) -> Config:

    from openood.utils.config import merge_configs

    config = merge_configs(run.config)
    config["cache.image.mode"] = "read_write"
    return config


def logged_cache_paths(
    run: RecordedRun,
) -> dict[tuple[str, str], Path]:

    log = run.dir / "log.txt"
    found: dict[tuple[str, str], Path] = {}

    if not log.is_file():
        return found

    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _CACHE_LINE.search(line)
        if match is None:
            continue

        path = Path(match.group("path").replace("\\", "/"))
        name = _CACHE_NAME.search(path.name)
        if name is None:
            continue

        found[(name.group("source"), name.group("split"))] = path if path.is_absolute() else REPO_ROOT / path

    return found


def features_of(
    run: RecordedRun,
    device: str | torch.device = "cuda",
    *,
    source: str = "auto",
    allow_extraction: bool = False,
) -> RunFeatures:

    import torch
    import torch.nn.functional as functional

    from openood.datasets import get_dataloader, get_ood_dataloader
    from openood.evaluators.ood_evaluator_clip_tta import OODEvaluatorClipTTA
    from openood.networks import get_network
    from openood.utils.config import get_config
    from openood.utils.vlm.text import load_text_features

    if source not in FEATURE_SOURCES:
        raise ValueError(f"source must be one of {list(FEATURE_SOURCES)}, got {source!r}.")

    config = replay_config(run)
    id_loaders = get_dataloader(config)
    ood_loaders = get_ood_dataloader(config)
    net = get_network(config.network)
    net.eval()
    dataset_name = str(config.dataset.name)
    id_splits = set(OODEvaluatorClipTTA.image_feature_cache_required_id_splits)
    ood_splits = set(OODEvaluatorClipTTA.image_feature_cache_required_ood_splits)
    wanted = [(dataset_name, str(split), loader.dataset) for split, loader in id_loaders.items() if split in id_splits]
    wanted += [(str(leaf), str(split), loader.dataset) for split, loaders in ood_loaders.items() if split in ood_splits
        for leaf, loader in loaders.items()]
    logged = logged_cache_paths(run)
    if source == "log" or (source == "auto" and all((name, split) in logged for name, split, _ in wanted)):
        missing = [(name, split) for name, split, _ in wanted if (name, split) not in logged]
        if missing:
            raise FileNotFoundError(f"{run.dir / 'log.txt'} names no image-feature cache for {missing}.")

        loaded = {(name, split): _features_from_cache(path=logged[(name, split)], dataset=dataset, name=name)
            for name, split, dataset in wanted}
        cache_paths = tuple(str(logged[(name, split)]) for name, split, _ in wanted)
    else:
        loaded, cache_paths = _features_from_manager(
            config=config,
            net=net,
            id_loaders=id_loaders,
            ood_loaders=ood_loaders,
            allow_extraction=allow_extraction,
        )

    missing = [(name, split) for name, split, _ in wanted if (name, split) not in loaded]
    if missing:
        raise FileNotFoundError(f"no image features for {missing} of {run.dir}.")

    groups = {str(split): {str(leaf): loaded[(str(leaf), str(split))] for leaf in loaders}
        for split, loaders in ood_loaders.items() if split in ood_splits}
    device = torch.device(device)
    batch_size = int(get_config(obj=config.dataset.test, name="batch_size", default=256))
    text = load_text_features(
        model=net.model,
        dataset=dataset_name,
        device=device,
        backbone=str(config.network.backbone.name),
        text_prompt=str(config.vlm.text.prompt),
        model_name=str(config.network.name),
        text_center=bool(get_config(obj=config.vlm.text, name="center", default=True)),
        batch_size=batch_size,
    )

    id_text_features = functional.normalize(text, dim=0)
    prototypes = functional.normalize(id_text_features, dim=0).float()
    return RunFeatures(
        run=run,
        config=config,
        id=loaded[(dataset_name, "test")],
        groups=groups,
        net=net,
        device=device,
        logit_scale=net.get_logit_scale().to(device),
        id_text_features=id_text_features,
        prototypes=prototypes,
        class_num=int(id_text_features.shape[1]),
        batch_size=batch_size,
        cache_paths=cache_paths,
    )


def _features_from_cache(
    path: Path,
    dataset: Any,
    name: str,
) -> Features:

    import torch

    from openood.evaluators.stream_protocol import dataset_labels
    from openood.utils.vlm.cache import load_cache

    if not path.is_file():
        raise FileNotFoundError(f"{name}: the run's log names the image-feature cache {path}, which does not exist.")

    cache = load_cache(str(path), weights_only=True)
    data = cache["image_features"]
    labels = cache["labels"].reshape(-1).to(torch.long)
    indices = cache["source_indices"].reshape(-1)
    size = len(dataset)
    if int(data.shape[0]) != size or not torch.equal(indices, torch.arange(size)):
        raise ValueError(
            f"{name}: {path} holds {int(data.shape[0])} rows for an imglist of {size} images in canonical "
            f"order.",
        )

    if not torch.equal(labels, torch.as_tensor(dataset_labels(dataset), dtype=torch.long)):
        raise ValueError(f"{name}: the labels stored in {path} differ from the imglist's labels.")

    return Features(name=name, data=data, labels=labels)


def _features_from_manager(
    config: Config,
    net: Any,
    id_loaders: Mapping[str, Any],
    ood_loaders: Mapping[str, Any],
    *,
    allow_extraction: bool,
) -> tuple[dict[tuple[str, str], Features], tuple[str, ...]]:

    import torch

    from openood.datasets.feature_dataset import CachedImageFeatureDataset
    from openood.evaluators.ood_evaluator_clip_tta import OODEvaluatorClipTTA
    from openood.postprocessors import get_postprocessor
    from openood.utils.vlm.image.policy import ImageFeatureCachePolicy

    manager = _recording_cache_manager_class()(
        config=config,
        policy=ImageFeatureCachePolicy.from_config(config),
        allow_extraction=allow_extraction,
    )

    prepared_id, prepared_ood = manager.prepare(
        net=net,
        id_loader_dict=id_loaders,
        ood_loader_dict=ood_loaders,
        postprocessor=get_postprocessor(config),
        evaluator=OODEvaluatorClipTTA(config),
    )

    loaded: dict[tuple[str, str], Features] = {}

    for split, loader in prepared_id.items():
        if isinstance(loader.dataset, CachedImageFeatureDataset):
            loaded[(str(config.dataset.name), str(split))] = Features(
                name=str(config.dataset.name),
                data=loader.dataset.data,
                labels=loader.dataset.labels.to(torch.long),
            )

    for split, loaders in prepared_ood.items():
        if not isinstance(loaders, Mapping):
            continue

        for leaf, loader in loaders.items():
            if isinstance(loader.dataset, CachedImageFeatureDataset):
                loaded[(str(leaf), str(split))] = Features(
                    name=str(leaf),
                    data=loader.dataset.data,
                    labels=loader.dataset.labels.to(torch.long),
                )

    return loaded, tuple(manager.cache_paths)


def _recording_cache_manager_class() -> type:

    from openood.utils.vlm.image.manager import ImageFeatureCacheManager

    class RecordingCacheManager(ImageFeatureCacheManager):

        def __init__(
            self,
            *,
            config: Any,
            policy: Any,
            allow_extraction: bool,
        ) -> None:

            super().__init__(config=config, policy=policy)
            self.allow_extraction = allow_extraction
            self.cache_paths: list[str] = []


        def _load_or_build(
            self,
            *,
            net: Any,
            source_loader: Any,
            spec: Any,
            refresh_spec: Any = None,
        ) -> Any:

            if not self.allow_extraction and not os.path.exists(spec.cache_path):
                raise FileNotFoundError(
                    f"no image-feature cache at {spec.cache_path}: this environment's cache identity differs from the "
                    f"run's, and features extracted here would not reproduce the recorded scores (pass "
                    f"allow_extraction=True to extract anyway)."
                )

            return super()._load_or_build(net=net, source_loader=source_loader, spec=spec, refresh_spec=refresh_spec)


        def _load_from_disk(
            self,
            spec: Any,
        ) -> Any:

            cached = super()._load_from_disk(spec)
            self.cache_paths.append(str(spec.cache_path))
            return cached

    return RecordingCacheManager


class _Indices:
    def __init__(
        self,
        size: int,
    ) -> None:
        self.size = int(size)


    def __len__(self) -> int:
        return self.size


    def __getitem__(
        self,
        index: int,
    ) -> int:
        return int(index)


def default_order(
    size: int,
    seed: int,
) -> np.ndarray:

    import torch
    from torch.utils.data import DataLoader

    from openood.utils.reproducibility import make_torch_generator

    loader = DataLoader(
        _Indices(size),
        batch_size=4096,
        shuffle=True,
        generator=make_torch_generator(seed),
        num_workers=0,
    )

    return torch.cat([torch.as_tensor(batch) for batch in loader]).numpy().astype(np.int64)


@dataclass(frozen=True, eq=False)
class Evaluation:

    leaf: str
    group: str
    positions: np.ndarray
    recorded: dict[str, np.ndarray] | None


@dataclass(frozen=True, eq=False)
class Stream:

    name: str
    group: str | None
    leaf: str | None
    raw: torch.Tensor
    labels: np.ndarray
    order: np.ndarray
    batch_size: int
    evaluations: tuple[Evaluation, ...]

    def __len__(self) -> int:
        return int(self.labels.shape[0])


    @cached_property
    def features(self) -> torch.Tensor:
        return normalize_features(self.raw)


    @property
    def is_ood(self) -> np.ndarray:
        return self.labels < 0


    @property
    def recorded_scores(self) -> dict[str, np.ndarray] | None:

        if len(self.evaluations) == 1 and len(self.evaluations[0].positions) == len(self):
            return self.evaluations[0].recorded

        return None


    def batches(self) -> Iterator[tuple[int, int]]:

        for start in range(0, len(self), self.batch_size):
            yield start, min(start + self.batch_size, len(self))


def streams(
    run: RecordedRun,
    features: RunFeatures,
    leaves: Sequence[str] | None = None,
) -> list[Stream]:

    import torch

    protocol = run.protocol
    wanted = None if leaves is None else set(leaves)
    result: list[Stream] = []

    if protocol.is_default:
        for group, members in features.groups.items():
            for leaf, ood in members.items():
                if wanted is not None and leaf not in wanted:
                    continue

                size = len(ood) + len(features.id)
                order = default_order(size=size, seed=run.seed)
                index = torch.as_tensor(order)
                raw = torch.cat([ood.data, features.id.data])[index]
                labels = torch.cat([ood.labels, features.id.labels])[index].numpy().astype(np.int64)
                evaluation = Evaluation(
                    leaf=leaf,
                    group=group,
                    positions=np.arange(size),
                    recorded=run.recorded_scores(leaf),
                )

                result.append(
                    Stream(
                        name=leaf,
                        group=group,
                        leaf=leaf,
                        raw=raw,
                        labels=labels,
                        order=order,
                        batch_size=features.batch_size,
                        evaluations=(evaluation,),
                    ),
                )

        return result

    from openood.evaluators.stream_protocol import build_stream_plans

    plans = build_stream_plans(
        protocol,
        id_dataset=features.id,
        groups={group: dict(members) for group, members in features.groups.items()},
        seed=run.seed,
    )
    for plan in plans:
        ood_parts = plan.ood_parts()
        if wanted is not None and not any(part.leaf in wanted for _, part in ood_parts):
            continue

        rows = [torch.as_tensor(np.asarray(part.rows, dtype=np.int64)) for part in plan.parts]
        order = np.asarray(plan.order, dtype=np.int64)
        index = torch.as_tensor(order)
        raw = torch.cat([part.dataset.data[select] for part, select in zip(plan.parts, rows)])[index]
        labels = torch.cat([part.dataset.labels[select] for part, select in zip(plan.parts, rows)])[index]
        evaluations = tuple(
            Evaluation(
                leaf=str(part.leaf),
                group=str(part.group),
                positions=np.asarray(plan.evaluation_positions(i)),
                recorded=run.recorded_scores(str(part.leaf)),
            )
            for i, part in ood_parts
        )

        single = ood_parts[0][1] if len(ood_parts) == 1 else None
        result.append(
            Stream(
                name=str(plan.name),
                group=None if single is None else str(single.group),
                leaf=None if single is None else str(single.leaf),
                raw=raw,
                labels=labels.numpy().astype(np.int64),
                order=order,
                batch_size=features.batch_size,
                evaluations=evaluations,
            ),
        )

    return result


def verify_stream(
    stream: Stream,
) -> int:

    checked = 0

    for evaluation in stream.evaluations:
        if evaluation.recorded is None:
            continue

        recorded = np.asarray(evaluation.recorded["label"]).astype(np.int64)
        replayed = stream.labels[evaluation.positions]
        if recorded.shape != replayed.shape:
            raise StreamMismatch(
                f"stream {stream.name!r}: scores/{evaluation.leaf}.npz holds {recorded.shape[0]} "
                f"samples, the replayed stream {replayed.shape[0]}.",
            )

        if not np.array_equal(a1=recorded, a2=replayed):
            differing = np.flatnonzero(recorded != replayed)
            first = int(differing[0])
            raise StreamMismatch(
                f"stream {stream.name!r}: the replayed labels differ from "
                f"scores/{evaluation.leaf}.npz at {differing.size} of {recorded.size} positions "
                f"(first at {first}: recorded {recorded[first]}, replayed {replayed[first]}); the "
                f"sample order or the feature cache differs from the recorded run.",
            )

        checked += 1

    if checked == 0:
        raise FileNotFoundError(f"stream {stream.name!r}: the run recorded no scores/<leaf>.npz to verify against.")

    return checked


@dataclass(frozen=True, eq=False)
class ReplayBatch:

    index: int
    start: int
    stop: int
    window: slice
    rows: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    geometry: Geometry | None
    score: np.ndarray
    pred: np.ndarray


@dataclass(frozen=True, eq=False)
class CAMPReplay:

    score: np.ndarray
    pred: np.ndarray
    final: Geometry | None
    final_window: slice
    batches: int
    final_rows: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def _camp_config(
    config: Config,
    policy: CAMPPolicy,
) -> Config:

    from dataclasses import asdict

    from openood.utils.config import Config, merge_configs

    inner = merge_configs(config)
    if "postprocessor" in inner:
        del inner["postprocessor"]

    inner["postprocessor"] = Config({"name": "camp", "postprocessor_args": asdict(policy)})
    return inner


def _window_rows(
    window: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:

    past = window.rows()
    return None if past is None else (past.features, past.msp, past.pseudo_class)


def replay_camp(
    stream: Stream,
    features: RunFeatures,
    policy: CAMPPolicy | None = None,
    *,
    every_batch: Callable[[ReplayBatch], None] | None = None,
) -> CAMPReplay:

    import torch

    from openood.postprocessors.camp.postprocessor import CAMPPostprocessor
    from openood.utils.vlm.image.types import CachedImageFeatures

    if policy is None and features.run.method == "camp":
        config = features.config
    else:
        config = _camp_config(config=features.config, policy=features.run.policy() if policy is None else policy)

    postprocessor = CAMPPostprocessor(config)
    postprocessor.setup(features.net, None, None)
    postprocessor.reset_memory()
    window = postprocessor.window
    scores: list[np.ndarray] = []
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for index, (start, stop) in enumerate(stream.batches()):
            count = int(window.count)
            rows = geometry = None

            if every_batch is not None and window.ready:
                rows = _window_rows(window)
                geometry = camp_geometry(*rows, **postprocessor.geometry_args)

            batch = CachedImageFeatures(values=stream.raw[start:stop].to(postprocessor.device), normalized=True)
            pred, conf = postprocessor.postprocess(features.net, batch)
            score = conf.detach().cpu().numpy()
            prediction = pred.detach().cpu().numpy()
            scores.append(score)
            preds.append(prediction)

            if every_batch is not None:
                every_batch(
                    ReplayBatch(
                        index=index,
                        start=start,
                        stop=stop,
                        window=slice(start - count, start),
                        rows=rows,
                        geometry=geometry,
                        score=score,
                        pred=prediction,
                    ),
                )

        count = int(window.count)
        final_rows = _window_rows(window)
        final = camp_geometry(*final_rows, **postprocessor.geometry_args) if window.ready else None

    return CAMPReplay(
        score=np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32),
        pred=np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64),
        final=final,
        final_window=slice(len(stream) - count, len(stream)),
        batches=len(scores),
        final_rows=final_rows,
    )


def check_against_recorded(
    stream: Stream,
    recomputed: np.ndarray,
    *,
    tolerance: float = 1e-3,
) -> float:

    recomputed = np.asarray(recomputed, dtype=np.float64)
    if recomputed.shape != (len(stream),):
        raise StreamMismatch(
            f"stream {stream.name!r}: the recomputed score has shape {recomputed.shape}, "
            f"expected ({len(stream)},).",
        )

    worst = None

    for evaluation in stream.evaluations:
        if evaluation.recorded is None:
            continue

        recorded = np.asarray(evaluation.recorded["conf"], dtype=np.float64)
        values = recomputed[evaluation.positions]
        if recorded.shape != values.shape:
            raise StreamMismatch(
                f"stream {stream.name!r}: scores/{evaluation.leaf}.npz holds {recorded.shape[0]} "
                f"scores, the replayed stream {values.shape[0]}.",
            )

        difference = float(np.max(np.abs(recorded - values))) if recorded.size else 0.0
        worst = difference if worst is None else max(worst, difference)

    if worst is None:
        raise FileNotFoundError(f"stream {stream.name!r}: the run recorded no scores/<leaf>.npz to compare with.")

    if not np.isfinite(worst) or worst > tolerance:
        warnings.warn(
            message=f"stream {stream.name!r}: the recomputed CAMP score differs from the recorded conf by up to "
            f"{worst:.3g} (tolerance {tolerance:g}).",
            category=RuntimeWarning,
            stacklevel=2,
        )

    return worst


def default_config_path(
    method: str,
    root: Path | None = None,
) -> Path:
    return (REPO_ROOT if root is None else Path(root)) / "configs" / "postprocessors" / method / "default.yml"


def method_config(
    config: Config,
    method: str,
    root: Path | None = None,
) -> Config:

    from openood.utils.config import Config, merge_configs

    defaults = load_yaml(default_config_path(method=method, root=root))
    inner = merge_configs(config)
    if "postprocessor" in inner:
        del inner["postprocessor"]

    inner["postprocessor"] = Config(dict(defaults["postprocessor"]))

    for key, block in dict(defaults.get("vlm") or {}).items():
        if key != "text":
            inner[f"vlm.{key}"] = Config(dict(block))

    return inner


class _NpzTrace:

    def save(
        self,
        path: Path | str,
    ) -> Path:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{field.name: np.asarray(getattr(self, field.name)) for field in fields(self)})
        return path


    @classmethod
    def load(
        cls,
        path: Path | str,
    ) -> Any:

        with np.load(path) as data:
            missing = [field.name for field in fields(cls) if field.name not in data.files]
            if missing:
                raise ValueError(f"{path} is not a {cls.__name__}: it lacks {missing}.")

            return cls(**{field.name: np.asarray(data[field.name]) for field in fields(cls)})


@dataclass(frozen=True, eq=False)
class TANLTrace(_NpzTrace):

    memory_score: np.ndarray
    threshold: np.ndarray
    id_admit: np.ndarray
    ood_admit: np.ndarray
    conf: np.ndarray
    pred: np.ndarray


@dataclass(frozen=True, eq=False)
class AdaNegTrace(_NpzTrace):

    id_admit: np.ndarray
    ood_admit: np.ndarray
    conf: np.ndarray
    pred: np.ndarray


TRACE_VERSION = 2


def trace_path(
    cache_dir: Path | str,
    run: RecordedRun,
    stream: Stream,
    method: str,
) -> Path:

    return Path(cache_dir) / (f"{method}_trace_v{TRACE_VERSION}__{run.dir.parent.name}__{run.dir.name}__"
        f"{stream.name}.npz")


def _tracing_tanl_class() -> type:

    import torch

    from openood.postprocessors.tanl import postprocessor as tanl

    class TracingTANL(tanl.TANLPostprocessor):

        def __init__(
            self,
            config: Any,
        ) -> None:

            super().__init__(config)
            self.records: list[tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []


        def postprocess(
            self,
            net: Any,
            data: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:

            scores: list[torch.Tensor] = []
            admitted: list[int] = []
            score, update = tanl.activation_aware_score, tanl.update_queue

            def observed_score(
                *args: Any,
                **kwargs: Any,
            ) -> torch.Tensor:

                value = score(*args, **kwargs)
                scores.append(value)
                return value

            def observed_update(
                to_add: torch.Tensor,
                queue: torch.Tensor,
                max_len: int = 1000,
                init: bool = False,
            ) -> torch.Tensor:

                if not init:
                    admitted.append(int(to_add.shape[0]))

                return update(to_add, queue, max_len, init)

            tanl.activation_aware_score, tanl.update_queue = observed_score, observed_update

            try:
                pred, conf = super().postprocess(net, data)
            finally:
                tanl.activation_aware_score, tanl.update_queue = score, update

            memory_score = scores[-2].detach().float()
            threshold = float(self.thres)
            id_admit = memory_score > threshold + self.gap * (1.0 - threshold)
            ood_admit = memory_score < threshold - self.gap * threshold
            expected = ([int(id_admit.sum())] * 2 if bool(id_admit.any()) else []) + \
                ([int(ood_admit.sum())] * 2 if bool(ood_admit.any()) else [])
            if admitted != expected:
                raise RuntimeError("the recorded admissions disagree with the rows TANL's queues took.")

            self.records.append(
                (
                    memory_score.cpu().numpy(),
                    threshold,
                    id_admit.cpu().numpy(),
                    ood_admit.cpu().numpy(),
                    conf.detach().float().cpu().numpy(),
                    pred.detach().cpu().numpy(),
                ),
            )

            return pred, conf

    return TracingTANL


def _tracing_adaneg_class() -> type:

    import torch

    from openood.postprocessors.adaneg.postprocessor import AdaNegPostprocessor

    class TracingAdaNeg(AdaNegPostprocessor):

        def __init__(
            self,
            config: Any,
        ) -> None:

            super().__init__(config)
            self.records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []


        def _vanilla_score(
            self,
            data: Any,
        ) -> torch.Tensor:

            net = self.network
            class_num = net.n_cls
            image_features, text_features, logit_scale = net(data, return_feat=True)
            output_vanilla = logit_scale * image_features @ text_features.t()
            pos_logit = output_vanilla[:, :class_num]
            neg_logit = output_vanilla[:, class_num:]
            drop = neg_logit.size(1) % self.group_num
            if drop > 0:
                neg_logit = neg_logit[:, :-drop]

            if self.random_permute:
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                idx = torch.randperm(neg_logit.shape[1]).to(output_vanilla.device)
                neg_logit = neg_logit.T
                neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
            else:
                neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()

            scores = []

            for i in range(self.group_num):
                full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1).softmax(dim=-1)
                scores.append(full_sim[:, :pos_logit.shape[1]].sum(dim=-1).unsqueeze(-1))

            return torch.cat(scores, dim=-1).mean(dim=-1)


        def postprocess(
            self,
            net: Any,
            data: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:

            with torch.no_grad():
                conf_in_vanilla = self._vanilla_score(data)

            id_admit = conf_in_vanilla > (self.thres + self.gap * (1 - self.thres))
            ood_admit = conf_in_vanilla < (self.thres - self.gap * self.thres)
            pred, conf = super().postprocess(net, data)
            self.records.append(
                (
                    id_admit.cpu().numpy(),
                    ood_admit.cpu().numpy(),
                    conf.detach().float().cpu().numpy(),
                    pred.detach().cpu().numpy(),
                ),
            )

            return pred, conf

    return TracingAdaNeg


def _run_tracer(
    tracer: Any,
    features: RunFeatures,
    stream: Stream,
) -> None:

    import torch

    from openood.utils.vlm.image.types import CachedImageFeatures

    torch.manual_seed(features.run.seed)
    tracer.setup(features.net, None, None)
    tracer.reset_memory()

    with torch.no_grad():
        for start, stop in stream.batches():
            batch = CachedImageFeatures(values=stream.raw[start:stop].to(tracer.device), normalized=True)
            tracer.postprocess(features.net, batch)


def trace_tanl(
    features: RunFeatures,
    stream: Stream,
    *,
    cache_dir: Path | str | None = None,
    root: Path | None = None,
) -> TANLTrace:

    if cache_dir is None:
        path = None
    else:
        path = trace_path(cache_dir=cache_dir, run=features.run, stream=stream, method="tanl")

    if path is not None and path.is_file():
        return TANLTrace.load(path)

    tracer = _tracing_tanl_class()(method_config(features.config, "tanl", root))
    _run_tracer(tracer=tracer, features=features, stream=stream)
    records = tracer.records
    trace = TANLTrace(
        memory_score=np.concatenate([record[0] for record in records]).astype(np.float32),
        threshold=np.concatenate([np.full(shape=record[0].shape[0], fill_value=record[1], dtype=np.float32) for record in records]),
        id_admit=np.concatenate([record[2] for record in records]).astype(bool),
        ood_admit=np.concatenate([record[3] for record in records]).astype(bool),
        conf=np.concatenate([record[4] for record in records]).astype(np.float32),
        pred=np.concatenate([record[5] for record in records]).astype(np.int64),
    )

    if path is not None:
        trace.save(path)

    return trace


def trace_adaneg(
    features: RunFeatures,
    stream: Stream,
    *,
    cache_dir: Path | str | None = None,
    root: Path | None = None,
) -> AdaNegTrace:

    if cache_dir is None:
        path = None
    else:
        path = trace_path(cache_dir=cache_dir, run=features.run, stream=stream, method="adaneg")

    if path is not None and path.is_file():
        return AdaNegTrace.load(path)

    tracer = _tracing_adaneg_class()(method_config(features.config, "adaneg", root))
    _run_tracer(tracer=tracer, features=features, stream=stream)
    records = tracer.records
    trace = AdaNegTrace(
        id_admit=np.concatenate([record[0] for record in records]).astype(bool),
        ood_admit=np.concatenate([record[1] for record in records]).astype(bool),
        conf=np.concatenate([record[2] for record in records]).astype(np.float32),
        pred=np.concatenate([record[3] for record in records]).astype(np.int64),
    )

    if path is not None:
        trace.save(path)

    return trace


__all__ = [
    "AdaNegTrace",
    "BACKBONE_NAMES",
    "BENCHMARKS",
    "CAMPReplay",
    "Evaluation",
    "FEATURE_SOURCES",
    "Features",
    "RECORDED_SCORE_KEYS",
    "REPO_ROOT",
    "RecordedRun",
    "ReplayBatch",
    "RunFeatures",
    "Stream",
    "StreamMismatch",
    "TANLTrace",
    "TRACE_VERSION",
    "check_against_recorded",
    "default_config_path",
    "default_order",
    "features_of",
    "logged_cache_paths",
    "method_config",
    "replay_camp",
    "replay_config",
    "streams",
    "trace_adaneg",
    "trace_path",
    "trace_tanl",
    "verify_stream",
]
