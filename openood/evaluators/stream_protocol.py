from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from openood.utils.config import get_config
from openood.utils.reproducibility import make_namespaced_generator

COMPOSITIONS = ("per_leaf", "sequential", "simultaneous", "mixed_groups", "temporal")
PHASED = ("sequential", "temporal")
ID_PRIORS = ("longtail",)
STREAM_KEYS = ("composition", "ood_ratio", "id_prior", "imbalance", "leaf_order")


@dataclass(frozen=True)
class StreamProtocol:
    composition: str = "per_leaf"
    ood_ratio: float | None = None
    id_prior: str | None = None
    imbalance: float = 100.0
    leaf_order: tuple[str, ...] | None = None

    @classmethod
    def from_config(
        cls,
        node: Any,
    ) -> "StreamProtocol":

        if node is None:
            return cls()

        if not hasattr(node, "keys"):
            raise ValueError(f"evaluator.stream must be a mapping, got {type(node).__name__}.")

        unknown = sorted(set(map(str, node.keys())) - set(STREAM_KEYS))
        if unknown:
            raise ValueError(f"evaluator.stream: unknown key(s) {unknown}; allowed keys are {list(STREAM_KEYS)}.")

        ratio, prior = get_config(obj=node, name="ood_ratio", default=None), get_config(obj=node, name="id_prior", default=None)
        imbalance, order = get_config(obj=node, name="imbalance", default=None), get_config(obj=node, name="leaf_order", default=None)
        if order is not None and not isinstance(order, (list, tuple)):
            raise ValueError(f"evaluator.stream.leaf_order must be a list of leaf names, got {order!r}.")

        return cls(
            composition=str(get_config(obj=node, name="composition", default=None) or "per_leaf"),
            ood_ratio=None if ratio is None else float(ratio),
            id_prior=None if prior is None else str(prior),
            imbalance=100.0 if imbalance is None else float(imbalance),
            leaf_order=None if order is None else tuple(str(name) for name in order)
        )


    def __post_init__(self) -> None:

        if self.composition not in COMPOSITIONS:
            raise ValueError(f"evaluator.stream.composition must be one of {list(COMPOSITIONS)}, "
                f"got {self.composition!r}.")

        if self.ood_ratio is not None and not 0.0 < self.ood_ratio < 1.0:
            raise ValueError(f"evaluator.stream.ood_ratio must lie in (0, 1), got {self.ood_ratio}.")

        if self.id_prior is not None and self.id_prior not in ID_PRIORS:
            raise ValueError(f"evaluator.stream.id_prior must be one of {list(ID_PRIORS)}, got {self.id_prior!r}.")

        if not self.imbalance > 1.0:
            raise ValueError(f"evaluator.stream.imbalance must be > 1, got {self.imbalance}.")

        if self.composition == "temporal" and (self.ood_ratio is not None or self.id_prior is not None):
            raise ValueError("evaluator.stream composition temporal keeps each phase's whole ID set and OOD leaf; it takes "
                "no ood_ratio or id_prior.")

        if self.leaf_order is not None and self.composition != "temporal":
            raise ValueError("evaluator.stream.leaf_order orders the phases of composition temporal only.")

        if self.leaf_order is not None and len(set(self.leaf_order)) != len(self.leaf_order):
            raise ValueError(f"evaluator.stream.leaf_order names a leaf twice: {list(self.leaf_order)}.")


    @property
    def is_default(self) -> bool:
        return self.composition == "per_leaf" and self.ood_ratio is None and self.id_prior is None


    def describe(self) -> str:

        words = [f"composition={self.composition}"]

        if self.ood_ratio is not None:
            words.append(f"ood_ratio={self.ood_ratio:g}")

        if self.id_prior is not None:
            words.append(f"id_prior={self.id_prior}(imbalance={self.imbalance:g})")

        if self.leaf_order is not None:
            words.append("leaf_order=" + ">".join(self.leaf_order))

        return ", ".join(words)


@dataclass(frozen=True)
class StreamPart:
    role: str
    group: str | None
    leaf: str | None
    phase: int
    dataset: Dataset
    rows: np.ndarray


@dataclass(frozen=True)
class StreamPlan:
    name: str
    composition: str
    parts: tuple[StreamPart, ...]
    order: np.ndarray
    part_of: np.ndarray
    dataset: ConcatDataset

    def __len__(self) -> int:
        return int(len(self.order))


    def ood_parts(self) -> list[tuple[int, StreamPart]]:

        rows = [
            (index, part)
            for index, part in enumerate(self.parts)
            if part.role == "ood"
        ]

        return rows


    def ood_dataset(self) -> ConcatDataset:
        return ConcatDataset([Subset(part.dataset, part.rows.tolist()) for _, part in self.ood_parts()])


    def evaluation_positions(
        self,
        part_index: int,
    ) -> np.ndarray:

        part = self.parts[part_index]
        reference = [
            index for index, other in enumerate(self.parts)
            if other.role == "id" and (self.composition not in PHASED or other.phase == part.phase)
        ]
        return np.flatnonzero(np.isin(element=self.part_of, test_elements=[part_index, *reference]))


    def phases(self) -> list[dict[str, Any]]:

        rows = []

        for phase in sorted({part.phase for part in self.parts}):
            members = [index for index, part in enumerate(self.parts) if part.phase == phase]
            positions = np.flatnonzero(np.isin(element=self.part_of, test_elements=members))
            leaves = [self.parts[index].leaf for index in members if self.parts[index].role == "ood"]
            rows.append({
                "stream": self.name, 
                "order_idx_start": int(positions.min()),
                "order_idx_end": int(positions.max()) + 1,
                "phase": leaves[0] if self.composition in PHASED else self.composition
            })

        return rows


    def parts_table(self) -> list[dict[str, Any]]:

        rows = [{
            "stream": self.name, 
            "composition": self.composition, 
            "role": part.role, 
            "group": part.group or "",
            "leaf": part.leaf or "",
            "phase": part.phase,
            "rows": int(len(part.rows))
        } for part in self.parts ]

        return rows


def dataset_labels(
    dataset: Dataset,
) -> np.ndarray:

    labels = getattr(dataset, "labels", None)
    if labels is not None:
        values = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        return values.reshape(-1).astype(np.int64)[:len(dataset)]

    imglist = getattr(dataset, "imglist", None)
    if imglist is not None:
        parsed = []

        for line in imglist[:len(dataset)]:
            value = ast.literal_eval(line.strip("\n").split(" ", 1)[1])
            parsed.append(0 if isinstance(value, dict) else int(value))     # as ImglistDataset.__getitem__ reads it

        return np.asarray(parsed, dtype=np.int64)

    return np.asarray([int(dataset[index]["label"]) for index in range(len(dataset))], dtype=np.int64)


def _permutation(
    size: int,
    seed: int,
    name: str,
) -> np.ndarray:
    return torch.randperm(int(size), generator=make_namespaced_generator(seed=seed, namespace=f"stream/{name}")).numpy()


def _allocate(
    total: int,
    weights: Sequence[int],
) -> list[int]:

    weights = np.asarray(weights, dtype=np.float64)
    if total <= 0 or weights.sum() <= 0:
        return [0] * len(weights)

    exact = total * weights / weights.sum()
    counts = np.floor(exact).astype(np.int64)
    
    remainder = int(total - counts.sum())
    if remainder:
        counts[np.argsort(-(exact - counts), kind="stable")[:remainder]] += 1

    return counts.tolist()


def _longtail_rows(
    dataset: Dataset,
    imbalance: float,
    seed: int,
) -> np.ndarray:

    labels = dataset_labels(dataset)
    classes = np.unique(labels[labels >= 0])
    if len(classes) < 2:
        raise ValueError("evaluator.stream.id_prior longtail needs an ID set with at least two labelled classes.")

    members = {int(label): np.flatnonzero(labels == label) for label in classes}
    ceiling = max(len(rows) for rows in members.values())
    ranking = classes[_permutation(size=len(classes), seed=seed, name="id/class_rank")]
    kept = []

    for rank, label in enumerate(ranking):
        rows = members[int(label)]
        target = max(1, int(np.floor(ceiling * imbalance ** (-rank / (len(classes) - 1)))))
        kept.append(rows[_permutation(size=len(rows), seed=seed, name=f"id/class/{int(label)}")[:min(target, len(rows))]])

    return np.sort(np.concatenate(kept))


def build_stream_plans(
    protocol: StreamProtocol,
    *,
    id_dataset: Dataset,
    groups: Mapping[str, Mapping[str, Dataset]],
    seed: int,
) -> list[StreamPlan]:

    leaves = [(group, leaf, dataset) for group, members in groups.items() for leaf, dataset in members.items()]
    if not leaves:
        return []

    if protocol.composition == "per_leaf":
        streams = [(leaf, [(group, leaf, dataset)]) for group, leaf, dataset in leaves]
    elif protocol.composition in ("sequential", "simultaneous", "temporal"):
        streams = [(f"{group}_{protocol.composition}", [(group, leaf, dataset) for leaf, dataset in members.items()])
            for group, members in groups.items() if members]
    else:
        streams = [("mixed_groups", leaves)]

    stream_plans = [
        _plan(protocol=protocol, name=name, id_dataset=id_dataset, leaves=members, seed=seed)
        for name, members in streams
    ]

    return stream_plans


def _plan(
    protocol: StreamProtocol,
    name: str,
    id_dataset: Dataset,
    leaves: Sequence[tuple[str, str, Dataset]],
    seed: int,
) -> StreamPlan:

    sizes = [len(dataset) for _, _, dataset in leaves]
    share = protocol.ood_ratio
    id_rows = np.arange(len(id_dataset))

    if protocol.id_prior == "longtail":
        if share is None:
            share = sum(sizes) / (sum(sizes) + len(id_rows))

        id_rows = _longtail_rows(dataset=id_dataset, imbalance=protocol.imbalance, seed=seed)

    ood_rows = [np.arange(size) for size in sizes]

    if share is not None:
        ood_total = sum(sizes)
        if ood_total / (ood_total + len(id_rows)) > share:
            keep = _allocate(total=int(round(share / (1.0 - share) * len(id_rows))), weights=sizes)
            ood_rows = [np.sort(_permutation(size=size, seed=seed, name=f"ood/{leaf}")[:count])
                for size, count, (_, leaf, _) in zip(sizes, keep, leaves)]
        else:
            count = int(round((1.0 - share) / share * ood_total))
            id_rows = np.sort(id_rows[_permutation(size=len(id_rows), seed=seed, name="id/prevalence")[:count]])

    if protocol.composition == "temporal":
        names = [leaf for _, leaf, _ in leaves]
        if protocol.leaf_order is None:
            order = names
        else:
            order = [leaf for leaf in protocol.leaf_order if leaf in names]
        if sorted(order) != sorted(names):
            raise ValueError(f"stream {name}: evaluator.stream.leaf_order {list(protocol.leaf_order or ())} must name "
                f"every leaf of the group once; the group holds {names}.")

        position = {leaf: index for index, leaf in enumerate(names)}
        parts = []

        for phase, leaf in enumerate(order):
            group, _, dataset = leaves[position[leaf]]
            parts.append(StreamPart("ood", group, leaf, phase, dataset, ood_rows[position[leaf]]))
            # every phase repeats the whole ID set: the phases are the per-leaf streams of TANL's temporal shift
            parts.append(StreamPart("id", None, None, phase, id_dataset, id_rows))
    elif protocol.composition == "sequential":
        leaf_order = _permutation(size=len(leaves), seed=seed, name=f"{name}/leaf_order")
        shuffled = id_rows[_permutation(size=len(id_rows), seed=seed, name=f"{name}/id_phases")]
        counts = _allocate(total=len(id_rows), weights=[len(ood_rows[index]) for index in leaf_order])
        parts, start = [], 0

        for phase, (index, count) in enumerate(zip(leaf_order, counts)):
            group, leaf, dataset = leaves[int(index)]
            parts.append(StreamPart("ood", group, leaf, phase, dataset, ood_rows[int(index)]))
            parts.append(StreamPart("id", None, None, phase, id_dataset, np.sort(shuffled[start:start + count])))
            start += count
    else:
        parts = [StreamPart("ood", group, leaf, 0, dataset, rows)
            for (group, leaf, dataset), rows in zip(leaves, ood_rows)]
        parts.append(StreamPart("id", None, None, 0, id_dataset, id_rows))

    if any(len(part.rows) == 0 for part in parts):
        raise ValueError(f"stream {name}: {protocol.describe()} leaves an empty ID or OOD part.")

    offsets = np.concatenate([[0], np.cumsum([len(part.rows) for part in parts])])
    phases = sorted({part.phase for part in parts})
    order = []

    for phase in phases:
        members = [index for index, part in enumerate(parts) if part.phase == phase]
        positions = np.concatenate([np.arange(offsets[index], offsets[index + 1]) for index in members])
        tag = f"{name}/order" if len(phases) == 1 else f"{name}/order/{phase}"
        order.append(positions[_permutation(size=len(positions), seed=seed, name=tag)])

    order = np.concatenate(order)
    part_of = np.searchsorted(a=offsets, v=order, side="right") - 1
    dataset = ConcatDataset([Subset(part.dataset, part.rows.tolist()) for part in parts])
    return StreamPlan(name, protocol.composition, tuple(parts), order, part_of, dataset)


__all__ = ["COMPOSITIONS", "ID_PRIORS", "PHASED", "StreamPart", "StreamPlan", "StreamProtocol",
    "build_stream_plans", "dataset_labels"]
