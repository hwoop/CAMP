from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from .geometry import Geometry


def normalized_weights(
    confidence: Any,
    pseudo: Any,
    *,
    id_reference: str = "classwise",
    weighting: str = "msp",
) -> np.ndarray:

    confidence = np.asarray(confidence, dtype=np.float64)
    pseudo = np.asarray(pseudo, dtype=np.int64)
    if confidence.ndim != 1 or pseudo.shape != confidence.shape or not len(confidence):
        raise ValueError("confidence and pseudo-class must be aligned, non-empty vectors")

    if not np.isfinite(confidence).all() or (confidence <= 0).any() or (pseudo < 0).any():
        raise ValueError("confidence must be finite and positive; pseudo-classes must be non-negative")

    if id_reference not in ("classwise", "global") or weighting not in ("msp", "uniform"):
        raise ValueError("expected classwise/global reference and msp/uniform weighting")

    weight = confidence if weighting == "msp" else np.ones_like(confidence)

    if id_reference == "global":
        return weight / weight.sum()

    mass = np.bincount(pseudo, weights=weight)
    return weight / (np.count_nonzero(mass) * mass[pseudo])


def weight_context(
    confidence: Any,
    pseudo: Any,
    is_ood: Any,
    *,
    id_reference: str = "classwise",
    weighting: str = "msp",
) -> dict[str, float | int]:

    a = normalized_weights(confidence=confidence, pseudo=pseudo, id_reference=id_reference, weighting=weighting)
    is_ood = np.asarray(is_ood, dtype=bool)
    if is_ood.shape != a.shape:
        raise ValueError("is_ood must line up with the window rows")

    class_weight = np.bincount(np.asarray(pseudo, dtype=np.int64), weights=a)
    return {
        "ood_sample_share": 100.0 * float(is_ood.mean()),
        "ood_weight_share": 100.0 * float(a[is_ood].sum()),
        "max_class_weight": 100.0 * float(class_weight.max()),
        "classes": int(np.count_nonzero(class_weight)),
    }


def separation_context(
    features: torch.Tensor,
    labels: Any,
    estimate: Geometry,
    *,
    class_num: int,
) -> dict[str, float | int]:

    import torch

    from . import geometry

    labels = torch.as_tensor(labels, device=features.device, dtype=torch.long)
    if labels.shape != (int(features.shape[0]),):
        raise ValueError("labels must line up with the stream features")

    is_id = labels >= 0
    if not bool(is_id.any()) or not bool((~is_id).any()):
        raise ValueError("separation context requires both ID and OOD samples")

    oracle = geometry.oracle_geometry(
        features[is_id],
        labels[is_id],
        mu_stream=estimate.mu_stream,
        eps=estimate.eps,
        class_num=class_num,
    )

    projection = geometry.project(features, estimate.u)
    id_projection, ood_projection = projection[is_id], projection[~is_id]
    gap = float(ood_projection.mean() - id_projection.mean())
    std_id = float(id_projection.std(unbiased=False))
    return {
        "delta_norm": float(estimate.delta.norm()),
        "cos_to_oracle": geometry.cosine(estimate.u, oracle.u),
        "projection_gap": gap,
        "gap_id_std": gap / std_id if std_id > 0 else math.nan,
        "n_id": int(is_id.sum()),
        "n_ood": int((~is_id).sum()),
    }


__all__ = [
    "normalized_weights",
    "separation_context",
    "weight_context",
]
