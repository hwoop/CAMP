"""
Window geometry of CAMP for the analyses, reusing the runtime implementation,
with oracle comparisons and OOD metrics.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from openood.postprocessors.camp.geometry import IDGeometry as Geometry

COVARIANCE_MODES = ("full", "identity")
ID_REFERENCE_MODES = ("classwise", "global")
WEIGHTING_MODES = ("msp", "uniform")


def __getattr__(
    name: str,
) -> Any:

    if name == "Geometry":
        from openood.postprocessors.camp.geometry import IDGeometry

        globals()[name] = IDGeometry
        return IDGeometry

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def camp_geometry(
    F: torch.Tensor,
    w: torch.Tensor,
    c: torch.Tensor,
    *,
    eps: float,
    class_num: int,
    covariance: str = "full",
    id_reference: str = "classwise",
    weighting: str = "msp",
    mu_stream: torch.Tensor | None = None,
) -> Geometry:

    from openood.postprocessors.camp.geometry import id_geometry

    return id_geometry(
        features=F,
        msp=w,
        pseudo_class=c,
        eps=eps,
        class_num=class_num,
        covariance=covariance,
        id_reference=id_reference,
        weighting=weighting,
        mu_stream=mu_stream,
    )


def oracle_geometry(
    F_id: torch.Tensor,
    labels: torch.Tensor,
    *,
    mu_stream: torch.Tensor,
    eps: float,
    class_num: int,
    covariance: str = "full",
    id_reference: str = "classwise",
) -> Geometry:

    import torch

    labels = labels.to(torch.long)
    if bool((labels < 0).any()):
        raise ValueError("oracle_geometry needs ID rows only (true class labels >= 0).")

    return camp_geometry(
        F=F_id,
        w=torch.ones(int(F_id.shape[0]), dtype=torch.float64, device=F_id.device),
        c=labels,
        eps=eps,
        class_num=class_num,
        covariance=covariance,
        id_reference=id_reference,
        weighting="uniform",
        mu_stream=mu_stream,
    )


def camp_score(
    features: torch.Tensor,
    u: torch.Tensor,
    m: float,
    v: float,
) -> torch.Tensor:

    from openood.postprocessors.camp.scores import projection_score

    return (projection_score(features=features, u=u) - m) / math.sqrt(v)


def normalize_features(
    raw: torch.Tensor,
) -> torch.Tensor:

    import torch.nn.functional as functional

    from openood.postprocessors.vlm_postprocessor import FEATURE_NORM_EPS

    return functional.normalize(raw.float(), dim=-1, eps=FEATURE_NORM_EPS)


def zero_shot(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor,
    torch.Tensor,
    torch.Tensor]:

    import torch

    id_logits = logit_scale * features @ prototypes
    msp = torch.softmax(id_logits.float(), dim=1).amax(dim=1)
    pred = id_logits.argmax(dim=1)
    return msp, pred, id_logits


def project(
    features: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:

    import torch

    return features.to(torch.float64) @ u.to(features.device, torch.float64)


def standardize(
    values: Any,
    reference: Any,
) -> torch.Tensor:

    import torch

    values = torch.as_tensor(values).to(torch.float64)
    reference = torch.as_tensor(reference).to(values.device, torch.float64)
    return (values - reference.mean()) / reference.std(unbiased=False).clamp_min(1e-12)


def _as_numpy(
    values: Any,
) -> np.ndarray:

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    return np.asarray(values).reshape(-1)


def auroc(
    score: Any,
    is_id: Any,
) -> float:

    from openood.evaluators.metrics import auc_and_fpr_recall

    label = np.where(_as_numpy(is_id).astype(bool), 0, -1)
    value, _, _, _ = auc_and_fpr_recall(conf=_as_numpy(score).astype(np.float64), label=label, tpr_th=0.95)
    return float(value)


def fpr95(
    score: Any,
    is_ood: Any,
) -> float:

    from openood.evaluators.metrics import auc_and_fpr_recall

    label = np.where(_as_numpy(is_ood).astype(bool), -1, 0)
    _, _, _, value = auc_and_fpr_recall(conf=_as_numpy(score).astype(np.float64), label=label, tpr_th=0.95)
    return float(value)


def mahalanobis(
    mu: torch.Tensor,
    mu_ref: torch.Tensor,
    operator_ref: torch.Tensor,
) -> float:

    import torch

    diff = mu.to(torch.float64) - mu_ref.to(mu.device, torch.float64)
    quadratic = diff @ torch.linalg.solve(operator_ref.to(mu.device, torch.float64), diff)
    return float(torch.sqrt(quadratic.clamp_min(0.0)))


def inverse_sqrt(
    matrix: torch.Tensor,
) -> torch.Tensor:

    import torch

    values, vectors = torch.linalg.eigh(matrix.to(torch.float64))
    return (vectors * values.clamp_min(1e-12).rsqrt()) @ vectors.T


def cosine(
    u: torch.Tensor,
    v: torch.Tensor,
) -> float:

    import torch

    u = u.to(torch.float64)
    v = v.to(u.device, torch.float64)
    return float(u @ v / (u.norm() * v.norm()).clamp_min(1e-12))


__all__ = [
    "COVARIANCE_MODES",
    "Geometry",
    "ID_REFERENCE_MODES",
    "WEIGHTING_MODES",
    "auroc",
    "camp_geometry",
    "camp_score",
    "cosine",
    "fpr95",
    "inverse_sqrt",
    "mahalanobis",
    "normalize_features",
    "oracle_geometry",
    "project",
    "standardize",
    "zero_shot",
]
