from __future__ import annotations

import math

import torch


def projection_score(
    features: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Projection score ``R(x) = -u^T f(x)`` for feature rows of shape [n, d]."""
    return -(features.detach().to(torch.float64) @ u)


def moments(
    reference: torch.Tensor,
) -> tuple[float, float]:
    """Return the mean and the population variance of ``reference``. The variance is floored at 1e-12."""

    reference = reference.to(torch.float64)
    return float(reference.mean()), max(float(reference.var(unbiased=False)), 1e-12)


def standardize(
    values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return ``(values - mean(reference)) / std(reference)``."""

    m, v = moments(reference)
    return (values.detach().to(torch.float64) - m) / math.sqrt(v)


def combine(
    z_semantic_score: torch.Tensor,
    z_camp_score: torch.Tensor,
    semantic_weight: float,
) -> torch.Tensor:
    """Return ``semantic_weight * z(S) + z(R)``."""

    if semantic_weight == 0.0:
        return torch.zeros_like(z_semantic_score) + z_camp_score

    if semantic_weight == 1.0:
        return z_semantic_score + z_camp_score

    return semantic_weight * z_semantic_score + z_camp_score


__all__ = ["combine", "moments", "projection_score", "standardize"]
