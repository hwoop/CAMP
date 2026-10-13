from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import COVARIANCE_MODES, ID_REFERENCE_MODES, WEIGHTING_MODES


@dataclass(frozen=True, eq=False)
class IDGeometry:
    mu_stream: torch.Tensor    # [d]
    mu_id: torch.Tensor        # [d]
    sigma: torch.Tensor        # [d, d]
    operator: torch.Tensor     # [d, d]
    delta: torch.Tensor        # [d]
    r: torch.Tensor            # [d]
    u: torch.Tensor            # [d]
    k: int
    weights: torch.Tensor      # [n]
    eps: float

    @property
    def e(self) -> torch.Tensor:
        """Unit vector of the mean difference, ``delta / |delta|``."""
        return self.delta / self.delta.norm().clamp_min(1e-12)


def _check_mode(
    name: str,
    value: str,
    choices: tuple[str, ...],
) -> None:

    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}.")


@torch.no_grad()
def id_geometry(
    features: torch.Tensor,
    msp: torch.Tensor,
    pseudo_class: torch.Tensor,
    *,
    eps: float,
    class_num: int,
    covariance: str = "full",
    id_reference: str = "classwise",
    weighting: str = "msp",
    mu_stream: torch.Tensor | None = None,
) -> IDGeometry:

    _check_mode(name="covariance", value=covariance, choices=COVARIANCE_MODES)
    _check_mode(name="id_reference", value=id_reference, choices=ID_REFERENCE_MODES)
    _check_mode(name="weighting", value=weighting, choices=WEIGHTING_MODES)

    if int(class_num) <= 0:
        raise ValueError(f"class_num must be > 0, got {class_num}.")

    features = features.to(torch.float64)
    msp = msp.to(features.device, torch.float64)
    pseudo_class = pseudo_class.to(features.device, torch.long)

    if weighting == "uniform":
        msp = torch.ones_like(msp)

    class_mass = torch.zeros(
        int(class_num),
        dtype=torch.float64,
        device=features.device
    ).index_add_(0, pseudo_class, msp)

    k = int((class_mass > 0).sum())
    if k == 0:
        raise ValueError("no pseudo-class has positive mass: the window yields no direction.")

    if id_reference == "global":
        a = msp / msp.sum()
    else:
        a = msp / (k * class_mass[pseudo_class])

    mu_id = a @ features
    eye = torch.eye(
        int(features.shape[1]),
        dtype=torch.float64,
        device=features.device
    )

    sigma = features.T @ (a[:, None] * features) - torch.outer(mu_id, mu_id)

    if covariance == "identity":
        operator = eye
    else:
        operator = sigma + float(eps) * eye

    if mu_stream is None:
        stream_mean = features.mean(0)
    else:
        stream_mean = mu_stream.to(features.device, torch.float64)

    delta = stream_mean - mu_id
    r = torch.linalg.solve(operator, delta)
    u = r / r.norm().clamp_min(1e-12)

    return IDGeometry(
        mu_stream=stream_mean.detach(),
        mu_id=mu_id.detach(),
        sigma=sigma.detach(),
        operator=operator.detach(),
        delta=delta.detach(),
        r=r.detach(),
        u=u.detach(),
        k=k,
        weights=a.detach(),
        eps=float(eps),
    )


__all__ = ["IDGeometry", "id_geometry"]
