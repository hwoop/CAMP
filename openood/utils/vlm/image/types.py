from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CachedImageFeatures:

    values: torch.Tensor
    normalized: bool

    def __post_init__(self) -> None:

        if self.values.ndim != 2:
            raise ValueError(
                "Cached image features must have shape [batch, feature_dim], "
                f"got {tuple(self.values.shape)}."
            )

        if not self.normalized:
            raise ValueError(
                "Image feature cache schema v1 requires normalized encode_image outputs."
            )


__all__ = ["CachedImageFeatures"]
