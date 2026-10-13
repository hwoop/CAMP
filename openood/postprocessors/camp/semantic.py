from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from openood.utils import Config
from openood.utils.vlm.image.types import CachedImageFeatures

from ..utils import get_inner_postprocessor


PROBABILITY_SCORE = {
    "tanl": lambda detector: True,
    "mcm": lambda detector: True,
    "neglabel": lambda detector: True,
    "csp": lambda detector: True,
    "adaneg": lambda detector: detector.in_score != "combine",
    "oodd": lambda detector: False,
    "coevo": lambda detector: True,
}

SEMANTIC_METHODS = tuple(PROBABILITY_SCORE)


@dataclass(frozen=True)
class SemanticScore:
    native: torch.Tensor    # [B] the detector's own score, larger = more ID
    logit: torch.Tensor     # [B] the same score on its additive axis
    pred: torch.Tensor      # [B] the detector's class prediction


def probability_to_logit(
    probability: torch.Tensor,
) -> torch.Tensor:

    p = probability.detach().float().clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


class SemanticPlugin:
    """Wrapper around another detector's postprocessor that exposes only its score."""

    def __init__(
        self,
        config: Config,
        method: str,
        args: Mapping[str, Any],
    ) -> None:

        if method not in SEMANTIC_METHODS:
            raise ValueError(f"Unknown semantic method: {method!r}.")

        self.name = method
        self.detector = get_inner_postprocessor(config=config, name=method, args=args)
        self.probability = PROBABILITY_SCORE[method](self.detector)


    def setup(
        self,
        net: nn.Module,
        id_loader_dict=None,
        ood_loader_dict=None,
    ) -> None:
        self.detector.setup(net, id_loader_dict, ood_loader_dict)


    def reset(self) -> None:
        self.detector.reset_memory()


    @torch.no_grad()
    def score(
        self,
        net: nn.Module,
        image_features: torch.Tensor,
    ) -> SemanticScore:

        # the detector reads the rows CAMP already encoded
        data = CachedImageFeatures(values=image_features, normalized=True)
        pred, conf = self.detector.postprocess(net, data)

        native = conf.detach()
        logit = probability_to_logit(native) if self.probability else native.float()

        return SemanticScore(
            native=native,
            logit=logit,
            pred=pred.detach(),
        )


    def summary(self) -> dict[str, Any]:

        detector = type(self.detector)

        return {
            "method": self.name,
            "implementation": f"{detector.__module__}.{detector.__name__}",
            "additive_axis": "logit" if self.probability else "native",
            **self.detector.policy_summary(),
        }


__all__ = ["SEMANTIC_METHODS", "SemanticPlugin", "SemanticScore", "probability_to_logit"]
