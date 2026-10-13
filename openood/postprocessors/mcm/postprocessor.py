"""MCM, ported from deeplearning-wisc/MCM@ea7130f851e7d462cacd21f0e87a127705700bd9. Also provides CAMP's warm-up score.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.networks.clip_fixed_ood_prompt import text_cache_settings
from openood.utils.vlm.image.types import CachedImageFeatures
from openood.utils.vlm.text import load_text_features

from ..base_postprocessor import BasePostprocessor
from ..method_arguments import as_mapping, boolean_argument, number_argument
from ..vlm_postprocessor import build_frozen_vlm_batch

SOURCE = "deeplearning-wisc/MCM@ea7130f851e7d462cacd21f0e87a127705700bd9"
MCM_ARGUMENTS: dict[str, Any] = {"tau": 1.0, "logit_scale": True}


def mcm_arguments(config: Any) -> SimpleNamespace:
    args = as_mapping(config.postprocessor.postprocessor_args, path="mcm.postprocessor_args")
    unknown = sorted(set(args) - set(MCM_ARGUMENTS))
    if unknown:
        raise ValueError(f"mcm.postprocessor_args: unknown key(s) {unknown}; allowed keys are {sorted(MCM_ARGUMENTS)}.")
    values = {**MCM_ARGUMENTS, **args}
    tau = number_argument("mcm", values, "tau")
    if tau <= 0.0:
        raise ValueError(f"mcm.postprocessor_args.tau must be > 0, got {tau}.")
    return SimpleNamespace(tau=tau, logit_scale=boolean_argument("mcm", values, "logit_scale"))


class MCMPostprocessor(BasePostprocessor):
    supports_image_feature_cache = True

    def __init__(self, config):
        super().__init__(config)
        self.args = mcm_arguments(config)
        self.tau = self.args.tau
        self.use_logit_scale = self.args.logit_scale
        self.device: torch.device | None = None
        self.text_features: torch.Tensor | None = None      # [C, D] float32
        self.logit_scale: torch.Tensor | None = None        # CAMP's warm-up: CLIP's logit scale
        self._id_prototypes: torch.Tensor | None = None     # CAMP's warm-up: [D, C] float32

    def setup(self, net: nn.Module, id_loader_dict=None, ood_loader_dict=None):
        if self.device is not None:
            return
        self.device = self._infer_device(net)
        settings = text_cache_settings(self.config)
        id_text_features = load_text_features(
            model=net.model, dataset=settings["dataset"], device=self.device, backbone=settings["backbone"],
            text_prompt=settings["prompt"], model_name=settings["model_name"], text_center=settings["text_center"],
            batch_size=settings["batch_size"],
        )
        if self.use_logit_scale:
            # CAMP's warm-up: the prototypes and logit scale of VLMPostprocessor.setup, as CAMP reads them
            self.logit_scale = net.get_logit_scale().to(self.device)
            self._id_prototypes = F.normalize(F.normalize(id_text_features, dim=0), dim=0).float()
        else:
            with torch.no_grad():
                text_features = id_text_features.t().float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features
        print(f"[MCMPostprocessor] setup complete | {self.policy_summary()}", flush=True)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if self.device is None:
            raise RuntimeError("setup() must be called before postprocess().")
        if self.use_logit_scale:
            return self._warmup(net, data)
        to_np = lambda x: x.data.cpu().numpy()
        with torch.no_grad():
            if isinstance(data, CachedImageFeatures):
                image_features = data.values.to(self.device).float()
                if image_features is data.values:
                    image_features = image_features.clone()
            else:
                images = data.to(self.device)
                image_features = net.model.encode_image(images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            output = image_features @ self.text_features.T
            smax = to_np(F.softmax(output/ self.tau, dim=1))
            score = -np.max(smax, axis=1)
        return output.argmax(dim=1), torch.from_numpy(-score).to(output.device)

    def _warmup(self, net: nn.Module, data: Any):
        """
        Warm-up score used by CAMP when ``logit_scale`` is true,
        the maximum softmax probability of the scaled zero-shot logits.
        """
        if isinstance(data, CachedImageFeatures):
            raw = data.values.to(self.device)
        else:
            raw = net.encode_image(data, normalize=True).to(self.device)
        vlm = build_frozen_vlm_batch(raw, self.logit_scale, self._id_prototypes)
        probability = torch.softmax((vlm.id_logits / self.tau).float(), dim=1)
        conf, pred = probability.max(dim=1)
        return pred, conf

    def policy_summary(self) -> dict[str, Any]:
        if self.use_logit_scale:
            return {"tau": self.tau, "logit_scale": True,
                    "score": "CAMP's warm-up: max softmax over the ID classes of logit_scale cos(f, t_c) / tau"}
        return {"source": SOURCE, "tau": self.tau, "logit_scale": False,
                "score": "official MCM: max softmax over the ID classes of cos(f, t_c) / T"}


__all__ = ["MCMPostprocessor", "MCM_ARGUMENTS", "SOURCE", "mcm_arguments"]
