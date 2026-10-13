from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
from openood.utils import get_config
from openood.utils.vlm.image.types import CachedImageFeatures
from openood.utils.vlm.text import load_text_features

# the eps of the frozen VLM view's normalisation (CAMP and its warm-up score), as its recorded runs used it
FEATURE_NORM_EPS = 1e-8


@dataclass(frozen=True)
class FrozenVLMBatch:
    raw: torch.Tensor
    features: torch.Tensor
    id_logits: torch.Tensor
    pred: torch.Tensor
    msp: torch.Tensor


def normalize_features(
    raw: torch.Tensor,
) -> torch.Tensor:
    return F.normalize(raw.float(), dim=-1, eps=FEATURE_NORM_EPS)


@torch.no_grad()
def build_frozen_vlm_batch(
    raw: torch.Tensor,
    logit_scale: torch.Tensor,
    id_prototypes: torch.Tensor,
) -> FrozenVLMBatch:

    features = normalize_features(raw)
    id_logits = logit_scale * features @ id_prototypes
    msp = torch.softmax(id_logits.float(), dim=1).amax(dim=1)
    pred = id_logits.argmax(dim=1)

    return FrozenVLMBatch(
        raw=raw,
        features=features,
        id_logits=id_logits,
        pred=pred,
        msp=msp,
    )


class VLMPostprocessor(BasePostprocessor):
    supports_image_feature_cache = True

    def __init__(
        self,
        config,
    ):

        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args


    def setup(
        self,
        net: nn.Module,
        id_loader_dict,
        ood_loader_dict,
    ):

        super().setup(net, id_loader_dict, ood_loader_dict)

        self.dataset_name = self.config.dataset.name
        self.model_name = self.config.network.name
        self.backbone_name = self.config.network.backbone.name
        self.text_prompt = get_config(obj=self.config.vlm.text, name="prompt", default=None)
        if self.text_prompt is None:
            raise ValueError("vlm.text.prompt is required: the prompt set of the ID class prototypes (e.g. nice).")

        self.text_center = get_config(obj=self.config.vlm.text, name="center", default=True)
        self.batch_size = get_config(obj=self.config.dataset.test, name='batch_size', default=256)
        self.device = self._infer_device(net)

        self.logit_scale = net.get_logit_scale().to(self.device)

        id_text_features = load_text_features(
            model=net.model,
            dataset=self.dataset_name,
            backbone=self.backbone_name,
            text_prompt=self.text_prompt,
            model_name=self.model_name,
            text_center=self.text_center,
            device=self.device,
            batch_size=self.batch_size,
        )

        self.id_txt_feat = F.normalize(id_text_features, dim=0)
        self.class_num = self.id_txt_feat.shape[1]

        print(
            "[VLMPostprocessor] setup complete | "
            f"dataset={self.dataset_name}, model={self.model_name}, "
            f"backbone={self.backbone_name}, prompt={self.text_prompt}, "
            f"text_center={self.text_center}, class_num={self.class_num}"
        )


    @torch.no_grad()
    def _extract_image_features(
        self,
        net: nn.Module,
        data: Any,
    ) -> torch.Tensor:

        if isinstance(data, CachedImageFeatures):
            image_feat = data.values
        else:
            image_feat = net.encode_image(data, normalize=True)

        return image_feat.to(self.device)
