"""CSP, ported from MengyuanChen21/NeurIPS2024-CSP@3a7bdabeb89301053ba8bfc56b227b6d81b273e9."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.networks.clip_fixed_ood_prompt import model_device, text_cache_settings
from openood.utils import get_config
from openood.utils.vlm.image.types import CachedImageFeatures
from openood.utils.vlm.negative_labels.csp import (
    CSP_ADJ_PROMPT,
    CSP_CORPUS_ROOT,
    CSP_EXCLUDED_FILES,
    CSP_NOUN_PROMPT,
    CSP_SUPERCLASS_PROMPT,
    load_csp_text_features,
    parse_excluded_files,
)
from openood.utils.vlm.text import load_text_features

from ..base_postprocessor import BasePostprocessor
from ..method_arguments import dataset_name, integer_argument, number_argument, resolve_method_arguments
from ..neglabel.postprocessor import ScalableClassifier as NegLabelScalableClassifier

SOURCE = "MengyuanChen21/NeurIPS2024-CSP@3a7bdabeb89301053ba8bfc56b227b6d81b273e9"
CSP_ARGUMENTS: dict[str, Any] = {"t": 1, "ngroup": 100, "group_fuse_num": None, "seed": 0}


def csp_pool_settings(config: Any) -> dict[str, Any]:
    csp = get_config(config.vlm, "csp", None)
    return {
        "corpus_root": str(get_config(csp, "corpus_root", CSP_CORPUS_ROOT)),
        "exclude": list(parse_excluded_files(get_config(csp, "exclude", CSP_EXCLUDED_FILES))),
        "noun_prompt": str(get_config(csp, "noun_prompt", CSP_NOUN_PROMPT)),
        "adj_prompt": str(get_config(csp, "adj_prompt", CSP_ADJ_PROMPT)),
        "superclass_prompt": str(get_config(csp, "superclass_prompt", CSP_SUPERCLASS_PROMPT)),
        "select_ratio": float(get_config(csp, "select_ratio", 0.15)),
        "cos_percentile": float(get_config(csp, "cos_percentile", 0.95)),
        "max_cos": float(get_config(csp, "max_cos", 0.95)),
    }


class CLIPScalableClassifier:
    """
    CSP's ``CLIPScalableClassifier`` with ``cls_mode=False`` and no ``ft_head``. 
    Provides the text features and ``simple_test``.
    """

    def __init__(self, net: Any, config: Any, seed: int = 0) -> None:
        self.device = model_device(net.model)
        self.clip_model = net.model
        self.clip_model.eval()
        self.model = self.clip_model
        self.cls_mode = False

        self.ft_head = None

        # of the encoded templates, then the pool built and selected), as [labels, D] rows
        settings = text_cache_settings(config)
        text_features = load_text_features(
            model=net.model, dataset=settings["dataset"], device=self.device, backbone=settings["backbone"],
            text_prompt=settings["prompt"], model_name=settings["model_name"], text_center=settings["text_center"],
            batch_size=settings["batch_size"],
        )
        cache = load_csp_text_features(
            model=net.model, dataset=settings["dataset"], model_name=settings["model_name"],
            backbone=settings["backbone"], prompt_name=settings["prompt"], text_center=settings["text_center"],
            id_text_features=F.normalize(text_features, dim=0), device=self.device,
            batch_size=settings["batch_size"], seed=seed, **csp_pool_settings(config),
        )
        with torch.no_grad():
            self.text_features_pos = text_features.to(self.device).t().to(torch.float32)
            self.feat_dim = self.text_features_pos.shape[-1]
            self.text_features_pos /= self.text_features_pos.norm(dim=-1, keepdim=True)

            self.text_features_neg = cache["neg_text_features"].to(self.device).t().contiguous()
            self.text_features_neg /= self.text_features_neg.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            self.text_features_neg = self.text_features_neg.to(torch.float32)

    def simple_test(self, img, img_metas=None, require_features=False, require_backbone_features=False, softmax=True,
                    **kwargs):
        """Test without augmentation."""
        with torch.no_grad():
            if isinstance(img, CachedImageFeatures):
                image_features = img.values.to(self.device)
            else:
                image_features = self.model.encode_image(img)
                image_features /= image_features.norm(dim=-1, keepdim=True)

        if self.cls_mode:
            image_features = image_features.to(torch.float32)
            self.text_features_pos = self.text_features_pos.to(torch.float32)
            if self.ft_head is None:
                pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            else:
                pos_sim = self.ft_head(image_features)
            pos_sim = list(pos_sim.softmax(dim=-1).detach().cpu().numpy())
            return pos_sim
        else:
            image_features = image_features.to(torch.float32)
            self.text_features_pos = self.text_features_pos.to(torch.float32)
            self.text_features_neg = self.text_features_neg.to(torch.float32)
            if self.ft_head is None:
                pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            else:
                pos_sim = self.ft_head(image_features)
            neg_sim = (100.0 * image_features @ self.text_features_neg.T)
            return pos_sim, neg_sim


class ScalableClassifier(NegLabelScalableClassifier):
    """
    CSP's ``ScalableClassifier``. 
    ``__init__`` and ``grouping`` are the same code as NegLabel's; 
    ``forward`` is CSP's own.
    """

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        if "dataset_name" in input and self.dump_score:
            dataset_name = input['dataset_name']
            dump_path = os.path.join(self.dump_score, "{}".format(dataset_name))
            os.makedirs(dump_path, exist_ok=True)
            del input['dataset_name']
        with torch.no_grad():
            pos, neg = self.classifier.simple_test(**input)
            full_sim = torch.cat([pos, neg], dim=-1) / self.t

            if self.ngroup > 1:
                score = self.grouping(pos, neg, num=self.group_fuse_num, ngroup=self.ngroup, random_permute=True)
            else:
                full_sim = full_sim.softmax(dim=-1)
                pos_score = full_sim[:, :pos.shape[1]].sum(dim=1)
                score = pos_score

        return score, pos


def csp_arguments(config: Any) -> SimpleNamespace:
    values, setting = resolve_method_arguments("csp", config.postprocessor.postprocessor_args, CSP_ARGUMENTS,
                                               dataset=dataset_name(config))
    fuse = values["group_fuse_num"]
    return SimpleNamespace(
        t=number_argument("csp", values, "t"),
        ngroup=integer_argument("csp", values, "ngroup", minimum=1),
        group_fuse_num=None if fuse is None else integer_argument("csp", values, "group_fuse_num", minimum=1),
        seed=integer_argument("csp", values, "seed", minimum=0),
        setting=setting,
    )


class CSPPostprocessor(BasePostprocessor):
    """
    Wrapper around the official ``ScalableClassifier`` over ``CLIPScalableClassifier``,
    built on this repository's CLIP model and caches.
    """

    supports_image_feature_cache = True

    def __init__(self, config):
        super().__init__(config)
        self.args = csp_arguments(config)
        self.model: ScalableClassifier | None = None
        self.device: torch.device | None = None

    def setup(self, net: nn.Module, id_loader_dict=None, ood_loader_dict=None):
        if self.model is None:
            classifier = CLIPScalableClassifier(net, self.config, seed=self.args.seed)
            self.model = ScalableClassifier(classifier, t=self.args.t, ngroup=self.args.ngroup,
                                            group_fuse_num=self.args.group_fuse_num)
            self.device = classifier.device
            print(f"[CSPPostprocessor] setup complete | {self.policy_summary()}", flush=True)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if self.model is None:
            raise RuntimeError("setup() must be called before postprocess().")
        score, pos = self.model.forward(img=data)
        return pos.argmax(dim=1), score

    def policy_summary(self) -> dict[str, Any]:
        summary = {"source": SOURCE, "t": self.args.t, "ngroup": self.args.ngroup,
                   "group_fuse_num": self.args.group_fuse_num, "seed": self.args.seed, "setting": self.args.setting}
        if self.model is not None:
            summary.update({"id_classes": int(self.model.classifier.text_features_pos.shape[0]),
                            "pool": int(self.model.classifier.text_features_neg.shape[0])})
        return summary


__all__ = ["CLIPScalableClassifier", "CSPPostprocessor", "CSP_ARGUMENTS", "SOURCE", "ScalableClassifier",
           "csp_arguments", "csp_pool_settings"]
