"""NegLabel, ported from tmlr-group/NegLabel@3253db684075b2db47844676eccdfda40d67a573."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from openood.networks.clip_fixed_ood_prompt import load_id_and_negative_features, model_device
from openood.utils.vlm.image.types import CachedImageFeatures

from ..base_postprocessor import BasePostprocessor
from ..method_arguments import dataset_name, integer_argument, number_argument, resolve_method_arguments

SOURCE = "tmlr-group/NegLabel@3253db684075b2db47844676eccdfda40d67a573"
NEGLABEL_ARGUMENTS: dict[str, Any] = {"t": 1, "ngroup": 100, "group_fuse_num": None}


class CLIPScalableClassifier:
    """
    NegLabel's ``CLIPScalableClassifier`` with ``cls_mode=False``. 
    Provides the text features and ``simple_test``.
    """

    def __init__(self, net: Any, config: Any) -> None:
        self.device = model_device(net.model)
        self.clip_model = net.model
        self.clip_model.eval()
        self.model = self.clip_model
        self.cls_mode = False

        text_features, cache = load_id_and_negative_features(net, config)
        with torch.no_grad():
            self.text_features_pos = text_features.t().to(torch.float32)
            self.text_features_pos /= self.text_features_pos.norm(dim=-1, keepdim=True)

            self.text_features_neg = cache["neg_text_features"].to(self.device).t().contiguous()
            self.text_features_neg /= self.text_features_neg.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            self.text_features_neg = self.text_features_neg.to(torch.float32)

    def simple_test(self, img, img_metas=None, require_features=False, require_backbone_features=False, softmax=True, **kwargs):
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
            pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            pos_sim = list(pos_sim.softmax(dim=-1).detach().cpu().numpy())
            return pos_sim
        else:
            image_features = image_features.to(torch.float32)
            self.text_features_pos = self.text_features_pos.to(torch.float32)
            self.text_features_neg = self.text_features_neg.to(torch.float32)
            pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            neg_sim = (100.0 * image_features @ self.text_features_neg.T)
            return pos_sim, neg_sim


class ScalableClassifier:
    def __init__(self, classifier, t=1, ngroup=20, group_fuse_num=None, dump_score=None, **kwargs):
        self.classifier = classifier
        self.dump_score = dump_score
        self.t = t
        self.ngroup=ngroup
        self.group_fuse_num = group_fuse_num

    def grouping(self, pos, neg, num, ngroup=10, random_permute=False):
        group = ngroup
        drop = neg.shape[1] % ngroup
        if drop > 0:
            neg = neg[:, :-drop]
        if random_permute:
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg.shape[1], device=self.classifier.device)
            neg = neg.T
            negs = neg[idx].T.reshape(pos.shape[0], group, -1).contiguous()
        else:
            negs = neg.reshape(pos.shape[0], group, -1).contiguous()
        scores = []
        for i in range(group):
            full_sim = torch.cat([pos, negs[:, i, :]], dim=-1) / self.t
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        if num is not None:
            scores = scores[:,0:num-1]
        score = scores.mean(dim=-1)
        return score

    def forward(self, **input):
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
                score = full_sim[:, :pos.shape[1]].sum(dim=-1)

        return score, pos


def neglabel_arguments(config: Any, method: str = "neglabel") -> SimpleNamespace:
    values, setting = resolve_method_arguments(method, config.postprocessor.postprocessor_args, NEGLABEL_ARGUMENTS,
                                               dataset=dataset_name(config))
    fuse = values["group_fuse_num"]
    return SimpleNamespace(
        t=number_argument(method, values, "t"),
        ngroup=integer_argument(method, values, "ngroup", minimum=1),
        group_fuse_num=None if fuse is None else integer_argument(method, values, "group_fuse_num", minimum=1),
        setting=setting,
    )


class NegLabelPostprocessor(BasePostprocessor):
    """
    Wrapper around the official ``ScalableClassifier`` over ``CLIPScalableClassifier``,
    built on this repository's CLIP model and caches.
    """

    supports_image_feature_cache = True

    def __init__(self, config):
        super().__init__(config)
        self.args = neglabel_arguments(config)
        self.model: ScalableClassifier | None = None
        self.device: torch.device | None = None

    def setup(self, net: nn.Module, id_loader_dict=None, ood_loader_dict=None):
        if self.model is None:
            classifier = CLIPScalableClassifier(net, self.config)
            self.model = ScalableClassifier(classifier, t=self.args.t, ngroup=self.args.ngroup,
                                            group_fuse_num=self.args.group_fuse_num)
            self.device = classifier.device
            print(f"[NegLabelPostprocessor] setup complete | {self.policy_summary()}", flush=True)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if self.model is None:
            raise RuntimeError("setup() must be called before postprocess().")
        score, pos = self.model.forward(img=data)
        return pos.argmax(dim=1), score

    def policy_summary(self) -> dict[str, Any]:
        summary = {"source": SOURCE, "t": self.args.t, "ngroup": self.args.ngroup,
                   "group_fuse_num": self.args.group_fuse_num, "setting": self.args.setting}
        if self.model is not None:
            summary.update({"id_classes": int(self.model.classifier.text_features_pos.shape[0]),
                            "negative_labels": int(self.model.classifier.text_features_neg.shape[0])})
        return summary


__all__ = ["CLIPScalableClassifier", "NEGLABEL_ARGUMENTS", "NegLabelPostprocessor", "SOURCE", "ScalableClassifier",
           "neglabel_arguments"]
