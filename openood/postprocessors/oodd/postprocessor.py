"""OODD, ported from zxk1212/OODD@edbb1a32e5fe81e443942156f0a9cafb0297d95b."""
from __future__ import annotations

import contextlib
import os
from queue import PriorityQueue
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from openood.utils.vlm.image.types import CachedImageFeatures

from ..base_postprocessor import BasePostprocessor
from ..neglabel.postprocessor import CLIPScalableClassifier as NegLabelClassifier
from ..neglabel.postprocessor import ScalableClassifier as NegLabelScalableClassifier
from ..neglabel.postprocessor import neglabel_arguments

SOURCE = "zxk1212/OODD@edbb1a32e5fe81e443942156f0a9cafb0297d95b"


class ScoreData:
    def __init__(self, score, data):
        self.score = score
        self.data = data

    def __lt__(self, other):
        return float(self.score) > float(other.score)

def kth_largest_per_column(matrix, k):
    """Compute the k-th largest element for each column of a matrix."""
    kth_values, _ = torch.kthvalue(matrix, matrix.size(0) - k + 1, dim=0)
    return kth_values.cpu().numpy()

def batched_matrix_multiply(ftrain, second_matrix, K, batch_size=9000):
    """Compute the product of the feature tensors in batches."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ftrain_tensor = torch.tensor(ftrain, device=device)
    second_matrix_tensor = torch.tensor(second_matrix, device=device)
    p, _ = second_matrix_tensor.shape

    num_batches = p // batch_size + (p % batch_size > 0)
    res = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, p)
        second_batch_tensor = second_matrix_tensor[start:end,:]
        batch_result = ftrain_tensor @ second_batch_tensor.T
        score = kth_largest_per_column(batch_result, K)
        res.append(score)
    return np.concatenate(res, axis=0)


class CLIPScalableClassifier(NegLabelClassifier):
    """
    OODD's ``CLIPScalableClassifier``. 
    The text features are NegLabel's; 
    ``simple_test`` is OODD's own (lines 175-194).
    """

    def simple_test(self, img, img_metas=None, require_features=False, require_backbone_features=False, softmax=True, **kwargs):
        """Test without augmentation."""
        with torch.no_grad():
            if isinstance(img, CachedImageFeatures):
                # norm; the 1e-10 is below fp16's resolution at the norm, so the rows are the same)
                image_features = img.values.to(self.device)
            else:
                image_features = self.model.encode_image(img)
                image_features /= image_features.norm(dim=-1, keepdim=True) + 1e-10

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

            return pos_sim, neg_sim, image_features


class ScalableClassifier(NegLabelScalableClassifier):
    """
    OODD's ``ScalableClassifier``. 
    ``__init__`` and ``grouping`` are NegLabel's; 
    ``forward`` is OODD's own (lines 53-68).
    """

    def forward(self, **input):
        if "dataset_name" in input and self.dump_score:
            dataset_name = input['dataset_name']
            dump_path = os.path.join(self.dump_score, "{}".format(dataset_name))
            os.makedirs(dump_path, exist_ok=True)
            del input['dataset_name']
        with torch.no_grad():
            pos, neg, image_features = self.classifier.simple_test(**input)
            full_sim = torch.cat([pos, neg], dim=-1) / self.t

            if self.ngroup > 1:
                score = self.grouping(pos, neg, num=self.group_fuse_num, ngroup=self.ngroup, random_permute=True)
            else:
                score = full_sim[:, :pos.shape[1]].sum(dim=-1)

        return score, image_features, pos


@contextlib.contextmanager
def kthvalue_allowed():
    """
    Temporarily allow non-deterministic algorithms. 
    ``main.py`` runs with ``torch.use_deterministic_algorithms(True)``, 
    and some PyTorch releases have no deterministic CUDA kernel for ``torch.kthvalue``.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if enabled:
        torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        if enabled:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)


class OODDPostprocessor(BasePostprocessor):
    """
    Wrapper around OODD's official streaming dictionary over its NegLabel model,
    fed by the evaluator's stream.
    """

    supports_image_feature_cache = True

    def __init__(self, config):
        super().__init__(config)
        self.args = neglabel_arguments(config, method="oodd")
        self.model: ScalableClassifier | None = None
        self.device: torch.device | None = None
        self.queue = PriorityQueue()

    def setup(self, net: nn.Module, id_loader_dict=None, ood_loader_dict=None):
        if self.model is None:
            classifier = CLIPScalableClassifier(net, self.config)
            self.model = ScalableClassifier(classifier, t=self.args.t, ngroup=self.args.ngroup,
                                            group_fuse_num=self.args.group_fuse_num)
            self.device = classifier.device
            print(f"[OODDPostprocessor] setup complete | {self.policy_summary()}", flush=True)

    def reset_memory(self):
        self.queue = PriorityQueue()

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if self.model is None:
            raise RuntimeError("setup() must be called before postprocess().")
        score, image_features, pos = self.model.forward(img=data)
        all_output = score.cpu().numpy()
        all_data = image_features.cpu().numpy()

        with kthvalue_allowed():
            queue_size =2048 # 2048
            queue = self.queue

            batch_size = 64

            num_batches = all_data.shape[0] // batch_size + (all_data.shape[0] % batch_size > 0)

            scores_list = []

            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, all_data.shape[0])
                batch_data = all_data[start:end]
                batch_score =  all_output[start:end]

                for j in range(batch_score.shape[0]):
                    queue.put(ScoreData(batch_score[j], batch_data[j]))
                    if queue.qsize() > queue_size:
                        queue.get()

                data_list = []
                for item in list(queue.queue):
                    data_list.append(item.data)
                new_food = np.array(data_list)
                ood_batch_score = batched_matrix_multiply(new_food, batch_data, 5) # 5
                batch_score = batch_score  - ood_batch_score
                scores_list.append(batch_score)
            scores_all = np.concatenate(scores_list, axis=0)

        return pos.argmax(dim=1), torch.from_numpy(scores_all).to(score.device)

    def policy_summary(self) -> dict[str, Any]:
        summary = {"source": SOURCE, "queue_size": 2048, "batch_size": 64, "k": 5,
                   "base": {"method": "neglabel", "t": self.args.t, "ngroup": self.args.ngroup,
                            "group_fuse_num": self.args.group_fuse_num}, "setting": self.args.setting}
        if self.model is not None:
            summary["negative_labels"] = int(self.model.classifier.text_features_neg.shape[0])
        return summary


__all__ = ["CLIPScalableClassifier", "OODDPostprocessor", "SOURCE", "ScalableClassifier", "ScoreData",
           "batched_matrix_multiply", "kth_largest_per_column", "kthvalue_allowed"]
