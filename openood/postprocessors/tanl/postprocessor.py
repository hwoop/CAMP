"""TANL, ported from PolyU-VCLab/OpenOOD-VLM@48b023365b68b54c87f84ab508de8a3222335985."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from openood.networks.clip_fixed_ood_prompt import FixedCLIP_NegOODPrompt

from ..base_postprocessor import BasePostprocessor
from ..method_arguments import (
    boolean_argument,
    dataset_name,
    integer_argument,
    number_argument,
    resolve_method_arguments,
)

SOURCE = "PolyU-VCLab/OpenOOD-VLM@48b023365b68b54c87f84ab508de8a3222335985"
# postprocessor_args and their defaults: the Table 1 setting (configs/postprocessors/tanl/default.yml); thres,
TANL_ARGUMENTS: dict[str, Any] = {
    "memleng": 300,
    "beta": 1000,
    "gamma": 1,
    "gap": 0.2,
    "alpha": 0.95,
    "thres": 0.5,
    "in_score": "sum",
    "cossim": False,
}


def compute_os_variance_tensor(os_tensor, th):
    """Calculate weighted variance for PyTorch tensors"""
    device = os_tensor.device

    mask = (os_tensor >= th).float()

    n_pixels = os_tensor.numel()
    n_pixels1 = torch.sum(mask)
    weight1 = n_pixels1 / n_pixels
    weight0 = 1 - weight1

    if weight1 == 0 or weight0 == 0:
        return torch.tensor(float('inf'), device=device)

    class1 = os_tensor[mask.bool()]
    class0 = os_tensor[~mask.bool()]

    var0 = torch.var(class0, unbiased=False) if class0.numel() > 0 else torch.tensor(0.0, device=device)
    var1 = torch.var(class1, unbiased=False) if class1.numel() > 0 else torch.tensor(0.0, device=device)

    return weight0 * var0 + weight1 * var1

def find_best_threshold(os_training_queue):
    """Find the best threshold on the GPU/CPU"""
    threshold_range = torch.arange(0, 1, 0.01, device=os_training_queue.device)

    criterias = torch.stack([compute_os_variance_tensor(os_training_queue, th) for th in threshold_range])

    min_val = torch.min(criterias)
    mask = (criterias == min_val)
    candidate_indices = torch.where(mask)[0]

    if len(candidate_indices) == 0:
        return threshold_range[torch.argmin(criterias)].item()

    mid_index = len(candidate_indices) // 2
    best_threshold = threshold_range[candidate_indices[mid_index]]

    return best_threshold.item()


def update_queue(to_add, queue, max_len=1000, init=False):
    if init:
        queue = queue
    else:
        queue = torch.cat((queue, to_add), dim=0)
    if queue.size(0) > max_len:
        queue = queue[-max_len:, :]
    return queue

def activation_aware_score(output, id_num, ood_num, step, weight_by_activation_score=False):
    if isinstance(weight_by_activation_score, torch.Tensor) and weight_by_activation_score.numel() > 1:
        use_weight = True
    elif bool(weight_by_activation_score):
        use_weight = True
    else:
        use_weight = False
    if use_weight:
        if step == 0:
            full_sim = output.softmax(dim=-1)
            conf_in = full_sim[:, :id_num].sum(dim=-1)
        elif step == 1:
            full_sim = (output * weight_by_activation_score.unsqueeze(0)).softmax(dim=-1)
            conf_in = full_sim[:, :id_num].sum(dim=-1)
        elif step == 2:
            weighted_logits = output.clone()
            weighted_logits = weighted_logits + weight_by_activation_score.log().unsqueeze(0)
            full_sim = torch.softmax(weighted_logits, dim=-1)
            conf_in = full_sim[:, :id_num].sum(dim=-1)
    elif step != 0:
        softmax_sums = []
        step = int(step)
        for i in range(id_num, id_num + ood_num, step):
            softmax_output = output[:,:i+step].softmax(dim=-1)
            sum_score = softmax_output[:, :id_num].sum(dim=-1)
            softmax_sums.append(sum_score)
        softmax_sums = torch.stack(softmax_sums, dim=-1)
        conf_in = softmax_sums.mean(dim=-1)
    else:
        full_sim = output.softmax(dim=-1)
        conf_in = full_sim[:, :id_num].sum(dim=-1)
    return conf_in


class ActivatedNegPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ActivatedNegPostprocessor, self).__init__(config)
        self.args = tanl_arguments(config)
        self.beta = int(self.args.beta)
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = True
        self.proj_flag = False
        self.alpha = self.args.alpha
        self.gamma = self.args.gamma
        self.reset = True
        self.thres = self.args.thres
        self.gap = self.args.gap
        self.cossim = self.args.cossim
        self.queue_len = self.args.memleng
        self.score_queue_len = 20000
        self.mute_mutual_enhancement = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        ###
        return

    def reset_memory(self):
        self.reset = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):

        net.eval()
        class_num = net.n_cls
        if self.reset:  ## For each ID-OOD dataset pair.
            self.class_count = torch.ones(net.n_cls) * self.beta

            self.online_text_features = net.text_features.clone() ## 512* [ID class + neg class]
            self.online_text_features_all = net.text_features_all.clone()

            ############## use the ID labels as positive samples, and the noise image features as negative samples
            self.score_from_pos = net.logit_scale * net.text_features_all[:, :class_num].t() @ net.text_features_all ## id num * all num
            if self.cossim:
                self.score_from_pos = self.score_from_pos[torch.randperm(self.score_from_pos.size(0))]
            else:
                self.score_from_pos = torch.softmax(self.score_from_pos, dim=1)
                self.score_from_pos = self.score_from_pos[torch.randperm(self.score_from_pos.size(0))]
            self.score_from_pos = self.score_from_pos[:, class_num:].float()
            self.score_from_pos = update_queue(self.score_from_pos, self.score_from_pos, self.queue_len, init=True)

            self.cached_pos_features = net.text_features_all[:, :class_num].clone().t() ## 1000*512
            self.cached_neg_features = net.noise_image_features.clone()                 ## 15*512
            print('cached_pos_features shape:', self.cached_pos_features.shape)
            print('cached_neg_features shape:', self.cached_neg_features.shape)
            self.cached_pos_features = update_queue(self.cached_pos_features, self.cached_pos_features, self.queue_len, init=True)
            self.cached_neg_features = update_queue(self.cached_neg_features, self.cached_neg_features, self.queue_len, init=True)

            self.score_from_neg = net.logit_scale * net.noise_image_features @ net.text_features_all
            if self.cossim:
                self.score_from_neg =  self.score_from_neg
            else:
                self.score_from_neg = torch.softmax(self.score_from_neg, dim=1)
            self.score_from_neg = self.score_from_neg[torch.randperm(self.score_from_neg.size(0))] ##

            self.score_from_neg = self.score_from_neg[:, class_num:].float()
            self.score_from_neg = update_queue(self.score_from_neg, self.score_from_neg, self.queue_len, init=True)
            combined_score = self.score_from_neg.mean(0) - self.score_from_pos.mean(0)
            selected_text_features = torch.cat([net.text_features[:,:class_num], self.online_text_features_all[:,class_num:].t()[combined_score.sort(descending=True)[1][:self.beta]].t()], dim=-1)
            if self.mute_mutual_enhancement:
                output = net.logit_scale * net.text_features_all[:, :class_num].t() @ self.online_text_features ## use negative labels of NegLabel
            else:
                output = net.logit_scale * net.text_features_all[:, :class_num].t() @ selected_text_features
            conf_in_id = activation_aware_score(output, class_num, self.beta, step=self.gamma)

            if self.mute_mutual_enhancement:
                output = net.logit_scale * net.noise_image_features @ self.online_text_features
            else:
                output = net.logit_scale * net.noise_image_features @ selected_text_features
            conf_in_ood = activation_aware_score(output, class_num, self.beta, step=self.gamma)

            self.scoretensor = torch.cat((conf_in_id, conf_in_ood), dim=0)
            self.reset = False
        image_features, text_features, logit_scale = net(data, return_feat=True)

        output = logit_scale * image_features @ self.online_text_features # batch * class.
        output_all = logit_scale * image_features @ self.online_text_features_all  # batch * dim   @ dim * num

        _, pred_in = torch.max(output[:, :class_num], dim=1)

        if self.cossim:
            actscore = output_all
        else:
            actscore = torch.softmax(output_all.float(), dim=1)  # batch * [id+ood class num]

        combined_score = self.score_from_neg.mean(0) - self.score_from_pos.mean(0)
        selected_text_features = torch.cat([net.text_features[:,:class_num], self.online_text_features_all[:,class_num:].t()[combined_score.sort(descending=True)[1][:self.beta]].t()], dim=-1)
        if self.mute_mutual_enhancement:
            output = logit_scale * image_features @ self.online_text_features
        else:
            output = logit_scale * image_features @ selected_text_features
        output = output.float()
        conf_in_domain = activation_aware_score(output, class_num, self.beta, step=self.gamma)

        self.scoretensor = torch.cat((self.scoretensor, conf_in_domain), dim=0)
        if self.scoretensor.size(0) > self.score_queue_len:
            self.scoretensor = self.scoretensor[-self.score_queue_len:]

        best_threshold = find_best_threshold(self.scoretensor)
        self.thres = best_threshold

        activate_indicator_id = conf_in_domain > (self.thres + self.gap * (1-self.thres)) ## select high confident samples
        activate_indicator_ood = conf_in_domain < (self.thres - self.gap * self.thres)  ## only select high confident samples
        to_cache_pos_features = image_features[activate_indicator_id]
        to_cache_neg_features = image_features[activate_indicator_ood]
        used_cluster_num = 0

        if torch.any(activate_indicator_id):
            score_pos_in_batch = actscore[activate_indicator_id][:, class_num:]
            ins_adaptive_pos_score = self.alpha * self.score_from_pos.mean(0) + (1 - self.alpha) * score_pos_in_batch.mean(0)
        else:
            ins_adaptive_pos_score = self.score_from_pos.mean(0)

        if torch.any(activate_indicator_ood):
            score_neg_in_batch = actscore[activate_indicator_ood][:, class_num:]
            ins_adaptive_neg_score = self.alpha * self.score_from_neg.mean(0) + (1 - self.alpha) * score_neg_in_batch.mean(0)
        else:
            ins_adaptive_neg_score = self.score_from_neg.mean(0)

        combined_score = ins_adaptive_neg_score - ins_adaptive_pos_score  ## 69554.

        if used_cluster_num > 0:
            selected_text_features = torch.cat([net.text_features[:,:class_num], reconst_clusters.t(), self.online_text_features_all[:,class_num:].t()[combined_score.sort(descending=True)[1][:self.beta-used_cluster_num]].t()], dim=-1)
        else:
            selected_text_features = torch.cat([net.text_features[:,:class_num], self.online_text_features_all[:,class_num:].t()[combined_score.sort(descending=True)[1][:self.beta]].t()], dim=-1) ## 512*2000

        output = logit_scale * image_features @ selected_text_features  ## batch * [id+ood class num]

        conf_in = activation_aware_score(output, class_num, self.beta, step=self.gamma)
        if torch.any(activate_indicator_id):
            self.score_from_pos = update_queue(actscore[activate_indicator_id][:, class_num:], self.score_from_pos, self.queue_len, init=False)
            self.cached_pos_features = update_queue(to_cache_pos_features, self.cached_pos_features, self.queue_len, init=False)
        if torch.any(activate_indicator_ood):
            self.score_from_neg = update_queue(actscore[activate_indicator_ood][:, class_num:], self.score_from_neg, self.queue_len, init=False)
            self.cached_neg_features = update_queue(to_cache_neg_features, self.cached_neg_features, self.queue_len, init=False)

        # max in prob - max out prob
        if self.in_score == 'sum':
            conf_in = conf_in  ## = 1-conf_out
        elif self.in_score == 'div_count':
            for i in range(len(conf_in)):
                conf_in[i] = conf_in[i] / normalized_count[pred_in[i]] / net.n_cls
        else:
            raise NotImplementedError
        if torch.isnan(conf_in).any():
            raise FloatingPointError("TANL produced a NaN score.")

        return pred_in, conf_in


def tanl_arguments(config: Any) -> SimpleNamespace:
    values, setting = resolve_method_arguments("tanl", config.postprocessor.postprocessor_args, TANL_ARGUMENTS,
                                               dataset=dataset_name(config))
    in_score = str(values["in_score"])
    if in_score != "sum":
        raise ValueError(f"tanl.postprocessor_args.in_score must be 'sum' (the official code's working score), "
                         f"got {in_score!r}.")
    return SimpleNamespace(
        memleng=integer_argument("tanl", values, "memleng", minimum=1),
        beta=integer_argument("tanl", values, "beta", minimum=1),
        gamma=integer_argument("tanl", values, "gamma", minimum=0),
        gap=number_argument("tanl", values, "gap"),
        alpha=number_argument("tanl", values, "alpha"),
        thres=number_argument("tanl", values, "thres"),
        in_score=in_score,
        cossim=boolean_argument("tanl", values, "cossim"),
        setting=setting,
    )


class TANLPostprocessor(ActivatedNegPostprocessor):
    """
    Wrapper around the official ``ActivatedNegPostprocessor``. 
    Its network is built on this repository's CLIP model and caches.
    """

    supports_image_feature_cache = True

    def __init__(self, config):
        super().__init__(config)
        self.network: FixedCLIP_NegOODPrompt | None = None
        self.device: torch.device | None = None
        self.class_num: int | None = None

    def setup(self, net: nn.Module, id_loader_dict=None, ood_loader_dict=None):
        if self.network is None:
            self.network = FixedCLIP_NegOODPrompt(net, self.config)
            self.device = self.network.device
            self.class_num = self.network.n_cls
            print(f"[TANLPostprocessor] setup complete | {self.policy_summary()}", flush=True)
        return super().setup(self.network, id_loader_dict, ood_loader_dict)

    def postprocess(self, net: nn.Module, data: Any):
        if self.network is None:
            raise RuntimeError("setup() must be called before postprocess().")
        return super().postprocess(self.network, data)

    def policy_summary(self) -> dict[str, Any]:
        summary = {"source": SOURCE, **{key: getattr(self.args, key) for key in TANL_ARGUMENTS},
                   "score_queue_len": self.score_queue_len, "setting": self.args.setting}
        if self.network is not None:
            summary.update({"labels": int(self.network.text_features.shape[1]),
                            "corpus": int(self.network.text_features_all.shape[1]) - self.network.n_cls,
                            "noise_images": int(self.network.noise_image_features.shape[0])})
        return summary


__all__ = ["ActivatedNegPostprocessor", "SOURCE", "TANL_ARGUMENTS", "TANLPostprocessor", "activation_aware_score",
           "compute_os_variance_tensor", "find_best_threshold", "tanl_arguments", "update_queue"]
