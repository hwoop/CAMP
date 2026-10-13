"""CoEvo, ported from yuleoliu/CoEvo@fec8d38c5bd804f018ec8cc541c5d0fe1b642114."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from openood.networks.clip_fixed_ood_prompt import FixedCLIP_CoEvo_Neglabel
from openood.utils import get_config

from ..base_postprocessor import BasePostprocessor
from ..method_arguments import (
    boolean_argument,
    dataset_name,
    integer_argument,
    number_argument,
    resolve_method_arguments,
)

SOURCE = "yuleoliu/CoEvo@fec8d38c5bd804f018ec8cc541c5d0fe1b642114"
COEVO_ARGUMENTS: dict[str, Any] = {
    "thres": 0.5,
    "gap": 0.8,
    "memleng": 10,
    "lambdaval": 0.2,
    "beta": 5.5,
    "group_num": 5,
    "random_permute": True,
    "samada": True,
    "in_score": "combine",
    "topk_neg": 5,
    "vision_logit_scale": 100.0,
    "text_prompt": None,
    "ood_number": None,
}
IN_SCORES = ("combine", "adaonly", "vanillaonly")


class TTAPromptPostprocessor_CoEvo(BasePostprocessor):
    def __init__(self, config):
        super(TTAPromptPostprocessor_CoEvo, self).__init__(config)
        self.args = coevo_arguments(config)
        self.beta = int(self.args.beta)
        self.in_score = self.args.in_score  # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.reset = True  ## reset after each dataset.
        self.memory_size = self.args.memleng
        self.lambda_val = self.args.lambdaval
        self.thres = self.args.thres
        self.samada = self.args.samada
        self.gap = self.args.gap
        self.group_num = self.args.group_num
        self.random_permute = self.args.random_permute
        self.adaptive_thres = True
        self.queue_length = 512
        self.os_detector_queue = []
    def reset_memory(self):
        self.reset = True

    def insert_memory_slot(self,features,num):
        # insert new memory for new text feat
        new_memory = features.unsqueeze(1) ## new * 1 *512
        extented_empty_memory = torch.zeros_like(new_memory).repeat(1, self.memory_size, 1)
        new_memory = torch.cat((new_memory, extented_empty_memory), dim=1)
        self.feat_memory = torch.cat((self.feat_memory,new_memory),dim=0)
        new_indice_memory = torch.ones(num)
        self.indice_memory = torch.cat((self.indice_memory,new_indice_memory),dim = 0)

        new_entropy_memory = torch.zeros(num, self.feat_memory.size(1)).to(
            features.device)
        self.entropy_memory = torch.cat((self.entropy_memory, new_entropy_memory),dim = 0)


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        image_features, text_features, logit_scale, logit_scale_vision = net(data, return_feat=True)
        logit_scale = 100.0
        if self.reset:
            self.feat_memory = text_features.unsqueeze(1)
            extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
            self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
            self.indice_memory = torch.ones(text_features.size(0))
            self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(
                text_features.device)
            self.reset = False

        conf_in_vanilla_old, output_vanilla_old = self.computer_neglabel_score(image_features,text_features,logit_scale,class_num)
        conf_in_tta_old, output_tta_old = self.compute_tta_score(image_features,logit_scale_vision,class_num)
        conf_old = (1 - self.lambda_val) * conf_in_vanilla_old + self.lambda_val * conf_in_tta_old   ### bs
        output_all = (1 - self.lambda_val) * output_vanilla_old + self.lambda_val * output_tta_old

        if self.adaptive_thres == True:
            self.os_detector_queue.extend(conf_old.detach().cpu().tolist())
            self.os_detector_queue = self.os_detector_queue[-self.queue_length:]
            threshold_range = np.arange(0, 1, 0.01)
            criterias = [self.compute_os_variance(np.array(self.os_detector_queue), th) for th in threshold_range]
            self.thres = threshold_range[np.argmin(criterias)]

        activate_indicator_id = conf_old > (
                    self.thres + self.gap * (1 - self.thres))

        id_image_feat = image_features[activate_indicator_id]
        text_features,new_text_features,new_text_features_num = net.get_id_text_feature(id_image_feat)
        self.insert_memory_slot(new_text_features, new_text_features_num)
        activate_indicator_ood = conf_old < (
                    self.thres - self.gap * self.thres)

        ood_image_feat = image_features[activate_indicator_ood]
        text_features, new_text_features, new_text_features_num = net.get_ood_text_feature(ood_image_feat)  ### y_2
        self.insert_memory_slot(new_text_features,new_text_features_num)

        activate_indicator = conf_old > (
                    self.thres + self.gap * (1 - self.thres))
        _, pred_all = torch.max(output_all[:, :class_num], dim=1)
        prob_id = torch.softmax(output_all[:, :class_num], dim=1)
        for i in range(activate_indicator.size(0)):
            if activate_indicator[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass
                    else:
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][
                        self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass

        activate_indicator = conf_old < (
                    self.thres - self.gap * self.thres)
        _, pred_all = torch.max(output_all[:, class_num:], dim=1)
        prob_ood = torch.softmax(output_all[:, class_num:], dim=1)
        for i in range(activate_indicator.size(0)):
            if activate_indicator[i].item():
                predicted_cate = pred_all[i].item() + class_num
                predicted_prob = prob_ood[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][
                        self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass

        conf_in_vanilla_new, output_vanilla_new = self.computer_neglabel_score(image_features, text_features,
                                                                               logit_scale, class_num)
        conf_in_tta_new, output_tta_new = self.compute_tta_score(image_features, logit_scale_vision, class_num)
        prob_tta = torch.softmax(output_tta_new, dim=1)
        prob_vanilla = torch.softmax(output_vanilla_new, dim=1)
        prob_all = prob_vanilla
        _, pred_in = torch.max(prob_all[:, :class_num], dim=1)
        if self.in_score == 'adaonly':
            conf = conf_in_tta_new
        elif self.in_score == 'vanillaonly':
            conf = conf_in_vanilla_new
        elif self.in_score == 'combine':
            conf = (1 - self.lambda_val) * conf_in_tta_new + conf_in_vanilla_new * self.lambda_val
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            raise FloatingPointError("CoEvo produced a NaN confidence.")

        return pred_in, conf, conf_in_vanilla_new, conf_in_tta_new

    def computer_neglabel_score(self,image_features,text_features,logit_scale,class_num):
        output_vanilla = logit_scale * image_features @ text_features.t()
        prob_vanilla = torch.softmax(output_vanilla, dim=1)
        pos_logit = output_vanilla[:, :class_num]  ## B*C
        neg_logit = output_vanilla[:, class_num:]  ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:

            idx = torch.randperm(neg_logit.shape[1]).to(output_vanilla.device)
            neg_logit = neg_logit.T  ## total_neg_num*B
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        scores = []
        for i in range(self.group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1)
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in_vanilla = scores.mean(dim=-1)  ### the mean ID score of multiple groups.
        return conf_in_vanilla,output_vanilla

    def compute_tta_score(self,image_features,logit_scale,class_num):
        sim = self.feat_memory @ image_features.t()
        sim = torch.exp(-self.beta * (-sim + 1))
        if self.samada:
            sa_text_features_list = []
            split_num = int(self.feat_memory.size(0) / 1000.0)
            for i in range(split_num):
                temp = sim[1000 * i:1000 * (i + 1)].unsqueeze(0).transpose(0, -1) * self.feat_memory[
                                                                                    1000 * i:1000 * (i + 1)].unsqueeze(
                    0)
                sa_text_features_split = temp.sum(2)
                sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)
                sa_text_features_list.append(sa_text_features_split)
            if self.feat_memory.size(0) % 1000.0 > 0:
                temp = sim[1000 * split_num:].unsqueeze(0).transpose(0, -1) * self.feat_memory[
                                                                              1000 * split_num:].unsqueeze(
                    0)
                sa_text_features_split = temp.sum(2)
                sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)
                sa_text_features_list.append(sa_text_features_split)
            sa_text_features = torch.cat(sa_text_features_list, dim=1)
        else:
            sa_text_features = self.feat_memory.mean(1)
            sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)
        output = logit_scale * (image_features.unsqueeze(1) * sa_text_features).sum(-1)
        pos_logit = output[:, :class_num]  ## B*C
        neg_logit = output[:, class_num:]
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            idx = torch.randperm(neg_logit.shape[1]).to(output.device)
            neg_logit = neg_logit.T
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        scores = []
        for i in range(self.group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1)
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in = scores.mean(dim=-1)
        return conf_in, output

    def compute_os_variance(self, os, th):
        thresholded_os = np.zeros(os.shape)
        thresholded_os[os >= th] = 1
        nb_pixels = os.size
        nb_pixels1 = np.count_nonzero(thresholded_os)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1
        if weight1 == 0 or weight0 == 0:
            return np.inf
        val_pixels1 = os[thresholded_os == 1]
        val_pixels0 = os[thresholded_os == 0]
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        return weight0 * var0 + weight1 * var1


def coevo_arguments(config: Any) -> SimpleNamespace:
    values, setting = resolve_method_arguments("coevo", config.postprocessor.postprocessor_args, COEVO_ARGUMENTS,
                                               dataset=dataset_name(config))
    in_score = str(values["in_score"])
    if in_score not in IN_SCORES:
        raise ValueError(f"coevo.postprocessor_args.in_score must be one of {list(IN_SCORES)}, got {in_score!r}.")
    return SimpleNamespace(
        thres=number_argument("coevo", values, "thres"),
        gap=number_argument("coevo", values, "gap"),
        memleng=integer_argument("coevo", values, "memleng", minimum=1),
        lambdaval=number_argument("coevo", values, "lambdaval"),
        beta=number_argument("coevo", values, "beta"),
        group_num=integer_argument("coevo", values, "group_num", minimum=1),
        random_permute=boolean_argument("coevo", values, "random_permute"),
        samada=boolean_argument("coevo", values, "samada"),
        in_score=in_score,
        topk_neg=integer_argument("coevo", values, "topk_neg", minimum=1),
        vision_logit_scale=number_argument("coevo", values, "vision_logit_scale"),
        text_prompt=None if values["text_prompt"] is None else str(values["text_prompt"]),
        ood_number=None if values["ood_number"] is None else integer_argument("coevo", values, "ood_number",
                                                                              minimum=1),
        setting=setting,
    )


class CoEvoPostprocessor(TTAPromptPostprocessor_CoEvo):
    """
    Wrapper around the official ``TTAPromptPostprocessor_CoEvo``. 
    Its network is built on this repository's CLIP model and caches.
    """

    supports_image_feature_cache = True

    def __init__(self, config):
        super().__init__(config)
        self.network: FixedCLIP_CoEvo_Neglabel | None = None
        self.device: torch.device | None = None
        self.class_num: int | None = None

    def setup(self, net: nn.Module, id_loader_dict=None, ood_loader_dict=None):
        if self.network is None:
            ood_number = self.args.ood_number
            if ood_number is None:
                ood_number = int(get_config(get_config(self.config.vlm, "neglabel", None), "ood_number", 10000))
            self.network = FixedCLIP_CoEvo_Neglabel(net, self.config, text_prompt=self.args.text_prompt,
                                                    ood_number=ood_number, topk_neg=self.args.topk_neg,
                                                    logit_scale_vision=self.args.vision_logit_scale)
            self.device = self.network.device
            self.class_num = self.network.n_cls
            print(f"[CoEvoPostprocessor] setup complete | {self.policy_summary()}", flush=True)

    def reset_memory(self):
        super().reset_memory()
        if self.network is not None:
            self.network.reset_text_feature()

    def postprocess(self, net: nn.Module, data: Any):
        if self.network is None:
            raise RuntimeError("setup() must be called before postprocess().")
        pred_in, conf, _, _ = super().postprocess(self.network, data)
        return pred_in, conf

    def policy_summary(self) -> dict[str, Any]:
        summary = {"source": SOURCE, **{key: getattr(self.args, key) for key in COEVO_ARGUMENTS},
                   "queue_length": self.queue_length, "setting": self.args.setting}
        if self.network is not None:
            summary["pool"] = int(self.network.pool_feature.shape[0])
        return summary


__all__ = ["COEVO_ARGUMENTS", "CoEvoPostprocessor", "IN_SCORES", "SOURCE", "TTAPromptPostprocessor_CoEvo",
           "coevo_arguments"]
