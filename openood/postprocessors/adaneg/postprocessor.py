"""AdaNeg, ported from PolyU-VCLab/OpenOOD-VLM@48b023365b68b54c87f84ab508de8a3222335985."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from openood.networks.clip_fixed_ood_prompt import FixedCLIP_NegOODPrompt
from openood.utils import get_config

from ..base_postprocessor import BasePostprocessor
from ..method_arguments import (
    boolean_argument,
    dataset_name,
    integer_argument,
    number_argument,
    resolve_method_arguments,
)

SOURCE = "PolyU-VCLab/OpenOOD-VLM@48b023365b68b54c87f84ab508de8a3222335985"
ADANEG_ARGUMENTS: dict[str, Any] = {
    "thres": 0.5,
    "gap": 0.5,
    "memleng": 10,
    "lambdaval": 0.1,
    "beta": 5.5,
    "group_num": 5,
    "random_permute": True,
    "samada": True,
    "in_score": "combine",
    "text_prompt": None,
    "ood_number": None,
    "samada_layout": "released",
}
IN_SCORES = ("combine", "adaonly", "vanillaonly", "multiply")
SAMADA_LAYOUTS = ("released", "factored")


def sample_adaptive_cosine(feat_memory: torch.Tensor, image_features: torch.Tensor, beta: int) -> torch.Tensor:
    memory = feat_memory.float()
    x = image_features.float()
    similarity = memory @ x.t()
    weight = torch.exp(-beta * (-similarity + 1))
    numerator = (weight * similarity).sum(1)
    gram = memory @ memory.transpose(1, 2)
    squared_norm = (weight * torch.bmm(gram, weight)).sum(1)
    return (numerator / squared_norm.clamp_min(1e-12).sqrt()).t()


class TTAPromptPostprocessor_noadagap(BasePostprocessor):
    def __init__(self, config):
        super(TTAPromptPostprocessor_noadagap, self).__init__(config)
        self.args = adaneg_arguments(config)
        self.beta = int(self.args.beta)
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.reset = True ## reset after each dataset.
        self.memory_size = self.args.memleng
        self.lambda_val = self.args.lambdaval
        self.thres = self.args.thres
        self.samada = self.args.samada
        self.gap = self.args.gap
        self.group_num = self.args.group_num
        self.random_permute = self.args.random_permute

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        ### get the image feature of each classes, construct the image feature classifier/memory.
        net.eval()
        # net.text_features 11k*512, if empty, fill with net.
        out_dim = net.n_output

    def reset_memory(self):
        self.reset = True

    # Store (high confident) test features into the feature memory, and get new classifier by merging image features in the same memory slot.
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        image_features, text_features, logit_scale = net(data, return_feat=True)
        if self.reset:
            ## reset feat memory to vanilla text features, for each ID/OOD pair.
            self.feat_memory = text_features.unsqueeze(1) ## C*1*D
            extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
            self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
            self.indice_memory = torch.ones(text_features.size(0))
            self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(text_features.device) ## C*(1+memory_size)
            self.reset=False
        ## image_features: 256*512
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention
        output_vanilla = logit_scale * image_features @ text_features.t()  # batch * class, to decide assign image to which category and the corresponding confidence.
        prob_vanilla = torch.softmax(output_vanilla, dim=1)
        pos_logit = output_vanilla[:, :class_num] ## B*C
        neg_logit = output_vanilla[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output_vanilla.device)
            neg_logit = neg_logit.T ## total_neg_num*B
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
        conf_in_vanilla = scores.mean(dim=-1) ### the mean ID score of multiple groups.

        activate_indicator = conf_in_vanilla > (self.thres + self.gap * (1-self.thres)) ## only store high confident samples into feature memory.

        _, pred_all = torch.max(output_vanilla[:, :class_num], dim=1)
        prob_id = torch.softmax(output_vanilla[:, :class_num], dim=1)
        for i in range(activate_indicator.size(0)):
            if activate_indicator[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
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
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass

        activate_indicator = conf_in_vanilla < (self.thres - self.gap * self.thres) ## only store high confident samples into feature memory.
        ####################################### in the above formulation, the default middle point is 0.5; thres - 0.5 is the unsed gap.
        _, pred_all = torch.max(output_vanilla[:, class_num:], dim=1)
        prob_ood = torch.softmax(output_vanilla[:, class_num:], dim=1)
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
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass

        ###predicting final ood confident and ID prediction with self.feat_memory
        sim = self.feat_memory @ image_features.t()  ## 11k*30*256
        sim = torch.exp(-self.beta * (-sim + 1))  ##
        if self.samada and self.args.samada_layout == "factored":
            output = logit_scale * sample_adaptive_cosine(self.feat_memory, image_features, self.beta)
        else:
            if self.samada:
                sa_text_features_list = []
                split_num = int(self.feat_memory.size(0) / 1000.0)
                for i in range(split_num):
                    temp = sim[1000*i:1000*(i+1)].unsqueeze(0).transpose(0,-1) * self.feat_memory[1000*i:1000*(i+1)].unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
                    sa_text_features_split = temp.sum(2) ## 256*1k*512
                    sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)  ## renorm.
                    sa_text_features_list.append(sa_text_features_split)
                if self.feat_memory.size(0) % 1000.0 > 0:
                    ### processing the remaining one.
                    temp = sim[1000*split_num:].unsqueeze(0).transpose(0,-1) * self.feat_memory[1000*split_num:].unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
                    sa_text_features_split = temp.sum(2) ## 256*1k*512
                    sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)  ## renorm.
                    sa_text_features_list.append(sa_text_features_split)
                sa_text_features = torch.cat(sa_text_features_list, dim=1)
            else:
                sa_text_features = self.feat_memory.mean(1)
                sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.

            output = logit_scale * (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k
        prob_tta = torch.softmax(output, dim=1)
        prob_all = prob_vanilla + prob_tta * self.lambda_val
        _, pred_in = torch.max(prob_all[:, :class_num], dim=1)

        pos_logit = output[:, :class_num] ## B*C
        neg_logit = output[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output.device)
            neg_logit = neg_logit.T ## total_neg_num*B
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

        # max in prob - max out prob
        if self.in_score == 'adaonly':
            conf = conf_in  ## = 1-conf_out
        elif self.in_score == 'vanillaonly':
            conf = conf_in_vanilla
        elif self.in_score == 'combine':
            conf = conf_in + conf_in_vanilla * self.lambda_val
        elif self.in_score == 'multiply':
            conf = conf_in * conf_in_vanilla
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            raise FloatingPointError("AdaNeg produced a NaN confidence.")

        return pred_in, conf


def adaneg_arguments(config: Any) -> SimpleNamespace:
    values, setting = resolve_method_arguments("adaneg", config.postprocessor.postprocessor_args, ADANEG_ARGUMENTS,
                                               dataset=dataset_name(config))
    in_score = str(values["in_score"])
    if in_score not in IN_SCORES:
        raise ValueError(f"adaneg.postprocessor_args.in_score must be one of {list(IN_SCORES)}, got {in_score!r}.")
    layout = str(values["samada_layout"])
    if layout not in SAMADA_LAYOUTS:
        raise ValueError(f"adaneg.postprocessor_args.samada_layout must be one of {list(SAMADA_LAYOUTS)}, "
                         f"got {layout!r}.")
    return SimpleNamespace(
        thres=number_argument("adaneg", values, "thres"),
        gap=number_argument("adaneg", values, "gap"),
        memleng=integer_argument("adaneg", values, "memleng", minimum=1),
        lambdaval=number_argument("adaneg", values, "lambdaval"),
        beta=number_argument("adaneg", values, "beta"),
        group_num=integer_argument("adaneg", values, "group_num", minimum=1),
        random_permute=boolean_argument("adaneg", values, "random_permute"),
        samada=boolean_argument("adaneg", values, "samada"),
        in_score=in_score,
        text_prompt=None if values["text_prompt"] is None else str(values["text_prompt"]),
        ood_number=None if values["ood_number"] is None else integer_argument("adaneg", values, "ood_number",
                                                                              minimum=1),
        samada_layout=layout,
        setting=setting,
    )


class AdaNegPostprocessor(TTAPromptPostprocessor_noadagap):
    """Wrapper around the official ``TTAPromptPostprocessor_noadagap``. Its network is built on this repository's CLIP
    model and caches.
    """

    supports_image_feature_cache = True

    def __init__(self, config):
        super().__init__(config)
        self.network: FixedCLIP_NegOODPrompt | None = None
        self.device: torch.device | None = None
        self.class_num: int | None = None

    def setup(self, net: nn.Module, id_loader_dict=None, ood_loader_dict=None):
        if self.network is None:
            ood_number = self.args.ood_number
            if ood_number is None:
                ood_number = int(get_config(get_config(self.config.vlm, "neglabel", None), "ood_number", 10000))
            self.network = FixedCLIP_NegOODPrompt(net, self.config, text_prompt=self.args.text_prompt,
                                                  ood_number=ood_number, corpus=False)
            self.device = self.network.device
            self.class_num = self.network.n_cls
            print(f"[AdaNegPostprocessor] setup complete | {self.policy_summary()}", flush=True)
        return super().setup(self.network, id_loader_dict, ood_loader_dict)

    def postprocess(self, net: nn.Module, data: Any):
        if self.network is None:
            raise RuntimeError("setup() must be called before postprocess().")
        return super().postprocess(self.network, data)

    def policy_summary(self) -> dict[str, Any]:
        summary = {"source": SOURCE, **{key: getattr(self.args, key) for key in ADANEG_ARGUMENTS},
                   "setting": self.args.setting}
        if self.network is not None:
            summary["labels"] = int(self.network.text_features.shape[1])
        return summary


__all__ = ["ADANEG_ARGUMENTS", "AdaNegPostprocessor", "IN_SCORES", "SAMADA_LAYOUTS", "SOURCE",
           "TTAPromptPostprocessor_noadagap", "adaneg_arguments", "sample_adaptive_cosine"]
