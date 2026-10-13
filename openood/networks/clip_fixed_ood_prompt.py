from __future__ import annotations

from typing import Any

import torch

from openood.networks.clip import clip
from openood.utils import get_config
from openood.utils.vlm.classnames import get_class_names
from openood.utils.vlm.image.types import CachedImageFeatures
from openood.utils.vlm.negative_labels import load_neglabel_text_features
from openood.utils.vlm.negative_labels.wordnet import load_or_build_wordnet_candidates
from openood.utils.vlm.text import load_text_features


def text_cache_settings(config: Any, text_prompt: str | None = None) -> dict[str, Any]:
    prompt = text_prompt if text_prompt is not None else get_config(config.vlm.text, "prompt", None)
    if prompt is None:
        raise ValueError("vlm.text.prompt is required: the prompt set of the ID class prototypes (e.g. nice).")
    return {
        "dataset": str(config.dataset.name),
        "model_name": str(config.network.name),
        "backbone": str(config.network.backbone.name),
        "text_center": bool(get_config(config.vlm.text, "center", True)),
        "batch_size": int(get_config(config.dataset.test, "batch_size", 256)),
        "prompt": str(prompt),
    }


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_id_and_negative_features(net: Any, config: Any, text_prompt: str | None = None,
                                  ood_number: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    settings = text_cache_settings(config, text_prompt)
    device = model_device(net.model)
    neglabel = get_config(config.vlm, "neglabel", None)
    if ood_number is None:
        ood_number = int(get_config(neglabel, "ood_number", 10000))
    text_features = load_text_features(
        model=net.model, dataset=settings["dataset"], device=device, backbone=settings["backbone"],
        text_prompt=settings["prompt"], model_name=settings["model_name"], text_center=settings["text_center"],
        batch_size=settings["batch_size"],
    )
    cache = load_neglabel_text_features(
        model=net.model, dataset=settings["dataset"], prompt_name=settings["prompt"], backbone=settings["backbone"],
        model_name=settings["model_name"], text_center=settings["text_center"], device=device,
        batch_size=settings["batch_size"], ood_number=int(ood_number),
        corpus=str(get_config(neglabel, "corpus", "wordnet")),
        cos_percentile=float(get_config(neglabel, "cos_percentile", 0.95)),
        candidate_mode=str(get_config(neglabel, "candidate_mode", "selected")),
        encode_unselected=True,
    )
    return text_features.to(device), cache


def unselected_text_features(cache: dict[str, Any], like: torch.Tensor) -> torch.Tensor:
    encodings = cache["unselected_text_features"]
    if encodings is None:
        return like.new_empty((like.shape[0], 0))
    return encodings


def get_text_features_neg(net: Any, config: Any, text_prompt: str | None,
                          ood_number: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of OpenOOD-VLM's ``get_text_features_neg`` (lines 616-718). Returns the ID and selected negative text
    features together with the features of the unselected labels.
    """
    text_features, cache = load_id_and_negative_features(net, config, text_prompt, ood_number)
    device = text_features.device
    text_features_neg = cache["neg_text_features"].to(device)
    text_features_unselected = unselected_text_features(cache, text_features_neg).to(device)
    text_features_id_ood = torch.cat((text_features, text_features_neg), dim=1)
    return text_features_id_ood, text_features_unselected


class FixedCLIP_NegOODPrompt:
    """Port of OpenOOD-VLM's ``FixedCLIP_NegOODPrompt`` (clip_fixed_ood_prompt.py, lines 721-773) onto this repository's
    CLIP model.
    """

    def __init__(self, net: Any, config: Any, *, text_prompt: str | None = None, ood_number: int | None = None,
                 corpus: bool = True) -> None:
        backbone = str(config.network.backbone.name) # 'ViT-B/16'   (wrap: cfg.backbone of this repository's config)
        classnames = get_class_names(str(config.dataset.name)) # imagenet
        self.n_cls = len(classnames)
        self.n_output = self.n_cls + 10000
        assert backbone in clip.available_models()
        clip_model = net.model
        self.model = clip_model
        self.device = model_device(clip_model)
        self.logit_scale = clip_model.logit_scale.data

        # text_features: feat_dim * text_nums
        self.text_features, self.text_features_unselected = get_text_features_neg(net, config, text_prompt, ood_number)

        if corpus:
            self.text_features_all = torch.cat((self.text_features, self.text_features_unselected), dim=1)  ## 512* (11k + 80k)
            gaussian_noise_image = torch.randn((3, 3, 224, 224), dtype=torch.float).to(self.device)  ## Gaussian Noise
            gaussian_noise_image = (gaussian_noise_image - gaussian_noise_image.min()) / (gaussian_noise_image.max() - gaussian_noise_image.min())
            uniform_noise_image = torch.rand((3, 3, 224, 224), dtype=torch.float).to(self.device)
            poisson_noise_image = torch.poisson(5.0 * torch.ones((3, 3, 224, 224))).to(self.device)
            poisson_noise_image = (poisson_noise_image - poisson_noise_image.min()) / (poisson_noise_image.max() - poisson_noise_image.min())
            gamma_noise = torch.distributions.Gamma(2, 1).sample((3,3,224,224)).to(self.device)
            gamma_noise = (gamma_noise - gamma_noise.min()) / (gamma_noise.max() - gamma_noise.min())
            salt_and_pepper_noise = torch.rand((3, 3, 224, 224), dtype=torch.float).to(self.device)
            salt_and_pepper_noise[salt_and_pepper_noise >= 0.5] = 1.0
            salt_and_pepper_noise[salt_and_pepper_noise < 0.5] = 0
            all_noise = torch.cat((gaussian_noise_image, uniform_noise_image, poisson_noise_image, gamma_noise, salt_and_pepper_noise), dim=0) ## 15*3*224*224
            with torch.no_grad():
                noise_image_features = self.model.encode_image(all_noise)
                self.noise_image_features = noise_image_features / noise_image_features.norm(dim=-1, keepdim=True)
        else:
            self.text_features_unselected = None

        self.logit_scale = self.logit_scale.exp()

    def forward(self, x: Any, return_feat: bool = False) -> Any:
        if isinstance(x, CachedImageFeatures):
            image_features = x.values.to(self.device)
        else:
            image_features = self.model.encode_image(x)
            image_features /= image_features.norm(dim=-1, keepdim=True)  ## batch * dim

        if return_feat:
            return image_features, self.text_features.transpose(0,1), self.logit_scale ## 512*11K --> 11K*512
        else:
            logits = self.logit_scale * image_features @ self.text_features  ## when eval acc, the forward pass is here.
            return logits

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def eval(self) -> FixedCLIP_NegOODPrompt:
        self.model.eval()
        return self


def coevo_pool(net: Any, config: Any, text_prompt: str | None,
               ood_number: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of CoEvo's ``get_text_features_neg`` (lines 483-539). Returns the ID features, ``neg_idx`` and
    ``pool_feature``.
    """
    text_features, cache = load_id_and_negative_features(net, config, text_prompt, ood_number)
    device = text_features.device
    wordnet = load_or_build_wordnet_candidates(dataset=str(config.dataset.name))
    labels = list(cache["neg_labels"]) + list(cache["unselected_labels"])
    encodings = torch.cat((cache["neg_text_features"],
                           unselected_text_features(cache, cache["neg_text_features"])), dim=1).to(device)
    column = {label: index for index, label in enumerate(labels)}
    pool = list(wordnet["text_list_adj"]) + list(wordnet["text_list_noun"])
    if len(column) != len(labels) or sorted(column) != sorted(pool):
        raise ValueError(f"the NegLabel cache of {config.dataset.name} and its WordNet candidate file hold different "
                         "labels; CoEvo's pool cannot be ordered as its file")
    order = torch.tensor([column[label] for label in pool], dtype=torch.long, device=device)
    pool_feature = encodings.t()[order].contiguous()
    position = {label: index for index, label in enumerate(pool)}
    neg_idx = torch.tensor([position[label] for label in cache["neg_labels"]], dtype=torch.long)
    return text_features.t(), neg_idx, pool_feature


class FixedCLIP_CoEvo_Neglabel:
    """Port of CoEvo's ``FixedCLIP_CoEvo_Neglabel`` (clip_fixed_ood_prompt.py, lines 542-662) onto this repository's CLIP
    model.
    """

    def __init__(self, net: Any, config: Any, *, text_prompt: str | None, ood_number: int | None, topk_neg: int,
                 logit_scale_vision: float, logit_scale: float = 100) -> None:
        backbone = str(config.network.backbone.name)  # 'ViT-B/16'   (wrap: cfg.backbone of this repository's config)
        classnames = get_class_names(str(config.dataset.name))  # imagenet
        self.n_cls = len(classnames)
        self.n_output = self.n_cls + 10000
        assert backbone in clip.available_models()
        clip_model = net.model
        clip_model.eval()
        self.model = clip_model
        self.device = model_device(clip_model)
        self.logit_scale = logit_scale                  # cfg.backbone.ls (configs/networks/fixed_clip.yml: 100)
        self.logit_scale_vision = logit_scale_vision    # cfg.backbone.ls_vision
        self.topk_neg = topk_neg

        self.text_features_pos, self.neg_idx, self.pool_feature = coevo_pool(net, config, text_prompt, ood_number)
        self.pool_feature = self.pool_feature.to(self.device)
        self.neg_idx = self.neg_idx.to(self.device)

        self.neg_idx_origin = self.neg_idx
        self.text_features = torch.cat((self.text_features_pos, self.pool_feature[self.neg_idx]))
        self.text_features_origin = self.text_features

    def reset_text_feature(self) -> None:
        self.neg_idx = self.neg_idx_origin
        self.text_features = self.text_features_origin

    ### 100.0 is csp logit
    def forward(self, x: Any, return_feat: bool = False) -> Any:
        with torch.no_grad():
            if isinstance(x, CachedImageFeatures):
                image_features = x.values.to(self.device)
            else:
                image_features = self.model.encode_image(x)
                image_features /= image_features.norm(dim=-1, keepdim=True)  ## batch * dim

        if return_feat:
            return image_features.half(), self.text_features.half(), self.logit_scale,self.logit_scale_vision  ## 7*11k*512 --> 11K*7*512
        else:
            logits = 100.0 * image_features @ self.text_features  ## when eval acc, the forward pass is here.
            return logits

    def get_id_text_feature(self, id_image_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        with torch.no_grad():

            id_wordnet_idx = []
            text_batch_size = 1000
            topk_neg = self.topk_neg
            img_batch = id_image_feat
            batch_size = img_batch.size(0)

            topk_values = torch.full((batch_size, topk_neg), float('inf'),
                                     device=img_batch.device, dtype=torch.half)
            topk_indices = torch.full((batch_size, topk_neg), -1,
                                      device=img_batch.device, dtype=torch.long)

            for j in range(0, self.pool_feature.size(0), text_batch_size):
                text_batch = self.pool_feature[j:j + text_batch_size]
                scores = img_batch @ text_batch.T
                text_indices = torch.arange(j, j + text_batch.size(0),
                                            device=scores.device).expand(batch_size, -1)
                combined = torch.cat([topk_values, scores], dim=1)
                combined_indices = torch.cat([topk_indices, text_indices], dim=1)

                sorted_idx = torch.argsort(combined, dim=1, descending=False)[:, :topk_neg]
                topk_values = torch.gather(combined, 1, sorted_idx)
                topk_indices = torch.gather(combined_indices, 1, sorted_idx)

            id_wordnet_idx = topk_indices.view(-1)
            unique_new_neg_indices = torch.unique(id_wordnet_idx)
            is_new = ~torch.isin(unique_new_neg_indices, self.neg_idx)
            final_new_neg_indices = unique_new_neg_indices[is_new]

            self.neg_idx = torch.cat([self.neg_idx,final_new_neg_indices])
            self.text_features = torch.cat([self.text_features, self.pool_feature[final_new_neg_indices]],
                                                          dim=0).half()

        return self.text_features,self.pool_feature[final_new_neg_indices],final_new_neg_indices.shape[0]

    def get_ood_text_feature(self, ood_image_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        img_batch = ood_image_feat  # [N_ood, D]
        batch_size = img_batch.size(0)
        text_batch_size = 1000

        topk_neg = self.topk_neg
        topk_values = torch.full((batch_size, topk_neg), -float('inf'),
                                 device=img_batch.device, dtype=torch.half)
        topk_indices = torch.full((batch_size, topk_neg), -1,
                                  device=img_batch.device, dtype=torch.long)

        for j in range(0, self.pool_feature.size(0), text_batch_size):
            text_batch = self.pool_feature[j:j + text_batch_size]
            scores = img_batch @ text_batch.T
            text_indices = torch.arange(j, j + text_batch.size(0),
                                        device=scores.device).expand(batch_size, -1)
            combined = torch.cat([topk_values, scores], dim=1)
            combined_indices = torch.cat([topk_indices, text_indices], dim=1)

            sorted_idx = torch.argsort(combined, dim=1, descending=True)[:, :topk_neg]
            topk_values = torch.gather(combined, 1, sorted_idx)
            topk_indices = torch.gather(combined_indices, 1, sorted_idx)

        ood_wordnet_idx = topk_indices.view(-1)

        unique_new_neg_indices = torch.unique(ood_wordnet_idx)
        is_new = ~torch.isin(unique_new_neg_indices, self.neg_idx)
        final_new_neg_indices = unique_new_neg_indices[is_new]
        self.neg_idx = torch.cat([self.neg_idx, final_new_neg_indices])

        self.text_features = torch.cat([self.text_features, self.pool_feature[final_new_neg_indices]],
                                       dim=0).half()
        return self.text_features, self.pool_feature[final_new_neg_indices], final_new_neg_indices.shape[0]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def eval(self) -> FixedCLIP_CoEvo_Neglabel:
        self.model.eval()
        return self


__all__ = ["FixedCLIP_CoEvo_Neglabel", "FixedCLIP_NegOODPrompt", "coevo_pool", "get_text_features_neg",
           "load_id_and_negative_features", "model_device", "text_cache_settings"]
