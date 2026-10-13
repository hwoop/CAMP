import os
from typing import Dict, Tuple

import torch

from openood.utils.vlm.cache import make_cache_path, save_cache, load_cache
from openood.utils.vlm.classnames import get_class_names
from openood.utils.vlm.prompts import get_templates
from openood.utils.vlm.text.encoder import encode_text_list


def get_cache_path(
    dataset: str,
    model_name: str,
    backbone: str,
    prompt_name: str,
    text_center: bool = True,
    return_type: str = "features",
) -> str:

    return make_cache_path(
        dataset=dataset,
        model_name=model_name,
        backbone=backbone,
        cache_group='text',
        params={
            "prompt": prompt_name,
            "center": text_center,
            "all": False,
            "return": return_type,
        },
    )


@torch.no_grad()
def build_text_features(
    model,
    dataset: str,
    prompt_name: str,
    device: str = "cuda",
    text_center: bool = True,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:

    classnames = get_class_names(dataset)
    templates = get_templates(prompt_name)

    print(f"[TextFeatures] dataset={dataset}")
    print(f"[TextFeatures] prompt={prompt_name}")
    print(f"[TextFeatures] num_classes={len(classnames)}")
    print(f"[TextFeatures] num_templates={len(templates)}")

    text_features = encode_text_list(
        model=model,
        texts=classnames,
        templates=templates,
        device=device,
        text_center=text_center,
        batch_size=batch_size,
    )

    text_labels = torch.arange(len(classnames), device=device)

    return text_features, text_labels


def save_text_feature_cache(
    cache_path: str,
    metadata: Dict,
    text_features: torch.Tensor,
    text_labels: torch.Tensor,
) -> None:

    save_cache(
        cache_path=cache_path,
        cache_obj={
            "metadata": metadata,
            "text_features": text_features,
            "text_labels": text_labels,
        },
    )


def load_text_feature_cache(
    cache_path: str,
    device: str = "cuda",
) -> torch.Tensor:

    cache = load_cache(cache_path, device=device)

    return cache["text_features"]


def load_text_features(
    model,
    dataset: str,
    device,
    backbone: str,
    text_prompt: str,
    model_name: str,
    text_center: bool = True,
    batch_size: int = 256,
) -> torch.Tensor:

    cache_path = get_cache_path(
        dataset=dataset,
        model_name=model_name,
        backbone=backbone,
        prompt_name=text_prompt,
        text_center=text_center,
        return_type="features",
    )
    if not os.path.exists(cache_path):
        print(f"[TextFeatures] Building cache: {cache_path}")

        text_features, text_labels = build_text_features(
            model=model,
            dataset=dataset,
            prompt_name=text_prompt,
            device=device,
            text_center=text_center,
            batch_size=batch_size,
        )

        metadata: Dict = {
            "dataset": dataset,
            "model_name": model_name,
            "backbone": backbone,
            "prompt_name": text_prompt,
            "text_center": text_center,
            "batch_size": batch_size,
            "cache_type": "text",
            "return": "features",
            "cache_path": cache_path,
        }

        save_text_feature_cache(
            cache_path=cache_path,
            metadata=metadata,
            text_features=text_features,
            text_labels=text_labels,
        )

        print(f"[TextFeatures] Saved cache: {cache_path}")

    print(f"[TextFeatures] Loaded cache: {cache_path}")
    return load_text_feature_cache(
        cache_path=cache_path,
        device=device,
    )
