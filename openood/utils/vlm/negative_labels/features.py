import os
from typing import Dict

import torch

from openood.utils.vlm.cache import make_cache_path, save_cache, load_cache
from openood.utils.vlm.prompts import get_templates
from openood.utils.vlm.text.encoder import encode_text_list
from openood.utils.vlm.negative_labels.wordnet import (
    load_wordnet_negative_label_lists,
)


def get_cache_path(
    dataset: str,
    model_name: str,
    backbone: str,
    prompt_name: str,
    text_center: bool = True,
    ood_number: int = 10000,
    corpus: str = "wordnet",
    cos_percentile: float = 0.95,
    candidate_mode: str = "selected",
    encode_unselected: bool = False,
    cache_group: str = "neglabel",
    return_type: str = "features",
) -> str:

    return make_cache_path(
        dataset=dataset,
        model_name=model_name,
        backbone=backbone,
        cache_group=cache_group,
        params={
            "prompt": prompt_name,
            "center": text_center,
            "ood": ood_number,
            "corpus": corpus,
            "cos": cos_percentile,
            "mode": candidate_mode,
            "unselected": encode_unselected,
            "return": return_type,
        },
    )


def _select_candidate_labels(
    label_dict: Dict,
    candidate_mode: str,
) -> Dict:

    candidate_mode = candidate_mode.lower()
    if candidate_mode == "selected":
        return {
            "adj": label_dict["selected_adj"],
            "noun": label_dict["selected_noun"],
            "labels": label_dict["selected"],
        }

    if candidate_mode == "unselected":
        return {
            "adj": label_dict["unselected_adj"],
            "noun": label_dict["unselected_noun"],
            "labels": label_dict["unselected"],
        }

    if candidate_mode == "all":
        return {
            "adj": label_dict["selected_adj"] + label_dict["unselected_adj"],
            "noun": label_dict["selected_noun"] + label_dict["unselected_noun"],
            "labels": label_dict["selected"] + label_dict["unselected"],
        }

    raise NotImplementedError(
        f"Unknown candidate_mode='{candidate_mode}'. "
        "Use one of: selected, unselected, all."
    )


@torch.no_grad()
def build_neglabel_text_features(
    model,
    dataset: str,
    prompt_name: str,
    text_center: bool = True,
    ood_number: int = 10000,
    device: str = "cuda",
    corpus: str = "wordnet",
    cos_percentile: float = 0.95,
    candidate_mode: str = "selected",
    encode_unselected: bool = False,
    batch_size: int = 256,
) -> Dict:

    corpus = corpus.lower()
    if corpus != "wordnet":
        raise NotImplementedError(
            f"Only corpus='wordnet' is supported now, got corpus='{corpus}'."
        )

    label_dict = load_wordnet_negative_label_lists(
        dataset=dataset,
        total_ood_num=ood_number,
        cos_percentile=cos_percentile
    )

    candidate_dict = _select_candidate_labels(
        label_dict=label_dict,
        candidate_mode=candidate_mode,
    )

    neg_adj_labels = candidate_dict["adj"]
    neg_noun_labels = candidate_dict["noun"]
    neg_labels = candidate_dict["labels"]

    noun_templates = get_templates(prompt_name)
    adj_templates = get_templates("neg_adj")

    print(f"[NegLabelFeatures] dataset={dataset}")
    print(f"[NegLabelFeatures] corpus={corpus}")
    print(f"[NegLabelFeatures] mode={candidate_mode}")
    print(f"[NegLabelFeatures] adj labels={len(neg_adj_labels)}")
    print(f"[NegLabelFeatures] noun labels={len(neg_noun_labels)}")

    neg_adj_features = None
    neg_noun_features = None

    if len(neg_adj_labels) > 0:
        neg_adj_features = encode_text_list(
            model=model,
            texts=neg_adj_labels,
            templates=adj_templates,
            device=device,
            text_center=text_center,
            batch_size=batch_size,
        )

    if len(neg_noun_labels) > 0:
        neg_noun_features = encode_text_list(
            model=model,
            texts=neg_noun_labels,
            templates=noun_templates,
            device=device,
            text_center=text_center,
            batch_size=batch_size,
        )

    if neg_adj_features is not None and neg_noun_features is not None:
        neg_text_features = torch.cat([neg_adj_features, neg_noun_features], dim=1)
    elif neg_adj_features is not None:
        neg_text_features = neg_adj_features
    elif neg_noun_features is not None:
        neg_text_features = neg_noun_features
    else:
        raise ValueError("No negative labels were selected.")

    unselected_text_features = None
    unselected_labels = label_dict["unselected"]

    if encode_unselected:
        unselected_adj = label_dict["unselected_adj"]
        unselected_noun = label_dict["unselected_noun"]

        unselected_adj_features = None
        unselected_noun_features = None

        if len(unselected_adj) > 0:
            unselected_adj_features = encode_text_list(
                model=model,
                texts=unselected_adj,
                templates=adj_templates,
                device=device,
                text_center=text_center,
                batch_size=batch_size,
            )

        if len(unselected_noun) > 0:
            unselected_noun_features = encode_text_list(
                model=model,
                texts=unselected_noun,
                templates=noun_templates,
                device=device,
                text_center=text_center,
                batch_size=batch_size,
            )

        if unselected_adj_features is not None and unselected_noun_features is not None:
            unselected_text_features = torch.cat(
                [unselected_adj_features, unselected_noun_features],
                dim=1,
            )
        elif unselected_adj_features is not None:
            unselected_text_features = unselected_adj_features
        elif unselected_noun_features is not None:
            unselected_text_features = unselected_noun_features

    metadata = {
        "dataset": dataset,
        "prompt_name": prompt_name,
        "text_center": text_center,
        "ood_number": ood_number,
        "corpus": corpus,
        "cos_percentile": cos_percentile,
        "candidate_mode": candidate_mode,
        "encode_unselected": encode_unselected,
        "neg_adj_num": len(neg_adj_labels),
        "neg_noun_num": len(neg_noun_labels),
        "neg_total_num": len(neg_labels),
        "unselected_num": len(unselected_labels),
        "return": "features",
    }

    return {
        "neg_text_features": neg_text_features,
        "unselected_text_features": unselected_text_features,
        "neg_labels": neg_labels,
        "neg_adj_labels": neg_adj_labels,
        "neg_noun_labels": neg_noun_labels,
        "unselected_labels": unselected_labels,
        "label_metadata": label_dict["metadata"],
        "metadata": metadata,
    }


def load_neglabel_text_features(
    model,
    dataset: str,
    prompt_name: str,
    backbone: str,
    model_name: str,
    text_center: bool = True,
    device: str = "cuda",
    batch_size: int = 256,

    ood_number: int = 10000,
    corpus: str = "wordnet",
    cos_percentile: float = 0.95,
    candidate_mode: str = "selected",
    encode_unselected: bool = False,
) -> Dict:

    cache_path = get_cache_path(
        dataset=dataset,
        model_name=model_name,
        backbone=backbone,
        prompt_name=prompt_name,
        text_center=text_center,
        ood_number=ood_number,
        corpus=corpus,
        cos_percentile=cos_percentile,
        candidate_mode=candidate_mode,
        encode_unselected=encode_unselected,
        cache_group="neglabel",
        return_type="features",
    )
    if not os.path.exists(cache_path):
        print(f"[NegLabelFeatures] Building cache: {cache_path}")

        built = build_neglabel_text_features(
            model=model,
            dataset=dataset,
            prompt_name=prompt_name,
            text_center=text_center,
            ood_number=ood_number,
            device=device,
            corpus=corpus,
            cos_percentile=cos_percentile,
            candidate_mode=candidate_mode,
            encode_unselected=encode_unselected,
            batch_size=batch_size,
        )

        metadata = dict(built["metadata"])
        metadata.update(
            {
                "model_name": model_name,
                "backbone": backbone,
                "cache_type": "neglabel",
                "cache_path": cache_path,
            }
        )

        cache = {
            "metadata": metadata,
            "label_metadata": built["label_metadata"],
            "neg_text_features": built["neg_text_features"],
            "unselected_text_features": built["unselected_text_features"],
            "neg_labels": built["neg_labels"],
            "neg_adj_labels": built["neg_adj_labels"],
            "neg_noun_labels": built["neg_noun_labels"],
            "unselected_labels": built["unselected_labels"],
        }

        save_cache(cache_path=cache_path, cache_obj=cache)
        print(f"[NegLabelFeatures] Saved cache: {cache_path}")

    print(f"[NegLabelFeatures] Loaded cache: {cache_path}")

    cache = load_cache(cache_path, device=device)
    cache["cache_path"] = cache_path

    return cache
