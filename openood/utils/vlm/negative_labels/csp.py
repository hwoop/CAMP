"""
Negative labels of CSP. Where the paper leaves details unspecified, 
the official code (NeurIPS2024-CSP@3a7bdab, multi_modal.py) is followed.
"""

import hashlib
import os
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from openood.utils.reproducibility import derive_seed, make_torch_generator
from openood.utils.vlm.cache import make_cache_path, load_cache, save_cache
from openood.utils.vlm.prompts import get_templates
from openood.utils.vlm.text.encoder import encode_text_list


CSP_EXCLUDED_FILES: Tuple[str, ...] = (
    "noun.person.txt",
    "noun.quantity.txt",
    "noun.group.txt",
    "adj.pert.txt",
)

CSP_CORPUS_ROOT = "./data/txtfiles"

CSP_CONJUGATION_NAMESPACE = "vlm.negative_labels.csp.conjugation"

CSP_NOUN_PROMPT = "csp_noun"
CSP_ADJ_PROMPT = "csp_adj"
CSP_SUPERCLASS_PROMPT = "csp_superclass"


def parse_excluded_files(
    exclude: Any,
) -> Tuple[str, ...]:

    if exclude is None:
        return ()

    if isinstance(exclude, str):
        names = exclude.split(",")
    elif isinstance(exclude, Iterable):
        names = [str(name) for name in exclude]
    else:
        raise TypeError(
            f"exclude must be a comma-separated string or a sequence of file names, got {type(exclude).__name__}."
        )

    return tuple(sorted({name.strip() for name in names if str(name).strip()}))


def load_csp_candidate_labels(
    corpus_root: str = CSP_CORPUS_ROOT,
    exclude: Any = CSP_EXCLUDED_FILES,
    seed: int = 0,
    superclass_prompt: str = CSP_SUPERCLASS_PROMPT,
) -> Dict:

    excluded = set(parse_excluded_files(exclude))
    files = [name for name in sorted(os.listdir(corpus_root)) if name not in excluded]

    seen: Dict[str, None] = {}
    nouns: List[str] = []
    adjectives: List[str] = []
    read_files: List[str] = []

    for name in files:
        part_of_speech = name.split(".")[0]
        if part_of_speech not in ("noun", "adj"):
            continue

        read_files.append(name)

        with open(os.path.join(corpus_root, name), "r") as handle:
            lines = handle.readlines()

        for line in lines:
            # Official order: replace "_" before strip(); strip() also drops the CRLF of this repository's copy.
            label = line.replace("_", " ").strip()
            if label in seen:
                continue

            seen[label] = None

            if part_of_speech == "noun":
                nouns.append(label)
            else:
                adjectives.append(label)

    superclass_templates = list(get_templates(superclass_prompt))
    generator = make_torch_generator(derive_seed(seed=seed, namespace=CSP_CONJUGATION_NAMESPACE))
    superclass_index = torch.randint(
        len(superclass_templates),
        (len(adjectives),),
        generator=generator,
    ).tolist()

    conjugated = [
        superclass_templates[index].format(label)
        for label, index in zip(adjectives, superclass_index)
    ]

    return {
        "nouns": nouns,
        "adjectives": adjectives,
        "conjugated": conjugated,
        "superclass_index": superclass_index,
        "files": read_files,
        "metadata": {
            "corpus_root": str(corpus_root),
            "excluded_files": list(parse_excluded_files(exclude)),
            "superclass_prompt": superclass_prompt,
            "superclass_num": len(superclass_templates),
            "seed": int(seed),
            "conjugation_namespace": CSP_CONJUGATION_NAMESPACE,
            "noun_candidate_num": len(nouns),
            "adj_candidate_num": len(adjectives),
        },
    }


@torch.no_grad()
def id_proximity(
    features: torch.Tensor,
    id_prototypes: torch.Tensor,
    cos_percentile: float = 0.95,
    max_cos: float = 0.95,
    chunk_size: int = 1000,
) -> torch.Tensor:

    proximity = []

    for start in range(0, features.shape[1], chunk_size):
        cosine = (features[:, start:start + chunk_size].T @ id_prototypes).float()
        quantile = torch.quantile(cosine, q=cos_percentile, dim=-1)
        quantile[cosine.max(dim=1).values > max_cos] = 1.0
        proximity.append(quantile)

    return torch.cat(proximity, dim=0)


def select_least_similar(
    proximity: torch.Tensor,
    select_ratio: float,
) -> torch.Tensor:
    """
    Select the ``int(n * ratio)`` least similar candidates, the count used by the official code. 
    ``stable=True`` is this repository's addition; it fixes the order of tied proximities.
    """

    keep = int(proximity.numel() * float(select_ratio))
    if keep <= 0:
        raise ValueError(
            f"select_ratio={select_ratio} keeps no candidate out of {int(proximity.numel())}."
        )

    return torch.argsort(proximity, stable=True)[:keep]


def get_cache_path(
    dataset: str,
    model_name: str,
    backbone: str,
    prompt_name: str,
    text_center: bool = True,
    corpus_root: str = CSP_CORPUS_ROOT,
    exclude: Any = CSP_EXCLUDED_FILES,
    noun_prompt: str = CSP_NOUN_PROMPT,
    adj_prompt: str = CSP_ADJ_PROMPT,
    superclass_prompt: str = CSP_SUPERCLASS_PROMPT,
    select_ratio: float = 0.15,
    cos_percentile: float = 0.95,
    max_cos: float = 0.95,
    seed: int = 0,
    cache_group: str = "csp",
    return_type: str = "features",
) -> str:

    # The path holds a digest of the excluded names because the names carry dots and commas.
    digest = hashlib.sha256(",".join(parse_excluded_files(exclude)).encode("utf-8")).hexdigest()[:8]

    return make_cache_path(
        dataset=dataset,
        model_name=model_name,
        backbone=backbone,
        cache_group=cache_group,
        params={
            "prompt": prompt_name,
            "center": text_center,
            "noun": noun_prompt,
            "adj": adj_prompt,
            "super": superclass_prompt,
            "exclude": digest,
            "ratio": select_ratio,
            "cos": cos_percentile,
            "max": max_cos,
            "seed": seed,
            "return": return_type,
        },
    )


@torch.no_grad()
def build_csp_text_features(
    model,
    id_text_features: torch.Tensor,
    corpus_root: str = CSP_CORPUS_ROOT,
    exclude: Any = CSP_EXCLUDED_FILES,
    noun_prompt: str = CSP_NOUN_PROMPT,
    adj_prompt: str = CSP_ADJ_PROMPT,
    superclass_prompt: str = CSP_SUPERCLASS_PROMPT,
    select_ratio: float = 0.15,
    cos_percentile: float = 0.95,
    max_cos: float = 0.95,
    seed: int = 0,
    device: str = "cuda",
    batch_size: int = 256,
    chunk_size: int = 1000,
) -> Dict:
    """
    Build the CSP text features. 
    ``id_text_features`` must be encoded with the ``csp_noun`` templates,
    which is how the official code builds its ID prototypes.
    """

    candidates = load_csp_candidate_labels(
        corpus_root=corpus_root,
        exclude=exclude,
        seed=seed,
        superclass_prompt=superclass_prompt,
    )

    noun_labels = candidates["nouns"]
    conjugated_labels = candidates["conjugated"]

    print(f"[CSPFeatures] corpus_root={corpus_root}")
    print(f"[CSPFeatures] files={len(candidates['files'])} (excluded {len(parse_excluded_files(exclude))})")
    print(f"[CSPFeatures] noun candidates={len(noun_labels)}")
    print(f"[CSPFeatures] conjugated adjective candidates={len(conjugated_labels)}")

    # CSP averages the raw prompt embeddings and normalizes once, unlike the rest of this repository.
    noun_features = encode_text_list(
        model=model,
        texts=noun_labels,
        templates=get_templates(noun_prompt),
        device=device,
        text_center=True,
        batch_size=batch_size,
        normalize_prompts=False,
    )

    conjugated_features = encode_text_list(
        model=model,
        texts=conjugated_labels,
        templates=get_templates(adj_prompt),
        device=device,
        text_center=True,
        batch_size=batch_size,
        normalize_prompts=False,
    )

    noun_features = F.normalize(noun_features.float(), dim=0)
    conjugated_features = F.normalize(conjugated_features.float(), dim=0)
    id_prototypes = F.normalize(id_text_features.to(device).float(), dim=0)

    noun_keep = select_least_similar(
        proximity=id_proximity(features=noun_features, id_prototypes=id_prototypes, cos_percentile=cos_percentile, max_cos=max_cos, chunk_size=chunk_size),
        select_ratio=select_ratio,
    )

    conjugated_keep = select_least_similar(
        proximity=id_proximity(features=conjugated_features, id_prototypes=id_prototypes, cos_percentile=cos_percentile, max_cos=max_cos, chunk_size=chunk_size),
        select_ratio=select_ratio,
    )

    selected_noun = [noun_labels[index] for index in noun_keep.tolist()]
    selected_conjugated = [conjugated_labels[index] for index in conjugated_keep.tolist()]

    neg_text_features = torch.cat(
        [noun_features[:, noun_keep], conjugated_features[:, conjugated_keep]],
        dim=1,
    )

    print(f"[CSPFeatures] selected nouns={len(selected_noun)}")
    print(f"[CSPFeatures] selected conjugated adjectives={len(selected_conjugated)}")
    print(f"[CSPFeatures] negative labels={neg_text_features.shape[1]}")

    metadata = {
        "corpus_root": str(corpus_root),
        "excluded_files": list(parse_excluded_files(exclude)),
        "noun_prompt": noun_prompt,
        "adj_prompt": adj_prompt,
        "superclass_prompt": superclass_prompt,
        "select_ratio": float(select_ratio),
        "cos_percentile": float(cos_percentile),
        "max_cos": float(max_cos),
        "seed": int(seed),
        "noun_candidate_num": len(noun_labels),
        "adj_candidate_num": len(conjugated_labels),
        "neg_noun_num": len(selected_noun),
        "neg_adj_num": len(selected_conjugated),
        "neg_total_num": int(neg_text_features.shape[1]),
        "id_class_num": int(id_prototypes.shape[1]),
        "return": "features",
    }

    return {
        "neg_text_features": neg_text_features,
        "neg_noun_labels": selected_noun,
        "neg_adj_labels": selected_conjugated,
        "neg_labels": selected_noun + selected_conjugated,
        "label_metadata": candidates["metadata"],
        "metadata": metadata,
    }


def load_csp_text_features(
    model,
    dataset: str,
    prompt_name: str,
    backbone: str,
    id_text_features: torch.Tensor,
    model_name: str,
    text_center: bool = True,
    device: str = "cuda",
    batch_size: int = 256,

    corpus_root: str = CSP_CORPUS_ROOT,
    exclude: Any = CSP_EXCLUDED_FILES,
    noun_prompt: str = CSP_NOUN_PROMPT,
    adj_prompt: str = CSP_ADJ_PROMPT,
    superclass_prompt: str = CSP_SUPERCLASS_PROMPT,
    select_ratio: float = 0.15,
    cos_percentile: float = 0.95,
    max_cos: float = 0.95,
    seed: int = 0,
) -> Dict:

    cache_path = get_cache_path(
        dataset=dataset,
        model_name=model_name,
        backbone=backbone,
        prompt_name=prompt_name,
        text_center=text_center,
        corpus_root=corpus_root,
        exclude=exclude,
        noun_prompt=noun_prompt,
        adj_prompt=adj_prompt,
        superclass_prompt=superclass_prompt,
        select_ratio=select_ratio,
        cos_percentile=cos_percentile,
        max_cos=max_cos,
        seed=seed,
        cache_group="csp",
        return_type="features",
    )
    if not os.path.exists(cache_path):
        print(f"[CSPFeatures] Building cache: {cache_path}")

        built = build_csp_text_features(
            model=model,
            id_text_features=id_text_features,
            corpus_root=corpus_root,
            exclude=exclude,
            noun_prompt=noun_prompt,
            adj_prompt=adj_prompt,
            superclass_prompt=superclass_prompt,
            select_ratio=select_ratio,
            cos_percentile=cos_percentile,
            max_cos=max_cos,
            seed=seed,
            device=device,
            batch_size=batch_size,
        )

        metadata = dict(built["metadata"])
        metadata.update(
            {
                "dataset": dataset,
                "prompt_name": prompt_name,
                "text_center": text_center,
                "model_name": model_name,
                "backbone": backbone,
                "cache_type": "csp",
                "cache_path": cache_path,
            }
        )

        cache = {
            "metadata": metadata,
            "label_metadata": built["label_metadata"],
            "neg_text_features": built["neg_text_features"],
            "neg_labels": built["neg_labels"],
            "neg_adj_labels": built["neg_adj_labels"],
            "neg_noun_labels": built["neg_noun_labels"],
        }

        save_cache(cache_path=cache_path, cache_obj=cache)
        print(f"[CSPFeatures] Saved cache: {cache_path}")

    print(f"[CSPFeatures] Loaded cache: {cache_path}")

    cache = load_cache(cache_path, device=device)
    cache["cache_path"] = cache_path

    return cache


__all__ = [
    "CSP_ADJ_PROMPT",
    "CSP_CONJUGATION_NAMESPACE",
    "CSP_CORPUS_ROOT",
    "CSP_EXCLUDED_FILES",
    "CSP_NOUN_PROMPT",
    "CSP_SUPERCLASS_PROMPT",
    "build_csp_text_features",
    "id_proximity",
    "load_csp_candidate_labels",
    "load_csp_text_features",
    "parse_excluded_files",
    "select_least_similar",
]
