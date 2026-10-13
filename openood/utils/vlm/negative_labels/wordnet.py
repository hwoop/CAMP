import os
from typing import Dict, List, Tuple

import torch

from openood.networks.clip_for_wordnet_prepare import (
    generate_cossim_idname_wordnet_dedup,
)
from openood.utils.vlm.classnames import get_class_names


def make_wordnet_cache_paths(
    wordnet_cache_root: str,
    dataset: str,
    total_ood_num: int,
    cos_percentile: float = 0.95,
) -> Tuple[str, str, str]:

    os.makedirs(wordnet_cache_root, exist_ok=True)

    wordnet_processed_path = os.path.join(
        wordnet_cache_root,
        f"wordnet_{dataset}_cossim_dedup.pth",
    )

    selected_path = os.path.join(
        wordnet_cache_root,
        f"{dataset}_selected_adj_noun_{cos_percentile}_{total_ood_num}_dedup.pth",
    )

    unselected_path = os.path.join(
        wordnet_cache_root,
        f"{dataset}_unselected_adj_noun_{cos_percentile}_{total_ood_num}_dedup.pth",
    )

    return wordnet_processed_path, selected_path, unselected_path


def load_or_build_wordnet_candidates(
    dataset: str,
    wordnet_cache_root: str = "./data/txtfiles_output",
) -> Dict:

    os.makedirs(wordnet_cache_root, exist_ok=True)

    wordnet_processed_path = os.path.join(
        wordnet_cache_root,
        f"wordnet_{dataset}_cossim_dedup.pth",
    )
    if not os.path.exists(wordnet_processed_path):
        classnames = get_class_names(dataset)
        generate_cossim_idname_wordnet_dedup(
            classnames=classnames,
            save_path=wordnet_processed_path,
        )

    return torch.load(wordnet_processed_path, map_location="cpu")


def select_wordnet_negative_labels(
    wordnet_dict: Dict,
    total_ood_num: int = 10000,
    cos_percentile: float = 0.95,
) -> Dict[str, List[str]]:

    can_list_adj = wordnet_dict["text_list_adj"]
    can_cos_adj = wordnet_dict["cos_sim_adj"]

    can_list_noun = wordnet_dict["text_list_noun"]
    can_cos_noun = wordnet_dict["cos_sim_noun"]

    if len(can_list_adj) == 0 or len(can_list_noun) == 0:
        raise ValueError("WordNet adjective/noun candidate lists must be non-empty.")

    adj_num = int(
        total_ood_num
        * (len(can_list_adj) / (len(can_list_adj) + len(can_list_noun)))
    )

    noun_num = total_ood_num - adj_num

    num_id_classes = can_cos_adj.size(1)
    cos_index = int(num_id_classes * cos_percentile)

    cos_index = min(max(cos_index, 0), num_id_classes - 1)

    value_cos_adj, _ = can_cos_adj.sort(dim=1)
    value_cos_noun, _ = can_cos_noun.sort(dim=1)

    selected_value_cos_adj = value_cos_adj[:, cos_index]
    selected_value_cos_noun = value_cos_noun[:, cos_index]

    _, indice_sim_adj = selected_value_cos_adj.sort(dim=0)
    _, indice_sim_noun = selected_value_cos_noun.sort(dim=0)

    selected_adj = [can_list_adj[i] for i in indice_sim_adj[:adj_num]]
    selected_noun = [can_list_noun[i] for i in indice_sim_noun[:noun_num]]

    unselected_adj = [can_list_adj[i] for i in indice_sim_adj[adj_num:]]
    unselected_noun = [can_list_noun[i] for i in indice_sim_noun[noun_num:]]

    return {
        "selected_adj": selected_adj,
        "selected_noun": selected_noun,
        "unselected_adj": unselected_adj,
        "unselected_noun": unselected_noun,
    }


def load_wordnet_negative_label_lists(
    dataset: str,
    total_ood_num: int = 10000,
    wordnet_cache_root: str = "./data/txtfiles_output",
    cos_percentile: float = 0.95,
) -> Dict:

    (
        _,
        selected_path,
        unselected_path,
    ) = make_wordnet_cache_paths(
        wordnet_cache_root=wordnet_cache_root,
        dataset=dataset,
        total_ood_num=total_ood_num,
        cos_percentile=cos_percentile,
    )
    if (
        os.path.exists(selected_path)
        and os.path.exists(unselected_path)
    ):
        selected = torch.load(selected_path, map_location="cpu")
        unselected = torch.load(unselected_path, map_location="cpu")

        selected_adj = selected["adj"]
        selected_noun = selected["noun"]
        unselected_adj = unselected["adj"]
        unselected_noun = unselected["noun"]

    else:
        wordnet_dict = load_or_build_wordnet_candidates(
            dataset=dataset,
            wordnet_cache_root=wordnet_cache_root,
        )

        selected_dict = select_wordnet_negative_labels(
            wordnet_dict=wordnet_dict,
            total_ood_num=total_ood_num,
            cos_percentile=cos_percentile,
        )

        selected_adj = selected_dict["selected_adj"]
        selected_noun = selected_dict["selected_noun"]
        unselected_adj = selected_dict["unselected_adj"]
        unselected_noun = selected_dict["unselected_noun"]

        torch.save(
            obj={
                "adj": selected_adj,
                "noun": selected_noun,
                "metadata": {
                    "dataset": dataset,
                    "total_ood_num": total_ood_num,
                    "cos_percentile": cos_percentile,
                    "return": "selected_labels",
                    "source": "wordnet",
                },
            },
            f=selected_path,
        )

        torch.save(
            obj={
                "adj": unselected_adj,
                "noun": unselected_noun,
                "metadata": {
                    "dataset": dataset,
                    "total_ood_num": total_ood_num,
                    "cos_percentile": cos_percentile,
                    "return": "unselected_labels",
                    "source": "wordnet",
                },
            },
            f=unselected_path,
        )

    metadata = {
        "dataset": dataset,
        "total_ood_num": total_ood_num,
        "cos_percentile": cos_percentile,
        "source": "wordnet",
        "selected_adj_num": len(selected_adj),
        "selected_noun_num": len(selected_noun),
        "unselected_adj_num": len(unselected_adj),
        "unselected_noun_num": len(unselected_noun),
        "selected_path": selected_path,
        "unselected_path": unselected_path,
    }

    return {
        "selected_adj": selected_adj,
        "selected_noun": selected_noun,
        "unselected_adj": unselected_adj,
        "unselected_noun": unselected_noun,
        "selected": selected_adj + selected_noun,
        "unselected": unselected_adj + unselected_noun,
        "metadata": metadata,
    }
