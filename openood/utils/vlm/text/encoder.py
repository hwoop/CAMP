from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from openood.networks.clip import clip


@torch.no_grad()
def encode_text_list(
    model,
    texts: List[str],
    templates: List[str],
    device: str = "cuda",
    text_center: bool = True,
    batch_size: int = 256,
    normalize_prompts: bool = True,
) -> torch.Tensor:

    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("texts must be a non-empty list of strings.")

    if not isinstance(templates, list) or len(templates) == 0:
        raise ValueError("templates must be a non-empty list of strings.")

    for template in templates:
        if "{}" not in template:
            raise ValueError(
                f"Each template must contain '{{}}' placeholder: {template}"
            )

    if not text_center:
        raise NotImplementedError(
            "encode_text_list currently supports text_center=True only. "
            "For text_center=False, define the exact output shape needed by the method."
        )

    model.eval()

    text_features = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="[VLM/Text] encode",
    ):
        batch_texts = texts[start:start + batch_size]

        prompt_texts = []
        slices = []

        cursor = 0

        for name in batch_texts:
            cur_prompts = [template.format(name) for template in templates]
            prompt_texts.extend(cur_prompts)

            slices.append((cursor, cursor + len(cur_prompts)))
            cursor += len(cur_prompts)

        tokens = clip.tokenize(prompt_texts, truncate=True).to(device)

        embeddings = model.encode_text(tokens)

        if normalize_prompts:
            embeddings = F.normalize(embeddings, dim=-1)

        for s, e in slices:
            feature = embeddings[s:e].mean(dim=0)
            feature = F.normalize(feature, dim=0)
            text_features.append(feature)

    text_features = torch.stack(text_features, dim=1).to(device)

    return text_features
