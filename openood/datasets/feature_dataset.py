from typing import Optional

import torch
from torch.utils.data import Dataset


class CachedImageFeatureDataset(Dataset):

    def __init__(
        self,
        *,
        image_features: torch.Tensor,
        labels: torch.Tensor,
        source_indices: torch.Tensor,
        source_dataset: str = "",
        source_split: str = "",
        is_ood: Optional[bool] = None,
    ) -> None:

        if image_features.ndim != 2:
            raise ValueError("image_features must have shape [sample, feature_dim].")

        self.data = image_features.detach().cpu()
        self.labels = labels.detach().cpu().reshape(-1)
        self.source_indices = source_indices.detach().cpu().reshape(-1)
        self.source_dataset = str(source_dataset)
        self.source_split = str(source_split)
        self.is_ood = is_ood
        self.len = int(self.data.shape[0])
        if self.len != int(self.labels.numel()) or self.len != int(self.source_indices.numel()):
            raise ValueError("Cached feature, label, and source-index lengths must match.")

        if self.source_indices.dtype != torch.long:
            raise TypeError("Cached source indices must use torch.long dtype.")


    def __len__(self):
        return self.len


    def __getitem__(
        self,
        idx,
    ):

        label = self.labels[idx]
        source_index = int(self.source_indices[idx])
        return {
            'data': self.data[idx],
            'label': label,
            'index': source_index,
            'source_index': source_index,
            'source_dataset': self.source_dataset,
            'source_split': self.source_split,
            'is_ood': int(label.item() < 0) if self.is_ood is None else int(self.is_ood),
            'image_features_cached': True,
            'image_features_normalized': True,
        }
