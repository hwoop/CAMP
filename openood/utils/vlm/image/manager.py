from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from openood.datasets.feature_dataset import CachedImageFeatureDataset
from openood.utils import get_config
from openood.utils.reproducibility import make_torch_generator, seed_worker

from .features import (
    ImageFeatureCache,
    ImageFeatureCacheSpec,
    build_image_encoder_identity,
    extract_image_features,
    load_image_feature_cache,
    save_image_feature_cache,
)
from .policy import ImageFeatureCachePolicy
from .source_seal import (
    ImageFeatureCacheSpecResolution,
    build_image_feature_cache_spec_with_source_seal,
)


class ImageFeatureCacheManager:
    """
    Replaces the data loaders with loaders over cached image features.
    The stream order must be the same as with the original loaders.
    """

    def __init__(
        self,
        *,
        config: Any,
        policy: ImageFeatureCachePolicy,
    ) -> None:

        self.config = config
        self.policy = policy


    def prepare(
        self,
        *,
        net: torch.nn.Module,
        id_loader_dict: Mapping[str, DataLoader],
        ood_loader_dict: Mapping[str, Any],
        postprocessor: Optional[Any] = None,
        evaluator: Optional[Any] = None,
    ) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:

        if not self.policy.enabled:
            return dict(id_loader_dict), dict(ood_loader_dict)

        if postprocessor is None or not bool(
            getattr(postprocessor, "supports_image_feature_cache", False)
        ):
            raise TypeError(
                "Managed image feature cache requires a VLM postprocessor with "
                "an explicit cached-feature input contract."
            )

        evaluator_supports_cache = (
            False
            if evaluator is None
            else bool(
                evaluator.__class__.__dict__.get(
                    "supports_image_feature_cache",
                    False,
                )
            )
        )
        if not evaluator_supports_cache:
            raise TypeError(
                "Managed image feature cache requires an evaluator with an "
                "explicit cached-feature input contract."
            )

        encoder_identity = build_image_encoder_identity(
            config=self.config,
            net=net,
            encoder_revision=self.policy.encoder_revision,
        )

        required_id_splits, required_ood_splits = self._required_splits(evaluator)
        prepared_id: Dict[str, DataLoader] = dict(id_loader_dict)

        for split, loader in id_loader_dict.items():
            if str(split).lower() not in required_id_splits:
                continue

            prepared_id[split] = self._prepare_loader(
                net=net,
                loader=loader,
                source_dataset=str(self.config.dataset.name),
                source_split=str(split),
                encoder_identity=encoder_identity,
                is_ood=False,
            )

        prepared_ood: Dict[str, Any] = dict(ood_loader_dict)

        for split, loaders in ood_loader_dict.items():
            if str(split).lower() not in required_ood_splits:
                continue

            prepared_ood[split] = {
                source_dataset: self._prepare_loader(
                    net=net,
                    loader=loader,
                    source_dataset=str(source_dataset),
                    source_split=str(split),
                    encoder_identity=encoder_identity,
                    is_ood=True,
                )
                for source_dataset, loader in loaders.items()
            }

        return prepared_id, prepared_ood


    def _required_splits(
        self,
        evaluator: Any,
    ) -> Tuple[set[str], set[str]]:

        id_splits = set(
            getattr(
                evaluator,
                "image_feature_cache_required_id_splits",
                {"test"},
            )
        )

        ood_splits = set(
            getattr(
                evaluator,
                "image_feature_cache_required_ood_splits",
                {"nearood", "farood"},
            )
        )

        return id_splits, ood_splits


    def _prepare_loader(
        self,
        *,
        net: torch.nn.Module,
        loader: DataLoader,
        source_dataset: str,
        source_split: str,
        encoder_identity: Mapping[str, Any],
        is_ood: bool,
    ) -> DataLoader:

        resolution = self._build_spec(
            net=net,
            loader=loader,
            source_dataset=source_dataset,
            source_split=source_split,
            encoder_identity=encoder_identity,
            refresh_source_seal=False,
        )

        spec = resolution.spec
        refresh_spec: Optional[Callable[[], ImageFeatureCacheSpec]] = None

        if resolution.source_seal_hit:
            # Refresh the seal before any rebuild so new features are never written under a stale source digest.
            refresh_spec = lambda: self._build_spec(
                net=net,
                loader=loader,
                source_dataset=source_dataset,
                source_split=source_split,
                encoder_identity=encoder_identity,
                refresh_source_seal=True,
            ).spec

        cached = self._load_or_build(
            net=net,
            source_loader=loader,
            spec=spec,
            refresh_spec=refresh_spec,
        )

        dataset = CachedImageFeatureDataset(
            image_features=cached.image_features,
            labels=cached.labels,
            source_indices=cached.source_indices,
            source_dataset=source_dataset,
            source_split=source_split,
            is_ood=is_ood,
        )

        batch_size = loader.batch_size
        if batch_size is None:
            raise ValueError("Managed image cache requires a DataLoader batch_size.")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=loader.sampler,
            num_workers=0,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            generator=loader.generator,
        )


    def _build_spec(
        self,
        *,
        net: torch.nn.Module,
        loader: DataLoader,
        source_dataset: str,
        source_split: str,
        encoder_identity: Mapping[str, Any],
        refresh_source_seal: bool,
    ) -> ImageFeatureCacheSpecResolution:

        return build_image_feature_cache_spec_with_source_seal(
            config=self.config,
            net=net,
            dataset=loader.dataset,
            source_dataset=source_dataset,
            source_split=source_split,
            cache_root=self.policy.cache_root,
            extraction_batch_size=self._extraction_batch_size(loader),
            source_validation=self.policy.source_validation,
            dataset_revision=self.policy.dataset_revision,
            encoder_identity=encoder_identity,
            refresh_source_seal=refresh_source_seal,
        )


    def _load_or_build(
        self,
        *,
        net: torch.nn.Module,
        source_loader: DataLoader,
        spec: ImageFeatureCacheSpec,
        refresh_spec: Optional[Callable[[], ImageFeatureCacheSpec]] = None,
    ) -> ImageFeatureCache:

        if os.path.exists(spec.cache_path):
            try:
                cached = self._load_from_disk(spec)
                print(f"[ImageFeatures] Loaded cache: {spec.cache_path}")
                return cached
            except (
                EOFError,
                OSError,
                pickle.PickleError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                print(
                    "[ImageFeatures] Rebuilding invalid cache: "
                    f"{spec.cache_path} ({error})"
                )

        if refresh_spec is not None:
            refreshed_spec = refresh_spec()
            if refreshed_spec.cache_path != spec.cache_path:
                return self._load_or_build(
                    net=net,
                    source_loader=source_loader,
                    spec=refreshed_spec,
                )

            spec = refreshed_spec

        print(f"[ImageFeatures] Building cache: {spec.cache_path}")
        extraction_loader = self._canonical_loader(source_loader)
        image_features, labels, source_indices = extract_image_features(
            net=net,
            data_loader=extraction_loader,
        )

        save_image_feature_cache(
            spec=spec,
            image_features=image_features,
            labels=labels,
            source_indices=source_indices,
        )

        print(f"[ImageFeatures] Saved cache: {spec.cache_path}")
        return self._load_from_disk(spec)


    def _load_from_disk(
        self,
        spec: ImageFeatureCacheSpec,
    ) -> ImageFeatureCache:

        return load_image_feature_cache(
            spec=spec,
            load_validation=self.policy.load_validation,
        )


    def _extraction_batch_size(
        self,
        loader: DataLoader,
    ) -> int:

        batch_size = self.policy.extraction_batch_size or loader.batch_size
        if batch_size is None:
            raise ValueError("Managed image cache requires a DataLoader batch_size.")

        return int(batch_size)


    def _canonical_loader(
        self,
        source_loader: DataLoader,
    ) -> DataLoader:

        batch_size = self._extraction_batch_size(source_loader)
        seed = int(get_config(obj=self.config, name="seed", default=0))
        return DataLoader(
            source_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=source_loader.num_workers,
            drop_last=False,
            collate_fn=source_loader.collate_fn,
            pin_memory=source_loader.pin_memory,
            generator=make_torch_generator(seed),
            worker_init_fn=seed_worker,
        )


__all__ = ["ImageFeatureCacheManager", "ImageFeatureCachePolicy"]
