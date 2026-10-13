from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openood.utils import get_config


IMAGE_CACHE_MODES = {"disabled", "read_write"}
IMAGE_CACHE_LOAD_VALIDATION_MODES = {"fast", "strict"}
IMAGE_SOURCE_VALIDATION_MODES = {"auto", "revision", "sha256", "stat"}


@dataclass(frozen=True)
class ImageFeatureCachePolicy:
    """Settings of the ``cache.image`` block of the config."""

    mode: str = "disabled"
    cache_root: str = "./.cache/"
    source_validation: str = "auto"
    dataset_revision: str | None = None
    load_validation: str = "strict"
    encoder_revision: str | None = None
    extraction_batch_size: int | None = None

    def __post_init__(self) -> None:

        if self.extraction_batch_size is not None and int(self.extraction_batch_size) < 1:
            raise ValueError(
                f"cache.image.extraction_batch_size must be a positive integer, got {self.extraction_batch_size!r}."
            )

        if self.mode not in IMAGE_CACHE_MODES:
            raise ValueError(
                f"Unsupported image cache mode {self.mode!r}; "
                f"expected one of {sorted(IMAGE_CACHE_MODES)}."
            )

        if self.load_validation not in IMAGE_CACHE_LOAD_VALIDATION_MODES:
            raise ValueError(
                f"Unsupported image cache load validation {self.load_validation!r}; "
                f"expected one of {sorted(IMAGE_CACHE_LOAD_VALIDATION_MODES)}."
            )

        if self.source_validation not in IMAGE_SOURCE_VALIDATION_MODES:
            raise ValueError(
                f"Unsupported image source validation {self.source_validation!r}; "
                f"expected one of {sorted(IMAGE_SOURCE_VALIDATION_MODES)}."
            )

        if (
            self.enabled and
            self.source_validation == "revision" and
            not self.dataset_revision
        ):
            raise ValueError(
                "cache.image.dataset_revision is required when "
                "cache.image.source_validation=revision."
            )

        if self.encoder_revision is not None and not self.encoder_revision.strip():
            raise ValueError("cache.image.encoder_revision cannot be empty.")


    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"


    @classmethod
    def from_config(
        cls,
        config: Any,
    ) -> "ImageFeatureCachePolicy":

        cache_config = get_config(obj=config, name="cache", default=None)
        image_config = get_config(obj=cache_config, name="image", default=None)
        raw_source_validation = get_config(obj=image_config, name="source_validation", default="auto")
        source_validation = (
            str(raw_source_validation).lower()
            if raw_source_validation is not None
            else "auto"
        )
        encoder_revision = get_config(obj=image_config, name="encoder_revision", default=None)
        raw_extraction = get_config(obj=image_config, name="extraction_batch_size", default=None)
        # null, 0 or an empty string mean "the stream loader's batch size" (the behaviour before the option existed)
        extraction_batch_size = (
            None
            if raw_extraction is None or str(raw_extraction).strip() in ("", "0", "null", "None")
            else int(str(raw_extraction).strip())
        )
        return cls(
            extraction_batch_size=extraction_batch_size,
            mode=str(get_config(obj=image_config, name="mode", default="disabled")).lower(),
            cache_root=str(get_config(obj=cache_config, name="root", default="./.cache/")),
            load_validation=str(
                get_config(obj=image_config, name="load_validation", default="strict")
            ).lower(),
            source_validation=source_validation,
            dataset_revision=(
                None
                if get_config(obj=image_config, name="dataset_revision", default=None) is None
                else str(get_config(obj=image_config, name="dataset_revision", default=None))
            ),
            encoder_revision=(
                None
                if encoder_revision is None or not str(encoder_revision).strip()
                else str(encoder_revision).strip()
            ),
        )


__all__ = [
    "IMAGE_CACHE_LOAD_VALIDATION_MODES",
    "IMAGE_CACHE_MODES",
    "IMAGE_SOURCE_VALIDATION_MODES",
    "ImageFeatureCachePolicy",
]
