from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

from openood.utils.vlm.cache import (
    load_cache,
    make_cache_path,
    safe_cache_value,
    save_cache,
)

from .features import (
    ImageFeatureCacheSpec,
    _canonical_json,
    _is_sha256,
    _json_value,
    _stable_digest,
    build_image_feature_cache_spec,
)


IMAGE_SOURCE_SEAL_SCHEMA_VERSION = 1
IMAGE_SOURCE_SEAL_SENTINEL_COUNT = 64
_AUTO_SEAL_REVISION = "managed-auto-source-seal-v1"


@dataclass(frozen=True)
class SourceIdentitySeal:
    ordered_source_identity_sha256: str
    cache_hit: bool


@dataclass(frozen=True)
class ImageFeatureCacheSpecResolution:
    spec: ImageFeatureCacheSpec
    source_seal_hit: bool = False


def build_image_feature_cache_spec_with_source_seal(
    *,
    config: Any,
    net: Any,
    dataset: Any,
    source_dataset: str,
    source_split: str,
    cache_root: str = "./.cache/",
    extraction_batch_size: int | None = None,
    source_validation: str = "stat",
    dataset_revision: str | None = None,
    encoder_identity: Mapping[str, Any] | None = None,
    encoder_revision: str | None = None,
    refresh_source_seal: bool = False,
) -> ImageFeatureCacheSpecResolution:

    if source_validation != "auto":
        return ImageFeatureCacheSpecResolution(
            spec=build_image_feature_cache_spec(
                config=config,
                net=net,
                dataset=dataset,
                source_dataset=source_dataset,
                source_split=source_split,
                cache_root=cache_root,
                extraction_batch_size=extraction_batch_size,
                source_validation=source_validation,
                dataset_revision=dataset_revision,
                encoder_identity=encoder_identity,
                encoder_revision=encoder_revision,
            )
        )

    # Revision mode yields every identity field except the costly per-image stat digest; its path is discarded.
    provisional = build_image_feature_cache_spec(
        config=config,
        net=net,
        dataset=dataset,
        source_dataset=source_dataset,
        source_split=source_split,
        cache_root=cache_root,
        extraction_batch_size=extraction_batch_size,
        source_validation="revision",
        dataset_revision=_AUTO_SEAL_REVISION,
        encoder_identity=encoder_identity,
        encoder_revision=encoder_revision,
    )

    provisional_dataset = dict(provisional.identity["dataset"])
    source_descriptor = {
        "schema_version": IMAGE_SOURCE_SEAL_SCHEMA_VERSION,
        "source_dataset": str(source_dataset),
        "source_split": str(source_split),
        "dataset": {
            key: value
            for key, value in provisional_dataset.items()
            if key
            not in {
                "ordered_source_identity_sha256",
                "source_validation",
                "dataset_revision",
            }
        },
    }
    sample_count = int(len(dataset))
    ordered_image_names = [
        str(line).rstrip("\n").split(" ", 1)[0]
        for line in dataset.imglist[:sample_count]
    ]
    data_root = Path(str(getattr(dataset, "data_dir", ""))).expanduser().resolve()
    source_seal = resolve_auto_source_identity(
        cache_root=cache_root,
        source_dataset=source_dataset,
        source_split=source_split,
        source_descriptor=source_descriptor,
        ordered_image_names=ordered_image_names,
        data_root=data_root,
        refresh=refresh_source_seal,
    )

    dataset_identity = dict(provisional_dataset)
    dataset_identity.update(
        {
            "ordered_source_identity_sha256": (
                source_seal.ordered_source_identity_sha256
            ),
            "source_validation": "auto",
            "dataset_revision": dataset_revision,
        }
    )

    identity = dict(provisional.identity)
    identity["dataset"] = dataset_identity
    identity_digest = _stable_digest(identity)
    cache_path = make_cache_path(
        dataset=str(config.dataset.name),
        model_name=str(config.network.name),
        backbone=str(config.network.backbone.name),
        cache_group="image",
        cache_root=cache_root,
        params={
            "source": source_dataset,
            "split": source_split,
            "data": _stable_digest(dataset_identity)[:12],
            "preprocess": _stable_digest(identity["preprocess"])[:12],
            "encoder": _stable_digest(identity["encoder"])[:12],
            "identity": identity_digest,
            "normalized": True,
            "return": "features",
        },
    )

    return ImageFeatureCacheSpecResolution(
        spec=ImageFeatureCacheSpec(cache_path=cache_path, identity=identity),
        source_seal_hit=source_seal.cache_hit,
    )


def resolve_auto_source_identity(
    *,
    cache_root: str,
    source_dataset: str,
    source_split: str,
    source_descriptor: Mapping[str, Any],
    ordered_image_names: Sequence[str],
    data_root: Path,
    refresh: bool = False,
) -> SourceIdentitySeal:

    descriptor = _json_value(dict(source_descriptor))
    descriptor_sha256 = _stable_digest(descriptor)
    seal_path = _source_seal_path(
        cache_root=cache_root,
        source_dataset=source_dataset,
        source_split=source_split,
        descriptor_sha256=descriptor_sha256,
    )

    if not refresh:
        cached_digest = _load_source_seal(
            seal_path=seal_path,
            descriptor=descriptor,
            descriptor_sha256=descriptor_sha256,
            ordered_image_names=ordered_image_names,
            data_root=data_root,
        )
        if cached_digest is not None:
            return SourceIdentitySeal(
                ordered_source_identity_sha256=cached_digest,
                cache_hit=True,
            )

    source_digest, sentinels = _stat_source_identity(
        ordered_image_names=ordered_image_names,
        data_root=data_root,
        descriptor_sha256=descriptor_sha256,
    )

    save_cache(
        cache_path=str(seal_path),
        cache_obj={
            "schema_version": IMAGE_SOURCE_SEAL_SCHEMA_VERSION,
            "descriptor": descriptor,
            "descriptor_sha256": descriptor_sha256,
            "source_validation": "stat",
            "ordered_source_identity_sha256": source_digest,
            "sentinels": sentinels,
        },
    )

    return SourceIdentitySeal(
        ordered_source_identity_sha256=source_digest,
        cache_hit=False,
    )


def _source_seal_path(
    *,
    cache_root: str,
    source_dataset: str,
    source_split: str,
    descriptor_sha256: str,
) -> Path:

    filename = (
        f"source={safe_cache_value(source_dataset)}_"
        f"split={safe_cache_value(source_split)}_"
        f"descriptor={descriptor_sha256}.seal"
    )
    return Path(cache_root).expanduser() / ".source_seals" / filename


def _load_source_seal(
    *,
    seal_path: Path,
    descriptor: Mapping[str, Any],
    descriptor_sha256: str,
    ordered_image_names: Sequence[str],
    data_root: Path,
) -> str | None:

    if not seal_path.is_file():
        return None

    try:
        payload = load_cache(str(seal_path), weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("Image source seal payload must be a dictionary.")

        if payload.get("schema_version") != IMAGE_SOURCE_SEAL_SCHEMA_VERSION:
            raise ValueError("Image source seal schema version is incompatible.")

        if payload.get("descriptor_sha256") != descriptor_sha256:
            raise ValueError("Image source seal descriptor fingerprint is inconsistent.")

        if _canonical_json(payload.get("descriptor")) != _canonical_json(descriptor):
            raise ValueError("Image source seal descriptor is inconsistent.")

        if payload.get("source_validation") != "stat":
            raise ValueError("Image source seal validation method is inconsistent.")

        source_digest = payload.get("ordered_source_identity_sha256")
        if not _is_sha256(source_digest):
            raise ValueError("Image source seal digest is invalid.")

        _validate_sentinels(
            payload.get("sentinels"),
            ordered_image_names=ordered_image_names,
            data_root=data_root,
        )

        return str(source_digest)
    except (
        EOFError,
        OSError,
        pickle.PickleError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


def _stat_source_identity(
    *,
    ordered_image_names: Sequence[str],
    data_root: Path,
    descriptor_sha256: str,
) -> tuple[str, list[dict[str, Any]]]:

    sentinel_indices = _sentinel_indices(
        sample_count=len(ordered_image_names),
        descriptor_sha256=descriptor_sha256,
    )

    sentinels: list[dict[str, Any]] = []
    digest = hashlib.sha256()

    for index, image_name in enumerate(ordered_image_names):
        source_record = _source_record(
            index=index,
            image_name=image_name,
            data_root=data_root,
        )

        source_encoded = _canonical_json(
            {key: value for key, value in source_record.items() if key != "index"}
        ).encode("utf-8")

        digest.update(len(source_encoded).to_bytes(8, byteorder="big"))
        digest.update(source_encoded)

        if index in sentinel_indices:
            sentinels.append(source_record)

    return digest.hexdigest(), sentinels


def _validate_sentinels(
    value: Any,
    *,
    ordered_image_names: Sequence[str],
    data_root: Path,
) -> None:

    if not isinstance(value, list) or not value:
        raise ValueError("Image source seal sentinels are missing.")

    for expected in value:
        if not isinstance(expected, dict):
            raise ValueError("Image source seal sentinel must be a dictionary.")

        index = expected.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Image source seal sentinel index is invalid.")

        if index < 0 or index >= len(ordered_image_names):
            raise ValueError("Image source seal sentinel index is out of range.")

        image_name = ordered_image_names[index]
        actual = _source_record(
            index=index,
            image_name=image_name,
            data_root=data_root,
        )
        if _canonical_json(actual) != _canonical_json(expected):
            raise ValueError("Image source seal sentinel changed.")


def _source_record(
    *,
    index: int,
    image_name: str,
    data_root: Path,
) -> dict[str, Any]:

    source_stat = (data_root / image_name).stat()
    return {
        "index": int(index),
        "image_name": image_name,
        "size": int(source_stat.st_size),
        "mtime_ns": int(source_stat.st_mtime_ns),
        "ctime_ns": int(source_stat.st_ctime_ns),
    }


def _sentinel_indices(
    *,
    sample_count: int,
    descriptor_sha256: str,
) -> set[int]:

    if sample_count <= 0:
        raise ValueError("Image source seal cannot describe an empty dataset.")

    if sample_count <= IMAGE_SOURCE_SEAL_SENTINEL_COUNT:
        return set(range(sample_count))

    indices = {0, sample_count // 2, sample_count - 1}
    nonce = 0

    while len(indices) < IMAGE_SOURCE_SEAL_SENTINEL_COUNT:
        candidate = int(
            hashlib.sha256(
                f"{descriptor_sha256}:{nonce}".encode("utf-8")
            ).hexdigest(),
            16,
        ) % sample_count

        indices.add(candidate)
        nonce += 1

    return indices


__all__ = [
    "ImageFeatureCacheSpecResolution",
    "build_image_feature_cache_spec_with_source_seal",
]
