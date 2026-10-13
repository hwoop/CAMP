from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
import platform
from pathlib import Path
from typing import Any, Dict, Mapping, MutableSet, Tuple

import PIL
import torch
import torchvision
from torch.utils.data import DataLoader

from openood.networks.clip import clip
from openood.utils.vlm.cache import load_cache, make_cache_path, save_cache

from .policy import IMAGE_CACHE_LOAD_VALIDATION_MODES


IMAGE_FEATURE_CACHE_SCHEMA_VERSION = 1
IMAGE_FEATURE_ENCODER_CONTRACT = "encode_image(normalize=True):normalized_features:v1"


@dataclass(frozen=True)
class ImageFeatureCacheSpec:
    cache_path: str
    identity: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return _stable_digest(self.identity)


@dataclass(frozen=True)
class ImageFeatureCache:
    image_features: torch.Tensor
    labels: torch.Tensor
    source_indices: torch.Tensor


def build_image_feature_cache_spec(
    *,
    config: Any,
    net: torch.nn.Module,
    dataset: Any,
    source_dataset: str,
    source_split: str,
    cache_root: str = "./.cache/",
    extraction_batch_size: int | None = None,
    source_validation: str = "stat",
    dataset_revision: str | None = None,
    encoder_identity: Mapping[str, Any] | None = None,
    encoder_revision: str | None = None,
) -> ImageFeatureCacheSpec:

    if not hasattr(dataset, "imglist"):
        raise TypeError(
            "Managed image feature cache currently requires a dataset with an "
            "ordered 'imglist' attribute."
        )

    if int(len(dataset)) <= 0:
        raise ValueError("Cannot build an image feature cache for an empty dataset.")

    if str(source_split).lower() == "train":
        raise ValueError(
            "Managed image feature cache only supports deterministic evaluation splits."
        )

    _validate_deterministic_preprocessor(dataset)

    benchmark_dataset = str(config.dataset.name)
    model_name = str(config.network.name)
    backbone = str(config.network.backbone.name)
    dataset_identity = _dataset_identity(
        dataset,
        source_validation=source_validation,
        dataset_revision=dataset_revision,
    )

    preprocess_identity = _preprocess_identity(dataset)
    resolved_encoder_identity = (
        dict(encoder_identity)
        if encoder_identity is not None
        else build_image_encoder_identity(
            config=config,
            net=net,
            encoder_revision=encoder_revision,
        )
    )

    identity: Dict[str, Any] = {
        "schema_version": IMAGE_FEATURE_CACHE_SCHEMA_VERSION,
        "cache_type": "image",
        "benchmark_dataset": benchmark_dataset,
        "source_dataset": str(source_dataset),
        "source_split": str(source_split),
        "dataset": dataset_identity,
        "preprocess": preprocess_identity,
        "encoder": resolved_encoder_identity,
        "encoder_contract": IMAGE_FEATURE_ENCODER_CONTRACT,
        "extraction_batch_size": _json_value(extraction_batch_size),
        "normalized": True,
        "storage_dtype": "preserve",
    }

    identity_digest = _stable_digest(identity)
    cache_path = make_cache_path(
        dataset=benchmark_dataset,
        model_name=model_name,
        backbone=backbone,
        cache_group="image",
        cache_root=cache_root,
        params={
            "source": source_dataset,
            "split": source_split,
            "data": _stable_digest(dataset_identity)[:12],
            "preprocess": _stable_digest(preprocess_identity)[:12],
            "encoder": _stable_digest(resolved_encoder_identity)[:12],
            "identity": identity_digest,
            "normalized": True,
            "return": "features",
        },
    )

    return ImageFeatureCacheSpec(cache_path=cache_path, identity=identity)


@torch.no_grad()
def extract_image_features(
    *,
    net: torch.nn.Module,
    data_loader: DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if not hasattr(net, "encode_image"):
        raise TypeError(
            "Managed image feature cache requires a network with "
            "encode_image(data, normalize=True)."
        )

    device = _module_device(net)
    previous_training = [(module, bool(module.training)) for module in net.modules()]
    net.eval()
    feature_batches = []
    label_batches = []
    index_batches = []

    try:
        for batch in data_loader:
            images = batch["data"].to(device)
            labels = torch.as_tensor(batch["label"]).reshape(-1).detach().cpu()
            features = net.encode_image(images, normalize=True)
            if not torch.is_tensor(features) or features.ndim != 2:
                shape = tuple(features.shape) if torch.is_tensor(features) else type(features)
                raise ValueError(
                    "encode_image(normalize=True) must return [batch, feature_dim], "
                    f"got {shape}."
                )

            if int(features.shape[0]) != int(labels.numel()):
                raise ValueError("Image feature and label batch sizes do not match.")

            source_indices = torch.as_tensor(batch["index"]).reshape(-1)
            if source_indices.dtype != torch.long:
                raise TypeError("Dataset source indices must use torch.long dtype.")

            source_indices = source_indices.detach().cpu()

            feature_batches.append(features.detach().cpu())
            label_batches.append(labels)
            index_batches.append(source_indices)
    finally:
        for module, training in previous_training:
            module.training = training

    if not feature_batches:
        raise ValueError("Image feature extraction produced no batches.")

    image_features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    source_indices = torch.cat(index_batches, dim=0)
    order = torch.argsort(source_indices)
    image_features = image_features[order]
    labels = labels[order]
    source_indices = source_indices[order]

    expected_indices = torch.arange(len(data_loader.dataset), dtype=torch.long)
    if not torch.equal(source_indices, expected_indices):
        raise ValueError(
            "Image feature extraction must cover each canonical dataset index exactly once."
        )

    if not bool(torch.isfinite(image_features).all()):
        raise ValueError("Image feature cache cannot contain non-finite values.")

    return image_features, labels, source_indices


def save_image_feature_cache(
    *,
    spec: ImageFeatureCacheSpec,
    image_features: torch.Tensor,
    labels: torch.Tensor,
    source_indices: torch.Tensor,
) -> None:

    image_features, labels, source_indices = _validate_payload_tensors(
        image_features=image_features,
        labels=labels,
        source_indices=source_indices,
        expected_count=int(spec.identity["dataset"]["sample_count"]),
    )

    metadata = {
        "identity": dict(spec.identity),
        "fingerprint": spec.fingerprint,
        "payload_sha256": _payload_sha256(
            image_features=image_features,
            labels=labels,
            source_indices=source_indices,
        ),
        "sample_count": int(image_features.shape[0]),
        "feature_dim": int(image_features.shape[1]),
        "feature_dtype": str(image_features.dtype),
        "label_dtype": str(labels.dtype),
        "normalized": True,
        "cache_path": spec.cache_path,
    }

    save_cache(
        cache_path=spec.cache_path,
        cache_obj={
            "metadata": metadata,
            "image_features": image_features,
            "labels": labels,
            "source_indices": source_indices,
        },
    )


def load_image_feature_cache(
    *,
    spec: ImageFeatureCacheSpec,
    load_validation: str = "strict",
) -> ImageFeatureCache:

    if load_validation not in IMAGE_CACHE_LOAD_VALIDATION_MODES:
        raise ValueError(
            f"Unsupported image cache load validation: {load_validation!r}."
        )

    fast_load = load_validation == "fast"
    cache = load_cache(
        spec.cache_path,
        device=None,
        weights_only=True,
        mmap=fast_load,
    )
    if not isinstance(cache, dict):
        raise ValueError("Image feature cache payload must be a dictionary.")

    required = {"metadata", "image_features", "labels", "source_indices"}
    missing = sorted(required.difference(cache.keys()))
    if missing:
        raise ValueError(f"Image feature cache is missing fields: {missing}")

    metadata = cache["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Image feature cache metadata must be a dictionary.")

    if metadata.get("fingerprint") != spec.fingerprint:
        raise ValueError("Image feature cache fingerprint does not match the requested source.")

    if _canonical_json(metadata.get("identity")) != _canonical_json(spec.identity):
        raise ValueError("Image feature cache identity does not match the requested source.")

    image_features, labels, source_indices = _validate_payload_tensors(
        image_features=cache["image_features"],
        labels=cache["labels"],
        source_indices=cache["source_indices"],
        expected_count=int(spec.identity["dataset"]["sample_count"]),
        validate_values=not fast_load,
    )

    if fast_load:
        payload_sha256 = metadata.get("payload_sha256")
        if not _is_sha256(payload_sha256):
            raise ValueError("Image feature cache payload digest metadata is invalid.")
    else:
        payload_sha256 = _payload_sha256(
            image_features=image_features,
            labels=labels,
            source_indices=source_indices,
        )
        if metadata.get("payload_sha256") != payload_sha256:
            raise ValueError("Image feature cache payload digest is inconsistent.")

    if metadata.get("sample_count") != int(image_features.shape[0]):
        raise ValueError("Image feature cache sample_count metadata is inconsistent.")

    if metadata.get("feature_dim") != int(image_features.shape[1]):
        raise ValueError("Image feature cache feature_dim metadata is inconsistent.")

    if metadata.get("feature_dtype") != str(image_features.dtype):
        raise ValueError("Image feature cache feature_dtype metadata is inconsistent.")

    if metadata.get("label_dtype") != str(labels.dtype):
        raise ValueError("Image feature cache label_dtype metadata is inconsistent.")

    if metadata.get("normalized") is not True:
        raise ValueError("Image feature cache normalized metadata is inconsistent.")

    return ImageFeatureCache(
        image_features=image_features,
        labels=labels,
        source_indices=source_indices,
    )


def _validate_payload_tensors(
    *,
    image_features: Any,
    labels: Any,
    source_indices: Any,
    expected_count: int,
    validate_values: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if not all(torch.is_tensor(value) for value in (image_features, labels, source_indices)):
        raise TypeError("Image features, labels, and source indices must be tensors.")

    image_features = image_features.detach().cpu()
    labels = labels.detach().cpu().reshape(-1)
    source_indices = source_indices.detach().cpu().reshape(-1)

    if image_features.ndim != 2:
        raise ValueError("Cached image features must have shape [sample, feature_dim].")

    if not torch.is_floating_point(image_features):
        raise TypeError("Cached image features must use a floating-point dtype.")

    if labels.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("Cached image labels must use an integral dtype.")

    if source_indices.dtype != torch.long:
        raise TypeError("Cached source_indices must use torch.long dtype.")

    if int(image_features.shape[0]) != expected_count:
        raise ValueError(
            f"Image feature cache row count mismatch: expected {expected_count}, "
            f"got {int(image_features.shape[0])}."
        )

    if int(labels.numel()) != expected_count or int(source_indices.numel()) != expected_count:
        raise ValueError("Image feature cache labels/source_indices have invalid lengths.")

    if validate_values:
        if not torch.equal(
            source_indices,
            torch.arange(expected_count, dtype=torch.long),
        ):
            raise ValueError(
                "Image feature cache source_indices must be canonical 0..N-1."
            )

        if not bool(torch.isfinite(image_features).all()):
            raise ValueError("Image feature cache contains non-finite values.")

        norm_tolerance = (
            2e-2
            if image_features.dtype == torch.bfloat16
            else 5e-3
            if image_features.dtype == torch.float16
            else 1e-4
        )
        feature_norms = image_features.float().norm(p=2, dim=1)
        if not bool(
            torch.allclose(
                feature_norms,
                torch.ones_like(feature_norms),
                rtol=norm_tolerance,
                atol=norm_tolerance,
            )
        ):
            raise ValueError(
                "Image feature cache requires normalized encode_image outputs."
            )

    return image_features, labels, source_indices


def _payload_sha256(
    *,
    image_features: torch.Tensor,
    labels: torch.Tensor,
    source_indices: torch.Tensor,
) -> str:

    digest = hashlib.sha256()

    for name, value in (
        ("image_features", image_features),
        ("labels", labels),
        ("source_indices", source_indices),
    ):
        tensor = value.detach().contiguous().cpu()
        metadata = _canonical_json(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        ).encode("utf-8")

        digest.update(len(metadata).to_bytes(8, byteorder="big"))
        digest.update(metadata)
        byte_view = tensor.reshape(-1).view(torch.uint8)
        digest.update(memoryview(byte_view.numpy()))

    return digest.hexdigest()


def _dataset_identity(
    dataset: Any,
    *,
    source_validation: str,
    dataset_revision: str | None,
) -> Dict[str, Any]:

    sample_count = int(len(dataset))
    ordered_lines = list(dataset.imglist[:sample_count])
    imglist_digest = hashlib.sha256()

    if source_validation not in {"revision", "sha256", "stat", "auto"}:
        raise ValueError(f"Unsupported image source validation: {source_validation}.")

    if source_validation == "revision" and not dataset_revision:
        raise ValueError(
            "dataset_revision is required for revision-based image source validation."
        )

    resolved_validation = (
        "stat" if source_validation == "auto" else source_validation
    )

    source_file_digest = hashlib.sha256()
    revision_record_prefix = None

    if source_validation == "revision":
        revision_record_prefix = (
            b'{"dataset_revision":'
            + _canonical_json(dataset_revision).encode("utf-8")
            + b',"image_name":'
        )

    data_root = Path(str(getattr(dataset, "data_dir", ""))).expanduser().resolve()

    for line in ordered_lines:
        line_value = str(line).rstrip("\n")
        encoded = line_value.encode("utf-8")
        imglist_digest.update(len(encoded).to_bytes(8, byteorder="big"))
        imglist_digest.update(encoded)

        image_name = line_value.split(" ", 1)[0]

        if resolved_validation == "stat":
            image_path = data_root / image_name
            source_stat = image_path.stat()
            source_record = {
                "image_name": image_name,
                "size": int(source_stat.st_size),
                "mtime_ns": int(source_stat.st_mtime_ns),
                "ctime_ns": int(source_stat.st_ctime_ns),
            }
            source_encoded = _canonical_json(source_record).encode("utf-8")
        elif resolved_validation == "sha256":
            image_path = data_root / image_name
            source_encoded = _canonical_json(
                {
                    "image_name": image_name,
                    "sha256": _file_sha256(image_path),
                }
            ).encode("utf-8")
        else:
            if revision_record_prefix is None:
                raise RuntimeError("Revision source identity is not initialized.")
            source_encoded = (
                revision_record_prefix
                + json.encoder.encode_basestring_ascii(image_name).encode("utf-8")
                + b"}"
            )

        source_file_digest.update(len(source_encoded).to_bytes(8, byteorder="big"))
        source_file_digest.update(source_encoded)

    data_dir = str(data_root)
    imglist_digest.update(data_dir.encode("utf-8"))
    return {
        "class": _qualified_name(dataset),
        "class_source_sha256": _source_sha256(dataset.__class__),
        "ordered_imglist_sha256": imglist_digest.hexdigest(),
        "ordered_source_identity_sha256": source_file_digest.hexdigest(),
        "source_validation": source_validation,
        "dataset_revision": _json_value(dataset_revision),
        "data_dir": data_dir,
        "sample_count": sample_count,
        "maxlen": _json_value(getattr(dataset, "maxlen", None)),
        "few_shot": _json_value(getattr(dataset, "few_shot", 0)),
    }


def _preprocess_identity(
    dataset: Any,
) -> Dict[str, Any]:

    preprocessor = getattr(dataset, "preprocessor", None)
    if preprocessor is None:
        raise ValueError("Image feature cache requires dataset.preprocessor metadata.")

    interpolation = getattr(preprocessor, "interpolation", None)
    interpolation_value = getattr(interpolation, "value", interpolation)
    return {
        "class": _qualified_name(preprocessor),
        "class_source_sha256": _source_sha256(preprocessor.__class__),
        "pre_size": _json_value(getattr(preprocessor, "pre_size", None)),
        "image_size": _json_value(getattr(preprocessor, "image_size", None)),
        "interpolation": _json_value(interpolation_value),
        "mean": _json_value(getattr(preprocessor, "mean", None)),
        "std": _json_value(getattr(preprocessor, "std", None)),
        "transform_graph": _object_identity(getattr(preprocessor, "transform", None)),
        "pillow_version": str(PIL.__version__),
        "torchvision_version": str(torchvision.__version__),
    }


def build_image_encoder_identity(
    *,
    config: Any,
    net: torch.nn.Module,
    encoder_revision: str | None = None,
) -> Dict[str, Any]:

    module = net.module if hasattr(net, "module") else net
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError("Managed image feature cache requires a frozen image encoder.")

    encode_image = getattr(module, "encode_image", None)
    if encode_image is None:
        raise TypeError("Managed image feature cache requires net.encode_image().")

    network_config = config.network
    backbone = str(network_config.backbone.name)
    checkpoint = getattr(network_config, "checkpoint", None)
    checkpoint_value = str(checkpoint) if checkpoint is not None else "none"
    checkpoint_path = Path(checkpoint_value).expanduser()

    if encoder_revision is not None:
        trusted_revision = str(encoder_revision).strip()
        if not trusted_revision:
            raise ValueError("encoder_revision cannot be empty when configured.")

        weight_identity = {
            "kind": "trusted_encoder_revision",
            "value": trusted_revision,
            "configured_checkpoint": checkpoint_value,
        }
    elif checkpoint_path.is_file():
        weight_identity = {
            "kind": "checkpoint_sha256",
            "value": _file_sha256(checkpoint_path),
        }
    elif backbone in clip._MODELS:
        model_url = clip._MODELS[backbone]
        weight_identity = {
            "kind": "official_clip_sha256",
            "value": model_url.split("/")[-2],
        }
    elif checkpoint_value.lower() not in {"", "none", "null"}:
        weight_identity = {
            "kind": "configured_revision",
            "value": checkpoint_value,
        }
    else:
        raise ValueError(
            "Cannot identify image encoder weights. Configure a checkpoint/revision "
            "or use a registered CLIP backbone."
        )

    if encoder_revision is None:
        weight_identity["loaded_state_sha256"] = _module_state_sha256(module)

    try:
        source = inspect.getsource(encode_image)
    except (OSError, TypeError):
        source = _qualified_name(encode_image)

    device = _module_device(module)
    compute_device: Dict[str, Any] = {
        "type": device.type,
        "platform_machine": platform.machine(),
    }

    if device.type == "cuda":
        compute_device.update(
            {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            }
        )

    identity = {
        "network_name": str(network_config.name),
        "backbone": backbone,
        "network_class": _qualified_name(module),
        "encode_image_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "weights": weight_identity,
        "parameter_dtypes": sorted({str(value.dtype) for value in module.parameters()}),
        "buffer_dtypes": sorted({str(value.dtype) for value in module.buffers()}),
        "default_float_dtype": str(torch.get_default_dtype()),
        "compute_device": compute_device,
        "torch_version": str(torch.__version__),
        "cuda_version": _json_value(torch.version.cuda),
    }
    return identity


def _validate_deterministic_preprocessor(
    dataset: Any,
) -> None:

    preprocessor = getattr(dataset, "preprocessor", None)
    if preprocessor is None:
        raise ValueError("Managed image feature cache requires dataset.preprocessor.")

    supported_preprocessor_names = {"TestStandardPreProcessor"}
    if preprocessor.__class__.__name__ not in supported_preprocessor_names:
        raise ValueError(
            "Managed image feature cache only supports an explicit deterministic "
            f"preprocessor contract; unsupported preprocessor: "
            f"{preprocessor.__class__.__name__}."
        )

    transform = getattr(preprocessor, "transform", None)
    if transform is None:
        raise ValueError("Deterministic preprocessor contract requires a transform graph.")

    supported_transform_names = {
        "CenterCrop",
        "Compose",
        "Convert",
        "ConvertImageDtype",
        "Identity",
        "Normalize",
        "PILToTensor",
        "Resize",
        "ToTensor",
    }
    stack = [transform]
    while stack:
        current = stack.pop()
        if current is None:
            continue

        name = current.__class__.__name__
        if name not in supported_transform_names:
            raise ValueError(
                "Managed image feature cache only supports an explicit deterministic "
                f"transform contract; unsupported transform: {name}."
            )

        children = getattr(current, "transforms", None)
        if children is not None:
            stack.extend(list(children))


def _module_device(
    module: torch.nn.Module,
) -> torch.device:

    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _module_state_sha256(
    module: torch.nn.Module,
) -> str:

    digest = hashlib.sha256()

    for name, value in sorted(module.state_dict().items()):
        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported non-tensor encoder state: {name}.")

        tensor = value.detach().contiguous().cpu()
        metadata = _canonical_json(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        ).encode("utf-8")

        digest.update(len(metadata).to_bytes(8, byteorder="big"))
        digest.update(metadata)
        byte_view = tensor.reshape(-1).view(torch.uint8)
        digest.update(byte_view.numpy().tobytes())

    return digest.hexdigest()


def _object_identity(
    value: Any,
    seen: MutableSet[int] | None = None,
) -> Any:

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Enum):
        return {
            "enum": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "name": value.name,
            "value": _json_value(value.value),
        }

    if isinstance(value, Path):
        return str(value)

    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        return {
            "tensor_dtype": str(tensor.dtype),
            "tensor_shape": list(tensor.shape),
            "tensor_sha256": hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }

    if isinstance(value, Mapping):
        table = {
            str(key): _object_identity(value=item, seen=seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

        return table

    if isinstance(value, (list, tuple)):
        result = [
            _object_identity(value=item, seen=seen)
            for item in value
        ]

        return result

    if inspect.isroutine(value):
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "source_sha256": _source_sha256(value),
            "defaults": _object_identity(value=getattr(value, "__defaults__", None), seen=seen),
        }

    if seen is None:
        seen = set()

    object_id = id(value)
    if object_id in seen:
        return {"cycle": _qualified_name(value)}

    seen.add(object_id)

    try:
        public_state = {
            str(key): _object_identity(value=item, seen=seen)
            for key, item in sorted(vars(value).items())
            if not str(key).startswith("_")
        }
        return {
            "class": _qualified_name(value),
            "class_source_sha256": _source_sha256(value.__class__),
            "state": public_state,
        }
    except TypeError:
        return {
            "class": _qualified_name(value),
            "class_source_sha256": _source_sha256(value.__class__),
            "value": str(value),
        }
    finally:
        seen.remove(object_id)


def _qualified_name(
    value: Any,
) -> str:

    if inspect.isroutine(value):
        module = getattr(value, "__module__", value.__class__.__module__)
        qualname = getattr(value, "__qualname__", value.__class__.__qualname__)
        return f"{module}.{qualname}"

    cls = value if inspect.isclass(value) else value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _source_sha256(
    value: Any,
) -> str:

    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        source = _qualified_name(value)

    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _json_value(
    value: Any,
) -> Any:

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (list, tuple)):
        result = [
            _json_value(item)
            for item in value
        ]

        return result

    if isinstance(value, dict):
        table = {
            str(key): _json_value(item)
            for key, item in value.items()
        }

        return table

    return str(value)


def _canonical_json(
    value: Any,
) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_digest(
    value: Any,
) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(
    value: Any,
) -> bool:

    return (
        isinstance(value, str) and
        len(value) == 64 and
        all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "IMAGE_FEATURE_CACHE_SCHEMA_VERSION",
    "ImageFeatureCache",
    "ImageFeatureCacheSpec",
    "build_image_encoder_identity",
    "build_image_feature_cache_spec",
    "extract_image_features",
    "load_image_feature_cache",
    "save_image_feature_cache",
]
