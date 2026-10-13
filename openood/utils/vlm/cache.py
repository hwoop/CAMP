import inspect
import os
import stat
import uuid
from typing import Any, Dict, Optional

import torch

try:
    _TORCH_LOAD_PARAMETERS = frozenset(inspect.signature(torch.load).parameters)
except (TypeError, ValueError):
    _TORCH_LOAD_PARAMETERS = frozenset()


def safe_cache_value(
    value: Any,
) -> str:

    if isinstance(value, bool):
        return str(int(value))

    if value is None:
        return "none"

    value = str(value)
    value = value.replace("/", "_")
    value = value.replace("\\", "_")
    value = value.replace(" ", "")
    value = value.replace(".", "p")
    value = value.replace(":", "-")
    value = value.replace("=", "-")

    return value


def make_model_cache_dir(
    cache_root: str,
    dataset: str,
    model_name: str,
    backbone: str,
) -> str:

    dir_name = (
        f"{safe_cache_value(dataset)}_"
        f"{safe_cache_value(model_name)}_"
        f"{safe_cache_value(backbone)}"
    )

    return os.path.join(cache_root, dir_name)


def make_cache_path(
    dataset: str,
    model_name: str,
    backbone: str,
    cache_group: str,
    params: Dict[str, Any],
    cache_root: str = './.cache/',
    suffix: str = ".pt",
) -> str:

    if not suffix.startswith("."):
        suffix = "." + suffix

    model_cache_dir = make_model_cache_dir(
        cache_root=cache_root,
        dataset=dataset,
        model_name=model_name,
        backbone=backbone,
    )

    cache_dir = os.path.join(model_cache_dir, safe_cache_value(cache_group))

    parts = []

    for key, value in params.items():
        if key == "return":
            continue

        parts.append(f"{safe_cache_value(key)}={safe_cache_value(value)}")

    if "return" in params:
        parts.append(f"return={safe_cache_value(params['return'])}")

    filename = "_".join(parts) + suffix

    return os.path.join(cache_dir, filename)


def _to_cpu(
    obj: Any,
) -> Any:

    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        table = {
            k: _to_cpu(v)
            for k, v in obj.items()
        }

        return table

    if isinstance(obj, list):
        result = [
            _to_cpu(v)
            for v in obj
        ]

        return result

    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)

    return obj


def _to_device(
    obj: Any,
    device: str,
) -> Any:

    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        table = {
            k: _to_device(obj=v, device=device)
            for k, v in obj.items()
        }

        return table

    if isinstance(obj, list):
        result = [
            _to_device(obj=v, device=device)
            for v in obj
        ]

        return result

    if isinstance(obj, tuple):
        return tuple(_to_device(obj=v, device=device) for v in obj)

    return obj


def _fsync_directory(
    path: str,
) -> None:
    """
    Fsync a directory. 
    Skipped on platforms where a directory cannot be opened, such as Windows.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)

    try:
        descriptor = os.open(path=path, flags=flags)
    except OSError:
        return

    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def save_cache(
    cache_path: str,
    cache_obj: Dict[str, Any],
) -> None:

    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)

    try:
        existing_mode = stat.S_IMODE(os.stat(cache_path).st_mode)
    except FileNotFoundError:
        existing_mode = None

    temporary_path = os.path.join(
        cache_dir,
        f".{os.path.basename(cache_path)}.{uuid.uuid4().hex}.tmp",
    )

    file_descriptor = os.open(
        path=temporary_path,
        flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode=0o666,
    )

    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            torch.save(obj=_to_cpu(cache_obj), f=handle)
            handle.flush()
            os.fsync(handle.fileno())

        if existing_mode is not None:
            os.chmod(path=temporary_path, mode=existing_mode)

        os.replace(src=temporary_path, dst=cache_path)
        _fsync_directory(cache_dir)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_cache(
    cache_path: str,
    device: Optional[str] = None,
    *,
    weights_only: Optional[bool] = None,
    mmap: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Load a cache file with ``torch.load``. 
    The ``weights_only`` and ``mmap`` hints are passed only when the installed PyTorch accepts them, 
    and ``mmap`` is dropped for legacy files that do not support it.
    """

    load_options: Dict[str, Any] = {}

    if weights_only is not None and "weights_only" in _TORCH_LOAD_PARAMETERS:
        load_options["weights_only"] = weights_only

    if mmap is not None and "mmap" in _TORCH_LOAD_PARAMETERS:
        load_options["mmap"] = mmap

    try:
        cache = torch.load(cache_path, map_location="cpu", **load_options)
    except RuntimeError as error:
        if not load_options.get("mmap") or "mmap can only be used" not in str(error):
            raise

        del load_options["mmap"]
        cache = torch.load(cache_path, map_location="cpu", **load_options)

    if device is not None:
        cache = _to_device(obj=cache, device=device)

    return cache
