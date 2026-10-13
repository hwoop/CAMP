from __future__ import annotations

import hashlib
import os
import random
from typing import Union

import numpy as np
import torch

SeedLike = Union[int, np.integer]
_MAX_TORCH_SEED = (1 << 63) - 1


def normalize_seed(
    seed: SeedLike,
) -> int:
    return int(seed) % _MAX_TORCH_SEED


def derive_seed(
    seed: SeedLike,
    namespace: str,
) -> int:
    """
    Derive a seed for ``namespace`` from ``seed`` with sha256. 
    The result is the same in every process, 
    unlike the salted ``hash()``, and no global RNG state is consumed.
    """

    namespace = str(namespace).strip()
    if not namespace:
        raise ValueError("namespace must be a non-empty string")

    payload = f"{normalize_seed(seed)}::{namespace}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little") % _MAX_TORCH_SEED


def configure_determinism_environment(
    seed: SeedLike,
) -> int:
    """
    Set the environment variables for deterministic runs. Call it before CUDA is initialized, 
    because cuBLAS reads its workspace configuration once; ``PYTHONHASHSEED`` only takes effect in child processes.
    """

    normalized = normalize_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(normalized % (1 << 32))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return normalized


def seed_everything(
    seed: SeedLike,
) -> int:
    """
    Seed Python, NumPy and torch and enable deterministic algorithms. 
    ``get_network`` (openood/networks/utils.py) turns ``cudnn.benchmark`` back on afterwards.
    """

    normalized = configure_determinism_environment(seed)

    random.seed(normalized)
    np.random.seed(normalized % (1 << 32))
    torch.manual_seed(normalized)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)

    return normalized


def make_torch_generator(
    seed: SeedLike,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator:

    generator = torch.Generator(device=device)
    generator.manual_seed(normalize_seed(seed))
    return generator


def make_namespaced_generator(
    seed: SeedLike,
    namespace: str,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator:

    return make_torch_generator(
        derive_seed(seed=seed, namespace=namespace),
        device=device,
    )


def seed_worker(
    worker_id: int,
) -> None:

    del worker_id
    worker_seed = int(torch.initial_seed()) % (1 << 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


__all__ = [
    "configure_determinism_environment",
    "derive_seed",
    "make_namespaced_generator",
    "make_torch_generator",
    "seed_everything",
    "seed_worker",
]
