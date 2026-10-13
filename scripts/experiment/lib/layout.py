"""
Repository paths, benchmark, method and leaf names, 
and the naming of run directories.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
RESULTS: Path = ROOT / "results"
EXPERIMENT: Path = RESULTS / "experiment"
PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]

BENCHMARKS: dict[str, str] = {
    "imagenet_openood": "imagenet",
    "imagenet_standard": "imagenet",
    "cifar10_openood": "cifar10",
    "cifar100_openood": "cifar100",
}
BENCHMARK_NAMES: dict[str, str] = {
    "imagenet_openood": "OpenOOD ImageNet-1K",
    "imagenet_standard": "ImageNet-1K",
    "cifar10_openood": "OpenOOD CIFAR-10",
    "cifar100_openood": "OpenOOD CIFAR-100",
}
IMAGENET_BENCHMARKS: tuple[str, ...] = ("imagenet_standard", "imagenet_openood")

BACKBONES: tuple[str, ...] = ("vitb16", "vitb32", "vitl14", "rn50")
BACKBONE_NAMES: dict[str, str] = {
    "vitb16": "ViT-B/16",
    "vitb32": "ViT-B/32",
    "vitl14": "ViT-L/14",
    "rn50": "ResNet-50",
}

ZERO_SHOT = "zero-shot"
TEST_TIME = "test-time"


@dataclass(frozen=True)
class Method:

    key: str
    name: str
    kind: str


METHODS: dict[str, Method] = {
    "mcm": Method("mcm", "MCM", ZERO_SHOT),
    "neglabel": Method("neglabel", "NegLabel", ZERO_SHOT),
    "csp": Method("csp", "CSP", ZERO_SHOT),
    "adaneg": Method("adaneg", "AdaNeg", TEST_TIME),
    "oodd": Method("oodd", "OODD", TEST_TIME),
    "coevo": Method("coevo", "CoEvo", TEST_TIME),
    "tanl": Method("tanl", "TANL", TEST_TIME),
    "camp": Method("camp", "CAMP (Proposed)", TEST_TIME),
}
COMPARED_METHODS: tuple[str, ...] = tuple(key for key in METHODS if key != "camp")

GROUPS: dict[str, str] = {
    "nearood": "Near-OOD",
    "farood": "Far-OOD",
}

LEAVES: dict[str, dict[str, tuple[str, ...]]] = {
    "imagenet_openood": {"nearood": ("ssb_hard", "ninco"), "farood": ("inaturalist", "textures", "openimageo")},
    "imagenet_standard": {"farood": ("inaturalist", "sun", "places", "dtd")},
    "cifar10_openood": {"nearood": ("cifar100", "tin"), "farood": ("mnist", "svhn", "texture", "place365")},
    "cifar100_openood": {"nearood": ("cifar10", "tin"), "farood": ("mnist", "svhn", "texture", "places365")},
}

LEAF_NAMES: dict[str, str] = {
    "inaturalist": "iNaturalist",
    "sun": "SUN",
    "places": "Places",
    "dtd": "Textures",
    "textures": "Textures",
    "texture": "Textures",
    "ssb_hard": "SSB-hard",
    "ninco": "NINCO",
    "openimageo": "OpenImage-O",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tin": "TinyImageNet",
    "mnist": "MNIST",
    "svhn": "SVHN",
    "place365": "Places365",
    "places365": "Places365",
}

RUN_FAMILY = "vanilla_clip_test_ood_ood_clip_tta"


def leaf_group(
    benchmark: str,
    leaf: str,
) -> str:

    if leaf in GROUPS:
        return leaf

    for group, leaves in LEAVES[benchmark].items():
        if leaf in leaves:
            return group

    raise KeyError(f"{leaf!r} is not an OOD set of {benchmark}")


def mark(
    benchmark: str,
    backbone: str = "vitb16",
    label: str | None = None,
) -> str:

    if benchmark not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {benchmark!r}; one of {', '.join(BENCHMARKS)}")

    if backbone not in BACKBONES:
        raise KeyError(f"unknown backbone {backbone!r}; one of {', '.join(BACKBONES)}")

    text = benchmark if backbone == "vitb16" else f"{benchmark}_{backbone}"
    return text if label is None else f"{text}_{label}"


def results_family(
    method: str,
    benchmark: str,
) -> str:
    return f"{BENCHMARKS[benchmark]}_{RUN_FAMILY}_{method}"


def run_dir(
    method: str,
    benchmark: str,
    seed: int,
    backbone: str = "vitb16",
    label: str | None = None,
    results: Path | None = None,
) -> Path:

    base = RESULTS if results is None else Path(results)
    return base / results_family(method=method, benchmark=benchmark) / f"{mark(benchmark, backbone, label)}-s{seed}"


def item_dir(
    name: str,
) -> Path:
    return EXPERIMENT / name


def relative(
    path: Path,
) -> str:

    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()
