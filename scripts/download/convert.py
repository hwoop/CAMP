"""
Build what the configs read but OpenOOD does not ship, from an OpenOOD download:

- images_largescale/dtd/images/: OpenOOD's images_classic/texture/ under the name dtd, all 47 classes
- benchmark_imglist/imagenet/test_dtd.txt, test_sun.txt, test_places.txt and test_imagenet_all.txt

The lists are written byte for byte as the paper's runs read them and checked by SHA-256, since the evaluator draws its
seeded stream order from the list order. Then every image of every list in configs/datasets/ is checked; --dry_run
writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import string
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "datasets"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
JUNK_NAMES = {".DS_Store"}  # macOS metadata in OpenOOD's texture.zip

# The lists the paper's runs read: (entries, SHA-256 of the file).
EXPECTED = {
    "test_dtd.txt": (5640, "33c6750f78487e186784606edc5aa1bb737f2a88bb4f00e373d24e96991a433d"),
    "test_sun.txt": (10000, "14bb8fd2e3aa9cdd2982ab80451b16141667d580e3d9d72df482fd04ef3614dc"),
    "test_places.txt": (10000, "25f3a4d32d78ff4ffa3275c11d22a5152174249eae6a0f7db5e6a40d8c596624"),
    "test_imagenet_all.txt": (50000, "bdcff9e1f269db7fe93e896886833b1d1f119d0c5be3c075bc8c4219ba1a8a42"),
}

_ASCII_UPPER = str.maketrans(string.ascii_lowercase, string.ascii_uppercase)


class ConversionError(RuntimeError):
    """A step that cannot be done with the data at hand."""


def ntfs_order(
    name: str,
) -> str:
    """
    Sort key of an NTFS listing (names compared in upper case), the order the original lists were written in;
    listing on another file system gives another order.
    """

    return name.translate(_ASCII_UPPER)


def image_files(
    folder: Path,
    hint: str,
) -> list[str]:
    """The image files of a folder, in NTFS order."""

    if not folder.is_dir():
        raise ConversionError(f"{folder} is missing; {hint}")

    with os.scandir(folder) as entries:
        names = [entry.name for entry in entries if entry.is_file() and entry.name.lower().endswith(IMAGE_SUFFIXES)]

    return sorted(names, key=ntfs_order)


def tree_files(
    folder: Path,
) -> dict[str, int]:
    """Relative path -> size of every file under folder, without the macOS metadata files."""

    found: dict[str, int] = {}
    for parent, _, files in os.walk(folder):
        for name in files:
            if name not in JUNK_NAMES:
                path = Path(parent) / name
                found[path.relative_to(folder).as_posix()] = path.stat().st_size

    return found


def build_dtd(
    data: Path,
    *,
    link: str,
    workers: int,
    dry_run: bool,
) -> str:
    """Put the DTD images of images_classic/texture/ at images_largescale/dtd/images/."""

    source = data / "images_classic" / "texture"
    target = data / "images_largescale" / "dtd" / "images"
    if not source.is_dir():
        raise ConversionError(
            f"{source} is missing; download it first: python scripts/download/download.py "
            f"--contents datasets --dataset_mode dataset --datasets texture --save_dir {data}",
        )

    files = tree_files(source)
    if target.exists():
        present = tree_files(target)
        if present == files:
            return f"in place ({len(files):,} files)"

        missing = sorted(set(files) - set(present))
        changed = sorted(name for name in set(files) & set(present) if files[name] != present[name])
        extra = sorted(set(present) - set(files))
        raise ConversionError(
            f"{target} exists but differs from {source} (missing: {len(missing)}, other size: {len(changed)}, "
            f"extra: {len(extra)}); remove it and rerun to rebuild it",
        )

    if dry_run:
        return f"would be built from {source} ({len(files):,} files)"

    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)

    for name in files:
        (staging / name).parent.mkdir(parents=True, exist_ok=True)

    mode = link

    def place(name: str) -> None:
        nonlocal mode
        if mode == "hardlink":
            try:
                os.link(source / name, staging / name)
                return
            except OSError:
                mode = "copy"  # no hard links on this file system: copy the rest

        shutil.copy2(source / name, staging / name)

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(place, files))

        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    how = "hard links" if mode == "hardlink" else "copies"
    return f"built from {source} ({len(files):,} files, {how})"


def write_list(
    path: Path,
    lines: list[str],
    *,
    final_newline: bool,
    overwrite: bool,
    allow_mismatch: bool,
    dry_run: bool,
) -> str:
    """Write an image list unless it is in place, after checking it against the paper's."""

    data = ("\n".join(lines) + ("\n" if final_newline else "")).encode("utf-8")
    entries, sha256 = EXPECTED[path.name]
    if hashlib.sha256(data).hexdigest() != sha256:
        message = (
            f"{path.name} would differ from the list the paper's runs read ({len(lines):,} entries, {entries:,} "
            f"expected; same count means another order or other names)"
        )
        if not allow_mismatch:
            raise ConversionError(f"{message}; pass --allow_mismatch to write it anyway")

        print(f"  warning: {message}; writing it anyway (--allow_mismatch)", flush=True)

    if path.is_file():
        if path.read_bytes() == data:
            return f"in place ({len(lines):,} entries)"

        if not overwrite:
            raise ConversionError(f"{path} exists with other content; pass --overwrite to replace it")

    if dry_run:
        return f"would be written ({len(lines):,} entries)"

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return f"written ({len(lines):,} entries)"


def dtd_lines(
    data: Path,
) -> list[str]:

    images = data / "images_largescale" / "dtd" / "images"
    if not images.is_dir():  # dry run before the dtd step: same files as texture/
        images = data / "images_classic" / "texture"

    if not images.is_dir():
        raise ConversionError(f"{images} is missing; download texture first (download.py)")

    with os.scandir(images) as entries:
        classes = sorted((entry.name for entry in entries if entry.is_dir()), key=ntfs_order)

    return [f"dtd/images/{name}/{file} -1" for name in classes for file in image_files(images / name, "")]


def flat_lines(
    data: Path,
    dataset: str,
) -> list[str]:

    hint = (
        f"download it first: python scripts/download/download.py --contents datasets --dataset_mode dataset "
        f"--datasets {dataset} --save_dir {data}"
    )
    files = image_files(data / "images_largescale" / dataset / "images", hint)
    return [f"{dataset}/images/{file} -1" for file in files]


def imagenet_all_lines(
    data: Path,
) -> list[str]:
    """test_imagenet.txt followed by val_imagenet.txt, as scripts/make_imagenet_test_all.py joins them."""

    script = ROOT / "scripts" / "make_imagenet_test_all.py"
    spec = importlib.util.spec_from_file_location("make_imagenet_test_all", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = data / "benchmark_imglist" / "imagenet"
    lines: list[str] = []
    for name in module.PARTS:
        if not (folder / name).is_file():
            raise ConversionError(f"{folder / name} is missing; download the image lists first (download.py)")

        lines += module.read_list(folder / name)

    return lines


def config_lists(
    data: Path,
) -> list[tuple[Path, Path]]:
    """(image list, image folder) pairs of configs/datasets/; an overlay without data_dir takes the same key's."""

    import yaml

    def resolve(value: str) -> Path:
        parts = Path(value).parts
        if parts and parts[0] == "data":
            return data.joinpath(*parts[1:])

        return (ROOT / value).resolve()

    found: list[tuple[tuple[str, ...], str, str | None]] = []
    data_dirs: dict[tuple[str, ...], str] = {}

    def walk(node: object, keys: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            if "imglist_pth" in node:
                found.append((keys, str(node["imglist_pth"]), node.get("data_dir")))
                if node.get("data_dir"):
                    data_dirs[keys] = str(node["data_dir"])

            for key, value in node.items():
                walk(value, keys + (str(key),))

    for config in sorted(CONFIGS.rglob("*.yml")):
        walk(yaml.safe_load(config.read_text(encoding="utf-8")), ())

    pairs: list[tuple[Path, Path]] = []
    for keys, imglist, data_dir in found:
        folder = data_dir or data_dirs.get(keys)
        if folder is not None and (resolve(imglist), resolve(folder)) not in pairs:
            pairs.append((resolve(imglist), resolve(folder)))

    return pairs


def check(
    data: Path,
) -> int:
    """Check every image of every list the configs read; return the number of incomplete lists."""

    listings: dict[Path, set[str]] = {}
    incomplete = 0
    for imglist, folder in config_lists(data):
        shown = imglist.relative_to(data).as_posix() if imglist.is_relative_to(data) else str(imglist)
        if not imglist.is_file():
            print(f"  MISSING  {shown} (the list itself)", flush=True)
            incomplete += 1
            continue

        names = [line.split(" ", 1)[0] for line in imglist.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing: list[str] = []
        for name in names:
            path = folder / name
            if path.parent not in listings:
                try:
                    with os.scandir(path.parent) as entries:
                        listings[path.parent] = {entry.name for entry in entries}
                except OSError:
                    listings[path.parent] = set()

            if path.name not in listings[path.parent]:
                missing.append(name)

        if missing:
            incomplete += 1
            print(f"  MISSING  {shown}: {len(missing):,} of {len(names):,} images, e.g. {missing[0]}", flush=True)
        else:
            print(f"  ok       {shown}: {len(names):,} images", flush=True)

    return incomplete


def main() -> int:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=str(ROOT / "data"), help="the data folder (default: data/)")
    parser.add_argument(
        "--link",
        default="copy",
        choices=["copy", "hardlink"],
        help="put the DTD images as copies or as hard links to images_classic/texture/ (default: copy)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing list with other content")
    parser.add_argument("--allow_mismatch", action="store_true", help="write a list that differs from the paper's")
    parser.add_argument("--skip_check", action="store_true", help="do not check the images of the configs' lists")
    parser.add_argument("--workers", type=int, default=4, help="threads placing the DTD images (default: 4)")
    parser.add_argument("--dry_run", action="store_true", help="write nothing: report what would change, then check")
    args = parser.parse_args()

    data = Path(args.data_dir).expanduser().resolve()
    imagenet = data / "benchmark_imglist" / "imagenet"
    options = {"overwrite": args.overwrite, "allow_mismatch": args.allow_mismatch, "dry_run": args.dry_run}
    steps = [
        ("images_largescale/dtd/images", lambda: build_dtd(
            data, link=args.link, workers=args.workers, dry_run=args.dry_run,
        )),
        ("test_dtd.txt", lambda: write_list(imagenet / "test_dtd.txt", dtd_lines(data), final_newline=True, **options)),
        ("test_sun.txt", lambda: write_list(
            imagenet / "test_sun.txt", flat_lines(data, "sun"), final_newline=False, **options,
        )),
        ("test_places.txt", lambda: write_list(
            imagenet / "test_places.txt", flat_lines(data, "places"), final_newline=False, **options,
        )),
        ("test_imagenet_all.txt", lambda: write_list(
            imagenet / "test_imagenet_all.txt", imagenet_all_lines(data), final_newline=True, **options,
        )),
    ]

    print(f"data: {data}" + (" (dry run: nothing is written)" if args.dry_run else ""), flush=True)
    failures = 0
    for name, step in steps:
        start = time.time()
        try:
            result = step()
            print(f"  {name}: {result} ({time.time() - start:.0f} s)", flush=True)
        except ConversionError as error:
            failures += 1
            print(f"  {name}: FAILED, {error}", flush=True)

    incomplete = 0
    if not args.skip_check:
        print("images of the lists in configs/datasets/:", flush=True)
        incomplete = check(data)

    if failures or incomplete:
        print(f"\n{failures} step(s) failed, {incomplete} list(s) incomplete", flush=True)
        return 1

    print("\ndone: the data folder is ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
