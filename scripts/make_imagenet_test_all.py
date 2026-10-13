"""
Build the ID image list of the ImageNet-1K benchmark (imagenet_standard): the whole ILSVRC-2012
validation set, 50,000 images, as OpenOOD's test_imagenet.txt (45,000) followed by val_imagenet.txt (5,000).
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "benchmark_imglist" / "imagenet"
PARTS: tuple[str, ...] = ("test_imagenet.txt", "val_imagenet.txt")
OUTPUT = "test_imagenet_all.txt"
EXPECTED = 50_000


def read_list(
    path: Path,
) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_DIR),
        help="the ImageNet image-list directory (default: data/benchmark_imglist/imagenet)",
    )
    args = parser.parse_args()
    folder = Path(args.dir)
    lines: list[str] = []

    for name in PARTS:
        path = folder / name
        if not path.is_file():
            raise SystemExit(f"{path} is missing; run `bash scripts/download/download.sh` first.")

        lines += read_list(path)

    images = {line.split()[0] for line in lines}
    if len(lines) != EXPECTED or len(images) != EXPECTED:
        raise SystemExit(
            f"expected {EXPECTED:,} distinct images in {' + '.join(PARTS)}, "
            f"found {len(lines):,} lines and {len(images):,} distinct images.",
        )

    output = folder / OUTPUT
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(lines):,} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
