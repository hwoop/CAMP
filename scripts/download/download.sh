#!/usr/bin/env bash
# Usage: bash scripts/download/download.sh   (OpenOOD v1.5 image lists and the benchmarks' datasets into ./data)
set -Eeuo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/download/download.py \
    --contents datasets \
    --datasets cifar-10 cifar-100 imagenet-1k \
    --dataset_mode benchmark \
    --save_dir ./data ./results
