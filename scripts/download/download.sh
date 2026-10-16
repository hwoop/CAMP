#!/usr/bin/env bash
# Usage: bash scripts/download/download.sh [DATA_DIR]
#   Downloads the data of the four CAMP benchmarks into DATA_DIR (default ./data), then runs convert.py.
set -Eeuo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${1:-./data}"

"${PYTHON_BIN}" scripts/download/download.py \
    --contents datasets \
    --datasets cifar-10 cifar-100 imagenet-1k \
    --dataset_mode benchmark \
    --save_dir "${DATA_DIR}" ./results
"${PYTHON_BIN}" scripts/download/convert.py --data_dir "${DATA_DIR}"
