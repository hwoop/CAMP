#!/usr/bin/env bash
# Usage: [BACKBONE=vitb32|vitl14|rn50] [DRY_RUN=1] bash scripts/run.sh <method> <benchmark> [seed ...]
set -Eeuo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

method="${1:?usage: bash scripts/run.sh <method> <benchmark> [seed ...]}"
benchmark="${2:?usage: bash scripts/run.sh <method> <benchmark> [seed ...]}"
shift 2
seeds=("$@")
if [ "${#seeds[@]}" -eq 0 ]; then
    seeds=(0 1 42)
fi
BACKBONE="${BACKBONE:-vitb16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$method" in
    camp|mcm|neglabel|csp|tanl|adaneg|oodd|coevo) ;;
    *) echo "unknown method: $method" >&2; exit 2 ;;
esac

case "$benchmark" in
    imagenet_openood)
        dataset=imagenet
        datasets=(configs/datasets/imagenet/imagenet_traditional_four_ood.yml configs/datasets/imagenet/imagenet_ood.yml) ;;
    imagenet_standard)
        dataset=imagenet
        datasets=(configs/datasets/imagenet/imagenet_traditional_four_ood.yml) ;;
    cifar10_openood)
        dataset=cifar10
        datasets=(configs/datasets/cifar10/cifar10_clip.yml configs/datasets/cifar10/cifar10_ood.yml) ;;
    cifar100_openood)
        dataset=cifar100
        datasets=(configs/datasets/cifar100/cifar100_clip.yml configs/datasets/cifar100/cifar100_ood.yml) ;;
    *) echo "unknown benchmark: $benchmark" >&2; exit 2 ;;
esac

case "$BACKBONE" in
    vitb16) network=configs/networks/vanilla_clip.yml; mark="$benchmark" ;;
    vitb32|vitl14|rn50) network="configs/networks/vanilla_clip_${BACKBONE}.yml"; mark="${benchmark}_${BACKBONE}" ;;
    *) echo "unknown BACKBONE: $BACKBONE (vitb16 | vitb32 | vitl14 | rn50)" >&2; exit 2 ;;
esac

for seed in "${seeds[@]}"; do
    out="results/${dataset}_vanilla_clip_test_ood_ood_clip_tta_${method}/${mark}-s${seed}"
    if [ -f "$out/ood.csv" ]; then
        echo "[skip] $out/ood.csv exists"
        continue
    fi
    cmd=("$PYTHON_BIN" main.py
         --config "${datasets[@]}" "$network"
                  configs/pipelines/test/test_ood_clip_tta.yml
                  "configs/postprocessors/${method}/default.yml"
         --num_workers "$NUM_WORKERS"
         --cache.image.mode read_write --cache.image.load_validation fast --cache.image.source_validation auto
         --mark "$mark" --seed "$seed")
    echo "[run] ${cmd[*]}"
    if [ -z "${DRY_RUN:-}" ]; then
        "${cmd[@]}"
    fi
done
