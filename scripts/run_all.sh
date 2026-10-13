#!/usr/bin/env bash
# Usage: bash scripts/run_all.sh [seed ...]   (every main-table run; finished runs are skipped, so it can be restarted)
set -Eeuo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

benchmarks=(imagenet_openood imagenet_standard cifar10_openood cifar100_openood)
methods=(camp mcm neglabel csp tanl adaneg oodd coevo)

for method in "${methods[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        bash scripts/run.sh "$method" "$benchmark" "$@"
    done
done
