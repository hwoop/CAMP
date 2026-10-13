# Experiments

Every table and figure of the paper is one *item*: a package under `scripts/experiment/<item>/` that knows which
runs it needs, how to turn them into tables, and how to draw its figure. Installation, datasets and the published
results are in the repository [README](../../README.md). Run every command from the repository root.

## Command line

```bash
python -m scripts.experiment list                        # every item with its benchmarks and seeds
python -m scripts.experiment <item> plan                 # recorded / missing runs, with each missing command
python -m scripts.experiment <item> run [--dry-run]      # launch the missing runs
python -m scripts.experiment <item> analyze              # tables  -> results/experiment/<item>/tables/
python -m scripts.experiment <item> figure               # figures -> results/experiment/<item>/figures/
python -m scripts.experiment <item> all                  # run, analyze, figure
python -m scripts.experiment <item> status               # runs recorded? tables fresh? figures present?
```

| Option | Meaning |
|---|---|
| `--benchmarks <b> ...` | `imagenet_standard`, `imagenet_openood`, `cifar10_openood`, `cifar100_openood` |
| `--seeds <s> ...` | default `0 1 42` |
| `--backbone` | `vitb16` (default), `vitb32`, `vitl14`, `rn50` |
| `--device` | `cuda`, `cuda:<i>` or `cpu` |
| `--dry-run` | print what would happen |
| `--opt key=value` | item-specific switches, repeatable |
| `--out <dir>` | write elsewhere than `results/experiment/<item>` |

Every stage is idempotent: `run` skips a run whose `ood.csv` exists, `analyze` works with whatever runs are
recorded and lists missing seeds in the `seeds` column, and tables and figures are rewritten in place.

## Items

| Item | Benchmarks | What it produces |
|---|---|---|
| `main_results` | all four | the published tables (96 runs) |
| `ablation` | ImageNet-1K ×2 | CAMP without class balancing, covariance adjustment, or confidence weighting |
| `complementarity` | ImageNet-1K ×2 | each detector alone and with CAMP's standardized score added, `z(S) + z(R)` |
| `backbones` | ImageNet-1K ×2 | ResNet-50, ViT-B/32, ViT-L/14 |
| `sensitivity` | ImageNet-1K ×2 | window size `N`, warm-up length `N_min`, ridge constant `eps` |
| `ood_ratio` | ImageNet-1K ×2 | streams resampled to a fixed OOD ratio (1 % to 99 %), CAMP vs MCM |
| `direction_warmup` | ImageNet-1K ×2 | direction from the first *t* samples scored on the whole stream, vs MCM |
| `warmup_batch1` | ImageNet-1K ×2 | warm-up length at batch size 1 |
| `motivation` | OpenOOD ImageNet-1K | ID means and stream mean of the last iNaturalist window |
| `id_approximation` | OpenOOD ImageNet-1K | window estimate of the ID distribution vs the full-ID oracle and TANL's ID memory |
| `id_oracle` | all four | CAMP rescored with oracle ID statistics: μ_ID and Σ_ID from the window's true ID rows or from every ID row of the stream, plus the rank-one ceiling (tables only) |
| `selection` | OpenOOD ImageNet-1K | CAMP's window restricted to the samples the TANL rule admits |
| `covariance_adjustment` | OpenOOD ImageNet-1K | ID/OOD distributions along the mean difference before and after adjustment |
| `estimation_context` | ImageNet-1K ×2, CIFAR-10 | confidence weight concentration and the CIFAR-10 near-OOD shortfall (tables only) |
| `stream_composition` | all four | ID/OOD counts and OOD ratio of every stream (no GPU) |
| `cost_methods` | ImageNet-1K, seed 0 | end-to-end and postprocessor time, GPU memory, per method |

Items that only replay recorded CAMP runs (`motivation`, `id_approximation`, `id_oracle`, `selection`,
`covariance_adjustment`, `direction_warmup`, `estimation_context`) launch no new runs: `analyze` replays the streams from the cached image
features and stores its trace under the item's `cache/`.

`cost_methods` is a timing measurement: run it with nothing else on the GPU. Each run times the copy + image encoder
every method shares and each method's postprocessor once, in an order that changes between runs; the table reports
the mean ± standard deviation of the runs' medians. The default three runs take about 75 minutes.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.experiment cost_methods all --device cuda
```

## Custom runs

`scripts/run.sh` and the items call `main.py` with a stack of configs, a later file overriding an earlier one.
`bash scripts/run.sh camp imagenet_openood 0` runs

```bash
python main.py --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
    configs/datasets/imagenet/imagenet_ood.yml configs/networks/vanilla_clip.yml \
    configs/pipelines/test/test_ood_clip_tta.yml configs/postprocessors/camp/default.yml \
    --num_workers 8 --cache.image.mode read_write --cache.image.load_validation fast \
    --cache.image.source_validation auto --mark imagenet_openood --seed 0
```

- Override an existing key on the command line (`--postprocessor.postprocessor_args.camp.covariance identity`).
- Add a new key in an overlay YAML appended to `--config`; `evaluator.stream` overlays change the streams
  (`openood/evaluators/stream_protocol.py`).
- Give each run its own `--mark`; the result directory is keyed by mark and seed.

To add CAMP's score to another detector, append an overlay after `camp/default.yml` plus that detector's `vlm`
settings:

```yaml
postprocessor:
  postprocessor_args:
    semantic: {method: tanl, args: {}}   # tanl | mcm | neglabel | csp | adaneg | oodd | coevo
    fusion: {semantic_weight: 1.0}
```

## Outputs

A run writes `ood.csv` (FPR@95, AUROC, AUPR, ACC per OOD set and group), `scores/<OOD set>.npz`, `config.yml` and
`log.txt` to

```text
results/<dataset>_vanilla_clip_test_ood_ood_clip_tta_<method>/<benchmark>[_<backbone>][_<arm>]-s<seed>/
```

An item adds only derived outputs under `results/experiment/<item>/`: `tables/` (CSV and Markdown), `figures/`
(SVG and PNG), `cache/` (replay traces) and `manifest.json` (inputs, outputs, argv, git HEAD). `results/` is not
tracked by git.

`id_approximation`, `selection`, `covariance_adjustment`, `sensitivity`, `direction_warmup` and `ood_ratio` also
write a column-width variant of their figure (`<stem>_column.svg` / `.png`, 3.3 in wide, the panels stacked) for a
two-column layout.

Each table value is the mean over seeds with the sample standard deviation in parentheses; Markdown tables bold the
best value of each column and underline the second best. FPR@95 follows the OpenOOD evaluator, so it can differ from
the compared methods' papers.

## Diagnostics

Diagnostics are off by default and never change a score. Append `configs/diagnostics/camp_samples.yml` (per-sample
scores), `camp_summary.yml` (plus per-batch scalars) or `camp_full.yml` (plus window vectors) after the method
config with a distinct `--mark`. The recorder writes `run_metadata.json`, `sample_scores.parquet`,
`batch_diagnostics.parquet` and `selected_vectors.npz` under the run's `camp_diagnostics/<stream>/`. All batch
geometry describes the window **before** the scored batch enters it.

## Built-in checks

There is no test suite; the checks live in the stages. `plan` validates every run list and override key against the
merged config stack, `analyze` replays the recorded streams and writes the largest score difference into
`manifest.json`, and `status` flags tables older than the runs they came from. Keep a run's `config.yml`, `log.txt`,
`scores/*.npz` and its cached features together: a replay needs all four.
