"""
Print FPR@95 and AUROC of every method on every benchmark as mean (std) over seeds,
from the ood.csv files written by run.sh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment.lib import layout, tables  # noqa: E402
from scripts.experiment.lib.runs import RunSpec  # noqa: E402


def tolerant_stdout() -> None:
    """
    Make stdout replace characters it cannot encode, 
    such as the arrows in the tables, 
    instead of raising on a legacy code page.
    """

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")


def main() -> int:

    tolerant_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(layout.RESULTS), help="the results directory (default: results/)")
    parser.add_argument("--benchmarks", nargs="+", default=list(layout.BENCHMARKS), choices=list(layout.BENCHMARKS))
    parser.add_argument("--methods", nargs="+", default=list(layout.METHODS), choices=list(layout.METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--backbone", default="vitb16", choices=layout.BACKBONES)
    parser.add_argument("--leaves", action="store_true", help="also list every OOD set, not only the group means")
    parser.add_argument("--csv", default=None, help="write every row (OOD sets included) to this CSV file")
    args = parser.parse_args()

    results = Path(args.results)
    seeds_text = ", ".join(map(str, args.seeds))
    frames = []

    for benchmark in args.benchmarks:
        specs = [RunSpec(method, benchmark, seed, backbone=args.backbone)
            for method in tables.method_order(args.methods) for seed in args.seeds]
        frame = tables.summarize(specs, results=results)
        frames.append(frame)
        title = f"{benchmark} (seeds {seeds_text}; mean (std) over seeds, best in bold, second best underlined)"

        if frame.empty:
            print(tables.markdown_table(frame, title=title))
            continue

        columns = tables.default_columns(frame, benchmark, groups_only=True, backbone=args.backbone)

        if args.leaves:
            columns += [c for c in tables.default_columns(frame, benchmark, backbone=args.backbone)
                if c.row not in tables.GROUPS]

        print(tables.markdown_table(frame, columns=columns, title=title))

    if args.csv:
        import pandas as pd

        path = tables.write_csv(tables.wide(pd.concat(frames, ignore_index=True)), Path(args.csv))
        print(f"written {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
