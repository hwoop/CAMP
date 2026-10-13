from __future__ import annotations

import argparse
import shlex
import sys
import traceback
from collections.abc import Sequence

from scripts.experiment.lib import items, layout, runs

EXPECTED_ERRORS = (KeyError, ValueError, FileNotFoundError, RuntimeError, TypeError, NotImplementedError)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="python -m scripts.experiment",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("item", help="an item name (see `list`)")
    parser.add_argument("stage", choices=items.STAGES)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(layout.BENCHMARKS),
        default=None,
        help="benchmarks to evaluate (default: the item's)",
    )

    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="seeds (default: the item's, 0 1 42)")
    parser.add_argument("--backbone", choices=layout.BACKBONES, default="vitb16")
    parser.add_argument("--device", default=None, help="cuda | cuda:<index> | cpu (default: cuda when available)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands of the missing runs only")
    parser.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="an item-specific option; repeatable",
    )

    parser.add_argument("--out", default=None, help="output directory (default: results/experiment/<item>)")
    return parser


def print_list(
    found: Sequence[items.Item],
) -> None:

    if not found:
        print("no items (no scripts/experiment/<item>/__init__.py defines ITEM)")
        return

    width = max(len(item.name) for item in found)

    for item in found:
        print(
            f"{item.name:<{width}}  {', '.join(item.benchmarks)}  "
            f"seeds {','.join(map(str, item.seeds))}  {item.title}",
        )


def print_plan(
    rows: Sequence[items.PlanRow],
) -> None:

    if not rows:
        print("no runs: this item reads no recorded run")
        return

    for row in rows:
        state = "recorded" if row.recorded else "missing "
        print(f"[{state}] {layout.relative(row.dir)}")

        if not row.recorded:
            print(f"    {shlex.join(row.command)}")

    missing = sum(not row.recorded for row in rows)
    print(f"{len(rows) - missing}/{len(rows)} runs recorded, {missing} missing")


def tolerant_stdout() -> None:
    """
    Make stdout replace characters it cannot encode,
    such as the arrows in the tables, instead of raising on a legacy code page.
    """

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")


def main(
    argv: Sequence[str] | None = None,
) -> int:

    argv = list(sys.argv[1:] if argv is None else argv)
    tolerant_stdout()

    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0 if argv else 2

    if argv[0] == "list":
        try:
            print_list(items.discover_items())
        except Exception as error:
            print(f"error: {error}", file=sys.stderr)
            traceback.print_exc()
            return 1

        return 0

    args = build_parser().parse_args(argv)

    try:
        module = items.load_item(args.item)
        device = args.device
        if device is None:
            device = items.default_device() if args.stage in items.EXECUTING_STAGES else "cpu"
        else:
            runs.cuda_visible_devices(device)

        ctx = items.make_context(
            module.ITEM,
            benchmarks=args.benchmarks,
            seeds=args.seeds,
            backbone=args.backbone,
            device=device,
            dry_run=args.dry_run,
            options=items.parse_options(args.opt),
            out=args.out,
            stage=args.stage,
        )

        if args.stage == "plan":
            print_plan(items.plan(module, ctx))
            return 0

        if args.stage == "status":
            print("\n".join(items.status(module, ctx).lines()))
            return 0

        written = items.run_stage(module, args.stage, ctx)
        for path in written:
            print(layout.relative(path))

        if args.dry_run and not written:
            print("nothing to do")

        return 0
    except EXPECTED_ERRORS as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"error: {message}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1
