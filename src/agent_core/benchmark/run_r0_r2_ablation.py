#!/usr/bin/env python3
"""运行可复现的 R0/R1/R2 Benchmark 并构建对比报告。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_core.benchmark.ablation import (
    console_progress,
    run_r0_r2_ablation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run identical cases sequentially with R0, R1 and R2 policies, "
            "then generate a Markdown report and SVG charts."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("agent_config.toml"))
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmark/workloads/r0_full.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark/results"),
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        metavar="CASE_ID",
        help="Repeat to select several case IDs.",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--measured-runs", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    engine_overrides = {
        key: value
        for key, value in {
            "model_path": args.model_path,
            "n_gpu_layers": args.n_gpu_layers,
            "n_ctx": args.n_ctx,
        }.items()
        if value is not None
    }
    try:
        result = run_r0_r2_ablation(
            config_file=args.config,
            suite_file=args.suite,
            output_root=args.output_root,
            case_ids=tuple(args.case_ids),
            engine_overrides=engine_overrides,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            progress_callback=console_progress,
        )
    except Exception as exc:
        print(f"R0-R2 ablation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Ablation completed: {result.output_dir}")
    print(f"Comparison report: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
