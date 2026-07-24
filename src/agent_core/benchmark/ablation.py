"""One-command sequential R0/R1/R2 ablation runner."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_core.benchmark.comparison import write_comparison_report
from agent_core.benchmark.runner import BenchmarkRunner


ROUND_MEMORY_OVERRIDES: dict[str, dict[str, bool]] = {
    "R0": {
        "artifact_virtualization": False,
        "lifecycle_context": False,
        "kv_lifecycle": False,
        "branch_management": False,
    },
    "R1": {
        "artifact_virtualization": True,
        "lifecycle_context": False,
        "kv_lifecycle": False,
        "branch_management": False,
    },
    "R2": {
        "artifact_virtualization": True,
        "lifecycle_context": True,
        "kv_lifecycle": False,
        "branch_management": False,
    },
}


@dataclass(frozen=True)
class AblationResult:
    output_dir: Path
    run_dirs: dict[str, Path]
    report_path: Path


def run_r0_r2_ablation(
    *,
    config_file: Path,
    suite_file: Path,
    output_root: Path = Path("benchmark/results"),
    case_ids: tuple[str, ...] = (),
    engine_overrides: dict[str, Any] | None = None,
    warmup_runs: int | None = None,
    measured_runs: int | None = None,
    round_order: tuple[str, ...] = ("R0", "R1", "R2"),
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> AblationResult:
    """Run the same suite sequentially under R0, R1 and R2 policies."""
    if len(round_order) != 3 or set(round_order) != set(ROUND_MEMORY_OVERRIDES):
        raise ValueError(
            "round_order must contain R0, R1 and R2 exactly once"
        )
    group_id = (
        f"R0-R2-ablation-{suite_file.stem}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    output_dir = output_root.resolve() / group_id
    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=False)
    run_dirs: dict[str, Path] = {}
    _write_manifest(
        output_dir,
        config_file=config_file,
        suite_file=suite_file,
        case_ids=case_ids,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        round_order=round_order,
        run_dirs=run_dirs,
        status="running",
    )

    try:
        for round_name in round_order:
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "ablation_round_started",
                        "round": round_name,
                        "output_dir": str(output_dir),
                    }
                )
            run_dir = BenchmarkRunner(
                config_file=config_file,
                suite_file=suite_file,
                output_root=runs_root,
                round_name=round_name,
                engine_overrides=engine_overrides,
                memory_overrides=ROUND_MEMORY_OVERRIDES[round_name],
                case_ids=case_ids,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                progress_callback=progress_callback,
            ).run()
            run_dirs[round_name] = run_dir
            _write_manifest(
                output_dir,
                config_file=config_file,
                suite_file=suite_file,
                case_ids=case_ids,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                round_order=round_order,
                run_dirs=run_dirs,
                status="running",
            )
    except Exception as exc:
        _write_manifest(
            output_dir,
            config_file=config_file,
            suite_file=suite_file,
            case_ids=case_ids,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            round_order=round_order,
            run_dirs=run_dirs,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    report_path = write_comparison_report(output_dir, run_dirs)
    _write_manifest(
        output_dir,
        config_file=config_file,
        suite_file=suite_file,
        case_ids=case_ids,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        round_order=round_order,
        run_dirs=run_dirs,
        status="completed",
        report_path=report_path,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "ablation_finished",
                "output_dir": str(output_dir),
                "report_path": str(report_path),
            }
        )
    return AblationResult(output_dir, dict(run_dirs), report_path)


def console_progress(event: dict[str, Any]) -> None:
    """Human-readable progress suitable for long local-model runs."""
    kind = event.get("event")
    if kind == "ablation_round_started":
        print(f"\n[{event['round']}] 开始运行", flush=True)
    elif kind == "sample_started":
        sample_kind = "测量" if event.get("measured") else "预热"
        print(
            f"[{event['round']}] {event['case_id']} "
            f"rep-{int(event['repetition']):03d} ({sample_kind}) ...",
            flush=True,
        )
    elif kind == "sample_finished":
        result = "PASS" if event.get("passed") else "FAIL"
        duration = float(event.get("duration_ms") or 0) / 1000
        print(
            f"[{event['round']}] {event['case_id']} "
            f"rep-{int(event['repetition']):03d} {result} "
            f"{duration:.2f}s",
            flush=True,
        )
    elif kind == "run_finished":
        print(
            f"[{event['round']}] 完成：{event['passed_count']}/"
            f"{event['sample_count']} 通过",
            flush=True,
        )
    elif kind == "ablation_finished":
        print(f"\n跨轮报告：{event['report_path']}", flush=True)


def _write_manifest(
    output_dir: Path,
    *,
    config_file: Path,
    suite_file: Path,
    case_ids: tuple[str, ...],
    warmup_runs: int | None,
    measured_runs: int | None,
    round_order: tuple[str, ...],
    run_dirs: dict[str, Path],
    status: str,
    report_path: Path | None = None,
    error: str | None = None,
) -> None:
    value = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_file.resolve()),
        "suite_file": str(suite_file.resolve()),
        "case_ids": list(case_ids),
        "warmup_runs_override": warmup_runs,
        "measured_runs_override": measured_runs,
        "round_order": list(round_order),
        "round_memory_overrides": ROUND_MEMORY_OVERRIDES,
        "run_dirs": {
            name: str(path)
            for name, path in run_dirs.items()
        },
        "report_path": str(report_path) if report_path else None,
        "error": error,
    }
    (output_dir / "ablation_manifest.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "AblationResult",
    "ROUND_MEMORY_OVERRIDES",
    "console_progress",
    "run_r0_r2_ablation",
]
