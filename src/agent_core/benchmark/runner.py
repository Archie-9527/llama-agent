"""Parent-process benchmark orchestrator.

Every repetition runs in a fresh Python process so llama.cpp allocator state
cannot leak into the next measured sample.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_core.benchmark.aggregator import aggregate_run
from agent_core.benchmark.models import BenchmarkSuite
from agent_core.benchmark.report import write_report
from agent_core.config import load_engine_config, load_memory_config


class BenchmarkRunner:
    def __init__(
        self,
        *,
        config_file: Path,
        suite_file: Path,
        output_root: Path = Path("benchmark/results"),
        round_name: str = "R0",
        engine_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config_file = config_file.resolve()
        self.suite_file = suite_file.resolve()
        self.output_root = output_root.resolve()
        self.round_name = round_name
        self.engine_overrides = engine_overrides or {}

    def run(self) -> Path:
        suite = BenchmarkSuite.load(self.suite_file)
        run_id = (
            f"{self.round_name}-{suite.name}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        run_dir = self.output_root / run_id
        manifest = self._manifest(run_id, suite)
        if self.round_name == "R0" and any(manifest["memory_flags"].values()):
            raise ValueError(
                "R0 requires every [memory] optimization switch to be false"
            )
        cases_dir = run_dir / "cases"
        cases_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "manifest.json", manifest)

        failures_path = run_dir / "failures.jsonl"
        results_path = run_dir / "task_results.jsonl"
        total_runs = suite.warmup_runs + suite.measured_runs
        for case in suite.cases:
            for repetition in range(total_runs):
                measured = repetition >= suite.warmup_runs
                sample_dir = cases_dir / case.case_id / f"rep-{repetition:03d}"
                sample_dir.mkdir(parents=True)
                worker_case = _prepare_case(case, sample_dir)
                result_file = sample_dir / "result.json"
                command = [
                    sys.executable,
                    "-m",
                    "agent_core.benchmark.worker",
                    "--config",
                    str(self.config_file),
                    "--case-json",
                    json.dumps(worker_case, ensure_ascii=False),
                    "--output-dir",
                    str(sample_dir),
                    "--result-file",
                    str(result_file),
                    "--seed",
                    str(suite.seed + repetition),
                ]
                completed = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    text=True,
                    capture_output=True,
                    env=_worker_environment(self.engine_overrides),
                )
                if result_file.exists():
                    result = json.loads(result_file.read_text(encoding="utf-8"))
                else:
                    result = {
                        "case_id": case.case_id,
                        "category": case.category,
                        "status": "worker_failed",
                        "duration_ms": 0,
                        "evaluation": {"passed": False, "checks": {}},
                    }
                result.update(
                    {
                        "measured": measured,
                        "repetition": repetition,
                        "worker_exit_code": completed.returncode,
                    }
                )
                _append_jsonl(results_path, result)
                if completed.returncode != 0 or not result["evaluation"]["passed"]:
                    _append_jsonl(
                        failures_path,
                        {
                            **result,
                            "worker_stderr": completed.stderr[-8000:],
                        },
                    )

        _consolidate_telemetry(run_dir)
        for expected in (
            "inference_events.jsonl",
            "tool_events.jsonl",
            "lifecycle_events.jsonl",
            "system_memory.csv",
            "kv_metrics.csv",
            "failures.jsonl",
        ):
            (run_dir / expected).touch(exist_ok=True)
        summary = aggregate_run(run_dir)
        _write_json(run_dir / "summary.json", summary)
        write_report(run_dir, manifest, summary)
        return run_dir

    def _manifest(self, run_id: str, suite: BenchmarkSuite) -> dict[str, Any]:
        engine = load_engine_config(
            self.config_file, cli_overrides=self.engine_overrides
        )
        memory = load_memory_config(self.config_file)
        model_path = Path(engine.model_path)
        return {
            "run_id": run_id,
            "round": self.round_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "suite": {
                "name": suite.name,
                "path": str(self.suite_file),
                "warmup_runs": suite.warmup_runs,
                "measured_runs": suite.measured_runs,
                "seed": suite.seed,
                "case_count": len(suite.cases),
            },
            "model": {
                "path": str(model_path),
                "sha256": _sha256(model_path),
                "size_bytes": model_path.stat().st_size,
                "quantization": _infer_quantization(model_path.name),
            },
            "engine_config": {
                key: value
                for key, value in engine.__dict__.items()
                if key != "model_path"
            },
            "memory_flags": memory.__dict__,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "physical_memory_bytes": _physical_memory_bytes(),
                "llama_cpp_python": _package_version("llama-cpp-python"),
                "langchain": _package_version("langchain"),
                "langgraph": _package_version("langgraph"),
                "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
                "dirty_worktree": bool(
                    _git_value(["git", "status", "--porcelain"])
                ),
            },
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _worker_environment(
    engine_overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    source_dir = str((Path.cwd() / "src").resolve())
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_dir}{os.pathsep}{previous}" if previous else source_dir
    )
    for key, value in (engine_overrides or {}).items():
        environment[f"AGENT_{key.upper()}"] = str(value)
    return environment


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _infer_quantization(filename: str) -> str | None:
    import re

    match = re.search(r"(?:^|[-_.])(Q\d(?:_[A-Z0-9]+)*)", filename.upper())
    return match.group(1) if match else None


def _physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _git_value(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except OSError:
        return ""


def _consolidate_telemetry(run_dir: Path) -> None:
    """Create the documented run-level raw files from isolated samples."""
    jsonl_names = (
        "inference_events.jsonl",
        "tool_events.jsonl",
        "lifecycle_events.jsonl",
    )
    csv_names = ("system_memory.csv", "kv_metrics.csv")
    for name in jsonl_names:
        target = run_dir / name
        for source in sorted(run_dir.glob(f"cases/**/{name}")):
            parts = source.relative_to(run_dir / "cases").parts
            case_id, repetition = parts[0], parts[1]
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record.update({"case_id": case_id, "sample": repetition})
                _append_jsonl(target, record)

    import csv

    for name in csv_names:
        target = run_dir / name
        writer = None
        handle = None
        try:
            for source in sorted(run_dir.glob(f"cases/**/{name}")):
                parts = source.relative_to(run_dir / "cases").parts
                case_id, repetition = parts[0], parts[1]
                with source.open(encoding="utf-8") as source_handle:
                    for record in csv.DictReader(source_handle):
                        enriched = {
                            **record,
                            "case_id": case_id,
                            "sample": repetition,
                        }
                        if writer is None:
                            handle = target.open("w", encoding="utf-8", newline="")
                            writer = csv.DictWriter(
                                handle, fieldnames=list(enriched.keys())
                            )
                            writer.writeheader()
                        writer.writerow(enriched)
        finally:
            if handle is not None:
                handle.close()


def _prepare_case(case, sample_dir: Path) -> dict[str, Any]:
    """Materialize deterministic local fixtures requested by workload metadata."""
    raw = case.to_dict()
    size = int(case.metadata.get("fixture_size_bytes", 0))
    if size > 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
        marker = str(case.metadata.get("fixture_marker", "AGENTMEM_PAYLOAD"))
        unit = (marker + "\n").encode("utf-8")
        payload = (unit * (size // len(unit) + 1))[:size]
        fixture = sample_dir / "payload.txt"
        fixture.write_bytes(payload)
        raw["goal"] = raw["goal"].replace("{fixture_path}", str(fixture))
    return raw
