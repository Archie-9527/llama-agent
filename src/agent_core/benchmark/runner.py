"""Parent-process benchmark orchestrator.

Every repetition runs in a fresh Python process so llama.cpp allocator state
cannot leak into the next measured sample.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import uuid
import importlib.metadata
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_core.benchmark.aggregator import aggregate_run
from agent_core.benchmark.models import BenchmarkSuite
from agent_core.benchmark.report import write_report
from agent_core.config import load_engine_config, load_memory_config


def _filter_suite_cases(
    suite: BenchmarkSuite,
    case_ids: tuple[str, ...],
) -> BenchmarkSuite:
    """Return a suite containing only requested IDs, in suite-file order."""
    if not case_ids:
        return suite

    requested = set(case_ids)
    available = {case.case_id for case in suite.cases}
    unknown = sorted(requested - available)
    if unknown:
        choices = ", ".join(sorted(available))
        raise ValueError(
            f"Unknown benchmark case_id(s): {', '.join(unknown)}. "
            f"Available case_ids: {choices}"
        )

    selected = tuple(case for case in suite.cases if case.case_id in requested)
    return replace(suite, cases=selected)


class BenchmarkRunner:
    def __init__(
        self,
        *,
        config_file: Path,
        suite_file: Path,
        output_root: Path = Path("benchmark/results"),
        round_name: str = "R0",
        engine_overrides: dict[str, Any] | None = None,
        memory_overrides: dict[str, Any] | None = None,
        case_ids: tuple[str, ...] = (),
        warmup_runs: int | None = None,
        measured_runs: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config_file = config_file.resolve()
        self.suite_file = suite_file.resolve()
        self.output_root = output_root.resolve()
        self.round_name = round_name
        self.engine_overrides = engine_overrides or {}
        self.memory_overrides = memory_overrides or {}
        self.case_ids = tuple(dict.fromkeys(case_ids))
        if warmup_runs is not None and warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")
        if measured_runs is not None and measured_runs < 1:
            raise ValueError("measured_runs must be >= 1")
        self.warmup_runs = warmup_runs
        self.measured_runs = measured_runs
        self.progress_callback = progress_callback

    def run(self) -> Path:
        suite = BenchmarkSuite.load(self.suite_file)
        suite = _filter_suite_cases(suite, self.case_ids)
        suite = replace(
            suite,
            warmup_runs=(
                suite.warmup_runs
                if self.warmup_runs is None
                else self.warmup_runs
            ),
            measured_runs=(
                suite.measured_runs
                if self.measured_runs is None
                else self.measured_runs
            ),
        )
        run_id = (
            f"{self.round_name}-{suite.name}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        run_dir = self.output_root / run_id
        manifest = self._manifest(run_id, suite)
        if self.round_name.upper().startswith("R0") and any(
            manifest["memory_flags"].values()
        ):
            raise ValueError(
                "R0 requires every [memory] optimization switch to be false"
            )
        if self.round_name.upper().startswith("R1") and not manifest[
            "memory_flags"
        ]["artifact_virtualization"]:
            raise ValueError(
                "R1 requires memory.artifact_virtualization=true; set it in "
                "the config file or with "
                "AGENT_MEMORY_ARTIFACT_VIRTUALIZATION=true"
            )
        if self.round_name.upper().startswith("R2") and not (
            manifest["memory_flags"]["artifact_virtualization"]
            and manifest["memory_flags"]["lifecycle_context"]
        ):
            raise ValueError(
                "R2 requires memory.artifact_virtualization=true and "
                "memory.lifecycle_context=true"
            )
        cases_dir = run_dir / "cases"
        cases_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "manifest.json", manifest)
        self._progress(
            event="run_started",
            round=self.round_name,
            run_dir=str(run_dir),
            case_count=len(suite.cases),
            total_samples=len(suite.cases)
            * (suite.warmup_runs + suite.measured_runs),
        )

        failures_path = run_dir / "failures.jsonl"
        results_path = run_dir / "task_results.jsonl"
        total_runs = suite.warmup_runs + suite.measured_runs
        for case in suite.cases:
            for repetition in range(total_runs):
                measured = repetition >= suite.warmup_runs
                self._progress(
                    event="sample_started",
                    round=self.round_name,
                    case_id=case.case_id,
                    repetition=repetition,
                    measured=measured,
                )
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
                    env=_worker_environment(
                        self.engine_overrides,
                        memory_config=manifest["memory_config"],
                    ),
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
                self._progress(
                    event="sample_finished",
                    round=self.round_name,
                    case_id=case.case_id,
                    repetition=repetition,
                    measured=measured,
                    passed=bool(result.get("evaluation", {}).get("passed")),
                    status=result.get("status"),
                    duration_ms=result.get("duration_ms"),
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
        self._progress(
            event="run_finished",
            round=self.round_name,
            run_dir=str(run_dir),
            passed_count=summary["passed_count"],
            sample_count=summary["sample_count"],
        )
        return run_dir

    def _manifest(self, run_id: str, suite: BenchmarkSuite) -> dict[str, Any]:
        engine = load_engine_config(
            self.config_file, cli_overrides=self.engine_overrides
        )
        memory = load_memory_config(
            self.config_file,
            cli_overrides=self.memory_overrides,
        )
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
                "case_filter": list(self.case_ids),
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
            "memory_flags": {
                "artifact_virtualization": memory.artifact_virtualization,
                "lifecycle_context": memory.lifecycle_context,
                "kv_lifecycle": memory.kv_lifecycle,
                "branch_management": memory.branch_management,
            },
            "memory_config": memory.__dict__,
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

    def _progress(self, **event: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)


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
    memory_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    source_dir = str((Path.cwd() / "src").resolve())
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_dir}{os.pathsep}{previous}" if previous else source_dir
    )
    for key, value in (engine_overrides or {}).items():
        environment[f"AGENT_{key.upper()}"] = str(value)
    # Freeze the parent process's resolved memory policy into every fresh
    # worker.  Workers must not observe a config file edited halfway through a
    # long benchmark, and manifest.json must describe what they actually ran.
    for key, value in (memory_config or {}).items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        environment[f"AGENT_MEMORY_{key.upper()}"] = rendered
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
    sample_dir.mkdir(parents=True, exist_ok=True)
    replacements: dict[str, str] = {}

    filler_count = int(case.metadata.get("conversation_filler_turns", 0))
    if filler_count:
        seed_turn = str(
            case.metadata.get(
                "conversation_seed_turn",
                "记住项目代号是 AgentMem，校验码是 7319，并直接复述。",
            )
        )
        final_turn = str(
            case.metadata.get(
                "conversation_final_turn",
                "综合此前对话，项目代号和校验码分别是什么？",
            )
        )
        raw["turns"] = [
            seed_turn,
            *[
                f"这是上下文压力测试第 {index + 1} 轮。请只回复 ACK-{index + 1}。"
                for index in range(filler_count)
            ],
            final_turn,
        ]

    size = int(case.metadata.get("fixture_size_bytes", 0))
    if size > 0:
        marker = str(case.metadata.get("fixture_marker", "AGENTMEM_PAYLOAD"))
        fixture = sample_dir / "payload.txt"
        marker_position = str(
            case.metadata.get("fixture_marker_position", "end")
        )
        _write_payload_fixture(
            fixture,
            size=size,
            marker=marker,
            marker_position=marker_position,
        )
        replacements["{fixture_path}"] = str(fixture)

    if case.metadata.get("incident_fixture"):
        replacements.update(_write_incident_fixture(sample_dir))

    if case.metadata.get("document_fixture"):
        document = sample_dir / "service_notes.txt"
        document.write_text(
            "\n".join(
                [
                    "service=payment-api",
                    "owner=AgentMem-Team",
                    "region=cn-north-4",
                    "runbook=RB-2048",
                    "status=active",
                ]
            ),
            encoding="utf-8",
        )
        replacements["{document_path}"] = str(document)

    return _replace_placeholders(raw, replacements)


def _write_payload_fixture(
    path: Path,
    *,
    size: int,
    marker: str,
    marker_position: str = "end",
) -> None:
    """Create deterministic filler with one marker at end or middle."""
    if marker_position not in {"end", "middle"}:
        raise ValueError("fixture_marker_position must be 'end' or 'middle'")
    critical = (
        f"\nFINAL_EVIDENCE marker={marker} request_id=req-7319 "
        "root_cause=connection_pool_exhausted\n"
    ).encode("utf-8")
    if len(critical) > size:
        raise ValueError("fixture_size_bytes is too small for FINAL_EVIDENCE")
    filler_lines = []
    index = 0
    written = 0
    while written < size:
        line = (
            f"record={index:06d} status=ok latency_ms={20 + index % 17} "
            f"checksum={(index * 2654435761) & 0xFFFFFFFF:08x}\n"
        ).encode("utf-8")
        filler_lines.append(line)
        written += len(line)
        index += 1
    filler = b"".join(filler_lines)[:size]
    insertion = (
        size - len(critical)
        if marker_position == "end"
        else (size - len(critical)) // 2
    )
    payload = filler[:insertion] + critical + filler[insertion + len(critical) :]
    path.write_bytes(payload)


def _write_incident_fixture(sample_dir: Path) -> dict[str, str]:
    log_path = sample_dir / "incident.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-07-18T14:03:20 INFO code=REQUEST_START "
                "request_id=req-7319 node=api-1",
                "2026-07-18T14:03:24 WARN code=POOL_PRESSURE "
                "request_id=req-7319 node=db-worker-2 active=48",
                "2026-07-18T14:03:27 ERROR code=DB_POOL_EXHAUSTED "
                "request_id=req-7319 node=db-worker-2",
                "2026-07-18T14:03:28 ERROR code=PAYMENT_FAILED "
                "request_id=req-7319 node=payment-api",
                "2026-07-18T14:08:02 INFO code=POOL_RECOVERED "
                "request_id=req-7319 node=db-worker-2",
            ]
        ),
        encoding="utf-8",
    )

    db_path = sample_dir / "incidents.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE incidents (
                request_id TEXT PRIMARY KEY,
                service TEXT NOT NULL,
                severity TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                runbook TEXT NOT NULL,
                resolved INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO incidents VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "req-7319",
                    "payment-api",
                    "critical",
                    "connection_pool_exhausted",
                    "RB-2048",
                    1,
                ),
                (
                    "req-1002",
                    "catalog-api",
                    "warning",
                    "cache_miss_spike",
                    "RB-1001",
                    1,
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    config_path = sample_dir / "service.conf"
    config_path.write_text(
        "\n".join(
            [
                "service=payment-api",
                "db_pool_size=48",
                "db_pool_timeout_ms=3000",
                "runbook=RB-2048",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "{log_path}": str(log_path),
        "{db_path}": str(db_path),
        "{config_path}": str(config_path),
    }


def _replace_placeholders(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for marker, replacement in replacements.items():
            value = value.replace(marker, replacement)
        return value
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_placeholders(item, replacements)
            for key, item in value.items()
        }
    return value
