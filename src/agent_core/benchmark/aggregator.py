"""Aggregate raw benchmark results without hiding failures."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "stdev": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_run(run_dir: Path) -> dict[str, Any]:
    results = []
    path = run_dir / "task_results.jsonl"
    if path.exists():
        results = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    measured = [item for item in results if item.get("measured", True)]
    duration = [float(item["duration_ms"]) for item in measured]
    successes = [item for item in measured if item.get("evaluation", {}).get("passed")]

    by_category: dict[str, dict[str, int]] = {}
    for item in measured:
        category = item["category"]
        bucket = by_category.setdefault(category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(bool(item.get("evaluation", {}).get("passed")))

    measured_keys = {
        (str(item["case_id"]), f"rep-{int(item.get('repetition', 0)):03d}")
        for item in measured
    }
    rss_values: list[float] = []
    kv_tokens: list[float] = []
    for csv_path in run_dir.glob("cases/**/system_memory.csv"):
        parts = csv_path.relative_to(run_dir / "cases").parts
        if (parts[0], parts[1]) not in measured_keys:
            continue
        with csv_path.open(encoding="utf-8") as handle:
            rss_values.extend(
                float(row["rss_bytes"])
                for row in csv.DictReader(handle)
                if row.get("rss_bytes") not in ("", None)
            )
    for csv_path in run_dir.glob("cases/**/kv_metrics.csv"):
        parts = csv_path.relative_to(run_dir / "cases").parts
        if (parts[0], parts[1]) not in measured_keys:
            continue
        with csv_path.open(encoding="utf-8") as handle:
            kv_tokens.extend(
                float(row["logical_tokens"])
                for row in csv.DictReader(handle)
                if row.get("logical_tokens") not in ("", None)
            )
    inference_events = _measured_jsonl(
        run_dir / "inference_events.jsonl", measured_keys
    )
    tool_events = _measured_jsonl(run_dir / "tool_events.jsonl", measured_keys)
    input_tokens = [
        float(item.get("input_tokens", 0))
        for item in inference_events
        if item.get("event") == "inference_completed"
    ]
    output_tokens = [
        float(item.get("output_tokens", 0))
        for item in inference_events
        if item.get("event") == "inference_completed"
    ]
    prompt_eval_ms = [
        float(item["prompt_eval_ms"])
        for item in inference_events
        if item.get("prompt_eval_ms") is not None
    ]
    tool_output_bytes = sum(
        int(item.get("output_bytes", 0))
        for item in tool_events
        if item.get("event") == "tool_completed"
    )
    virtualization_events = [
        item
        for item in tool_events
        if item.get("event") == "artifact_virtualized"
    ]
    externalized_tool_output_bytes = sum(
        int(item.get("original_bytes", 0)) for item in virtualization_events
    )
    virtualized_inline_bytes = sum(
        int(item.get("inline_bytes", 0)) for item in virtualization_events
    )
    artifact_bytes_saved = sum(
        int(item.get("bytes_saved", 0)) for item in virtualization_events
    )

    process_gpu_values: list[float] = []
    root_memory = run_dir / "system_memory.csv"
    if root_memory.exists():
        with root_memory.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("case_id"), row.get("sample")) not in measured_keys:
                    continue
                if row.get("process_used_bytes") not in ("", None):
                    process_gpu_values.append(float(row["process_used_bytes"]))

    return {
        "sample_count": len(measured),
        "passed_count": len(successes),
        "failed_count": len(measured) - len(successes),
        "task_success_rate": len(successes) / len(measured) if measured else 0.0,
        "duration_ms": _distribution(duration),
        "peak_rss_bytes": max(rss_values) if rss_values else None,
        "peak_logical_kv_tokens": max(kv_tokens) if kv_tokens else None,
        "inference_call_count": len(input_tokens),
        "input_tokens": {
            "total": int(sum(input_tokens)),
            **_distribution(input_tokens),
        },
        "output_tokens": {
            "total": int(sum(output_tokens)),
            **_distribution(output_tokens),
        },
        "prompt_eval_ms": _distribution(prompt_eval_ms),
        "tool_output_bytes": tool_output_bytes,
        "virtualized_tool_result_count": len(virtualization_events),
        "externalized_tool_output_bytes": externalized_tool_output_bytes,
        "virtualized_inline_bytes": virtualized_inline_bytes,
        "artifact_bytes_saved": artifact_bytes_saved,
        "artifact_reduction_ratio": (
            artifact_bytes_saved / externalized_tool_output_bytes
            if externalized_tool_output_bytes
            else 0.0
        ),
        "artifact_storage_bytes": sum(
            path.stat().st_size
            for path in run_dir.glob("cases/**/artifacts/**/*")
            if path.is_file()
        ),
        "peak_gpu_process_bytes": (
            max(process_gpu_values) if process_gpu_values else None
        ),
        "checkpoint_bytes": sum(
            path.stat().st_size
            for path in run_dir.glob("cases/**/checkpoints.sqlite")
        ),
        "conversation_bytes": sum(
            path.stat().st_size
            for path in run_dir.glob("cases/**/conversations.sqlite")
        ),
        "categories": by_category,
    }


def _measured_jsonl(
    path: Path, measured_keys: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        record
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
        if (record.get("case_id"), record.get("sample")) in measured_keys
    ]
