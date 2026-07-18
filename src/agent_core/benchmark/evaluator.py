"""Deterministic benchmark correctness checks."""

from __future__ import annotations

from typing import Any

from agent_core.benchmark.models import BenchmarkCase


def evaluate(case: BenchmarkCase, result: dict[str, Any]) -> dict[str, Any]:
    final_answer = str(result.get("final_answer", ""))
    used_tools = {
        str(record.get("tool_used"))
        for record in result.get("execution_log", [])
        if record.get("tool_used")
    }
    checks = {
        "status": result.get("status") == case.expected_status,
        "answer_contains": all(
            expected.casefold() in final_answer.casefold()
            for expected in case.expected_contains
        ),
        "tools": set(case.expected_tools).issubset(used_tools),
        "forbidden_tools": set(case.forbidden_tools).isdisjoint(used_tools),
        "final_answer_nonempty": (
            bool(final_answer.strip()) if case.expected_status == "done" else True
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "used_tools": sorted(used_tools),
    }
