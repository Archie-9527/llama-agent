"""Deterministic benchmark correctness checks."""

from __future__ import annotations

from typing import Any

from agent_core.benchmark.models import BenchmarkCase


def evaluate(case: BenchmarkCase, result: dict[str, Any]) -> dict[str, Any]:
    final_answer = str(result.get("final_answer", ""))
    tool_trace = [
        str(record.get("tool_used"))
        for record in result.get("execution_log", [])
        if record.get("tool_used")
    ]
    used_tools = {
        str(record.get("tool_used"))
        for record in result.get("execution_log", [])
        if record.get("tool_used")
    }
    tool_evidence = "\n".join(
        str(record.get("result", ""))
        for record in result.get("execution_log", [])
        if record.get("tool_used")
    )
    checks = {
        "status": result.get("status") == case.expected_status,
        "answer_contains": all(
            expected.casefold() in final_answer.casefold()
            for expected in case.expected_contains
        ),
        "tool_result_contains": all(
            expected.casefold() in tool_evidence.casefold()
            for expected in case.expected_tool_result_contains
        ),
        "tools": set(case.expected_tools).issubset(used_tools),
        "tool_sequence": _is_subsequence(
            list(case.expected_tool_sequence), tool_trace
        ),
        "forbidden_tools": set(case.forbidden_tools).isdisjoint(used_tools),
        "min_tool_calls": len(tool_trace) >= case.min_tool_calls,
        "max_tool_calls": (
            len(tool_trace) <= case.max_tool_calls
            if case.max_tool_calls is not None
            else True
        ),
        "final_answer_nonempty": (
            bool(final_answer.strip()) if case.expected_status == "done" else True
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "used_tools": sorted(used_tools),
        "tool_trace": tool_trace,
    }


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return True
    cursor = iter(actual)
    return all(any(item == wanted for item in cursor) for wanted in expected)
