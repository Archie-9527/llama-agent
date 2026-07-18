"""Benchmark suite and result contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    goal: str = ""
    turns: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    concurrency: int = 1
    expected_status: str = "done"
    expected_contains: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(raw["case_id"]),
            category=str(raw.get("category", "general")),
            goal=str(raw.get("goal", "")),
            turns=tuple(str(item) for item in raw.get("turns", [])),
            tasks=tuple(str(item) for item in raw.get("tasks", [])),
            concurrency=int(raw.get("concurrency", 1)),
            expected_status=str(raw.get("expected_status", "done")),
            expected_contains=tuple(
                str(item) for item in raw.get("expected_contains", [])
            ),
            expected_tools=tuple(
                str(item) for item in raw.get("expected_tools", [])
            ),
            forbidden_tools=tuple(
                str(item) for item in raw.get("forbidden_tools", [])
            ),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "goal": self.goal,
            "turns": list(self.turns),
            "tasks": list(self.tasks),
            "concurrency": self.concurrency,
            "expected_status": self.expected_status,
            "expected_contains": list(self.expected_contains),
            "expected_tools": list(self.expected_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    cases: tuple[BenchmarkCase, ...]
    warmup_runs: int = 1
    measured_runs: int = 5
    seed: int = 42

    @classmethod
    def load(cls, path: Path) -> "BenchmarkSuite":
        raw = json.loads(path.read_text(encoding="utf-8"))
        suite = cls(
            name=str(raw.get("name", path.stem)),
            cases=tuple(BenchmarkCase.from_dict(item) for item in raw["cases"]),
            warmup_runs=int(raw.get("warmup_runs", 1)),
            measured_runs=int(raw.get("measured_runs", 5)),
            seed=int(raw.get("seed", 42)),
        )
        if not suite.cases:
            raise ValueError("benchmark suite must contain at least one case")
        if suite.measured_runs < 1 or suite.warmup_runs < 0:
            raise ValueError("invalid warmup_runs/measured_runs")
        return suite
