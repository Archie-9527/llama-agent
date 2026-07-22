from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.artifacts.runtime import (
    _reset_artifact_stores_for_testing,
    get_artifact_store,
    initialize_artifact_store,
)
from agent_core.artifacts.virtualizer import (
    _reset_artifact_virtualizer_for_testing,
    initialize_artifact_virtualizer,
    maybe_virtualize_tool_result,
)
from agent_core.benchmark.aggregator import aggregate_run
from agent_core.benchmark.models import BenchmarkSuite
from agent_core.capability_registry import Capability
from agent_core.config import MemoryConfig, load_memory_config
from agent_core.graph.react_agent_factory import _run_capability
from agent_core.telemetry import telemetry_task


@pytest.fixture(autouse=True)
def _reset_r1_runtime():
    _reset_artifact_virtualizer_for_testing()
    _reset_artifact_stores_for_testing()
    yield
    _reset_artifact_virtualizer_for_testing()
    _reset_artifact_stores_for_testing()


def test_disabled_virtualizer_preserves_r0_result_identity(tmp_path: Path):
    initialize_artifact_store(tmp_path / "artifacts")
    initialize_artifact_virtualizer(
        MemoryConfig(
            artifact_virtualization=False,
            artifact_inline_max_bytes=1,
        )
    )
    result = {"success": True, "content": "x" * 10_000}

    assert maybe_virtualize_tool_result("read_file", result) is result


def test_large_result_is_externalized_and_retrievable(tmp_path: Path):
    initialize_artifact_store(tmp_path / "artifacts")
    initialize_artifact_virtualizer(
        MemoryConfig(
            artifact_virtualization=True,
            artifact_inline_max_bytes=128,
            artifact_preview_chars=120,
            artifact_summary_chars=200,
        )
    )
    original = json.dumps(
        {
            "success": True,
            "path": "payload.txt",
            "content": "head\n" + "x" * 1000 + "\nFINAL_EVIDENCE=R1_OK",
        },
        ensure_ascii=False,
    )

    with telemetry_task("task-r1"):
        inline = maybe_virtualize_tool_result("read_file", original)
        descriptor = json.loads(str(inline))
        stored = get_artifact_store().retrieve(
            descriptor["artifact_id"],
            owner_id="task-r1",
            offset=0,
            length=len(original.encode("utf-8")),
        )

    assert descriptor["virtualized"] is True
    assert descriptor["original_bytes"] == len(original.encode("utf-8"))
    assert descriptor["artifact_id"].startswith("artifact://")
    assert "FINAL_EVIDENCE=R1_OK" in descriptor["tail_preview"]
    assert stored["content"] == original


def test_artifact_access_tools_are_never_virtualized(tmp_path: Path):
    initialize_artifact_store(tmp_path / "artifacts")
    initialize_artifact_virtualizer(
        MemoryConfig(
            artifact_virtualization=True,
            artifact_inline_max_bytes=1,
        )
    )
    result = "search result " * 100

    assert maybe_virtualize_tool_result("search_artifact", result) is result


def test_central_capability_bridge_applies_r1_policy(tmp_path: Path):
    initialize_artifact_store(tmp_path / "artifacts")
    initialize_artifact_virtualizer(
        MemoryConfig(
            artifact_virtualization=True,
            artifact_inline_max_bytes=64,
            artifact_preview_chars=80,
        )
    )
    capability = Capability(
        name="read_file",
        description="test",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: "A" * 500,
    )

    with telemetry_task("bridge-task"):
        result = _run_capability(capability, {})

    assert json.loads(str(result))["virtualized"] is True


def test_memory_config_supports_file_and_environment_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[memory]
artifact_virtualization = false
artifact_inline_max_bytes = 4096
artifact_preview_chars = 800
artifact_summary_chars = 300
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_MEMORY_ARTIFACT_VIRTUALIZATION", "true")
    monkeypatch.setenv("AGENT_MEMORY_ARTIFACT_INLINE_MAX_BYTES", "2048")

    config = load_memory_config(path)

    assert config.artifact_virtualization is True
    assert config.artifact_inline_max_bytes == 2048
    assert config.artifact_preview_chars == 800


def test_r1_suite_is_valid_and_covers_preview_and_on_demand():
    suite = BenchmarkSuite.load(
        Path("benchmark/workloads/r1_artifact_virtualization.json")
    )

    assert len(suite.cases) == 3
    assert {case.category for case in suite.cases} == {
        "R1-preview",
        "R1-on-demand",
    }
    assert "search_artifact" in suite.cases[-1].expected_tools


def test_aggregator_reports_r1_byte_reduction(tmp_path: Path):
    (tmp_path / "task_results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "r1",
                "category": "R1-preview",
                "measured": True,
                "repetition": 0,
                "duration_ms": 10,
                "evaluation": {"passed": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "tool_events.jsonl").write_text(
        json.dumps(
            {
                "case_id": "r1",
                "sample": "rep-000",
                "event": "artifact_virtualized",
                "original_bytes": 10_000,
                "inline_bytes": 1_000,
                "bytes_saved": 9_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_run(tmp_path)

    assert summary["virtualized_tool_result_count"] == 1
    assert summary["externalized_tool_output_bytes"] == 10_000
    assert summary["artifact_bytes_saved"] == 9_000
    assert summary["artifact_reduction_ratio"] == pytest.approx(0.9)
    assert summary["artifact_storage_bytes"] == 0
