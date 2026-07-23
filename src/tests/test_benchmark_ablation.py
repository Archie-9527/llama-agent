from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agent_core.benchmark.ablation import (
    ROUND_MEMORY_OVERRIDES,
    run_r0_r2_ablation,
)
from agent_core.benchmark.comparison import write_comparison_report


def _summary(round_name: str) -> dict:
    optimized = round_name != "R0"
    lifecycle = round_name == "R2"
    return {
        "sample_count": 1,
        "passed_count": 1,
        "task_success_rate": 1.0,
        "duration_ms": {"mean": 1200 if optimized else 1500, "p95": 1200},
        "inference_call_count": 2,
        "input_tokens": {"total": 700 if optimized else 1000},
        "peak_logical_kv_tokens": 600 if optimized else 900,
        "peak_rss_bytes": 1024**3,
        "peak_gpu_process_bytes": 512 * 1024**2,
        "checkpoint_bytes": 2000 if lifecycle else 3000,
        "virtualized_tool_result_count": 1 if optimized else 0,
        "externalized_tool_output_bytes": 16000 if optimized else 0,
        "artifact_bytes_saved": 12000 if optimized else 0,
        "artifact_reduction_ratio": 0.75 if optimized else 0.0,
        "context_reduction_ratio": 0.4 if lifecycle else 0.0,
        "context_compaction_count": 2 if lifecycle else 0,
        "context_compaction_bytes_saved": 1000 if lifecycle else 0,
        "context_recalled_turns": 1 if lifecycle else 0,
        "context_selected_turns": 3 if lifecycle else 0,
        "context_recalled_tokens": 120 if lifecycle else 0,
    }


def _make_run(tmp_path: Path, round_name: str) -> Path:
    run_dir = tmp_path / round_name
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "round": round_name,
                "suite": {
                    "name": "same-suite",
                    "warmup_runs": 0,
                    "measured_runs": 1,
                    "case_filter": ["case-one"],
                },
                "model": {"sha256": "same-model"},
                "engine_config": {"n_ctx": 4096},
                "environment": {"dirty_worktree": False},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(_summary(round_name)),
        encoding="utf-8",
    )
    (run_dir / "task_results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-one",
                "category": "test",
                "measured": True,
                "repetition": 0,
                "duration_ms": 1000,
                "status": "done",
                "evaluation": {"passed": True, "checks": {"status": True}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "inference_events.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-one",
                "sample": "rep-000",
                "event": "inference_completed",
                "input_tokens": 100,
                "output_tokens": 20,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (run_dir / "kv_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case_id", "sample", "logical_tokens"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case-one",
                "sample": "rep-000",
                "logical_tokens": 80,
            }
        )
    return run_dir


def test_ablation_policy_focuses_only_r1_and_r2():
    assert ROUND_MEMORY_OVERRIDES["R0"] == {
        "artifact_virtualization": False,
        "lifecycle_context": False,
        "kv_lifecycle": False,
        "branch_management": False,
    }
    assert ROUND_MEMORY_OVERRIDES["R1"]["artifact_virtualization"] is True
    assert ROUND_MEMORY_OVERRIDES["R1"]["lifecycle_context"] is False
    assert ROUND_MEMORY_OVERRIDES["R2"]["artifact_virtualization"] is True
    assert ROUND_MEMORY_OVERRIDES["R2"]["lifecycle_context"] is True
    assert all(
        not config["kv_lifecycle"] and not config["branch_management"]
        for config in ROUND_MEMORY_OVERRIDES.values()
    )


def test_comparison_report_writes_data_markdown_and_svg_charts(tmp_path: Path):
    run_dirs = {
        round_name: _make_run(tmp_path, round_name)
        for round_name in ("R0", "R1", "R2")
    }

    report = write_comparison_report(tmp_path / "comparison", run_dirs)

    assert report.exists()
    markdown = report.read_text(encoding="utf-8")
    assert "R0–R2 自动消融测试报告" in markdown
    assert "R3/KV 生命周期与分支管理" in markdown
    assert "R1 相对 R0" in markdown
    assert (report.parent / "comparison_data.json").exists()
    assert (report.parent / "charts" / "success_rate.svg").exists()
    assert (report.parent / "charts" / "latency_mean.svg").exists()
    assert "<svg" in (
        report.parent / "charts" / "input_tokens.svg"
    ).read_text(encoding="utf-8")


def test_ablation_records_failed_status_if_a_round_crashes(
    tmp_path: Path,
    monkeypatch,
):
    class FakeRunner:
        def __init__(self, **kwargs):
            self.round_name = kwargs["round_name"]
            self.output_root = kwargs["output_root"]

        def run(self) -> Path:
            if self.round_name == "R1":
                raise RuntimeError("synthetic failure")
            run_dir = self.output_root / self.round_name
            run_dir.mkdir()
            return run_dir

    monkeypatch.setattr(
        "agent_core.benchmark.ablation.BenchmarkRunner",
        FakeRunner,
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_r0_r2_ablation(
            config_file=tmp_path / "agent_config.toml",
            suite_file=tmp_path / "suite.json",
            output_root=tmp_path / "results",
        )

    manifests = list((tmp_path / "results").glob("*/ablation_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["run_dirs"].keys() == {"R0"}
    assert "synthetic failure" in manifest["error"]
