from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agent_core.benchmark.aggregator import aggregate_run
from agent_core.benchmark.evaluator import evaluate
from agent_core.benchmark.models import BenchmarkCase, BenchmarkSuite
from agent_core.benchmark.runner import _filter_suite_cases, _prepare_case
from agent_core.benchmark.worker import _run_conversation_case
from agent_core.conversation.models import Turn
from agent_core.conversation.manager import ConversationConfig, ConversationManager
from agent_core.conversation.store import ConversationStore
from agent_core.telemetry.kv_monitor import sample_kv
from agent_core.telemetry.models import TelemetryConfig
from agent_core.graph.finalizer import finalizer_node
from agent_core.llm_engine import (
    _append_no_think_marker,
    _convert_llama_response_to_aimessage,
    _strip_thinking_content,
)
from agent_core.config import load_memory_config, load_telemetry_config


def test_conversation_store_running_then_done(tmp_path: Path):
    store = ConversationStore(tmp_path / "conversations.sqlite")
    try:
        store.create_conversation("c1")
        running = store.create_running_turn(
            turn_id="t1",
            conversation_id="c1",
            thread_id="thread1",
            user_input="hello",
        )
        assert running.status == "running"
        done = store.finish_turn(
            "t1", status="done", assistant_output="world"
        )
        assert done.assistant_output == "world"
        assert store.get_turn_by_thread("thread1") == done
    finally:
        store.close()


def test_conversation_history_is_recent_and_budgeted(tmp_path: Path):
    store = ConversationStore(tmp_path / "conversations.sqlite")
    store.create_conversation("c1")
    for index in range(3):
        store.create_running_turn(
            turn_id=f"t{index}",
            conversation_id="c1",
            thread_id=f"thread{index}",
            user_input=f"u{index}",
        )
        store.finish_turn(
            f"t{index}", status="done", assistant_output=f"a{index}"
        )
    runner = MagicMock()
    engine = MagicMock()
    engine.get_num_tokens.side_effect = lambda text: len(text)
    manager = ConversationManager(
        runner,
        store,
        engine,
        ConversationConfig(history_turns=2, history_token_budget=10_000),
    )
    history = manager._render_history("c1")
    assert "u0" not in history
    assert "u1" in history and "u2" in history
    store.close()


def test_failed_turn_keeps_user_fact_but_not_assistant_output(tmp_path: Path):
    store = ConversationStore(tmp_path / "conversations.sqlite")
    store.create_conversation("c1")
    store.create_running_turn(
        turn_id="t1",
        conversation_id="c1",
        thread_id="thread1",
        user_input="项目代号是 AgentMem",
    )
    store.finish_turn(
        "t1",
        status="failed",
        assistant_output="错误答案是 OtherProject",
        error="empty response",
    )
    engine = MagicMock()
    engine.get_num_tokens.side_effect = lambda text: len(text)
    manager = ConversationManager(MagicMock(), store, engine)
    history = manager._render_history("c1")
    assert "AgentMem" in history
    assert "OtherProject" not in history
    assert "执行失败" in history
    store.close()


def test_benchmark_conversation_stops_after_first_failed_turn():
    manager = MagicMock()
    failed_turn = Turn(
        turn_id="t1",
        conversation_id="c1",
        thread_id="thread1",
        turn_index=0,
        user_input="记住 AgentMem",
        assistant_output="",
        status="failed",
        error="empty response",
        created_at="now",
        updated_at="now",
    )
    failed_result = {"status": "failed", "error": "empty response"}
    manager.start.return_value = ("c1", failed_turn, failed_result)

    result, turns, failed_index = _run_conversation_case(
        manager,
        ("记住 AgentMem", "上一轮的代号是什么？"),
    )

    assert result["status"] == failed_result["status"]
    assert result["error"] == failed_result["error"]
    assert result["execution_log"] == []
    assert failed_index == 0
    assert turns[0]["error"] == "empty response"
    assert turns[1]["status"] == "skipped_due_to_previous_failure"
    manager.continue_conversation.assert_not_called()


def test_kv_snapshot_falls_back_cleanly():
    class Client:
        n_tokens = 12

        @staticmethod
        def n_ctx():
            return 128

    client = Client()
    snapshot = sample_kv(client)
    assert snapshot.logical_tokens == 12
    assert snapshot.capacity_tokens == 128
    assert isinstance(snapshot.supported, bool)


def test_telemetry_config_rejects_busy_sampling():
    config = TelemetryConfig(sample_interval_ms=1)
    try:
        config.validate()
    except ValueError as exc:
        assert "sample_interval_ms" in str(exc)
    else:
        raise AssertionError("expected validation error")


def test_benchmark_suite_and_evaluator(tmp_path: Path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "tiny",
                "warmup_runs": 0,
                "measured_runs": 1,
                "cases": [
                    {
                        "case_id": "one",
                        "category": "W1",
                        "goal": "count",
                        "expected_contains": ["42"],
                        "expected_tools": ["count_lines"],
                        "forbidden_tools": ["query_sqlite"],
                    }
                ],
            }
        )
    )
    suite = BenchmarkSuite.load(suite_path)
    result = {
        "status": "done",
        "final_answer": "answer is 42",
        "execution_log": [{"tool_used": "count_lines"}],
    }
    assert evaluate(suite.cases[0], result)["passed"] is True


def test_benchmark_case_filter_preserves_suite_order():
    suite = BenchmarkSuite(
        name="filter-test",
        warmup_runs=0,
        measured_runs=1,
        cases=(
            BenchmarkCase(case_id="a", category="W1"),
            BenchmarkCase(case_id="b", category="W2"),
            BenchmarkCase(case_id="c", category="W3"),
        ),
    )
    filtered = _filter_suite_cases(suite, ("c", "a"))
    assert [case.case_id for case in filtered.cases] == ["a", "c"]


def test_benchmark_case_filter_rejects_unknown_id():
    suite = BenchmarkSuite(
        name="filter-test",
        cases=(BenchmarkCase(case_id="known", category="W1"),),
    )
    with pytest.raises(ValueError, match="Unknown benchmark case_id"):
        _filter_suite_cases(suite, ("missing",))


def test_evaluator_checks_tool_order_and_count():
    case = BenchmarkCase(
        case_id="ordered",
        category="tools",
        expected_tool_sequence=("search_log", "query_sqlite"),
        min_tool_calls=2,
        max_tool_calls=2,
    )
    correct = {
        "status": "done",
        "final_answer": "complete",
        "execution_log": [
            {"tool_used": "search_log"},
            {"tool_used": "query_sqlite"},
        ],
    }
    reversed_result = {
        **correct,
        "execution_log": list(reversed(correct["execution_log"])),
    }
    assert evaluate(case, correct)["passed"] is True
    assert evaluate(case, reversed_result)["checks"]["tool_sequence"] is False


def test_evaluator_enforces_tool_limits_per_conversation_turn():
    case = BenchmarkCase(
        case_id="turn-tools",
        category="conversation",
        expected_contains=("req-7319",),
        max_tool_calls=2,
        max_tool_calls_per_turn=(2, 0, 0),
    )
    result = {
        "status": "done",
        "final_answer": "req-7319",
        "execution_log": [
            {"tool_used": "search_log"},
            {"tool_used": "get_log_window"},
        ],
        "turn_results": [
            {
                "execution_log": [
                    {"tool_used": "search_log"},
                    {"tool_used": "get_log_window"},
                ]
            },
            {"execution_log": []},
            {"execution_log": []},
        ],
    }
    assert evaluate(case, result)["passed"] is True

    result["turn_results"][2]["execution_log"] = [
        {"tool_used": "search_log"}
    ]
    assert (
        evaluate(case, result)["checks"]["max_tool_calls_per_turn"] is False
    )


def test_prepare_case_builds_deterministic_local_evidence(tmp_path: Path):
    case = BenchmarkCase(
        case_id="fixture",
        category="fixture",
        goal="inspect {log_path} {db_path} {config_path} {fixture_path}",
        metadata={
            "incident_fixture": True,
            "fixture_size_bytes": 4096,
            "fixture_marker": "ONLY_ONCE",
        },
    )
    raw = _prepare_case(case, tmp_path / "sample")
    assert "{" not in raw["goal"]
    payload = (tmp_path / "sample" / "payload.txt").read_text()
    assert payload.count("ONLY_ONCE") == 1
    assert payload.endswith("root_cause=connection_pool_exhausted\n")
    with sqlite3.connect(tmp_path / "sample" / "incidents.sqlite") as connection:
        row = connection.execute(
            "SELECT root_cause FROM incidents WHERE request_id='req-7319'"
        ).fetchone()
    assert row == ("connection_pool_exhausted",)


def test_prepare_case_can_place_evidence_in_middle(tmp_path: Path):
    case = BenchmarkCase(
        case_id="middle-fixture",
        category="fixture",
        goal="inspect {fixture_path}",
        metadata={
            "fixture_size_bytes": 8192,
            "fixture_marker": "MIDDLE_ONLY",
            "fixture_marker_position": "middle",
        },
    )

    _prepare_case(case, tmp_path / "sample")
    payload = (tmp_path / "sample" / "payload.txt").read_text()

    assert len(payload.encode("utf-8")) == 8192
    assert payload.count("MIDDLE_ONLY") == 1
    assert "MIDDLE_ONLY" not in payload[:1000]
    assert "MIDDLE_ONLY" not in payload[-1000:]


def test_aggregator_keeps_failed_samples(tmp_path: Path):
    records = [
        {
            "case_id": "a",
            "category": "W1",
            "duration_ms": 10,
            "measured": True,
            "repetition": 0,
            "evaluation": {"passed": True},
        },
        {
            "case_id": "b",
            "category": "W1",
            "duration_ms": 20,
            "measured": True,
            "repetition": 0,
            "evaluation": {"passed": False},
        },
    ]
    (tmp_path / "task_results.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records)
    )
    summary = aggregate_run(tmp_path)
    assert summary["sample_count"] == 2
    assert summary["failed_count"] == 1
    assert summary["task_success_rate"] == 0.5


def test_case_round_trip():
    case = BenchmarkCase(
        case_id="c",
        category="W5",
        turns=("one", "two"),
        tasks=("a", "b"),
        concurrency=2,
    )
    assert BenchmarkCase.from_dict(case.to_dict()) == case


def test_finalizer_sets_stable_answer():
    engine = MagicMock()
    engine.invoke.return_value.content = "完整最终答案"
    state = {
        "task_goal": "goal",
        "plan_steps": ["step", "step 2"],
        "execution_log": [{"step": "step", "result": "ok", "tool_used": None}],
        "status": "done",
        "final_answer": "",
    }
    with (
        patch("agent_core.graph.finalizer.get_engine", return_value=engine),
        patch(
            "agent_core.graph.finalizer.assemble_finalization_prompt",
            return_value=[],
        ),
    ):
        result = finalizer_node(state)
    assert result["final_answer"] == "完整最终答案"


def test_llama_usage_is_preserved():
    message = _convert_llama_response_to_aimessage(
        {
            "model": "local",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        }
    )
    assert message.usage_metadata["input_tokens"] == 10
    assert message.response_metadata["finish_reason"] == "stop"


def test_thinking_blocks_are_not_user_visible():
    visible, reasoning = _strip_thinking_content(
        "<think>private reasoning</think>\n文件共有 698 行。"
    )
    assert visible == "文件共有 698 行。"
    assert reasoning == "private reasoning"


def test_unfinished_thinking_is_not_an_answer():
    visible, reasoning = _strip_thinking_content(
        "<think>Thinking Process: generation was truncated"
    )
    assert visible == ""
    assert "generation was truncated" in reasoning


def test_no_think_marker_does_not_mutate_input():
    messages = [{"role": "user", "content": "hello"}]
    output = _append_no_think_marker(messages)
    assert output[-1]["content"].endswith("/no_think")
    assert messages[-1]["content"] == "hello"


def test_stage1_config_sections_load(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[telemetry]
enabled = true
sample_interval_ms = 250
[memory]
artifact_virtualization = false
kv_lifecycle = false
"""
    )
    assert load_telemetry_config(config_file).enabled is True
    assert load_memory_config(config_file).kv_lifecycle is False
