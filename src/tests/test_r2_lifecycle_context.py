from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from agent_core.config import MemoryConfig, load_memory_config
from agent_core.benchmark.aggregator import aggregate_run
from agent_core.benchmark.models import BenchmarkSuite
from agent_core.benchmark.runner import _prepare_case
from agent_core.conversation.manager import ConversationManager
from agent_core.conversation.store import ConversationStore
from agent_core.memory.context_manager import (
    _reset_lifecycle_context_for_testing,
    get_lifecycle_context_manager,
    initialize_lifecycle_context,
)
from agent_core.memory.models import Lifecycle
from agent_core.memory.store import ContextStore
from agent_core.prompt_assembler import assemble_reflection_prompt


class _TokenEngine:
    n_ctx = 20000

    @staticmethod
    def get_num_tokens(text: str) -> int:
        return max(1, len(str(text).encode("utf-8")) // 4)


@pytest.fixture(autouse=True)
def _reset_runtime():
    _reset_lifecycle_context_for_testing()
    yield
    _reset_lifecycle_context_for_testing()


def _enabled_config(tmp_path: Path, **overrides) -> MemoryConfig:
    return replace(
        MemoryConfig(),
        lifecycle_context=True,
        context_store_path=tmp_path / "context.sqlite",
        **overrides,
    )


def test_memory_config_loads_r2_fields(tmp_path: Path):
    path = tmp_path / "agent.toml"
    path.write_text(
        """
[memory]
lifecycle_context = true
context_store_path = "data/custom-context.sqlite"
context_budget_tokens = 9000
context_trigger_ratio = 0.8
hot_execution_records = 3
summary_mode = "deterministic"
checkpoint_compaction = false
""",
        encoding="utf-8",
    )
    config = load_memory_config(path)
    assert config.lifecycle_context is True
    assert config.context_store_path == Path("data/custom-context.sqlite")
    assert config.context_budget_tokens == 9000
    assert config.context_trigger_ratio == 0.8
    assert config.hot_execution_records == 3
    assert config.summary_mode == "deterministic"
    assert config.checkpoint_compaction is False


def test_context_store_enforces_owner_isolation(tmp_path: Path):
    store = ContextStore(tmp_path / "context.sqlite")
    item = store.put(
        owner_type="conversation",
        owner_id="conversation-a",
        kind="fact",
        lifecycle=Lifecycle.PINNED,
        content="项目代号是 AgentMem",
        token_count=8,
        source_type="turn",
        source_id="turn-1",
    )
    assert store.get(
        item.item_id,
        owner_type="conversation",
        owner_id="conversation-a",
    ).content == "项目代号是 AgentMem"
    with pytest.raises(KeyError):
        store.get(
            item.item_id,
            owner_type="conversation",
            owner_id="conversation-b",
        )
    store.close()


def test_commit_archives_and_compacts_old_execution_records(tmp_path: Path):
    manager = initialize_lifecycle_context(
        _enabled_config(
            tmp_path,
            hot_execution_records=2,
            summary_target_chars=80,
        )
    )
    state = {
        "task_goal": "分析本地证据",
        "current_user_input": "记住项目代号是 AgentMem",
        "execution_log": [
            {
                "step": f"step-{index}",
                "result": "payload-" + str(index) * 500,
                "tool_used": "read_file",
            }
            for index in range(5)
        ],
        "reflection_notes": [],
        "pinned_facts": [],
        "context_summary": {},
        "archived_context_ids": [],
        "lifecycle_stats": {},
    }
    manager.commit(state, event="executor")
    assert all(record.get("_r2_compacted") for record in state["execution_log"][:3])
    assert all(not record.get("_r2_compacted") for record in state["execution_log"][-2:])
    assert len(state["execution_log"][0]["result"]) < 150
    assert len(state["archived_context_ids"]) == 3
    assert len(manager.store.list_owner("task", "__unscoped__")) == 3
    assert state["pinned_facts"][0]["text"] == "记住项目代号是 AgentMem"


def test_conversation_selection_keeps_pinned_and_recent_turns(tmp_path: Path):
    manager = initialize_lifecycle_context(
        _enabled_config(
            tmp_path,
            hot_conversation_turns=2,
            context_retrieval_top_k=1,
            context_retrieval_token_budget=4000,
        )
    )
    turns = [
        SimpleNamespace(
            turn_id=f"turn-{index}",
            turn_index=index,
            user_input=(
                "记住项目代号是 AgentMem，校验码是 7319"
                if index == 0
                else f"无关占位问题 {index}"
            ),
            assistant_output="已记录" if index == 0 else f"ACK-{index}",
            status="done",
        )
        for index in range(8)
    ]
    rendered = manager.select_conversation_context(
        conversation_id="conversation-1",
        turns=turns,
        current_input="此前项目代号和校验码是什么？",
        engine=_TokenEngine(),
    )
    assert "AgentMem" in rendered
    assert "7319" in rendered
    assert "历史第 8 轮" in rendered
    assert len(manager.store.list_owner("conversation", "conversation-1")) == 8


def test_prompt_assembler_injects_r2_context_only_when_enabled(tmp_path: Path):
    state = {
        "task_goal": "回答当前问题",
        "plan_steps": ["直接回答"],
        "execution_log": [],
        "pinned_facts": [{"text": "项目代号是 AgentMem"}],
        "context_summary": {"completed_steps": ["读取配置"]},
        "conversation_context": "用户此前给出校验码7319",
    }
    initialize_lifecycle_context(_enabled_config(tmp_path))
    messages = assemble_reflection_prompt(state, _TokenEngine())
    assert any(
        isinstance(message, HumanMessage) and "AgentMem" in str(message.content)
        for message in messages
    )

    _reset_lifecycle_context_for_testing()
    messages = assemble_reflection_prompt(state, _TokenEngine())
    assert not any(isinstance(message, HumanMessage) for message in messages)


def test_conversation_manager_passes_history_separately_in_r2(tmp_path: Path):
    initialize_lifecycle_context(
        _enabled_config(
            tmp_path,
            hot_conversation_turns=2,
            context_retrieval_token_budget=4000,
        )
    )
    store = ConversationStore(tmp_path / "conversations.sqlite")
    runner = MagicMock()
    runner.start_new_task.side_effect = [
        (
            "thread-1",
            {
                "status": "done",
                "final_answer": "我记住了 AgentMem 和 7319",
                "execution_log": [],
            },
        ),
        (
            "thread-2",
            {
                "status": "done",
                "final_answer": "项目代号是 AgentMem，校验码是7319",
                "execution_log": [],
            },
        ),
    ]
    manager = ConversationManager(runner, store, _TokenEngine())
    conversation_id, _, _ = manager.start("记住项目代号是 AgentMem，校验码是 7319")
    manager.continue_conversation(conversation_id, "上一轮的项目代号是什么？")

    second = runner.start_new_task.call_args_list[1]
    assert second.args[0] == "上一轮的项目代号是什么？"
    assert "AgentMem" in second.kwargs["conversation_context"]
    assert second.kwargs["conversation_id"] == conversation_id
    store.close()


def test_r2_workload_generates_requested_conversation_lengths(tmp_path: Path):
    suite = BenchmarkSuite.load(
        Path("benchmark/workloads/r2_lifecycle_context.json")
    )
    case16 = next(
        case for case in suite.cases
        if case.case_id == "r2-conversation-16-turn-recall"
    )
    case32 = next(
        case for case in suite.cases
        if case.case_id == "r2-conversation-32-turn-recall"
    )
    assert len(_prepare_case(case16, tmp_path / "case16")["turns"]) == 16
    assert len(_prepare_case(case32, tmp_path / "case32")["turns"]) == 32


def test_aggregator_reports_r2_lifecycle_metrics(tmp_path: Path):
    (tmp_path / "task_results.jsonl").write_text(
        '{"case_id":"r2","category":"R2","measured":true,'
        '"repetition":0,"duration_ms":10,"evaluation":{"passed":true}}\n',
        encoding="utf-8",
    )
    events = [
        {
            "case_id": "r2",
            "sample": "rep-000",
            "event": "context_budget_applied",
            "tokens_before": 1000,
            "tokens_after": 400,
        },
        {
            "case_id": "r2",
            "sample": "rep-000",
            "event": "context_compacted",
            "bytes_saved": 3000,
        },
        {
            "case_id": "r2",
            "sample": "rep-000",
            "event": "context_recalled",
            "recalled_count": 2,
        },
    ]
    import json

    (tmp_path / "lifecycle_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    context_file = (
        tmp_path / "cases" / "r2" / "rep-000" / "context_memory.sqlite"
    )
    context_file.parent.mkdir(parents=True)
    context_file.write_bytes(b"context")

    summary = aggregate_run(tmp_path)
    assert summary["context_tokens_before"] == 1000
    assert summary["context_tokens_after"] == 400
    assert summary["context_reduction_ratio"] == pytest.approx(0.6)
    assert summary["context_compaction_bytes_saved"] == 3000
    assert summary["context_recalled_turns"] == 2
    assert summary["context_storage_bytes"] == 7
