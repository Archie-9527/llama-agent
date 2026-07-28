"""测试 prompt_assembler.py。

覆盖设计文档中的四项验收标准：
  ① 结构有效性：SystemMessage 位于首位，不存在孤立的 ToolMessage，
     内容不为空（带 tool_calls 的 AIMessage 除外）。
  ② 预算有效性：应用 Token 裁剪，并在必要时抛出 ContextBudgetExceededError。
  ③ 一致性：工具描述来自 list_capabilities()。
  ④ 确定性：相同输入产生相同输出。

同时覆盖三个 [STABLE] 组装入口以及 [INTERNAL] _validate_message_sequence 检查。
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall

from agent_core.capability_registry import clear_registry, register
from agent_core.exceptions import (
    ContextBudgetExceededError,
    PromptAssemblyError,
)
from agent_core.knowledge_scope import TokenCounter
from agent_core.prompt_assembler import (
    _build_history_messages,
    _render_tools_section,
    _validate_message_sequence,
    assemble_execution_prompt,
    assemble_finalization_prompt,
    assemble_planning_prompt,
    assemble_reflection_prompt,
)


# ── 轻量级桩 Token 计数器 ───────────────────────────────────────────────────


class _CharTokenCounter:
    def get_num_tokens(self, text: str) -> int:
        return len(text)


# ── 测试夹具 ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def engine() -> TokenCounter:
    return _CharTokenCounter()


@pytest.fixture
def empty_state() -> dict:
    return {
        "task_goal": "",
        "plan_steps": [],
        "current_step_index": 0,
        "execution_log": [],
        "reflection_notes": [],
        "status": "planning",
        "max_iterations": 10,
        "current_iteration": 0,
        "n_ctx": 4096,
    }


def _register_test_tool():
    """为需要工具的测试注册一个最小工具。"""
    try:
        from agent_core.capability_registry import get_capability
        get_capability("test_search")
        return  # 已由其他测试注册
    except KeyError:
        pass

    @register(
        name="test_search",
        description="Search the test database",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    )
    def test_search(query: str) -> str:
        return f"results for {query}"


# ============================================================================
# ① 结构有效性：校验
# ============================================================================


class TestValidateMessageSequence:
    """验收：_validate_message_sequence 执行所有结构规则。"""

    def test_empty_list_raises(self):
        with pytest.raises(PromptAssemblyError, match="empty"):
            _validate_message_sequence([])

    def test_first_not_system_raises(self):
        with pytest.raises(PromptAssemblyError, match="must be SystemMessage"):
            _validate_message_sequence([HumanMessage(content="hi")])

    def test_valid_single_system_passes(self):
        _validate_message_sequence([SystemMessage(content="sys")])

    def test_empty_human_content_raises(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="   "),
        ]
        with pytest.raises(PromptAssemblyError, match="empty or whitespace"):
            _validate_message_sequence(msgs)

    def test_aimessage_empty_without_tool_calls_raises(self):
        msgs = [
            SystemMessage(content="sys"),
            AIMessage(content=""),
        ]
        with pytest.raises(PromptAssemblyError, match="empty content"):
            _validate_message_sequence(msgs)

    def test_aimessage_empty_with_tool_calls_passes(self):
        msgs = [
            SystemMessage(content="sys"),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="t", args={}, id="call_1")
                ],
            ),
            ToolMessage(content="result", tool_call_id="call_1"),
        ]
        _validate_message_sequence(msgs)

    def test_orphan_tool_message_raises(self):
        """缺少前置 AIMessage(tool_calls) 的 ToolMessage 应触发错误。"""
        msgs = [
            SystemMessage(content="sys"),
            ToolMessage(content="orphan", tool_call_id="no_match"),
        ]
        with pytest.raises(PromptAssemblyError, match="ToolMessage"):
            _validate_message_sequence(msgs)

    def test_tool_call_id_mismatch_raises(self):
        msgs = [
            SystemMessage(content="sys"),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="t", args={}, id="call_x")
                ],
            ),
            ToolMessage(content="result", tool_call_id="call_y"),
        ]
        with pytest.raises(PromptAssemblyError, match="does not match"):
            _validate_message_sequence(msgs)

    def test_tool_message_as_first_raises(self):
        with pytest.raises(PromptAssemblyError, match="ToolMessage"):
            _validate_message_sequence(
                [ToolMessage(content="x", tool_call_id="c1")]
            )


# ============================================================================
# ② 历史记录构建
# ============================================================================


class TestBuildHistoryMessages:
    """验收：execution_log 条目转换为正确的 BaseMessage 序列。"""

    def test_tool_used_entry_creates_pair(self):
        log = [{"step": "search", "result": "found it", "tool_used": "search_log"}]
        msgs = _build_history_messages(log)
        # HumanMessage（步骤）→ AIMessage（工具调用）→ ToolMessage（结果）
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "search"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].tool_calls
        assert msgs[1].tool_calls[0]["name"] == "search_log"
        assert isinstance(msgs[2], ToolMessage)
        assert msgs[2].content == "found it"

    def test_no_tool_entry_creates_ai_only(self):
        log = [{"step": "think", "result": "I think...", "tool_used": None}]
        msgs = _build_history_messages(log)
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "I think..."
        assert not msgs[1].tool_calls

    def test_tool_calls_have_unique_ids(self):
        log = [
            {"step": "s1", "result": "r1", "tool_used": "t1"},
            {"step": "s2", "result": "r2", "tool_used": "t2"},
        ]
        msgs = _build_history_messages(log)
        ids = [
            m.tool_call_id
            for m in msgs
            if isinstance(m, ToolMessage)
        ]
        assert len(set(ids)) == 2  # 全部唯一

    def test_empty_log_returns_empty(self):
        assert _build_history_messages([]) == []


# ============================================================================
# ③ 组装入口
# ============================================================================


class TestAssemblePlanningPrompt:
    """验收：assemble_planning_prompt 生成有效消息。"""

    def test_returns_list_of_base_messages(self, engine, empty_state):
        empty_state["task_goal"] = "Write a poem"
        msgs = assemble_planning_prompt(empty_state, engine)
        assert isinstance(msgs, list)
        assert all(isinstance(m, BaseMessage) for m in msgs)
        assert len(msgs) > 0
        assert isinstance(msgs[0], SystemMessage)

    def test_system_message_contains_task_goal(self, engine, empty_state):
        empty_state["task_goal"] = "Calculate the meaning of life"
        msgs = assemble_planning_prompt(empty_state, engine)
        assert "Calculate the meaning of life" in msgs[0].content

    def test_reflection_notes_appended(self, engine, empty_state):
        empty_state["task_goal"] = "test"
        empty_state["reflection_notes"] = ["Step 1 was wrong", "Need to replan"]
        msgs = assemble_planning_prompt(empty_state, engine)
        # 应出现包含反思记录的 HumanMessage。
        human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
        assert len(human_msgs) >= 1
        assert "评估反馈" in human_msgs[0].content
        assert "Step 1 was wrong" in human_msgs[0].content

    def test_tools_section_rendered(self, engine, empty_state):
        _register_test_tool()
        empty_state["task_goal"] = "search something"
        msgs = assemble_planning_prompt(empty_state, engine)
        assert "test_search" in msgs[0].content

    def test_structural_validity_checked(self, engine, empty_state):
        empty_state["task_goal"] = "ok"
        msgs = assemble_planning_prompt(empty_state, engine)
        # 不应抛出异常，应通过 _validate_message_sequence。
        assert isinstance(msgs[0], SystemMessage)


class TestAssembleExecutionPrompt:
    """验收：assemble_execution_prompt 生成有效消息。"""

    def test_returns_valid_structure(self, engine, empty_state):
        empty_state["task_goal"] = "do thing"
        empty_state["plan_steps"] = ["step one"]
        msgs = assemble_execution_prompt(empty_state, engine)
        assert isinstance(msgs[0], SystemMessage)
        assert "step one" in msgs[0].content

    def test_explicit_empty_tool_list_creates_tool_free_prompt(
        self, engine, empty_state
    ):
        _register_test_tool()
        empty_state["plan_steps"] = ["记住项目代号"]
        msgs = assemble_execution_prompt(
            empty_state,
            engine,
            available_tools=[],
        )
        assert "不允许调用任何工具" in msgs[0].content
        assert "test_search" not in msgs[0].content

    def test_execution_log_converted_to_messages(self, engine, empty_state):
        empty_state["task_goal"] = "test"
        empty_state["plan_steps"] = ["do it"]
        empty_state["execution_log"] = [
            {"step": "search", "result": "found", "tool_used": "test_search"},
        ]
        msgs = assemble_execution_prompt(empty_state, engine)
        # 应包含 ToolMessage。
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1

    def test_tool_result_appended(self, engine, empty_state):
        empty_state["task_goal"] = "test"
        empty_state["plan_steps"] = ["step"]
        empty_state["_last_tool_call_id"] = "call_abc"
        # 预先在 execution_log 中加入 AIMessage(tool_calls) 工具条目，
        # 使追加的 ToolMessage 能匹配前置 AIMessage。
        empty_state["execution_log"] = [
            {"step": "call a tool", "result": "got results from tool", "tool_used": "test_search"},
        ]
        _register_test_tool()
        msgs = assemble_execution_prompt(
            empty_state, engine, tool_result="success"
        )
        # 历史记录提供 AIMessage(tool_calls) 后，tool_result 对应的
        # ToolMessage 应通过校验并追加到末尾。
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1
        assert msgs[-1].content == "success"

    def test_structural_invalid_input_rejected(self, engine, empty_state):
        """若 _build_history_messages 产生孤立 ToolMessage，校验器应将其捕获。"""
        # 通过模拟 _build_history_messages 返回错误数据进行测试。
        bad_messages = [
            SystemMessage(content="sys"),
            ToolMessage(content="orphan", tool_call_id="no_match"),
        ]
        with patch(
            "agent_core.prompt_assembler._build_history_messages",
            return_value=bad_messages[1:],
        ):
            with pytest.raises(PromptAssemblyError):
                assemble_execution_prompt(empty_state, engine)


class TestAssembleReflectionPrompt:
    """验收：assemble_reflection_prompt 生成有效消息。"""

    def test_returns_valid_structure(self, engine, empty_state):
        empty_state["task_goal"] = "evaluate this"
        empty_state["plan_steps"] = ["step 1", "step 2"]
        empty_state["execution_log"] = [
            {"step": "step 1", "result": "done", "tool_used": None},
        ]
        msgs = assemble_reflection_prompt(empty_state, engine)
        assert isinstance(msgs[0], SystemMessage)
        assert "evaluate this" in msgs[0].content.lower() or "evaluate this" in msgs[0].content

    def test_no_tools_described(self, engine, empty_state):
        """反思器不会调用工具，因此不应包含工具部分。"""
        _register_test_tool()
        empty_state["task_goal"] = "x"
        msgs = assemble_reflection_prompt(empty_state, engine)
        # 反思 Prompt 中不应出现工具部分。
        content = msgs[0].content
        # 模板不会注入工具，这里进行验证。
        assert "test_search" not in content

    def test_large_tool_result_is_bounded_but_tail_evidence_survives(
        self, engine, empty_state
    ):
        raw_result = "HEAD" + ("x" * 60_000) + "FINAL_EVIDENCE=req-7319"
        empty_state["task_goal"] = "find final evidence"
        empty_state["plan_steps"] = ["read a large file"]
        empty_state["execution_log"] = [
            {
                "step": "read a large file",
                "result": raw_result,
                "tool_used": "read_file",
            },
            {
                "step": "summarize",
                "result": "FINAL_EVIDENCE=req-7319",
                "tool_used": None,
            },
        ]

        reflected = assemble_reflection_prompt(empty_state, engine)
        finalized = assemble_finalization_prompt(empty_state, engine)

        assert "FINAL_EVIDENCE=req-7319" in reflected[0].content
        assert "FINAL_EVIDENCE=req-7319" in finalized[0].content
        assert len(reflected[0].content) < len(raw_result)
        assert empty_state["execution_log"][0]["result"] == raw_result


# ============================================================================
# ④ 一致性与确定性
# ============================================================================


class TestConsistencyAndDeterminism:
    """验收：相同输入产生相同输出，工具来自单一数据源。"""

    def test_deterministic_output(self, engine, empty_state):
        empty_state["task_goal"] = "test"
        msgs_1 = assemble_planning_prompt(empty_state, engine)
        msgs_2 = assemble_planning_prompt(empty_state, engine)
        assert len(msgs_1) == len(msgs_2)
        for a, b in zip(msgs_1, msgs_2):
            assert a.content == b.content

    def test_tools_from_single_source(self, engine, empty_state):
        """验证系统 Prompt 中的工具描述来自 list_capabilities()，而非硬编码字符串。"""
        _register_test_tool()
        empty_state["task_goal"] = "search for X"

        from agent_core.capability_registry import list_capabilities
        caps = list_capabilities()
        msgs = assemble_planning_prompt(empty_state, engine)

        # 每个已注册工具的名称都应出现在系统 Prompt 中。
        for cap in caps:
            assert cap.name in msgs[0].content


# ============================================================================
# ⑤ 预算管理
# ============================================================================


class TestBudgetManagement:
    """验收：应用裁剪，超出预算时抛出 ContextBudgetExceededError。"""

    def test_long_messages_are_trimmed(self, engine, empty_state):
        empty_state["task_goal"] = "t"
        empty_state["execution_log"] = [
            {"step": "s" + "x" * 500, "result": "r" * 500, "tool_used": None},
        ]
        # 预算充足时应成功。
        msgs = assemble_execution_prompt(empty_state, engine, reserved_for_generation=0)
        assert isinstance(msgs[0], SystemMessage)

    def test_tiny_budget_raises(self, engine, empty_state):
        """若 n_ctx 小到无法容纳 SystemMessage，应抛出 ContextBudgetExceededError。"""
        empty_state["task_goal"] = "a" * 10000
        empty_state["n_ctx"] = 10
        with pytest.raises(ContextBudgetExceededError):
            assemble_planning_prompt(empty_state, engine, reserved_for_generation=0)


# ============================================================================
# ⑥ 渲染辅助函数
# ============================================================================


class TestRenderToolsSection:
    """验收：_render_tools_section 生成便于阅读的文本。"""

    def test_empty_list(self):
        assert _render_tools_section([]) == ""

    def test_single_tool(self, engine, empty_state):
        _register_test_tool()
        from agent_core.capability_registry import list_capabilities
        caps = list_capabilities()
        result = _render_tools_section(caps)
        assert "test_search" in result
        assert "Search the test database" in result
        assert "query" in result  # 参数名


# ============================================================================
# ⑦ 集成：完整 Prompt 流水线
# ============================================================================


class TestIntegration:
    """端到端：状态→组装→校验→传递给模拟引擎。"""

    def test_full_pipeline_planning(self, engine, empty_state):
        _register_test_tool()
        empty_state["task_goal"] = "Search for python docs"
        empty_state["reflection_notes"] = ["previous search was too narrow"]

        msgs = assemble_planning_prompt(empty_state, engine)

        # 结构检查
        assert isinstance(msgs[0], SystemMessage)
        _validate_message_sequence(msgs)

        # 内容检查
        assert "python docs" in msgs[0].content
        assert "test_search" in msgs[0].content

    def test_full_pipeline_execution(self, engine, empty_state):
        empty_state["task_goal"] = "Complete task"
        empty_state["plan_steps"] = ["search", "summarize"]
        empty_state["current_step_index"] = 0
        empty_state["execution_log"] = []

        msgs = assemble_execution_prompt(empty_state, engine)
        _validate_message_sequence(msgs)
        assert "search" in msgs[0].content
        assert "Complete task" in msgs[0].content

    def test_replanning_prompt_contains_failure_evidence(self, engine, empty_state):
        empty_state["task_goal"] = "query /absolute/incidents.sqlite"
        empty_state["plan_steps"] = ["query incidents.sqlite"]
        empty_state["reflection_notes"] = [
            "decision=continue reason=path does not exist"
        ]
        empty_state["execution_log"] = [
            {
                "step": "query incidents.sqlite",
                "result": "ValueError: path does not exist",
                "tool_used": "query_sqlite",
                "tool_args": {"db_path": "incidents.sqlite"},
            }
        ]

        msgs = assemble_planning_prompt(empty_state, engine)
        combined = "\n".join(str(msg.content) for msg in msgs)
        assert "path does not exist" in combined
        assert "/absolute/incidents.sqlite" in combined

    def test_full_pipeline_reflection(self, engine, empty_state):
        empty_state["task_goal"] = "Write report"
        empty_state["plan_steps"] = ["draft", "review"]
        empty_state["execution_log"] = [
            {"step": "draft", "result": "draft complete", "tool_used": None},
        ]

        msgs = assemble_reflection_prompt(empty_state, engine)
        _validate_message_sequence(msgs)
        assert "Write report" in msgs[0].content
