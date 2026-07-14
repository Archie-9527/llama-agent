"""Tests for prompt_assembler.py.

Covers the four design-doc acceptance criteria:
  ① Structural validity — SystemMessage first, no orphan ToolMessage,
     no empty content (except AIMessage with tool_calls).
  ② Budget validity — token trimming applied, ContextBudgetExceededError
     when needed.
  ③ Consistency — tool descriptions from list_capabilities().
  ④ Determinism — same input → same output.

Also covers the three [STABLE] assembly entry points and the
[INTERNAL] _validate_message_sequence checks.
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
    assemble_planning_prompt,
    assemble_reflection_prompt,
)


# ── Lightweight stub token counter ───────────────────────────────────────────


class _CharTokenCounter:
    def get_num_tokens(self, text: str) -> int:
        return len(text)


# ── Fixtures ─────────────────────────────────────────────────────────────────


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
    """Register a minimal tool for tests that need one."""
    try:
        from agent_core.capability_registry import get_capability
        get_capability("test_search")
        return  # already registered (leftover from another test)
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
# ① Structural validity — validation
# ============================================================================


class TestValidateMessageSequence:
    """Acceptance: _validate_message_sequence enforces all structural rules."""

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
        """ToolMessage without preceding AIMessage(tool_calls) → error."""
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
# ② History construction
# ============================================================================


class TestBuildHistoryMessages:
    """Acceptance: execution_log entries → proper BaseMessage sequence."""

    def test_tool_used_entry_creates_pair(self):
        log = [{"step": "search", "result": "found it", "tool_used": "web_search"}]
        msgs = _build_history_messages(log)
        # HumanMessage(step) → AIMessage(tool_calls) → ToolMessage(result)
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "search"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].tool_calls
        assert msgs[1].tool_calls[0]["name"] == "web_search"
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
        assert len(set(ids)) == 2  # all unique

    def test_empty_log_returns_empty(self):
        assert _build_history_messages([]) == []


# ============================================================================
# ③ Assembly entry points
# ============================================================================


class TestAssemblePlanningPrompt:
    """Acceptance: assemble_planning_prompt produces valid messages."""

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
        # A HumanMessage with reflection notes should appear
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
        # Should not raise — let _validate_message_sequence pass
        assert isinstance(msgs[0], SystemMessage)


class TestAssembleExecutionPrompt:
    """Acceptance: assemble_execution_prompt produces valid messages."""

    def test_returns_valid_structure(self, engine, empty_state):
        empty_state["task_goal"] = "do thing"
        empty_state["plan_steps"] = ["step one"]
        msgs = assemble_execution_prompt(empty_state, engine)
        assert isinstance(msgs[0], SystemMessage)
        assert "step one" in msgs[0].content

    def test_execution_log_converted_to_messages(self, engine, empty_state):
        empty_state["task_goal"] = "test"
        empty_state["plan_steps"] = ["do it"]
        empty_state["execution_log"] = [
            {"step": "search", "result": "found", "tool_used": "test_search"},
        ]
        msgs = assemble_execution_prompt(empty_state, engine)
        # Should contain a ToolMessage
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1

    def test_tool_result_appended(self, engine, empty_state):
        empty_state["task_goal"] = "test"
        empty_state["plan_steps"] = ["step"]
        empty_state["_last_tool_call_id"] = "call_abc"
        # Pre-populate execution_log with an AIMessage(tool_calls) tool entry
        # so the appended ToolMessage has a matching preceding AIMessage.
        empty_state["execution_log"] = [
            {"step": "call a tool", "result": "got results from tool", "tool_used": "test_search"},
        ]
        _register_test_tool()
        msgs = assemble_execution_prompt(
            empty_state, engine, tool_result="success"
        )
        # With history providing AIMessage(tool_calls), the tool_result
        # ToolMessage should validate and be appended last.
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1
        assert msgs[-1].content == "success"

    def test_structural_invalid_input_rejected(self, engine, empty_state):
        """If _build_history_messages somehow produces an orphan ToolMessage,
        _validate_message_sequence should catch it."""
        # We test this by mocking _build_history_messages to return bad data.
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
    """Acceptance: assemble_reflection_prompt produces valid messages."""

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
        """Reflector never calls tools, so tools section is absent."""
        _register_test_tool()
        empty_state["task_goal"] = "x"
        msgs = assemble_reflection_prompt(empty_state, engine)
        # tools section should NOT be present in reflector prompt
        content = msgs[0].content
        # Our template doesn't inject tools → verify
        assert "test_search" not in content


# ============================================================================
# ④ Consistency & determinism
# ============================================================================


class TestConsistencyAndDeterminism:
    """Acceptance: same input → same output; tools from single source."""

    def test_deterministic_output(self, engine, empty_state):
        empty_state["task_goal"] = "test"
        msgs_1 = assemble_planning_prompt(empty_state, engine)
        msgs_2 = assemble_planning_prompt(empty_state, engine)
        assert len(msgs_1) == len(msgs_2)
        for a, b in zip(msgs_1, msgs_2):
            assert a.content == b.content

    def test_tools_from_single_source(self, engine, empty_state):
        """Verify that the tool descriptions in the system prompt come from
        list_capabilities(), not from hardcoded strings."""
        _register_test_tool()
        empty_state["task_goal"] = "search for X"

        from agent_core.capability_registry import list_capabilities
        caps = list_capabilities()
        msgs = assemble_planning_prompt(empty_state, engine)

        # Every registered tool name should appear in the system prompt
        for cap in caps:
            assert cap.name in msgs[0].content


# ============================================================================
# ⑤ Budget management
# ============================================================================


class TestBudgetManagement:
    """Acceptance: trim applied, ContextBudgetExceededError on overflow."""

    def test_long_messages_are_trimmed(self, engine, empty_state):
        empty_state["task_goal"] = "t"
        empty_state["execution_log"] = [
            {"step": "s" + "x" * 500, "result": "r" * 500, "tool_used": None},
        ]
        # With a generous budget it should succeed
        msgs = assemble_execution_prompt(empty_state, engine, reserved_for_generation=0)
        assert isinstance(msgs[0], SystemMessage)

    def test_tiny_budget_raises(self, engine, empty_state):
        """If n_ctx is so small even the SystemMessage doesn't fit,
        we expect ContextBudgetExceededError."""
        empty_state["task_goal"] = "a" * 10000
        empty_state["n_ctx"] = 10
        with pytest.raises(ContextBudgetExceededError):
            assemble_planning_prompt(empty_state, engine, reserved_for_generation=0)


# ============================================================================
# ⑥ Rendering helpers
# ============================================================================


class TestRenderToolsSection:
    """Acceptance: _render_tools_section produces human-readable text."""

    def test_empty_list(self):
        assert _render_tools_section([]) == ""

    def test_single_tool(self, engine, empty_state):
        _register_test_tool()
        from agent_core.capability_registry import list_capabilities
        caps = list_capabilities()
        result = _render_tools_section(caps)
        assert "test_search" in result
        assert "Search the test database" in result
        assert "query" in result  # parameter name


# ============================================================================
# ⑦ Integration: full prompt pipeline
# ============================================================================


class TestIntegration:
    """End-to-end: state → assemble → validate → pass to engine (mock)."""

    def test_full_pipeline_planning(self, engine, empty_state):
        _register_test_tool()
        empty_state["task_goal"] = "Search for python docs"
        empty_state["reflection_notes"] = ["previous search was too narrow"]

        msgs = assemble_planning_prompt(empty_state, engine)

        # Structural checks
        assert isinstance(msgs[0], SystemMessage)
        _validate_message_sequence(msgs)

        # Content checks
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

    def test_full_pipeline_reflection(self, engine, empty_state):
        empty_state["task_goal"] = "Write report"
        empty_state["plan_steps"] = ["draft", "review"]
        empty_state["execution_log"] = [
            {"step": "draft", "result": "draft complete", "tool_used": None},
        ]

        msgs = assemble_reflection_prompt(empty_state, engine)
        _validate_message_sequence(msgs)
        assert "Write report" in msgs[0].content
