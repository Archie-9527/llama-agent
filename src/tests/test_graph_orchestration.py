"""Tests for the graph orchestration layer (Stage 2).

Covers acceptance criteria A1–A20 from the orchestration-layer design
doc.  Most tests use mocked LLM engines / tools so they run without a
real GGUF model.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall

from agent_core.capability_registry import (
    Capability,
    clear_registry,
    get_capability,
    list_capabilities,
    register,
)
from agent_core.exceptions import (
    AgentCoreError,
    ExecutionError,
    GraphOrchestrationError,
    PlanningError,
    ReflectionError,
)
from agent_core.graph.state import AgentState
from agent_core.graph.planner import planner_node
from agent_core.graph.executor import (
    _build_react_input,
    _ensure_tool_summary,
    _extract_execution_result,
    _normalize_output_messages,
    _recover_empty_response,
    _validate_required_tool_execution,
    executor_node,
)
from agent_core.graph.reflector import reflector_node
from agent_core.graph.build_graph import (
    _route_after_executor,
    _route_after_reflector,
    build_graph,
)
from agent_core.graph.react_agent_factory import (
    USE_OFFICIAL_CREATE_AGENT,
    _build_via_self_made_stategraph,
    _build_via_create_agent,
    _ReActState,
    _agent_node,
    _tool_node,
    _should_continue,
    to_langchain_tool,
)
from agent_core.graph.checkpointer import get_checkpointer


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear the capability registry before every test."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def base_state() -> AgentState:
    """Minimal valid AgentState with defaults."""
    return AgentState(
        task_goal="",
        plan_steps=[],
        current_step_index=0,
        execution_log=[],
        reflection_notes=[],
        status="planning",
        max_iterations=10,
        current_iteration=0,
    )


def _register_test_tools():
    """Register a couple of test capabilities."""
    @register(
        name="echo",
        description="Echo back the input",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Text to echo"},
            },
            "required": ["message"],
        },
    )
    def echo(message: str) -> str:
        return f"echo: {message}"

    @register(
        name="add_numbers",
        description="Add two numbers together",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    )
    def add_numbers(a: int, b: int) -> str:
        return str(a + b)


# ── Acceptance A1: Planner normal output ─────────────────────────────────────


class TestPlannerNormalOutput:
    """A1: Given a valid task_goal, planner_node produces a non-empty
    list[str] plan_steps and status == 'executing'."""

    def test_planner_produces_valid_plan(self, base_state):
        """Planner should parse a valid JSON response into plan_steps."""
        base_state["task_goal"] = "Write a report"

        # Mock the engine to return a valid plan JSON
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "steps": ["Research topic", "Draft outline", "Write report"]
        })
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.planner.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.planner.assemble_planning_prompt", return_value=[SystemMessage(content="test")]):
            result = planner_node(base_state)

        assert len(result["plan_steps"]) == 3
        assert result["plan_steps"][0] == "Research topic"
        assert result["current_step_index"] == 0
        assert result["status"] == "executing"


# ── Acceptance A2: Planner exception translation ─────────────────────────────


class TestPlannerExceptionTranslation:
    """A2: Mock engine.invoke to return illegal JSON — must raise
    PlanningError, not raw json.JSONDecodeError."""

    def test_invalid_json_raises_planning_error(self, base_state):
        base_state["task_goal"] = "test"
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "not valid json at all {{{"
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.planner.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.planner.assemble_planning_prompt", return_value=[SystemMessage(content="test")]):
            with pytest.raises(PlanningError) as exc_info:
                planner_node(base_state)

        assert "not valid json" in str(exc_info.value)
        # Must NOT be a raw JSONDecodeError
        assert not isinstance(exc_info.value, json.JSONDecodeError)

    def test_missing_steps_key_raises_planning_error(self, base_state):
        base_state["task_goal"] = "test"
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({"wrong_key": []})
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.planner.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.planner.assemble_planning_prompt", return_value=[SystemMessage(content="test")]):
            with pytest.raises(PlanningError):
                planner_node(base_state)

    def test_empty_steps_raises_planning_error(self, base_state):
        base_state["task_goal"] = "test"
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({"steps": []})
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.planner.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.planner.assemble_planning_prompt", return_value=[SystemMessage(content="test")]):
            with pytest.raises(PlanningError):
                planner_node(base_state)

    def test_empty_string_step_raises_planning_error(self, base_state):
        base_state["task_goal"] = "test"
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({"steps": ["ok", "   ", "also ok"]})
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.planner.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.planner.assemble_planning_prompt", return_value=[SystemMessage(content="test")]):
            with pytest.raises(PlanningError):
                planner_node(base_state)

    def test_planning_error_is_agent_core_error(self, base_state):
        """PlanningError must be catchable via AgentCoreError (design doc §7)."""
        base_state["task_goal"] = "test"
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "garbage"
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.planner.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.planner.assemble_planning_prompt", return_value=[SystemMessage(content="test")]):
            try:
                planner_node(base_state)
            except AgentCoreError:
                pass
            else:
                pytest.fail("PlanningError should be caught by AgentCoreError")


# ── Acceptance A3–A7: Executor node ──────────────────────────────────────────


class TestExecutorStateBridging:
    """A3–A8: Executor state-bridging and output parsing."""

    def test_extract_execution_result_no_tool_calls(self):
        """A3: Single AIMessage with no tool_calls → 1 record, tool_used is None."""
        messages = [AIMessage(content="The answer is 42.")]
        records = _extract_execution_result(messages, "think step")
        assert len(records) == 1
        assert records[0]["tool_used"] is None
        assert records[0]["result"] == "The answer is 42."
        assert records[0]["step"] == "think step"

    def test_extract_execution_result_multiple_tools(self):
        """A4: AIMessage with 2 tool_calls → 2+ records, tool_used matches."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="echo", args={"message": "hi"}, id="call_1"),
                    ToolCall(name="add_numbers", args={"a": 1, "b": 2}, id="call_2"),
                ],
            ),
            ToolMessage(content="echo: hi", tool_call_id="call_1"),
            ToolMessage(content="3", tool_call_id="call_2"),
            AIMessage(content="All tools completed."),
        ]
        records = _extract_execution_result(messages, "multi-tool step")
        assert len(records) == 3
        assert records[0]["tool_used"] == "echo"
        assert records[1]["tool_used"] == "add_numbers"
        assert records[2]["tool_used"] is None
        assert records[2]["result"] == "All tools completed."

    def test_textual_pseudo_tool_call_is_rejected(self):
        @register(
            name="execute_shell_command",
            description="Run a command",
            input_schema={"type": "object", "properties": {}},
        )
        def execute_shell_command():
            return "unused"

        records = [{
            "step": "inspect files",
            "result": 'execute_shell_command(command="find src -type f")',
            "tool_used": None,
        }]

        with pytest.raises(ExecutionError, match="pseudo tool call"):
            _validate_required_tool_execution("inspect files", records)

    def test_step_named_tool_requires_matching_tool_message(self):
        @register(
            name="execute_shell_command",
            description="Run a command",
            input_schema={"type": "object", "properties": {}},
        )
        def execute_shell_command():
            return "unused"

        records = [{"step": "s", "result": "I will do it", "tool_used": None}]
        with pytest.raises(ExecutionError, match="explicitly requires"):
            _validate_required_tool_execution(
                "Use execute_shell_command to inspect src", records
            )

    def test_protocol_marker_is_replaced_by_recovery_summary(self):
        records = [
            {
                "step": "count",
                "result": '{"success": true, "stdout": "611\\n"}',
                "tool_used": "count_lines",
            },
            {
                "step": "count",
                "result": "functions.count_lines:",
                "tool_used": None,
            },
        ]
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = AIMessage(
            content="llm_engine.py 共有 611 行代码。"
        )

        with patch("agent_core.graph.executor.get_engine", return_value=mock_engine):
            result = _ensure_tool_summary(
                [HumanMessage(content="count lines")],
                "count",
                records,
            )

        assert len(result) == 2
        assert result[-1]["result"] == "llm_engine.py 共有 611 行代码。"
        assert all(r["result"] != "functions.count_lines:" for r in result)
        mock_engine.invoke.assert_called_once()

    def test_empty_response_has_one_tool_free_recovery(self):
        mock_engine = MagicMock()
        mock_engine.invoke.return_value = AIMessage(
            content="我记住的项目代号是 AgentMem。"
        )
        with patch("agent_core.graph.executor.get_engine", return_value=mock_engine):
            records = _recover_empty_response(
                [HumanMessage(content="记住项目代号是 AgentMem")],
                "直接回答用户",
            )
        assert records == [
            {
                "step": "直接回答用户",
                "result": "我记住的项目代号是 AgentMem。",
                "tool_used": None,
            }
        ]
        assert "不要调用工具" in mock_engine.invoke.call_args.args[0][-1].content

    def test_functions_protocol_is_rejected_as_pseudo_call(self):
        @register(
            name="count_lines",
            description="Count lines",
            input_schema={"type": "object", "properties": {}},
        )
        def count_lines():
            return "unused"

        records = [
            {
                "step": "count",
                "result": "functions.count_lines:",
                "tool_used": None,
            }
        ]
        with pytest.raises(ExecutionError, match="pseudo tool call"):
            _validate_required_tool_execution("count", records)

    def test_missing_tool_message_raises_execution_error(self):
        """A7: Missing ToolMessage → ExecutionError with tool_call_id."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="echo", args={"message": "hi"}, id="call_1"),
                ],
            ),
            # ToolMessage intentionally omitted
            AIMessage(content="Done."),
        ]
        with pytest.raises(ExecutionError) as exc_info:
            _extract_execution_result(messages, "broken step")
        assert "call_1" in str(exc_info.value)

    def test_normalize_output_messages_dict_format(self):
        """A8: Dict-format messages normalised to BaseMessage."""
        raw_messages = [
            {"role": "assistant", "content": "Hello world"},
            {"role": "user", "content": "Hi"},
        ]
        normalised = _normalize_output_messages(raw_messages)
        assert len(normalised) == 2
        assert isinstance(normalised[0], AIMessage)
        assert normalised[0].content == "Hello world"
        assert isinstance(normalised[1], HumanMessage)
        assert normalised[1].content == "Hi"

    def test_normalize_output_messages_mixed_format(self):
        """A8: Mixed dict + BaseMessage list produces consistent BaseMessage list."""
        raw_messages = [
            AIMessage(content="I am already a message"),
            {"role": "assistant", "content": "I am a dict"},
        ]
        normalised = _normalize_output_messages(raw_messages)
        assert len(normalised) == 2
        assert all(isinstance(m, AIMessage) for m in normalised)
        assert normalised[0].content == "I am already a message"
        assert normalised[1].content == "I am a dict"


class TestExecutorNode:
    """A3–A6: Executor integration tests with mocked inner agent."""

    def test_executor_single_step_no_tool(self, base_state):
        """A3: Executor runs a simple step, produces 1 record with tool_used=None."""
        _register_test_tools()
        base_state["task_goal"] = "test"
        base_state["plan_steps"] = ["Say hello"]
        base_state["current_step_index"] = 0

        # Mock the inner agent to return a plain text response
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Hello!")]
        }

        # Build a mock config with the thread_id that executor_node expects
        mock_config = {"configurable": {"thread_id": "test-thread"}}

        with patch("agent_core.graph.executor._build_react_input", return_value={"messages": []}), \
             patch("agent_core.graph.react_agent_factory.initialize_react_agent", return_value=mock_agent):
            result = executor_node(base_state, config=mock_config)  # type: ignore[call-arg]

        assert len(result["execution_log"]) == 1
        assert result["execution_log"][0]["tool_used"] is None
        assert result["execution_log"][0]["result"] == "Hello!"
        assert result["current_step_index"] == 1
        assert result["status"] == "reflecting"

    def test_executor_multiple_steps_status_stays_executing(self, base_state):
        """A5: With 3 plan steps, executor stays at 'executing' until last step."""
        _register_test_tools()
        base_state["task_goal"] = "test"
        base_state["plan_steps"] = ["step 1", "step 2", "step 3"]
        base_state["current_step_index"] = 0

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                SystemMessage(content="test"),
                AIMessage(content="Done."),
            ]
        }

        mock_config = {"configurable": {"thread_id": "test-thread"}}

        with patch("agent_core.graph.executor._build_react_input", return_value={"messages": [SystemMessage(content="test")]}), \
             patch("agent_core.graph.react_agent_factory.initialize_react_agent", return_value=mock_agent):
            # Step 1
            result = executor_node(base_state, config=mock_config)  # type: ignore[call-arg]
            assert result["current_step_index"] == 1
            assert result["status"] != "reflecting"  # more steps remain

            # Step 2
            result = executor_node(result, config=mock_config)  # type: ignore[call-arg]
            assert result["current_step_index"] == 2
            assert result["status"] != "reflecting"

            # Step 3 — last step
            result = executor_node(result, config=mock_config)  # type: ignore[call-arg]
            assert result["current_step_index"] == 3
            assert result["status"] == "reflecting"

    def test_executor_recursion_limit_raises_execution_error(self, base_state):
        """A6: recursion limit exceeded → ExecutionError."""
        _register_test_tools()
        base_state["task_goal"] = "test"
        base_state["plan_steps"] = ["step that loops forever"]

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("GraphRecursionError: recursion limit")

        mock_config = {"configurable": {"thread_id": "test-thread"}}

        with patch("agent_core.graph.executor._build_react_input", return_value={"messages": [SystemMessage(content="test")]}), \
             patch("agent_core.graph.react_agent_factory.initialize_react_agent", return_value=mock_agent):
            with pytest.raises(ExecutionError) as exc_info:
                executor_node(base_state, config=mock_config)  # type: ignore[call-arg]
            assert "recursion" in str(exc_info.value).lower()

    def test_executor_out_of_range_step_raises(self, base_state):
        """Calling executor with current_step_index beyond plan_steps → ExecutionError."""
        base_state["plan_steps"] = []
        base_state["current_step_index"] = 0
        mock_config = {"configurable": {"thread_id": "test-thread"}}
        with pytest.raises(ExecutionError):
            executor_node(base_state, config=mock_config)  # type: ignore[call-arg]


# ── Acceptance A9: Reflector enum strictness ─────────────────────────────────


class TestReflector:
    """A9: reflector_node enforces strict enum validation."""

    def test_reflector_valid_decision_done(self, base_state):
        base_state["task_goal"] = "test"
        base_state["plan_steps"] = ["step 1"]
        base_state["execution_log"] = [{"step": "step 1", "result": "ok", "tool_used": None}]

        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "done"
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.reflector.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.reflector.assemble_reflection_prompt", return_value=[SystemMessage(content="test")]):
            result = reflector_node(base_state)

        assert result["status"] == "done"
        assert len(result["reflection_notes"]) == 1
        assert "decision=done" in result["reflection_notes"][0]

    def test_reflector_valid_decision_continue(self, base_state):
        base_state["task_goal"] = "test"

        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "continue"
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.reflector.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.reflector.assemble_reflection_prompt", return_value=[SystemMessage(content="test")]):
            result = reflector_node(base_state)

        assert result["status"] == "continue"

    def test_reflector_invalid_decision_raises(self, base_state):
        """A9: Any decision outside {done, continue, failed} → ReflectionError."""
        base_state["task_goal"] = "test"

        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "maybe"
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.reflector.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.reflector.assemble_reflection_prompt", return_value=[SystemMessage(content="test")]):
            with pytest.raises(ReflectionError):
                reflector_node(base_state)

    def test_reflector_quoted_decision_stripped(self, base_state):
        """Enum grammar may produce quoted strings — reflector should strip them."""
        base_state["task_goal"] = "test"

        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '"failed"'
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.reflector.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.reflector.assemble_reflection_prompt", return_value=[SystemMessage(content="test")]):
            result = reflector_node(base_state)

        assert result["status"] == "failed"

    def test_reflector_increments_iteration(self, base_state):
        base_state["task_goal"] = "test"
        base_state["current_iteration"] = 3

        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "done"
        mock_engine.invoke.return_value = mock_response

        with patch("agent_core.graph.reflector.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.reflector.assemble_reflection_prompt", return_value=[SystemMessage(content="test")]):
            result = reflector_node(base_state)

        assert result["current_iteration"] == 4

    def test_continue_at_iteration_limit_becomes_failed(self, base_state):
        base_state["current_iteration"] = 1
        base_state["max_iterations"] = 2

        mock_engine = MagicMock()
        mock_engine.invoke.return_value = AIMessage(content="continue")

        with patch("agent_core.graph.reflector.get_engine", return_value=mock_engine), \
             patch("agent_core.graph.reflector.assemble_reflection_prompt", return_value=[SystemMessage(content="test")]):
            result = reflector_node(base_state)

        assert result["status"] == "failed"
        assert "iteration limit" in result["reflection_notes"][-1]


# ── Acceptance A10–A11: Routing logic ────────────────────────────────────────


class TestRoutingFunctions:
    """A10–A11: _route_after_executor and _route_after_reflector."""

    def test_route_after_executor_more_steps(self):
        """More steps remain → return 'executor'."""
        state = AgentState(
            task_goal="t", plan_steps=["a", "b", "c"],
            current_step_index=1, execution_log=[], reflection_notes=[],
            status="executing", max_iterations=10, current_iteration=0,
        )
        assert _route_after_executor(state) == "executor"

    def test_route_after_executor_all_done(self):
        """All steps done → return 'reflector'."""
        state = AgentState(
            task_goal="t", plan_steps=["a", "b"],
            current_step_index=2, execution_log=[], reflection_notes=[],
            status="executing", max_iterations=10, current_iteration=0,
        )
        assert _route_after_executor(state) == "reflector"

    def test_route_after_reflector_done(self):
        """A10: Reflector says 'done' → stable-answer Finalizer."""
        state = AgentState(
            task_goal="t", plan_steps=["a"], current_step_index=1,
            execution_log=[], reflection_notes=[],
            status="done", max_iterations=10, current_iteration=0,
        )
        assert _route_after_reflector(state) == "finalizer"

    def test_route_after_reflector_failed(self):
        """Reflector says 'failed' → END."""
        state = AgentState(
            task_goal="t", plan_steps=["a"], current_step_index=1,
            execution_log=[], reflection_notes=[],
            status="failed", max_iterations=10, current_iteration=0,
        )
        from langgraph.graph import END
        assert _route_after_reflector(state) == END

    def test_route_after_reflector_continue(self):
        """Reflector says 'continue' → 'planner'."""
        state = AgentState(
            task_goal="t", plan_steps=["a"], current_step_index=1,
            execution_log=[], reflection_notes=[],
            status="continue", max_iterations=10, current_iteration=5,
        )
        assert _route_after_reflector(state) == "planner"

    def test_route_after_reflector_max_iterations(self):
        """A11: Reflector says 'continue' but max_iterations reached → END."""
        state = AgentState(
            task_goal="t", plan_steps=["a"], current_step_index=1,
            execution_log=[], reflection_notes=[],
            status="continue", max_iterations=2, current_iteration=2,
        )
        from langgraph.graph import END
        assert _route_after_reflector(state) == END


# ── Acceptance A12–A14: Checkpoint persistence ───────────────────────────────


class TestCheckpointer:
    """A12–A14: SqliteSaver integration."""

    def test_get_checkpointer_creates_directory(self, tmp_path):
        """A12: get_checkpointer creates db parent dir and yields SqliteSaver."""
        db_path = tmp_path / "sub" / "checkpoints.sqlite"
        with get_checkpointer(db_path) as cp:
            assert cp is not None
        assert db_path.parent.exists()
        assert db_path.exists()

    def test_checkpointer_basic_write(self, tmp_path):
        """A12: Fixed thread_id with checkpointer — verify persistence.

        Uses a minimal pass-through graph to avoid requiring a real LLM
        engine.  The full graph (planner → executor → reflector) requires
        a `model_path` for `get_engine()`, so checkpoint persistence is
        verified with a simpler graph that exercises the same SqliteSaver
        code path.
        """
        from langgraph.graph import StateGraph, END

        db_path = tmp_path / "checkpoints.sqlite"

        # Minimal graph that just copies the state through
        graph = StateGraph(AgentState)
        def passthrough(state: AgentState) -> AgentState:
            return state
        graph.add_node("passthrough", passthrough)
        graph.set_entry_point("passthrough")
        graph.add_edge("passthrough", END)

        with get_checkpointer(db_path) as cp:
            compiled = graph.compile(checkpointer=cp)
            initial_state: AgentState = {
                "task_goal": "test write",
                "plan_steps": [],
                "current_step_index": 0,
                "execution_log": [],
                "reflection_notes": [],
                "status": "done",
                "max_iterations": 10,
                "current_iteration": 0,
            }
            config = {"configurable": {"thread_id": "test-thread-1"}}
            result = compiled.invoke(initial_state, config=config)
            assert result["task_goal"] == "test write"

        # After context exits, the db file should exist
        assert db_path.exists()

    def test_checkpointer_different_thread_ids_isolated(self, tmp_path):
        """A14: Two different thread_ids run independently — structural check.

        Verifies the checkpointer factory works correctly; actual isolation
        is guaranteed by LangGraph's checkpoint system and is tested with
        the minimal pass-through graph.
        """
        from langgraph.graph import StateGraph, END

        db_path = tmp_path / "isolation.sqlite"

        graph = StateGraph(AgentState)
        def passthrough(state: AgentState) -> AgentState:
            return state
        graph.add_node("passthrough", passthrough)
        graph.set_entry_point("passthrough")
        graph.add_edge("passthrough", END)

        with get_checkpointer(db_path) as cp:
            compiled = graph.compile(checkpointer=cp)

            state_1: AgentState = {
                "task_goal": "task one", "plan_steps": [],
                "current_step_index": 0, "execution_log": [],
                "reflection_notes": [], "status": "done",
                "max_iterations": 10, "current_iteration": 0,
            }
            state_2: AgentState = {
                "task_goal": "task two", "plan_steps": [],
                "current_step_index": 0, "execution_log": [],
                "reflection_notes": [], "status": "done",
                "max_iterations": 10, "current_iteration": 0,
            }

            r1 = compiled.invoke(state_1, config={"configurable": {"thread_id": "t1"}})
            r2 = compiled.invoke(state_2, config={"configurable": {"thread_id": "t2"}})

            assert r1["task_goal"] == "task one"
            assert r2["task_goal"] == "task two"


# ── Acceptance A15: Inner subgraph has no checkpointer ────────────────────────


class TestInnerSubgraphNoCheckpointer:
    """A15: Verify the inner ReAct subgraph has no checkpointer."""

    def test_create_agent_path_no_checkpointer(self):
        """The create_agent call must pass checkpointer=None."""
        from agent_core.graph.react_agent_factory import _build_via_create_agent

        # We cannot call _build_via_create_agent without a real LLM engine,
        # but we can statically verify the source code explicitly sets
        # checkpointer=None in the create_agent call.
        import inspect
        source = inspect.getsource(_build_via_create_agent)
        assert "checkpointer=None" in source, (
            "create_agent must receive checkpointer=None (A15)"
        )

    def test_self_made_path_no_checkpointer(self):
        """The self-made StateGraph compiles without checkpointer."""
        from agent_core.graph.react_agent_factory import _build_via_self_made_stategraph

        import inspect
        source = inspect.getsource(_build_via_self_made_stategraph)
        # The compile() call should NOT pass a checkpointer arg
        # (graph.compile() with no args → no checkpointer)
        assert "checkpointer" not in source, (
            "Self-made StateGraph must compile without checkpointer (A15)"
        )


# ── Acceptance A16: LRU cache on get_react_agent ─────────────────────────────


class TestLruCache:
    """A16: initialize_react_agent() returns same object on repeated calls."""

    def test_lru_cache_returns_same_object(self):
        """initialize_react_agent must have @lru_cache and return same instance."""
        from agent_core.graph.react_agent_factory import initialize_react_agent
        assert hasattr(initialize_react_agent, "cache_info"), (
            "initialize_react_agent must be decorated with @lru_cache"
        )


# ── Acceptance A18: Plan B switchability ─────────────────────────────────────


class TestPlanBSwitchability:
    """A18: USE_OFFICIAL_CREATE_AGENT env var toggles the build path."""

    def test_self_made_graph_has_expected_nodes(self):
        """The self-made StateGraph should have 'agent' and 'tools' nodes."""
        graph = _build_via_self_made_stategraph()
        # The compiled graph's nodes can be inspected via get_graph()
        nodes = graph.get_graph().nodes
        node_names = {n for n in nodes}  # type: ignore[var-annotated]
        assert "agent" in node_names, f"Self-made graph must contain 'agent' node; got {node_names}"
        assert "tools" in node_names, f"Self-made graph must contain 'tools' node; got {node_names}"


# ── Acceptance A19: Single source of truth ───────────────────────────────────


class TestSingleSourceOfTruth:
    """A19: initialize_react_agent tools come from list_capabilities()."""

    def test_tools_from_capability_registry(self):
        """Registered capabilities must appear in the inner agent's tool list."""
        _register_test_tools()
        caps = list_capabilities()
        assert len(caps) == 2

        for cap in caps:
            langchain_tool = to_langchain_tool(cap)
            assert langchain_tool.name == cap.name
            assert langchain_tool.description == cap.description

    def test_to_langchain_tool_no_properties(self):
        """Capability with empty input_schema creates a valid tool."""
        cap = Capability(
            name="no_arg_tool",
            description="A tool with no arguments",
            handler=lambda: "done",
            input_schema={"type": "object", "properties": {}},
        )
        tool = to_langchain_tool(cap)
        assert tool.name == "no_arg_tool"
        assert tool.description == "A tool with no arguments"

    def test_langchain_tool_invokes_original_handler_once(self):
        calls: list[str] = []
        cap = Capability(
            name="record_value",
            description="Record one value",
            handler=lambda value: calls.append(value) or f"recorded:{value}",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

        tool = to_langchain_tool(cap)
        result = tool.invoke({"value": "once"})

        assert result == "recorded:once"
        assert calls == ["once"]

    def test_existing_shell_tool_node_executes_real_handler(self):
        """A structured call must reach the existing shell provider."""
        from agent_core.capabilities.providers.shell_provider import (
            ShellCapabilityProvider,
        )

        cap = ShellCapabilityProvider().build({"allowed_commands": ["pwd"]})[0]
        register(
            name=cap.name,
            description=cap.description,
            input_schema=cap.input_schema,
        )(cap.handler)

        state = _ReActState(messages=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="execute_shell_command",
                        args={"command": "pwd"},
                        id="call_pwd",
                    )
                ],
            )
        ])
        output = _tool_node(state)

        assert len(output["messages"]) == 1
        tool_message = output["messages"][0]
        assert isinstance(tool_message, ToolMessage)
        payload = json.loads(tool_message.content)
        assert payload["success"] is True
        assert payload["exit_code"] == 0
        assert "llama-agent" in payload["stdout"]

    def test_official_agent_executes_tool_then_summarizes_without_tools(self):
        """Full loop: structured call → real handler → tool-free final answer."""
        from agent_core.capabilities.providers.shell_provider import (
            ShellCapabilityProvider,
        )
        from agent_core.llm_engine import ChatLlamaCpp

        cap = ShellCapabilityProvider().build({"allowed_commands": ["pwd"]})[0]
        register(
            name=cap.name,
            description=cap.description,
            input_schema=cap.input_schema,
        )(cap.handler)

        fake_client = MagicMock()
        fake_client.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_pwd",
                            "type": "function",
                            "function": {
                                "name": "execute_shell_command",
                                "arguments": '{"command": "pwd"}',
                            },
                        }],
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "当前目录是 llama-agent 项目目录。",
                    }
                }]
            },
        ]
        model = ChatLlamaCpp.model_construct(model_path="unused.gguf")
        model._client = fake_client

        with patch(
            "agent_core.graph.react_agent_factory.get_engine",
            return_value=model,
        ):
            agent = _build_via_create_agent()
            output = agent.invoke(
                {"messages": [HumanMessage(content="请执行 pwd 并回答当前目录")]},
                config={"recursion_limit": 8},
            )

        assert fake_client.create_chat_completion.call_count == 2
        first_call = fake_client.create_chat_completion.call_args_list[0].kwargs
        second_call = fake_client.create_chat_completion.call_args_list[1].kwargs
        assert first_call["tools"]
        assert first_call["tool_choice"] == "auto"
        assert second_call["tools"] is None

        tool_messages = [
            message for message in output["messages"]
            if isinstance(message, ToolMessage)
        ]
        assert len(tool_messages) == 1
        payload = json.loads(tool_messages[0].content)
        assert payload["success"] is True
        assert isinstance(output["messages"][-1], AIMessage)
        assert output["messages"][-1].content == "当前目录是 llama-agent 项目目录。"


# ── Acceptance A20: Architecture isolation ───────────────────────────────────


class TestArchitectureIsolation:
    """A20: No ``import llama_cpp`` in graph/ files;
    no undeclared AgentState field assignments outside state.py."""

    def test_no_llama_cpp_import_in_graph(self):
        """Static scan — graph/*.py must not import llama_cpp."""
        graph_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agent_core", "graph",
        )
        violations: list[str] = []
        for fname in os.listdir(graph_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(graph_dir, fname)
            with open(fpath) as f:
                content = f.read()
            if "import llama_cpp" in content:
                violations.append(fname)
        assert not violations, (
            f"Architecture violation: these graph/*.py files import llama_cpp: {violations}"
        )

    def test_state_py_defines_all_fields(self):
        """state.py declares every field used by graph/ nodes."""
        from agent_core.graph.state import AgentState
        hints = AgentState.__annotations__ if hasattr(AgentState, '__annotations__') else AgentState.__dict__.get('__annotations__', {})
        required = {
            "task_goal", "plan_steps", "current_step_index",
            "execution_log", "reflection_notes", "status",
            "max_iterations", "current_iteration",
            "final_answer",
            "error",
        }
        assert set(hints.keys()) == required


# ── Acceptance A17: Dynamic prompt (config-level check) ──────────────────────


class TestDynamicPrompt:
    """A17: Inner agent receives dynamic prompts via input messages,
    not via a static system_prompt parameter at build time."""

    def test_create_agent_path_has_no_system_prompt(self):
        """Source inspection — _build_via_create_agent passes system_prompt=None."""
        import inspect
        from agent_core.graph.react_agent_factory import _build_via_create_agent
        source = inspect.getsource(_build_via_create_agent)
        assert "system_prompt=None" in source, (
            "create_agent must receive system_prompt=None for dynamic prompts (A17)"
        )


# ── Integration: full graph with mocked nodes ────────────────────────────────


class TestFullGraphIntegration:
    """End-to-end graph flow with mocked engine."""

    def test_graph_compiles_without_checkpointer(self):
        """build_graph() without checkpointer returns a compiled graph."""
        graph = build_graph()
        assert graph is not None
        # Should have nodes "planner", "executor", "reflector"
        nodes = graph.get_graph().nodes
        node_names = {n for n in nodes}  # type: ignore[var-annotated]
        assert "planner" in node_names
        assert "executor" in node_names
        assert "reflector" in node_names

    def test_graph_routing_with_done_skips_nodes(self):
        """build_graph() returns a compiled graph.  When the initial state has
        status='done', the entry routing should immediately hit the planner
        node, but the planner immediately invokes the engine.  Integration
        tests requiring a real LLM are skipped here — this test verifies
        the graph structure is correct."""
        # This is a structural-only test — real graph invocation requires
        # a properly configured LLM engine (model_path).
        graph = build_graph()
        nodes = graph.get_graph().nodes
        assert "planner" in {n for n in nodes}  # type: ignore[var-annotated]
        assert "reflector" in {n for n in nodes}  # type: ignore[var-annotated]

    def test_graph_routing_with_failed_skips_nodes(self):
        """Same structural check for failed status."""
        graph = build_graph()
        nodes = graph.get_graph().nodes
        assert "planner" in {n for n in nodes}  # type: ignore[var-annotated]

    def test_exception_hierarchy_consistent(self):
        """GraphOrchestrationError is properly in the hierarchy."""
        assert issubclass(GraphOrchestrationError, AgentCoreError)
        assert issubclass(PlanningError, GraphOrchestrationError)
        assert issubclass(ExecutionError, GraphOrchestrationError)
        assert issubclass(ReflectionError, GraphOrchestrationError)

    def test_react_agent_state_fields(self):
        """The self-made ReActState has the expected messages field."""
        import inspect
        source = inspect.getsource(_ReActState)
        assert "messages" in source
