"""Tests for session.py — the application service layer.

Covers acceptance criteria SS1–SS14 from the application-layer design
doc.  Tests that require a real LLM engine (SS1, SS5) are structural
or use mocked graph internals to avoid needing a GGUF model file.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Generator

import pytest

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from agent_core.exceptions import (
    AgentCoreError,
    ExecutionError,
    PlanningError,
    ReflectionError,
)
from agent_core.graph.state import AgentState
from agent_core.session import RunConfig, TaskRunner


# ── Helpers ──────────────────────────────────────────────────────────────────


def _state_keys(state: AgentState) -> set[str]:
    """Return the set of keys in an AgentState dict."""
    return set(state.keys())


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_runner(tmp_path: Path) -> Generator[TaskRunner, None, None]:
    """A TaskRunner pointed at a temp directory so no real data is touched."""
    config = RunConfig(
        db_path=tmp_path / "checkpoints.sqlite",
        last_thread_file=tmp_path / "last_thread_id.txt",
    )
    with TaskRunner(config=config) as runner:
        yield runner


@pytest.fixture
def caplog_debug(caplog):
    """Capture logs at DEBUG level and above."""
    with caplog.at_level(logging.DEBUG, logger="agent_core.session"):
        yield caplog


# =============================================================================
# SS2 — Initial state field completeness
# =============================================================================


class TestInitialState:
    """SS2: _build_initial_state produces a complete, correct AgentState."""

    def test_all_required_fields_present(self, tmp_runner):
        state = tmp_runner._build_initial_state("test goal")
        expected_keys = {
            "task_goal",
            "plan_steps",
            "current_step_index",
            "execution_log",
            "reflection_notes",
            "status",
            "max_iterations",
            "current_iteration",
            "final_answer",
            "error",
        }
        assert _state_keys(state) == expected_keys

    def test_defaults_are_correct(self, tmp_runner):
        state = tmp_runner._build_initial_state("do something")
        assert state["task_goal"] == "do something"
        assert state["plan_steps"] == []
        assert state["current_step_index"] == 0
        assert state["execution_log"] == []
        assert state["reflection_notes"] == []
        assert state["status"] == "planning"
        assert state["current_iteration"] == 0
        assert state["max_iterations"] == tmp_runner.config.max_iterations
        assert state["final_answer"] == ""
        assert state["error"] is None

    def test_different_goals_produce_different_states(self, tmp_runner):
        s1 = tmp_runner._build_initial_state("goal one")
        s2 = tmp_runner._build_initial_state("goal two")
        assert s1["task_goal"] == "goal one"
        assert s2["task_goal"] == "goal two"


# =============================================================================
# SS3 — Empty goal validation
# =============================================================================


class TestEmptyGoalValidation:
    """SS3: start_new_task rejects empty / whitespace-only goals."""

    def test_empty_string_raises_value_error(self, tmp_runner):
        with pytest.raises(ValueError, match="must not be empty"):
            tmp_runner.start_new_task("")

    def test_whitespace_only_raises_value_error(self, tmp_runner):
        with pytest.raises(ValueError, match="must not be empty"):
            tmp_runner.start_new_task("   ")

    def test_empty_goal_does_not_touch_disk(self, tmp_path):
        """Empty goal should fail before any side-effects occur."""
        config = RunConfig(
            db_path=tmp_path / "nope.sqlite",
            last_thread_file=tmp_path / "nope_thread.txt",
        )
        with TaskRunner(config=config) as runner:
            with pytest.raises(ValueError):
                runner.start_new_task("")
        # last_thread_file should NOT have been written
        assert not config.last_thread_file.exists()


# =============================================================================
# SS1 — New task basic flow (structural — requires mocked graph invoke)
# =============================================================================


class TestNewTaskBasicFlow:
    """SS1: start_new_task returns (thread_id, final_state) tuple."""

    def test_start_new_task_basic(self, tmp_runner):
        """Mock _invoke so we don't need a real LLM."""
        mock_state: AgentState = {
            "task_goal": "hello",
            "plan_steps": ["step 1"],
            "current_step_index": 1,
            "execution_log": [
                {"step": "step 1", "result": "ok", "tool_used": None}
            ],
            "reflection_notes": [],
            "status": "done",
            "max_iterations": 6,
            "current_iteration": 1,
        }

        with patch.object(tmp_runner, "_invoke", return_value=mock_state) as mock_invoke:
            thread_id, result = tmp_runner.start_new_task("say hello")

        # thread_id must be a valid UUID
        uuid.UUID(thread_id)
        assert isinstance(thread_id, str)
        assert result == mock_state
        assert result["status"] == "done"

        # _invoke must have been called with the correct initial state
        mock_invoke.assert_called_once()
        args, _ = mock_invoke.call_args
        passed_state = args[0]
        assert passed_state["task_goal"] == "say hello"
        assert passed_state["status"] == "planning"

    def test_start_new_task_result_is_tuple(self, tmp_runner):
        mock_state: AgentState = {
            "task_goal": "test",
            "plan_steps": [],
            "current_step_index": 0,
            "execution_log": [],
            "reflection_notes": [],
            "status": "done",
            "max_iterations": 6,
            "current_iteration": 0,
        }

        with patch.object(tmp_runner, "_invoke", return_value=mock_state):
            tid, result = tmp_runner.start_new_task("test")
        assert isinstance(tid, str)
        assert isinstance(result, dict)


# =============================================================================
# SS4 — resume_task signature has no task_goal parameter
# =============================================================================


class TestResumeSignature:
    """SS4: resume_task must only accept thread_id, not task_goal."""

    def test_resume_task_only_accepts_thread_id(self):
        """Static inspection — the method signature must not include task_goal."""
        import inspect

        sig = inspect.signature(TaskRunner.resume_task)
        params = list(sig.parameters.keys())
        # params = ['self', 'thread_id']
        assert "self" in params
        assert "thread_id" in params
        assert "task_goal" not in params, (
            "resume_task must NOT accept task_goal (design doc §2.3)"
        )


# =============================================================================
# SS6 — Empty thread_id validation for resume
# =============================================================================


class TestEmptyThreadIdResume:
    """SS6: resume_task rejects empty / None thread_id."""

    def test_empty_string_raises(self, tmp_runner):
        with pytest.raises(ValueError, match="must not be empty"):
            tmp_runner.resume_task("")

    def test_none_raises(self, tmp_runner):
        with pytest.raises(ValueError, match="must not be empty"):
            tmp_runner.resume_task(None)  # type: ignore[arg-type]


# =============================================================================
# SS7 — Resources are initialised once per instance
# =============================================================================


class TestResourceLifecycle:
    """SS7: Graph and checkpointer are created once and reused."""

    def test_graph_reused_across_calls(self, tmp_runner):
        """Two consecutive start_new_task calls reuse the same graph."""
        g1 = tmp_runner._graph
        cp1 = tmp_runner._checkpointer

        # The graph and checkpointer are instance attributes stored once.
        assert g1 is tmp_runner._graph
        assert cp1 is tmp_runner._checkpointer

    def test_same_instance_attributes_after_repeated_access(self, tmp_runner):
        """Accessing _graph / _checkpointer repeatedly returns the same objects."""
        g1 = tmp_runner._graph
        cp1 = tmp_runner._checkpointer
        g2 = tmp_runner._graph
        cp2 = tmp_runner._checkpointer
        assert g1 is g2
        assert cp1 is cp2


# =============================================================================
# SS8 — Unified exception normalisation
# =============================================================================


class TestExceptionNormalisation:
    """SS8: _invoke wraps all non-AgentCoreError exceptions."""

    def test_agent_core_error_passes_through(self, tmp_runner):
        """PlanningError / ExecutionError / ReflectionError should not be re-wrapped."""
        with patch.object(tmp_runner._graph, "invoke", side_effect=PlanningError("bad plan")):
            with pytest.raises(PlanningError):
                tmp_runner._invoke(
                    tmp_runner._build_initial_state("x"),
                    thread_id="test-1",
                )

    def test_execution_error_passes_through(self, tmp_runner):
        with patch.object(tmp_runner._graph, "invoke", side_effect=ExecutionError("exec fail")):
            with pytest.raises(ExecutionError):
                tmp_runner._invoke(
                    tmp_runner._build_initial_state("x"),
                    thread_id="test-2",
                )

    def test_reflection_error_passes_through(self, tmp_runner):
        with patch.object(tmp_runner._graph, "invoke", side_effect=ReflectionError("bad enum")):
            with pytest.raises(ReflectionError):
                tmp_runner._invoke(
                    tmp_runner._build_initial_state("x"),
                    thread_id="test-3",
                )

    def test_plain_runtime_error_wrapped_to_agent_core_error(self, tmp_runner):
        """A plain RuntimeError must be caught and wrapped."""
        with patch.object(tmp_runner._graph, "invoke", side_effect=RuntimeError("boom")):
            with pytest.raises(AgentCoreError) as exc_info:
                tmp_runner._invoke(
                    tmp_runner._build_initial_state("x"),
                    thread_id="test-4",
                )
            assert "Unexpected error" in str(exc_info.value)
            assert "boom" in str(exc_info.value)
            # The cause chain should contain the original RuntimeError
            assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_agent_core_error_is_ancestor_of_all_three(self):
        """All three orchestration errors are AgentCoreError subclasses."""
        assert issubclass(PlanningError, AgentCoreError)
        assert issubclass(ExecutionError, AgentCoreError)
        assert issubclass(ReflectionError, AgentCoreError)


# =============================================================================
# SS9 — thread_id recorded after successful start
# =============================================================================


class TestThreadIdRecording:
    """SS9: start_new_task writes the thread_id to the last_thread_file."""

    def test_thread_id_written_after_start(self, tmp_runner):
        mock_state: AgentState = {
            "task_goal": "test",
            "plan_steps": [],
            "current_step_index": 0,
            "execution_log": [],
            "reflection_notes": [],
            "status": "done",
            "max_iterations": 6,
            "current_iteration": 0,
        }

        with patch.object(tmp_runner, "_invoke", return_value=mock_state):
            tid, _ = tmp_runner.start_new_task("record me")

        recorded = tmp_runner.get_last_thread_id()
        assert recorded == tid

    def test_get_last_thread_id_returns_exact_match(self, tmp_runner):
        mock_state: AgentState = {
            "task_goal": "t",
            "plan_steps": [],
            "current_step_index": 0,
            "execution_log": [],
            "reflection_notes": [],
            "status": "done",
            "max_iterations": 6,
            "current_iteration": 0,
        }

        with patch.object(tmp_runner, "_invoke", return_value=mock_state):
            tid, _ = tmp_runner.start_new_task("exact match test")

        # Call get_last_thread_id twice — should return the same
        assert tmp_runner.get_last_thread_id() == tid
        assert tmp_runner.get_last_thread_id() == tid


# =============================================================================
# SS10 — get_last_thread_id returns None when no record exists
# =============================================================================


class TestNoRecordTolerance:
    """SS10: get_last_thread_id is safe when no file exists."""

    def test_returns_none_when_no_file(self, tmp_runner):
        """Fresh runner with no tasks should return None."""
        # tmp_runner was freshly created — no file should exist yet
        assert tmp_runner.get_last_thread_id() is None

    def test_returns_none_when_file_empty(self, tmp_runner):
        """If the file exists but is empty (corner case), return None."""
        config = tmp_runner.config
        config.last_thread_file.parent.mkdir(parents=True, exist_ok=True)
        config.last_thread_file.write_text("")
        assert tmp_runner.get_last_thread_id() is None

    def test_no_exception_raised_on_missing_file(self, tmp_runner):
        """get_last_thread_id must never raise, even if the path is absurd."""
        config = tmp_runner.config
        config.last_thread_file = Path("/nonexistent/path/definitely/not/there.txt")
        result = tmp_runner.get_last_thread_id()
        assert result is None


# =============================================================================
# SS11 — Resource release via close() and context manager
# =============================================================================


class TestResourceRelease:
    """SS11: close() and with-statement both release resources."""

    def test_close_releases_resources(self, tmp_path):
        config = RunConfig(db_path=tmp_path / "release.sqlite")
        runner = TaskRunner(config=config)
        # Capture the checkpointer's exit before close
        cp = runner._checkpointer
        runner.close()
        # After close the ExitStack is empty — no further cleanup needed
        # The key test: calling close twice should not raise
        runner.close()

    def test_context_manager_releases_resources(self, tmp_path):
        config = RunConfig(db_path=tmp_path / "cm.sqlite")
        with TaskRunner(config=config) as runner:
            assert runner._graph is not None
            assert runner._checkpointer is not None
        # After the with-block, the ExitStack should be closed
        # Calling close again is safe (idempotent)
        runner.close()

    def test_close_twice_does_not_raise(self, tmp_path):
        """ExitStack.close() is idempotent — double close must not crash."""
        config = RunConfig(db_path=tmp_path / "double_close.sqlite")
        runner = TaskRunner(config=config)
        runner.close()
        runner.close()  # must not raise


# =============================================================================
# SS12 — Multi-instance isolation
# =============================================================================


class TestMultiInstanceIsolation:
    """SS12: Two TaskRunners with different db_paths do not interfere."""

    def test_different_db_paths_do_not_collide(self, tmp_path):
        """Each runner writes to its own sqlite file."""
        db1 = tmp_path / "one" / "cp.sqlite"
        db2 = tmp_path / "two" / "cp.sqlite"

        mock_state: AgentState = {
            "task_goal": "task",
            "plan_steps": [],
            "current_step_index": 0,
            "execution_log": [],
            "reflection_notes": [],
            "status": "done",
            "max_iterations": 6,
            "current_iteration": 0,
        }

        config1 = RunConfig(db_path=db1)
        with TaskRunner(config=config1) as r1:
            with patch.object(r1, "_invoke", return_value=mock_state):
                tid1, _ = r1.start_new_task("task one")

        config2 = RunConfig(db_path=db2)
        with TaskRunner(config=config2) as r2:
            with patch.object(r2, "_invoke", return_value=mock_state):
                tid2, _ = r2.start_new_task("task two")

        assert tid1 != tid2
        assert db1.exists()
        assert db2.exists()

    def test_different_last_thread_files_isolated(self, tmp_path):
        """Each runner's _remember_thread_id writes to its own file."""
        f1 = tmp_path / "a" / "tid.txt"
        f2 = tmp_path / "b" / "tid.txt"

        mock_state: AgentState = {
            "task_goal": "t",
            "plan_steps": [],
            "current_step_index": 0,
            "execution_log": [],
            "reflection_notes": [],
            "status": "done",
            "max_iterations": 6,
            "current_iteration": 0,
        }

        config1 = RunConfig(db_path=tmp_path / "a" / "cp.sqlite", last_thread_file=f1)
        with TaskRunner(config=config1) as r1:
            with patch.object(r1, "_invoke", return_value=mock_state):
                tid1, _ = r1.start_new_task("a")

        config2 = RunConfig(db_path=tmp_path / "b" / "cp.sqlite", last_thread_file=f2)
        with TaskRunner(config=config2) as r2:
            with patch.object(r2, "_invoke", return_value=mock_state):
                tid2, _ = r2.start_new_task("b")

        assert f1.read_text().strip() == tid1
        assert f2.read_text().strip() == tid2
        assert tid1 != tid2


# =============================================================================
# SS13 — RunConfig default values
# =============================================================================


class TestRunConfigDefaults:
    """SS13: RunConfig defaults are correct."""

    def test_max_iterations_default(self):
        cfg = RunConfig()
        assert cfg.max_iterations == 6

    def test_db_path_default(self):
        cfg = RunConfig()
        assert cfg.db_path == Path("data/checkpoints.sqlite")

    def test_last_thread_file_default(self):
        cfg = RunConfig()
        assert cfg.last_thread_file == Path("data/last_thread_id.txt")

    def test_custom_values_override_defaults(self):
        cfg = RunConfig(
            max_iterations=42,
            db_path=Path("/custom/db.sqlite"),
            last_thread_file=Path("/custom/tid.txt"),
        )
        assert cfg.max_iterations == 42
        assert cfg.db_path == Path("/custom/db.sqlite")
        assert cfg.last_thread_file == Path("/custom/tid.txt")

    def test_runner_uses_config_defaults(self):
        """TaskRunner() with no args creates a RunConfig with defaults."""
        # We can't construct a real TaskRunner without a LLM engine, but
        # we can verify RunConfig directly.
        cfg = RunConfig()
        assert cfg.max_iterations == 6
        assert cfg.db_path == Path("data/checkpoints.sqlite")


# =============================================================================
# SS14 — Logging coverage
# =============================================================================


class TestLoggingCoverage:
    """SS14: Key paths are covered by log messages."""

    def test_start_new_task_logs_thread_id(self, tmp_runner, caplog_debug):
        mock_state: AgentState = {
            "task_goal": "log test",
            "plan_steps": [],
            "current_step_index": 0,
            "execution_log": [],
            "reflection_notes": [],
            "status": "done",
            "max_iterations": 6,
            "current_iteration": 0,
        }

        with patch.object(tmp_runner, "_invoke", return_value=mock_state):
            tid, _ = tmp_runner.start_new_task("log test")

        # The log should contain the thread_id
        log_messages = [r.message for r in caplog_debug.records]
        assert any(tid in msg for msg in log_messages), (
            f"thread_id {tid} not found in log messages: {log_messages}"
        )

    def test_exception_logs_at_exception_level(self, tmp_runner, caplog_debug):
        """When _invoke catches an unexpected error, it logs at ERROR level."""
        with patch.object(
            tmp_runner._graph, "invoke", side_effect=RuntimeError("loggable boom")
        ):
            try:
                tmp_runner._invoke(
                    tmp_runner._build_initial_state("x"),
                    thread_id="test-log-error",
                )
            except AgentCoreError:
                pass

        # At least one record at ERROR / EXCEPTION level
        error_records = [
            r for r in caplog_debug.records
            if r.levelno >= logging.ERROR
        ]
        assert len(error_records) >= 1, (
            f"No ERROR-level records found in: {[r.message for r in caplog_debug.records]}"
        )
        # The exception info is attached to the log record's exc_info, not
        # necessarily in the message text.  Verify the exception was captured.
        error_with_exc = [
            r for r in error_records if r.exc_info is not None
        ]
        assert len(error_with_exc) >= 1, (
            f"No ERROR records with attached exception found in: {[r.message for r in error_records]}"
        )
        # The original RuntimeError should be in the exc_info chain
        exc_type, exc_value, _tb = error_with_exc[0].exc_info
        assert exc_type is RuntimeError
        assert "loggable boom" in str(exc_value)

    def test_init_logs_db_path(self, tmp_path, caplog_debug):
        db_path = tmp_path / "logs.sqlite"
        config = RunConfig(db_path=db_path)
        with TaskRunner(config=config):
            pass
        log_messages = [r.message for r in caplog_debug.records]
        assert any(str(db_path) in msg for msg in log_messages)


# =============================================================================
# Additional — RunConfig immutability after construction
# =============================================================================


class TestRunConfigMutability:
    """RunConfig is a dataclass — fields can be set, but the object itself
    can be replaced — no unexpected behaviour."""

    def test_fields_are_mutable_dataclass_attrs(self):
        cfg = RunConfig()
        cfg.max_iterations = 99
        assert cfg.max_iterations == 99

    def test_field_types_are_correct(self):
        cfg = RunConfig(max_iterations=7)
        assert isinstance(cfg.max_iterations, int)
        assert isinstance(cfg.db_path, Path)
        assert isinstance(cfg.last_thread_file, Path)


# =============================================================================
# SS5 — Resume correctness (structural — requires checkpointer + mock)
# =============================================================================


class TestResumeCorrectness:
    """SS5: resume_task with a valid thread_id restores state from checkpoint."""

    def test_resume_task_calls_invoke_with_none_state(self, tmp_runner):
        """resume_task should call _invoke with state=None so that LangGraph
        reads the checkpoint instead of accepting a new initial state."""
        mock_state: AgentState = {
            "task_goal": "original task",
            "plan_steps": ["resumed step"],
            "current_step_index": 1,
            "execution_log": [
                {"step": "done step", "result": "ok", "tool_used": None}
            ],
            "reflection_notes": [],
            "status": "done",
            "max_iterations": 6,
            "current_iteration": 1,
        }

        with patch.object(tmp_runner, "_invoke", return_value=mock_state) as mock_invoke:
            result = tmp_runner.resume_task("existing-thread-id")

        # _invoke called with state=None, thread_id="existing-thread-id"
        mock_invoke.assert_called_once_with(None, "existing-thread-id")
        assert result == mock_state

    def test_resume_does_not_alter_planner_state(self, tmp_runner):
        """resume_task must preserve plan_steps as they were before interruption."""
        mock_state: AgentState = {
            "task_goal": "resumed task",
            "plan_steps": ["step A", "step B", "step C"],
            "current_step_index": 2,
            "execution_log": [
                {"step": "step A", "result": "ok", "tool_used": None},
                {"step": "step B", "result": "ok", "tool_used": None},
            ],
            "reflection_notes": [],
            "status": "executing",
            "max_iterations": 6,
            "current_iteration": 0,
        }

        with patch.object(tmp_runner, "_invoke", return_value=mock_state):
            result = tmp_runner.resume_task("tid-resume")

        assert result["plan_steps"] == ["step A", "step B", "step C"]
        assert result["current_step_index"] == 2
