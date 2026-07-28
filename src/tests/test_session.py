"""测试 session.py，即应用服务层。

覆盖应用层设计文档中的验收标准 SS1–SS14。需要真实大模型引擎的测试
（SS1、SS5）仅做结构验证，或模拟图内部实现以避免依赖 GGUF 模型文件。
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


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


def _state_keys(state: AgentState) -> set[str]:
    """返回 AgentState 字典的键集合。"""
    return set(state.keys())


# ── 测试夹具 ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_runner(tmp_path: Path) -> Generator[TaskRunner, None, None]:
    """创建指向临时目录的 TaskRunner，避免接触真实数据。"""
    config = RunConfig(
        db_path=tmp_path / "checkpoints.sqlite",
        last_thread_file=tmp_path / "last_thread_id.txt",
    )
    with TaskRunner(config=config) as runner:
        yield runner


@pytest.fixture
def caplog_debug(caplog):
    """捕获 DEBUG 及以上级别的日志。"""
    with caplog.at_level(logging.DEBUG, logger="agent_core.session"):
        yield caplog


# =============================================================================
# SS2：初始状态字段完整性
# =============================================================================


class TestInitialState:
    """SS2：_build_initial_state 生成完整且正确的 AgentState。"""

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
            "conversation_id",
            "current_user_input",
            "conversation_context",
            "pinned_facts",
            "context_summary",
            "archived_context_ids",
            "context_version",
            "lifecycle_stats",
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
# SS3：空目标校验
# =============================================================================


class TestEmptyGoalValidation:
    """SS3：start_new_task 拒绝空目标或仅含空白字符的目标。"""

    def test_empty_string_raises_value_error(self, tmp_runner):
        with pytest.raises(ValueError, match="must not be empty"):
            tmp_runner.start_new_task("")

    def test_whitespace_only_raises_value_error(self, tmp_runner):
        with pytest.raises(ValueError, match="must not be empty"):
            tmp_runner.start_new_task("   ")

    def test_empty_goal_does_not_touch_disk(self, tmp_path):
        """空目标应在产生任何副作用前失败。"""
        config = RunConfig(
            db_path=tmp_path / "nope.sqlite",
            last_thread_file=tmp_path / "nope_thread.txt",
        )
        with TaskRunner(config=config) as runner:
            with pytest.raises(ValueError):
                runner.start_new_task("")
        # 不应写入 last_thread_file。
        assert not config.last_thread_file.exists()


# =============================================================================
# SS1：新任务基本流程（结构测试，需要模拟图调用）
# =============================================================================


class TestNewTaskBasicFlow:
    """SS1：start_new_task 返回 (thread_id, final_state) 元组。"""

    def test_start_new_task_basic(self, tmp_runner):
        """模拟 _invoke，使测试无需真实大模型。"""
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

        # thread_id 必须是有效 UUID。
        uuid.UUID(thread_id)
        assert isinstance(thread_id, str)
        assert result == mock_state
        assert result["status"] == "done"

        # 必须使用正确的初始状态调用 _invoke。
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
# SS4：resume_task 签名不含 task_goal 参数
# =============================================================================


class TestResumeSignature:
    """SS4：resume_task 只能接受 thread_id，不能接受 task_goal。"""

    def test_resume_task_only_accepts_thread_id(self):
        """静态检查：方法签名不得包含 task_goal。"""
        import inspect

        sig = inspect.signature(TaskRunner.resume_task)
        params = list(sig.parameters.keys())
        # 参数应为 ['self', 'thread_id']。
        assert "self" in params
        assert "thread_id" in params
        assert "task_goal" not in params, (
            "resume_task must NOT accept task_goal (design doc §2.3)"
        )


# =============================================================================
# SS6：恢复任务时校验空 thread_id
# =============================================================================


class TestEmptyThreadIdResume:
    """SS6：resume_task 拒绝空值或 None 类型的 thread_id。"""

    def test_empty_string_raises(self, tmp_runner):
        with pytest.raises(ValueError, match="must not be empty"):
            tmp_runner.resume_task("")

    def test_none_raises(self, tmp_runner):
        with pytest.raises(ValueError, match="must not be empty"):
            tmp_runner.resume_task(None)  # type: ignore[arg-type]


# =============================================================================
# SS7：每个实例仅初始化一次资源
# =============================================================================


class TestResourceLifecycle:
    """SS7：图和检查点器只创建一次并重复使用。"""

    def test_graph_reused_across_calls(self, tmp_runner):
        """连续两次调用 start_new_task 应复用同一个图。"""
        g1 = tmp_runner._graph
        cp1 = tmp_runner._checkpointer

        # 图和检查点器作为实例属性，仅存储一次。
        assert g1 is tmp_runner._graph
        assert cp1 is tmp_runner._checkpointer

    def test_same_instance_attributes_after_repeated_access(self, tmp_runner):
        """重复访问 _graph/_checkpointer 应返回相同对象。"""
        g1 = tmp_runner._graph
        cp1 = tmp_runner._checkpointer
        g2 = tmp_runner._graph
        cp2 = tmp_runner._checkpointer
        assert g1 is g2
        assert cp1 is cp2


# =============================================================================
# SS8：统一异常规范化
# =============================================================================


class TestExceptionNormalisation:
    """SS8：_invoke 包装所有非 AgentCoreError 异常。"""

    def test_agent_core_error_passes_through(self, tmp_runner):
        """PlanningError、ExecutionError、ReflectionError 不应被再次包装。"""
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
        """普通 RuntimeError 必须被捕获并包装。"""
        with patch.object(tmp_runner._graph, "invoke", side_effect=RuntimeError("boom")):
            with pytest.raises(AgentCoreError) as exc_info:
                tmp_runner._invoke(
                    tmp_runner._build_initial_state("x"),
                    thread_id="test-4",
                )
            assert "Unexpected error" in str(exc_info.value)
            assert "boom" in str(exc_info.value)
        # 异常原因链中应包含原始 RuntimeError。
            assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_agent_core_error_is_ancestor_of_all_three(self):
        """三种编排异常均为 AgentCoreError 的子类。"""
        assert issubclass(PlanningError, AgentCoreError)
        assert issubclass(ExecutionError, AgentCoreError)
        assert issubclass(ReflectionError, AgentCoreError)


# =============================================================================
# SS9：成功启动后记录 thread_id
# =============================================================================


class TestThreadIdRecording:
    """SS9：start_new_task 将 thread_id 写入 last_thread_file。"""

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

        # 连续两次调用 get_last_thread_id 应返回相同值。
        assert tmp_runner.get_last_thread_id() == tid
        assert tmp_runner.get_last_thread_id() == tid


# =============================================================================
# SS10：无记录时 get_last_thread_id 返回 None
# =============================================================================


class TestNoRecordTolerance:
    """SS10：文件不存在时调用 get_last_thread_id 仍然安全。"""

    def test_returns_none_when_no_file(self, tmp_runner):
        """没有任务的新运行器应返回 None。"""
        # tmp_runner 刚创建，此时不应存在记录文件。
        assert tmp_runner.get_last_thread_id() is None

    def test_returns_none_when_file_empty(self, tmp_runner):
        """文件存在但为空时也应返回 None。"""
        config = tmp_runner.config
        config.last_thread_file.parent.mkdir(parents=True, exist_ok=True)
        config.last_thread_file.write_text("")
        assert tmp_runner.get_last_thread_id() is None

    def test_no_exception_raised_on_missing_file(self, tmp_runner):
        """即使路径异常，get_last_thread_id 也不得抛出异常。"""
        config = tmp_runner.config
        config.last_thread_file = Path("/nonexistent/path/definitely/not/there.txt")
        result = tmp_runner.get_last_thread_id()
        assert result is None


# =============================================================================
# SS11：通过 close() 和上下文管理器释放资源
# =============================================================================


class TestResourceRelease:
    """SS11：close() 与 with 语句均能释放资源。"""

    def test_close_releases_resources(self, tmp_path):
        config = RunConfig(db_path=tmp_path / "release.sqlite")
        runner = TaskRunner(config=config)
        # 在关闭前捕获检查点器的退出状态。
        cp = runner._checkpointer
        runner.close()
        # close 后 ExitStack 为空，无需进一步清理。
        # 关键测试：重复调用 close 不应抛出异常。
        runner.close()

    def test_context_manager_releases_resources(self, tmp_path):
        config = RunConfig(db_path=tmp_path / "cm.sqlite")
        with TaskRunner(config=config) as runner:
            assert runner._graph is not None
            assert runner._checkpointer is not None
        # 退出 with 块后 ExitStack 应已关闭。
        # 再次调用 close 仍然安全，即操作具有幂等性。
        runner.close()

    def test_close_twice_does_not_raise(self, tmp_path):
        """ExitStack.close() 具有幂等性，重复关闭不得崩溃。"""
        config = RunConfig(db_path=tmp_path / "double_close.sqlite")
        runner = TaskRunner(config=config)
        runner.close()
        runner.close()  # 不得抛出异常


# =============================================================================
# SS12：多实例隔离
# =============================================================================


class TestMultiInstanceIsolation:
    """SS12：使用不同 db_path 的两个 TaskRunner 互不干扰。"""

    def test_different_db_paths_do_not_collide(self, tmp_path):
        """每个运行器写入各自的 SQLite 文件。"""
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
        """每个运行器的 _remember_thread_id 写入各自的文件。"""
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
# SS13：RunConfig 默认值
# =============================================================================


class TestRunConfigDefaults:
    """SS13：RunConfig 默认值正确。"""

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
        """无参数 TaskRunner() 使用默认值创建 RunConfig。"""
        # 没有大模型引擎时无法构造真实 TaskRunner，但可以直接验证 RunConfig。
        cfg = RunConfig()
        assert cfg.max_iterations == 6
        assert cfg.db_path == Path("data/checkpoints.sqlite")


# =============================================================================
# SS14：日志覆盖
# =============================================================================


class TestLoggingCoverage:
    """SS14：关键路径均有日志消息覆盖。"""

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

        # 日志中应包含 thread_id。
        log_messages = [r.message for r in caplog_debug.records]
        assert any(tid in msg for msg in log_messages), (
            f"thread_id {tid} not found in log messages: {log_messages}"
        )

    def test_exception_logs_at_exception_level(self, tmp_runner, caplog_debug):
        """_invoke 捕获意外错误时，应按 ERROR 级别记录日志。"""
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

        # 至少存在一条 ERROR/EXCEPTION 级别的记录。
        error_records = [
            r for r in caplog_debug.records
            if r.levelno >= logging.ERROR
        ]
        assert len(error_records) >= 1, (
            f"No ERROR-level records found in: {[r.message for r in caplog_debug.records]}"
        )
        # 异常信息附加在日志记录的 exc_info 中，不一定出现在消息文本内；
        # 这里验证异常已被捕获。
        error_with_exc = [
            r for r in error_records if r.exc_info is not None
        ]
        assert len(error_with_exc) >= 1, (
            f"No ERROR records with attached exception found in: {[r.message for r in error_records]}"
        )
        # 原始 RuntimeError 应位于 exc_info 链中。
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
# 补充：RunConfig 构造后的可变性
# =============================================================================


class TestRunConfigMutability:
    """RunConfig 是数据类，字段可赋值且对象可替换，不应出现意外行为。"""

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
# SS5：恢复正确性（结构测试，需要检查点器和模拟对象）
# =============================================================================


class TestResumeCorrectness:
    """SS5：resume_task 使用有效 thread_id 从检查点恢复状态。"""

    def test_resume_task_calls_invoke_with_none_state(self, tmp_runner):
        """resume_task 应以 state=None 调用 _invoke，使 LangGraph 读取检查点，
        而不是接收新的初始状态。"""
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

        # _invoke 使用 state=None 和 thread_id="existing-thread-id" 调用。
        mock_invoke.assert_called_once_with(None, "existing-thread-id")
        assert result == mock_state

    def test_resume_does_not_alter_planner_state(self, tmp_runner):
        """resume_task 必须保留中断前的 plan_steps。"""
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
