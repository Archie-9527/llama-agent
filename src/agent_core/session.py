"""应用服务层——连接图编排层与 CLI/API 调用方。

``session.py`` 是唯一持有已编译图和 Checkpointer 连接、管理 ``thread_id``
生命周期、构造初始 ``AgentState`` 并规范化所有异常的位置，使调用方
（``cli.py``、Web Handler）只需处理 ``AgentCoreError``。

公共 API：
    * ``RunConfig``——近似不可变的配置 DataClass。
    * ``TaskRunner``——启动新任务、恢复中断任务、记住最近的 ``thread_id``，
      并正确释放资源。
"""

from __future__ import annotations

import logging
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from agent_core.exceptions import AgentCoreError
from agent_core.graph.build_graph import build_graph
from agent_core.graph.checkpointer import get_checkpointer
from agent_core.graph.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# [稳定接口] RunConfig——所有可调参数的统一位置
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """``TaskRunner`` 实例的所有可调参数。

    新参数应在此添加并提供合理默认值，避免影响现有调用方。

    属性：
        max_iterations：计划→反思循环的安全上限，默认为 6。
        db_path：SQLite Checkpoint 数据库所在位置。
        last_thread_file：记录最近 ``thread_id`` 的小文件，用于无参数恢复。
    """

    max_iterations: int = 6
    db_path: Path = field(default_factory=lambda: Path("data/checkpoints.sqlite"))
    last_thread_file: Path = field(
        default_factory=lambda: Path("data/last_thread_id.txt")
    )


# ---------------------------------------------------------------------------
# [稳定接口] TaskRunner——应用层任务执行器
# ---------------------------------------------------------------------------


class TaskRunner:
    """持有已编译的 LangGraph 图和 Checkpoint 连接，并提供简单的启动、恢复与
    关闭接口。

    典型用法::

        runner = TaskRunner()
        tid, result = runner.start_new_task("Summarise the meeting notes.")
        runner.close()

        # 也可以用作上下文管理器：
        with TaskRunner() as runner:
            tid, result = runner.start_new_task("Query tomorrow's weather")

        # 恢复崩溃的任务：
        with TaskRunner() as runner:
            result = runner.resume_task(tid)
    """

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config or RunConfig()
        self._exit_stack = ExitStack()
        self._checkpointer = self._init_checkpointer()
        self._graph = build_graph(checkpointer=self._checkpointer)
        logger.info("TaskRunner initialised  db_path=%s", self.config.db_path)

    # ------------------------------------------------------------------
    # 资源初始化
    # ------------------------------------------------------------------

    def _init_checkpointer(self):
        """以防御方式初始化 ``SqliteSaver``。

        ``get_checkpointer`` 是生成 Saver 的 ``@contextmanager``。这里使用
        ``ExitStack.enter_context``，以保证无论执行哪条路径都能清理 Saver，
        并让 ``close()`` 可以一次释放所有资源。
        """
        cm = get_checkpointer(self.config.db_path)
        return self._exit_stack.enter_context(cm)

    def close(self) -> None:
        """释放所有底层资源，例如数据库连接。

        可以安全地多次调用；``ExitStack`` 在第一次关闭后具有幂等性。
        """
        self._exit_stack.close()
        logger.info("TaskRunner resources released")

    def __enter__(self) -> "TaskRunner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # 初始状态构造
    # ------------------------------------------------------------------

    def _build_initial_state(
        self,
        task_goal: str,
        *,
        conversation_id: str | None = None,
        conversation_context: str = "",
        current_user_input: str | None = None,
    ) -> AgentState:
        """根据纯任务目标字符串构造有效的 ``AgentState``。

        所有默认值都在且仅在此处定义；调用方不应手工组装 ``AgentState`` 字典。
        """
        return AgentState(
            task_goal=task_goal,
            plan_steps=[],
            current_step_index=0,
            execution_log=[],
            reflection_notes=[],
            current_iteration=0,
            max_iterations=self.config.max_iterations,
            status="planning",
            final_answer="",
            error=None,
            conversation_id=conversation_id,
            current_user_input=current_user_input or task_goal,
            conversation_context=conversation_context,
            pinned_facts=[],
            context_summary={},
            archived_context_ids=[],
            context_version=0,
            lifecycle_stats={},
        )

    # ------------------------------------------------------------------
    # [稳定接口] 公共入口
    # ------------------------------------------------------------------

    def start_new_task(
        self,
        task_goal: str,
        *,
        thread_id: str | None = None,
        conversation_id: str | None = None,
        conversation_context: str = "",
        current_user_input: str | None = None,
    ) -> tuple[str, AgentState]:
        """启动全新任务。

        参数：
            task_goal：非空的 Agent 任务描述。

        返回：
            ``(thread_id, final_state)`` 元组。调用方应保存 ``thread_id``，
            ``resume_task`` 需要使用它。

        异常：
            ValueError：*task_goal* 为空或只包含空白时抛出。
            AgentCoreError：编排层发生任何错误时抛出。
        """
        if not task_goal or not task_goal.strip():
            raise ValueError("task_goal must not be empty")

        thread_id = thread_id or str(uuid.uuid4())
        initial_state = self._build_initial_state(
            task_goal,
            conversation_id=conversation_id,
            conversation_context=conversation_context,
            current_user_input=current_user_input,
        )
        logger.info("Starting new task  thread_id=%s  goal=%r", thread_id, task_goal)

        from agent_core.telemetry import get_telemetry, telemetry_task

        telemetry = get_telemetry()
        with telemetry_task(thread_id):
            telemetry.record_event("lifecycle_events.jsonl", "task_started")
            result = self._invoke(initial_state, thread_id)
            telemetry.record_event(
                "lifecycle_events.jsonl",
                "task_finished",
                status=result.get("status"),
                final_answer_bytes=len(
                    str(result.get("final_answer", "")).encode("utf-8")
                ),
            )
        self._remember_thread_id(thread_id)
        return thread_id, result

    def resume_task(self, thread_id: str) -> AgentState:
        """恢复因崩溃或手动终止而中断的任务。

        **本方法有意不接收 ``task_goal`` 参数。** LangGraph 会检测给定
        ``thread_id`` 已有的 Checkpoint，并从最后完成的节点恢复；任何新提供的
        初始状态都会被静默忽略，接受该参数会形成误导性的 API。

        参数：
            thread_id：之前调用 ``start_new_task`` 返回的标识符。

        返回：
            恢复运行完成后的最终 ``AgentState``。

        异常：
            ValueError：*thread_id* 为空或为 ``None`` 时抛出。
            AgentCoreError：编排层发生任何错误时抛出。
        """
        if not thread_id:
            raise ValueError("thread_id must not be empty")

        logger.info("Resuming task  thread_id=%s", thread_id)
        from agent_core.telemetry import telemetry_task

        with telemetry_task(thread_id):
            return self._invoke(None, thread_id)

    # ------------------------------------------------------------------
    # 内部调用——所有图调用的唯一入口
    # ------------------------------------------------------------------

    def _invoke(
        self, state_or_none: AgentState | None, thread_id: str
    ) -> AgentState:
        """带异常规范化的统一图调用入口。

        每次调用 ``self._graph.invoke`` 都必须经过此处，以统一应用异常处理。
        本层以上的调用方只需捕获 ``AgentCoreError``。
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            return self._graph.invoke(state_or_none, config=config)  # type: ignore[arg-type]
        except AgentCoreError:
            # 已经是语义异常，原样重新抛出。
            raise
        except Exception as exc:
            # 防御性兜底：包装所有穿透图层的原始异常，例如意外的 LangGraph 内部
            # 错误，使 cli.py 只会看到 AgentCoreError。
            logger.exception(
                "Unexpected error during task execution  thread_id=%s", thread_id
            )
            raise AgentCoreError(
                f"Unexpected error during task execution: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # thread_id 持久化（应用层职责，不属于 SqliteSaver）
    # ------------------------------------------------------------------

    def _remember_thread_id(self, thread_id: str) -> None:
        """持久化最近的 ``thread_id``，供 ``get_last_thread_id`` 后续读取。"""
        path = self.config.last_thread_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(thread_id)

    def get_last_thread_id(self) -> str | None:
        """返回最近持久化的 ``thread_id``，不存在时返回 ``None``。

        调用方借此支持“恢复最近任务”，用户无需记忆并重新输入 UUID。
        """
        path = self.config.last_thread_file
        if not path.exists():
            return None
        content = path.read_text().strip()
        return content or None
