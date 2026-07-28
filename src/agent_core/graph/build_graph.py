"""外层图组装——planner → executor ⇄ executor → reflector ⇄ planner。

本模块将三个节点函数连接为带条件路由的已编译 LangGraph ``StateGraph``，是定义
系统状态转换逻辑的唯一位置。

V3 改进：
    1. ``_normalize_entry_state``——初始状态默认值的唯一信息源，不再分散调用
       ``.get()`` 设置默认值。
    2. ``_with_error_isolation``——包装每个业务节点，将 ``AgentCoreError`` 子类
       转换为可控的 ``status="failed"``，而不是让 ``.invoke()`` 崩溃。
    3. 所有边都使用条件路由——每个下游节点之前都会检查
       ``status=="failed"``，防止失败状态继续扩散。
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Callable, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from agent_core.exceptions import AgentCoreError
from agent_core.graph.executor import executor_node
from agent_core.graph.finalizer import finalizer_node
from agent_core.graph.planner import planner_node
from agent_core.graph.reflector import reflector_node
from agent_core.graph.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# [内部实现] 初始状态契约——默认值的唯一信息源
# ---------------------------------------------------------------------------

_STATE_DEFAULTS: dict[str, Callable[[], object]] = {
    "plan_steps": list,
    "execution_log": list,
    "reflection_notes": list,
    "current_step_index": lambda: 0,
    "current_iteration": lambda: 0,
    "max_iterations": lambda: 10,
    "status": lambda: "planning",
    "final_answer": str,
    "error": lambda: None,
    "conversation_id": lambda: None,
    "current_user_input": str,
    "conversation_context": str,
    "pinned_facts": list,
    "context_summary": dict,
    "archived_context_ids": list,
    "context_version": lambda: 0,
    "lifecycle_stats": dict,
}


def _normalize_entry_state(state: AgentState) -> AgentState:
    """图入口节点：补齐所有缺失的可选字段。

    这里是外层图初始状态默认值的**唯一权威声明**。调用方（``session.py``）
    只需确保存在 ``task_goal``，其余字段都在此填充。

    此操作对 Checkpoint 恢复安全：如果旧 Checkpoint 缺少后来增加的字段，
    本节点会补齐该字段。
    """
    for key, default_factory in _STATE_DEFAULTS.items():
        if state.get(key) is None:
            state[key] = default_factory()
    return state


# ---------------------------------------------------------------------------
# [内部实现] 错误隔离——将 AgentCoreError 转换为可控失败
# ---------------------------------------------------------------------------


def _fail_state(state: AgentState, node_name: str, exc: Exception) -> AgentState:
    """在状态中记录节点失败，不再重新抛出异常。"""
    logger.error("Node '%s' failed — task will terminate: %s", node_name, exc)
    notes: list[str] = state.get("reflection_notes") or []
    notes.append(f"[error in {node_name}] {type(exc).__name__}: {exc}")
    state["reflection_notes"] = notes
    state["status"] = "failed"
    state["error"] = {
        "node": node_name,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    return state


def _with_error_isolation(
    node_fn: Callable, node_name: str
) -> Callable:
    """包装节点函数，使 ``AgentCoreError`` 子类转换为 ``status="failed"``，
    而不是继续向外传播。

    包装器会保留原函数的参数数量，使 LangGraph 能正确判断是否传入 ``config``。
    """
    accepts_config = len(inspect.signature(node_fn).parameters) >= 2

    if accepts_config:

        @wraps(node_fn)
        def wrapped(state: AgentState, config) -> AgentState:
            try:
                from agent_core.telemetry import telemetry_phase
                from agent_core.interactive.events import emit_interactive_event

                with telemetry_phase(node_name):
                    emit_interactive_event(
                        "phase_changed", phase=node_name, status="started"
                    )
                    result = node_fn(state, config)
                    from agent_core.memory import get_lifecycle_context_manager

                    committed = get_lifecycle_context_manager().commit(
                        result, event=node_name
                    )
                    emit_interactive_event(
                        "phase_changed",
                        phase=node_name,
                        status="completed",
                    )
                    return committed
            except AgentCoreError as exc:
                from agent_core.interactive.events import emit_interactive_event

                emit_interactive_event(
                    "phase_changed",
                    phase=node_name,
                    status="failed",
                    error=str(exc),
                )
                return _fail_state(state, node_name, exc)

        return wrapped

    @wraps(node_fn)
    def wrapped(state: AgentState) -> AgentState:
        try:
            from agent_core.telemetry import telemetry_phase
            from agent_core.interactive.events import emit_interactive_event

            with telemetry_phase(node_name):
                emit_interactive_event(
                    "phase_changed", phase=node_name, status="started"
                )
                result = node_fn(state)
                from agent_core.memory import get_lifecycle_context_manager

                committed = get_lifecycle_context_manager().commit(
                    result, event=node_name
                )
                emit_interactive_event(
                    "phase_changed",
                    phase=node_name,
                    status="completed",
                )
                return committed
        except AgentCoreError as exc:
            from agent_core.interactive.events import emit_interactive_event

            emit_interactive_event(
                "phase_changed",
                phase=node_name,
                status="failed",
                error=str(exc),
            )
            return _fail_state(state, node_name, exc)

    return wrapped


# ---------------------------------------------------------------------------
# [内部实现] 条件边路由函数
# ---------------------------------------------------------------------------


def _route_after_planner(state: AgentState) -> str:
    """规划后路由到 Executor；如果规划失败则终止。"""
    if state.get("status") == "failed":
        return END
    return "executor"


def _route_after_executor(state: AgentState) -> str:
    """执行一个步骤后，循环处理下一步或进入 Reflector；Executor 失败时立即终止。"""
    if state.get("status") == "failed":
        return END

    current_index: int = state.get("current_step_index", 0)
    plan_steps: list[str] = state.get("plan_steps", [])

    if current_index < len(plan_steps):
        return "executor"
    return "reflector"


def _route_after_reflector(state: AgentState) -> str:
    """Reflector 产生决策后，决定图的后续走向。

    * ``done`` / ``failed`` → ``END``。
    * ``continue`` → ``planner``，但达到 ``max_iterations`` 时会强制终止。
    """
    status: str = state.get("status", "failed")
    current_iteration: int = state.get("current_iteration", 0)
    max_iterations: int = state.get("max_iterations", 10)

    if status == "done":
        return "finalizer"
    if status == "failed":
        return END

    if current_iteration >= max_iterations:
        logger.info(
            "max_iterations (%d) reached — force-terminating", max_iterations
        )
        return END

    return "planner"


# ---------------------------------------------------------------------------
# [稳定接口] build_graph
# ---------------------------------------------------------------------------


def build_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """组装并编译外层“计划 → 执行 → 反思”图。

    图拓扑（V3）::

        entry → normalize_state → planner ──(failed?)───────┐
                                      │                       │
                                      └──(ok)→ executor ──(more steps?)──┐
                                                   │                    │
                                            (failed? / all done)        │
                                                   ↓                    │
                                               reflector ←──────────────┘
                                                   │
                                      (done/failed/limit) → END
                                      (continue) → planner

    相对 V1/V2 的主要变化：
        * 入口节点 ``normalize_state`` 补齐所有默认值。
        * 三个业务节点全部使用错误隔离包装。
        * ``planner → executor`` 现在是条件边。
    """
    graph = StateGraph(AgentState)

    # 为业务节点添加错误隔离包装
    safe_planner = _with_error_isolation(planner_node, "planner")
    safe_executor = _with_error_isolation(executor_node, "executor")
    safe_reflector = _with_error_isolation(reflector_node, "reflector")
    safe_finalizer = _with_error_isolation(finalizer_node, "finalizer")

    # 注册节点
    graph.add_node("normalize_state", _normalize_entry_state)
    graph.add_node("planner", safe_planner)
    graph.add_node("executor", safe_executor)
    graph.add_node("reflector", safe_reflector)
    graph.add_node("finalizer", safe_finalizer)

    # 入口点
    graph.set_entry_point("normalize_state")

    # normalize_state 始终进入 planner
    graph.add_edge("normalize_state", "planner")

    # planner 成功时进入 executor，失败时进入 END
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"executor": "executor", END: END},
    )

    # 还有步骤时 executor 自循环，全部完成后进入 reflector
    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"executor": "executor", "reflector": "reflector", END: END},
    )

    # reflector 决定继续时返回 planner，完成、失败或达到上限时进入 END
    graph.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {"planner": "planner", "finalizer": "finalizer", END: END},
    )
    graph.add_edge("finalizer", END)

    return graph.compile(checkpointer=checkpointer)
