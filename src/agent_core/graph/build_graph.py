"""Outer graph assembly — planner → executor ⇄ executor → reflector ⇄ planner.

This module wires the three node functions together into a compiled
LangGraph ``StateGraph`` with conditional routing.  It is the single
place where the system's state-transition logic is defined.

V3 improvements:
    1. ``_normalize_entry_state`` — single source of truth for initial
       state defaults (no more scattered ``.get()`` defaults).
    2. ``_with_error_isolation`` — wraps every business node so that
       ``AgentCoreError`` subclasses are converted to graceful
       ``status="failed"`` instead of crashing ``.invoke()``.
    3. All edges are conditional — a ``status=="failed"`` check
       appears before every downstream node to prevent a failed state
       from cascading.
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
# [INTERNAL] Initial-state contract — single source of truth for defaults
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
}


def _normalize_entry_state(state: AgentState) -> AgentState:
    """The graph's entry node: fill in every optional field that is missing.

    This is the **single authoritative declaration** of initial-state
    defaults for the outer graph.  Callers (``session.py``) only need
    to ensure ``task_goal`` is present — all other fields are
    populated here.

    Safe for checkpoint resume: if an older checkpoint is missing a
    field added later, this node backfills it.
    """
    for key, default_factory in _STATE_DEFAULTS.items():
        if state.get(key) is None:
            state[key] = default_factory()
    return state


# ---------------------------------------------------------------------------
# [INTERNAL] Error isolation — convert AgentCoreError into graceful failure
# ---------------------------------------------------------------------------


def _fail_state(state: AgentState, node_name: str, exc: Exception) -> AgentState:
    """Record a node failure on the state without re-raising."""
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
    """Wrap a node function so that ``AgentCoreError`` subclasses are
    converted to ``status="failed"`` instead of propagating.

    The wrapper respects the original function's parameter count so
    LangGraph correctly decides whether to pass ``config``.
    """
    accepts_config = len(inspect.signature(node_fn).parameters) >= 2

    if accepts_config:

        @wraps(node_fn)
        def wrapped(state: AgentState, config) -> AgentState:
            try:
                from agent_core.telemetry import telemetry_phase

                with telemetry_phase(node_name):
                    return node_fn(state, config)
            except AgentCoreError as exc:
                return _fail_state(state, node_name, exc)

        return wrapped

    @wraps(node_fn)
    def wrapped(state: AgentState) -> AgentState:
        try:
            from agent_core.telemetry import telemetry_phase

            with telemetry_phase(node_name):
                return node_fn(state)
        except AgentCoreError as exc:
            return _fail_state(state, node_name, exc)

    return wrapped


# ---------------------------------------------------------------------------
# [INTERNAL] Conditional-edge routing functions
# ---------------------------------------------------------------------------


def _route_after_planner(state: AgentState) -> str:
    """After planning, route to executor *or* terminate if planning failed."""
    if state.get("status") == "failed":
        return END
    return "executor"


def _route_after_executor(state: AgentState) -> str:
    """After executing one step, either loop for the next step or proceed
    to the Reflector.  Terminate immediately if executor failed."""
    if state.get("status") == "failed":
        return END

    current_index: int = state.get("current_step_index", 0)
    plan_steps: list[str] = state.get("plan_steps", [])

    if current_index < len(plan_steps):
        return "executor"
    return "reflector"


def _route_after_reflector(state: AgentState) -> str:
    """After the Reflector produces a decision, decide the graph's fate.

    * ``done`` / ``failed`` → ``END``.
    * ``continue`` → ``planner``, **unless** ``max_iterations`` has been
      reached (force-terminate).
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
# [STABLE] build_graph
# ---------------------------------------------------------------------------


def build_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Assemble and compile the outer Plan → Execute → Reflect graph.

    Graph topology (V3)::

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

    Key changes from V1/V2:
        * Entry node ``normalize_state`` fills in every default.
        * All three business nodes are wrapped with error isolation.
        * ``planner → executor`` is now a conditional edge.
    """
    graph = StateGraph(AgentState)

    # Wrap business nodes with error isolation
    safe_planner = _with_error_isolation(planner_node, "planner")
    safe_executor = _with_error_isolation(executor_node, "executor")
    safe_reflector = _with_error_isolation(reflector_node, "reflector")
    safe_finalizer = _with_error_isolation(finalizer_node, "finalizer")

    # Register nodes
    graph.add_node("normalize_state", _normalize_entry_state)
    graph.add_node("planner", safe_planner)
    graph.add_node("executor", safe_executor)
    graph.add_node("reflector", safe_reflector)
    graph.add_node("finalizer", safe_finalizer)

    # Entry point
    graph.set_entry_point("normalize_state")

    # normalize_state → planner (always)
    graph.add_edge("normalize_state", "planner")

    # planner → executor (ok) OR planner → END (failed)
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"executor": "executor", END: END},
    )

    # executor → executor (more steps) OR executor → reflector (all done)
    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"executor": "executor", "reflector": "reflector", END: END},
    )

    # reflector → planner (continue) OR reflector → END (done / failed / limit)
    graph.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {"planner": "planner", "finalizer": "finalizer", END: END},
    )
    graph.add_edge("finalizer", END)

    return graph.compile(checkpointer=checkpointer)
