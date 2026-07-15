"""Outer graph assembly — planner → executor ⇄ executor → reflector ⇄ planner.

This module wires the three node functions together into a compiled
LangGraph ``StateGraph`` with conditional routing.  It is the single
place where the system's state-transition logic is defined.
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_core.graph.state import AgentState
from agent_core.graph.planner import planner_node
from agent_core.graph.executor import executor_node
from agent_core.graph.reflector import reflector_node


# ---------------------------------------------------------------------------
# [INTERNAL] Conditional-edge routing functions
#
# These are the **only** places in the codebase that decide where the
# graph goes next.  Nodes themselves only write ``status`` — they do
# not contain routing logic.
# ---------------------------------------------------------------------------


def _route_after_executor(state: AgentState) -> str:
    """After executing one step, either loop back to execute the next step
    or proceed to the Reflector if all steps are done.

    A plan may contain multiple steps; each step is a single pass through
    the executor, which internally runs the ReAct inner subgraph.
    """
    current_index: int = state.get("current_step_index", 0)
    plan_steps: list[str] = state.get("plan_steps", [])

    if current_index < len(plan_steps):
        return "executor"
    return "reflector"


def _route_after_reflector(state: AgentState) -> str:
    """After the Reflector produces a decision, decide the graph's fate.

    * ``done`` / ``failed`` → ``END`` — the task is complete (or dead).
    * ``continue`` → ``planner`` — loop back for another planning cycle,
      **unless** ``max_iterations`` has been reached, in which case we
      force-terminate to prevent infinite looping.
    """
    status: str = state.get("status", "failed")
    current_iteration: int = state.get("current_iteration", 0)
    max_iterations: int = state.get("max_iterations", 10)

    if status in ("done", "failed"):
        return END

    if current_iteration >= max_iterations:
        # Safety cap — the Reflector wants to continue but we've
        # exhausted our iteration budget.  Force-terminate.
        return END

    return "planner"


# ---------------------------------------------------------------------------
# [STABLE] build_graph
# ---------------------------------------------------------------------------


def build_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Assemble and compile the outer Plan → Execute → Reflect graph.

    Graph topology::

        entry → planner → executor ──(more steps?)──┐
                        ↑     │                      │
                        │     └──(all done)→ reflector
                        │                         │
                        └──(continue)─────────────┘
                                  │
                            (done/failed/limit) → END

    Args:
        checkpointer: Optional ``BaseCheckpointSaver`` (e.g. ``SqliteSaver``).
            When provided, the outer graph persists its ``AgentState`` after
            each node completion, enabling crash recovery via ``thread_id``.
            The inner ReAct subgraph is never checkpointed.

    Returns:
        A compiled LangGraph graph ready for ``.invoke()`` or ``.stream()``.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reflector", reflector_node)

    # Entry point
    graph.set_entry_point("planner")

    # planner → executor (unconditional)
    graph.add_edge("planner", "executor")

    # executor → executor (next step) OR executor → reflector (all done)
    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {
            "executor": "executor",
            "reflector": "reflector",
        },
    )

    # reflector → planner (continue) OR reflector → END (done / failed / limit)
    graph.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {
            "planner": "planner",
            END: END,
        },
    )

    return graph.compile(checkpointer=checkpointer)
