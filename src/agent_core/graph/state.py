"""AgentState — the single, canonical TypedDict for the outer graph.

Every node in ``graph/`` must declare its input and output as
``AgentState`` and **must not** add undeclared keys.  Extend this file
first when a new field is needed.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    """Full state schema shared across planner / executor / reflector nodes.

    Attributes:
        task_goal: The user's original task description.
        plan_steps: Ordered list of steps produced by the Planner.
        current_step_index: Zero-based index of the step the Executor
            should work on next.
        execution_log: Per-step records — each dict has keys
            ``step`` (str), ``result`` (str), ``tool_used`` (str | None).
        reflection_notes: History of Reflector evaluation notes.
        status: Current lifecycle phase — ``planning``, ``executing``,
            ``reflecting``, ``done``, or ``failed``.
        max_iterations: Safety cap on planning-reflecting loops.
        current_iteration: How many plan→reflect cycles have run so far.
        final_answer: Stable, user-facing result created by the Finalizer.
        error: Structured terminal failure, or ``None``.
    """

    task_goal: str
    plan_steps: list[str]
    current_step_index: int
    execution_log: list[dict]
    reflection_notes: list[str]
    status: str  # "planning" | "executing" | "reflecting" | "done" | "failed"
    max_iterations: int
    current_iteration: int
    final_answer: str
    error: dict | None
