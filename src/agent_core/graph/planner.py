"""Planner node — decompose the user's task goal into ordered steps.

The Planner is the entry point of the outer graph.  It calls the LLM
with a GBNF grammar that constrains the output to a JSON object
containing a ``steps`` array, guaranteeing parseable output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent_core.exceptions import PlanningError
from agent_core.grammar_builder import build_json_grammar
from agent_core.llm_engine import get_engine
from agent_core.prompt_assembler import assemble_planning_prompt

if TYPE_CHECKING:
    from agent_core.graph.state import AgentState

# ---------------------------------------------------------------------------
# JSON Schema that constrains the Planner's LLM output.
# The model MUST emit ``{"steps": ["step 1", "step 2", ...]}``.
# ---------------------------------------------------------------------------

PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered list of executable steps.",
        }
    },
    "required": ["steps"],
}


# ---------------------------------------------------------------------------
# [STABLE] planner_node
# ---------------------------------------------------------------------------


def planner_node(state: "AgentState") -> "AgentState":
    """Generate or update the plan for the current task.

    Steps:
        1. Assemble the planning prompt (task goal + reflection notes + tools).
        2. Build a GBNF grammar that forces JSON ``{"steps": [...]}`` output.
        3. Invoke the engine and parse the result.
        4. Write ``plan_steps``, reset ``current_step_index`` to 0,
           and transition ``status`` to ``"executing"``.

    Returns:
        *state* mutated with the new plan.

    Raises:
        PlanningError: If the LLM output cannot be parsed as a valid plan
            (malformed JSON, missing ``steps`` key, or empty step list).
    """
    engine = get_engine()
    messages = assemble_planning_prompt(state, engine)
    grammar = build_json_grammar(PLAN_SCHEMA)

    response = engine.invoke(messages, grammar=grammar)

    # Parse the constrained JSON output
    try:
        parsed = json.loads(response.content)
        steps: list[str] = parsed["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps field is empty or not a list")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise PlanningError(
            f"Planner output could not be parsed as a valid plan: "
            f"{response.content!r}"
        ) from exc

    # Validate each step is a non-empty string
    for i, step in enumerate(steps):
        if not isinstance(step, str) or not step.strip():
            raise PlanningError(
                f"Plan step {i} is empty or not a string: {step!r}"
            )

    state["plan_steps"] = steps
    state["current_step_index"] = 0
    state["status"] = "executing"

    return state
