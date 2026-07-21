"""Reflector node — evaluate execution progress and decide the next action.

The Reflector is the decision-gate of the outer graph.  It examines
the task goal, the plan, and the execution log, then emits one of
three decisions via an enum-constrained LLM call:

    * ``done``     — the task goal has been fully achieved.
    * ``continue`` — more work is needed; loop back to the Planner.
    * ``failed``   — an unrecoverable error has been encountered.

The routing logic that acts on this decision lives in
``build_graph.py``, not here — the Reflector only **produces** the
decision, keeping the "how to route" logic in a single file.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent_core.exceptions import ReflectionError
from agent_core.grammar_builder import build_json_grammar
from agent_core.llm_engine import get_engine
from agent_core.prompt_assembler import assemble_reflection_prompt

if TYPE_CHECKING:
    from agent_core.graph.state import AgentState

# ---------------------------------------------------------------------------
# The set of decisions the Reflector is allowed to emit.
# ---------------------------------------------------------------------------

_VALID_DECISIONS = {"done", "continue", "failed"}

REFLECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": sorted(_VALID_DECISIONS),
        },
        "reason": {
            "type": "string",
            "description": "Concrete evidence for the decision and retry advice.",
        },
    },
    "required": ["decision", "reason"],
}


# ---------------------------------------------------------------------------
# [STABLE] reflector_node
# ---------------------------------------------------------------------------


def reflector_node(state: "AgentState") -> "AgentState":
    """Evaluate execution progress and emit a routing decision.

    Steps:
        1. Assemble the reflection prompt (task goal + plan + execution log).
        2. Build a JSON grammar containing ``decision`` and diagnostic
           ``reason`` fields.
        3. Invoke the engine and extract the structured decision.
        4. Append a note to ``reflection_notes``, increment
           ``current_iteration``, and set ``status`` to the decision.

    The ``status`` field is a **temporary** value — ``build_graph.py``'s
    conditional-edge function is the final arbiter that decides whether
    to honour it or force-terminate (e.g. due to ``max_iterations``).

    Returns:
        *state* mutated with a new reflection note and updated status.

    Raises:
        ReflectionError: If the model output is not a member of
            ``{"done", "continue", "failed"}``.
    """
    engine = get_engine()
    messages = assemble_reflection_prompt(state, engine)
    grammar = build_json_grammar(REFLECTION_SCHEMA)

    response = engine.invoke(messages, grammar=grammar)
    raw = str(response.content).strip()
    reason = ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw.strip('"').strip("'")
    if isinstance(parsed, dict):
        decision = str(parsed.get("decision", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
    else:
        # Backward compatibility for old checkpoints and lightweight tests.
        decision = str(parsed).strip()

    if decision not in _VALID_DECISIONS:
        raise ReflectionError(
            f"Reflector output '{decision}' is not a valid decision — "
            f"expected one of {sorted(_VALID_DECISIONS)}."
        )

    # Record the decision
    current_iteration: int = state.get("current_iteration", 0)
    note = f"[iteration {current_iteration}] decision={decision}"
    if reason:
        note += f" reason={reason}"
    reflection_notes: list[str] = state.get("reflection_notes", [])
    reflection_notes.append(note)

    next_iteration = current_iteration + 1
    state["reflection_notes"] = reflection_notes
    state["current_iteration"] = next_iteration

    # Do not return a misleading non-terminal ``continue`` status when the
    # safety cap has already been exhausted.  Mark the task as failed and
    # leave an explicit diagnostic for CLI/API consumers.
    max_iterations: int = state.get("max_iterations", 10)
    if decision == "continue" and next_iteration >= max_iterations:
        state["reflection_notes"].append(
            f"[iteration limit] max_iterations={max_iterations} reached "
            "before the task was completed"
        )
        state["status"] = "failed"
        state["error"] = {
            "node": "reflector",
            "type": "IterationLimitExceeded",
            "message": f"max_iterations={max_iterations} reached",
        }
    elif decision == "failed":
        state["status"] = "failed"
        state.setdefault(
            "error",
            {
                "node": "reflector",
                "type": "ReflectionRejected",
                "message": (
                    "Reflector rejected the execution result at "
                    f"iteration {current_iteration}."
                ),
            },
        )
    else:
        state["status"] = decision

    return state
