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

from typing import TYPE_CHECKING

from agent_core.exceptions import ReflectionError
from agent_core.grammar_builder import build_enum_grammar
from agent_core.llm_engine import get_engine
from agent_core.prompt_assembler import assemble_reflection_prompt

if TYPE_CHECKING:
    from agent_core.graph.state import AgentState

# ---------------------------------------------------------------------------
# The set of decisions the Reflector is allowed to emit.
# ---------------------------------------------------------------------------

_VALID_DECISIONS = {"done", "continue", "failed"}


# ---------------------------------------------------------------------------
# [STABLE] reflector_node
# ---------------------------------------------------------------------------


def reflector_node(state: "AgentState") -> "AgentState":
    """Evaluate execution progress and emit a routing decision.

    Steps:
        1. Assemble the reflection prompt (task goal + plan + execution log).
        2. Build an enum grammar that restricts output to exactly one of
           ``done`` / ``continue`` / ``failed``.
        3. Invoke the engine with the grammar and extract the decision.
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
    grammar = build_enum_grammar(list(_VALID_DECISIONS))

    response = engine.invoke(messages, grammar=grammar)
    decision = response.content.strip().strip('"').strip("'")

    if decision not in _VALID_DECISIONS:
        raise ReflectionError(
            f"Reflector output '{decision}' is not a valid decision — "
            f"expected one of {sorted(_VALID_DECISIONS)}."
        )

    # Record the decision
    current_iteration: int = state.get("current_iteration", 0)
    note = f"[iteration {current_iteration}] decision={decision}"
    reflection_notes: list[str] = state.get("reflection_notes", [])
    reflection_notes.append(note)

    state["reflection_notes"] = reflection_notes
    state["current_iteration"] = current_iteration + 1
    state["status"] = decision

    return state
