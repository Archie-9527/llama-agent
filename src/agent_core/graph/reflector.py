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
import re
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


def _parse_reflection_output(raw: str) -> object:
    """Parse one constrained value while tolerating harmless trailing text.

    Some local models emit a valid JSON object and then append a Markdown
    explanation despite grammar/prompt constraints.  ``raw_decode`` preserves
    strict validation of the first JSON value without accepting a fabricated
    decision from arbitrary prose.
    """
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidate_offsets = [0]
        candidate_offsets.extend(
            index for index, char in enumerate(stripped) if char == "{"
        )
        seen: set[int] = set()
        for offset in candidate_offsets:
            if offset in seen:
                continue
            seen.add(offset)
            try:
                value, _ = decoder.raw_decode(stripped[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "decision" in value:
                return value
        # Qwen occasionally closes the final JSON string with a typographic
        # quote (”) before appending prose.  Recover only the exact constrained
        # reflection shape and still validate the enum in ``reflector_node``.
        near_json = re.search(
            r'\{\s*"decision"\s*:\s*"(?P<decision>done|continue|failed)"\s*,'
            r'\s*"reason"\s*:\s*"(?P<reason>.*?)(?:"|”)\s*\}',
            stripped,
            re.DOTALL,
        )
        if near_json:
            return near_json.groupdict()
        # Backward compatibility for old checkpoints/tests that returned a
        # bare, optionally quoted enum instead of the structured schema.
        return stripped.strip('"').strip("'")


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
    parsed = _parse_reflection_output(raw)
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
