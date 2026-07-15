"""Executor node — run the current plan step inside the ReAct inner subgraph.

The executor bridges two state schemas:
    * **Outer** ``AgentState`` — business-level fields (plan_steps,
      execution_log, …).
    * **Inner** ``{"messages": [...]}`` — the message-driven state that
      the ReAct subgraph expects.

All bridging logic is encapsulated here so that ``planner_node`` and
``reflector_node`` never need to know the inner subgraph exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages

from agent_core.exceptions import ExecutionError
from agent_core.llm_engine import get_engine
from agent_core.prompt_assembler import assemble_execution_prompt

if TYPE_CHECKING:
    from agent_core.graph.state import AgentState


# ---------------------------------------------------------------------------
# [INTERNAL] State bridging — outer AgentState ↔ inner {"messages": [...]}
# ---------------------------------------------------------------------------


def _build_react_input(state: "AgentState") -> dict:
    """Translate outer ``AgentState`` into the inner subgraph's input format.

    Calls ``assemble_execution_prompt`` to produce a fully assembled,
    budget-compliant, structurally valid message list, then wraps it
    in the ``{"messages": [...]}`` dict that the inner subgraph expects.
    """
    engine = get_engine()
    messages = assemble_execution_prompt(state, engine)
    return {"messages": messages}


def _normalize_output_messages(raw_messages: list) -> list[BaseMessage]:
    """Defensively normalise the inner subgraph's output.

    Some LangGraph paths return plain ``dict`` entries inside the
    ``messages`` list (especially when a node uses ``add_messages``
    with dict-style returns).  This function converts everything to
    proper ``BaseMessage`` subclasses so downstream extraction code
    never has to guess the format.

    Args:
        raw_messages: The raw ``react_output["messages"]`` list —
            may contain ``dict``, ``BaseMessage``, or a mix.

    Returns:
        A list where every entry is a ``BaseMessage`` subclass.
    """
    # convert_to_messages handles mixed lists of dict / BaseMessage
    return list(convert_to_messages(raw_messages))


def _extract_execution_result(
    messages: list[BaseMessage],
    current_step: str,
) -> list[dict]:
    """Parse the inner subgraph's message list back into execution log records.

    Walk through the normalised message list and pair each
    ``AIMessage(tool_calls=…)`` with its following ``ToolMessage``(s).
    The final non-tool-call ``AIMessage`` (the step's conclusion) also
    produces a record.

    Args:
        messages: Normalised message list from ``_normalize_output_messages``.
        current_step: The text of the current plan step (for the ``step`` field).

    Returns:
        A list of execution-log dicts matching ``AgentState.execution_log``
        schema: ``{"step": str, "result": str, "tool_used": str | None}``.

    Raises:
        ExecutionError: If an ``AIMessage(tool_calls=…)`` references a
            ``tool_call_id`` that has no matching ``ToolMessage`` in the
            remaining messages.
    """
    records: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                # Find the matching ToolMessage in the remaining messages
                matching_tool_msg: ToolMessage | None = None
                for j in range(i + 1, len(messages)):
                    candidate = messages[j]
                    if (
                        isinstance(candidate, ToolMessage)
                        and candidate.tool_call_id == call["id"]
                    ):
                        matching_tool_msg = candidate
                        break

                if matching_tool_msg is None:
                    raise ExecutionError(
                        f"Step '{current_step}': tool call '{call['id']}' "
                        f"(name='{call['name']}') has no matching ToolMessage "
                        f"in the inner subgraph output."
                    )

                records.append(
                    {
                        "step": current_step,
                        "result": matching_tool_msg.content,
                        "tool_used": call["name"],
                    }
                )
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            # This is a plain-text response — the step's final conclusion
            if msg.content:
                records.append(
                    {
                        "step": current_step,
                        "result": msg.content,
                        "tool_used": None,
                    }
                )

        i += 1

    return records


# ---------------------------------------------------------------------------
# [STABLE] executor_node
# ---------------------------------------------------------------------------


def executor_node(state: "AgentState") -> "AgentState":
    """Execute the current plan step via the ReAct inner subgraph.

    1. Look up the current step from ``plan_steps[current_step_index]``.
    2. Build the inner subgraph input messages via ``_build_react_input``.
    3. Invoke the cached inner subgraph with a ``recursion_limit`` safety cap.
    4. Parse the output messages into ``execution_log`` records.
    5. Advance ``current_step_index`` and, if all steps are done,
       transition ``status`` to ``"reflecting"``.

    Returns:
        *state* mutated with new execution log entries and an updated
        step index.

    Raises:
        ExecutionError: If the inner subgraph exceeds the recursion
            limit, or if the output messages are malformed.
    """
    plan_steps: list[str] = state.get("plan_steps", [])
    current_step_index: int = state.get("current_step_index", 0)

    if current_step_index >= len(plan_steps):
        raise ExecutionError(
            f"executor_node called with current_step_index={current_step_index} "
            f"but plan_steps has only {len(plan_steps)} steps."
        )

    current_step = plan_steps[current_step_index]

    # Build input for the inner subgraph
    react_input = _build_react_input(state)

    # Get the cached inner agent
    from agent_core.graph.react_agent_factory import get_react_agent

    agent = get_react_agent()

    # Run the inner loop with a recursion-limit safety cap
    try:
        react_output = agent.invoke(react_input, config={"recursion_limit": 8})
    except Exception as exc:
        exc_type_name = type(exc).__name__
        if "recursion" in exc_type_name.lower() or "RecursionError" in exc_type_name:
            raise ExecutionError(
                f"Step '{current_step}': inner agent exceeded the "
                f"tool-call recursion limit."
            ) from exc
        raise ExecutionError(
            f"Step '{current_step}' execution failed: {exc}"
        ) from exc

    # Parse the inner output back to execution log records
    raw_messages: list = react_output.get("messages", [])
    normalised = _normalize_output_messages(raw_messages)
    new_records = _extract_execution_result(normalised, current_step)

    state["execution_log"].extend(new_records)
    state["current_step_index"] += 1

    # If all steps are done, transition to reflecting
    if state["current_step_index"] >= len(plan_steps):
        state["status"] = "reflecting"

    return state
