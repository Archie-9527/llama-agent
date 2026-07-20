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

import re
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables import RunnableConfig

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
                        "tool_args": dict(call.get("args") or {}),
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


def _validate_required_tool_execution(
    current_step: str,
    records: list[dict],
) -> None:
    """Reject textual tool imitations that were never actually executed.

    A real tool execution always produces a ToolMessage and therefore a log
    record with ``tool_used`` set.  Small local models sometimes understand
    that a tool is needed but print ``tool_name(...)`` or a JSON snippet in a
    plain AIMessage instead.  Treating that as success is unsafe and causes
    fabricated tool results in later planning/reflection rounds.
    """
    from agent_core.capability_registry import list_capabilities

    tool_names = [cap.name for cap in list_capabilities()]
    used_tools = {
        record["tool_used"] for record in records if record.get("tool_used")
    }

    required_by_step = [name for name in tool_names if name in current_step]
    missing = [name for name in required_by_step if name not in used_tools]
    if missing:
        raise ExecutionError(
            f"Step '{current_step}' explicitly requires tool(s) {missing}, "
            "but the model returned no matching structured tool_calls."
        )

    plain_text = "\n".join(
        str(record.get("result", ""))
        for record in records
        if record.get("tool_used") is None
    )
    pseudo_calls = [
        name
        for name in tool_names
        if re.search(rf"\b{re.escape(name)}\s*\(", plain_text)
        or re.search(rf"\bfunctions\.{re.escape(name)}\s*:", plain_text)
        or re.search(
            rf'["\'](?:tool|name)["\']\s*:\s*["\']{re.escape(name)}["\']',
            plain_text,
        )
    ]
    if pseudo_calls and not used_tools:
        raise ExecutionError(
            f"Step '{current_step}' returned textual pseudo tool call(s) "
            f"{pseudo_calls} instead of structured tool_calls; no tool was executed."
        )

    if used_tools:
        summaries = [
            str(record.get("result", "")).strip()
            for record in records
            if record.get("tool_used") is None
        ]
        valid_summaries = [
            summary for summary in summaries if _is_natural_language_summary(summary)
        ]
        if not valid_summaries:
            raise ExecutionError(
                f"Step '{current_step}' executed tool(s) {sorted(used_tools)} "
                "but produced no valid natural-language summary."
            )


def _is_natural_language_summary(content: str) -> bool:
    """Return False for empty or leaked llama.cpp function protocol markers."""
    stripped = content.strip()
    if not stripped:
        return False
    return re.fullmatch(r"functions\.[A-Za-z_][A-Za-z0-9_]*\s*:\s*", stripped) is None


def _recover_empty_response(
    messages: list[BaseMessage],
    current_step: str,
) -> list[dict]:
    """Retry once without tools when the inner agent returned no visible output."""
    response = get_engine().invoke(
        [
            *messages,
            HumanMessage(
                content=(
                    f"当前步骤是：{current_step}\n"
                    "刚才没有产生可见回答。请直接给出非空的自然语言最终回答。"
                    "不要调用工具，不要输出 JSON、functions.<name>: 或任何工具协议。"
                )
            ),
        ]
    )
    content = str(response.content or "").strip()
    if not _is_natural_language_summary(content):
        return []
    return [{"step": current_step, "result": content, "tool_used": None}]


def _ensure_tool_summary(
    messages: list[BaseMessage],
    current_step: str,
    records: list[dict],
) -> list[dict]:
    """Retry one tool-free model call when the inner agent omitted its summary."""
    has_tool_result = any(record.get("tool_used") for record in records)
    if has_tool_result:
        # Protocol-only text is an internal formatting leak, not a user-facing
        # conclusion.  Drop it before deciding whether recovery is needed.
        records = [
            record
            for record in records
            if record.get("tool_used") is not None
            or _is_natural_language_summary(str(record.get("result", "")))
        ]
    has_valid_summary = any(
        record.get("tool_used") is None
        and _is_natural_language_summary(str(record.get("result", "")))
        for record in records
    )
    if not has_tool_result or has_valid_summary:
        return records

    engine = get_engine()
    response = engine.invoke(
        [
            *messages,
            HumanMessage(
                content=(
                    f"当前步骤是：{current_step}\n"
                    "请只根据以上真实 ToolMessage 给出自然语言最终答案。"
                    "不要再次调用工具，不要输出 functions.<name>: 或调用格式。"
                )
            ),
        ]
    )
    if response.content:
        records.append(
            {
                "step": current_step,
                "result": response.content,
                "tool_used": None,
            }
        )
    return records


# ---------------------------------------------------------------------------
# [STABLE] executor_node
# ---------------------------------------------------------------------------


def executor_node(state: "AgentState", config: RunnableConfig) -> "AgentState":
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
    outer_thread_id = config["configurable"]["thread_id"]

    sub_config = {
        "configurable": {
            "thread_id": f"{outer_thread_id}::step-{current_step_index}"
        },
        "recursion_limit": 8
    }

    # Build input for the inner subgraph
    react_input = _build_react_input(state)

    # 记录输入消息的数量，以便我们稍后可以仅隔离由内部代理新生成的消息。
    # 内部子图可能会返回完整的对话历史记录（例如，当FakeReactAgent连接输入和输出时），
    # 我们不能重新从已经在先前步骤中记录的历史消息中提取记录。
    input_message_count = len(react_input["messages"])

    # Get the cached inner agent
    from agent_core.graph.react_agent_factory import initialize_react_agent

    agent = initialize_react_agent()

    # Run the inner loop with a recursion-limit safety cap
    try:
        react_output = agent.invoke(react_input, config=sub_config)
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

    # 将内部输出解析回执行日志记录
    # 仅从此次调用新生成的消息中提取
    # 历史输入消息已记录在日志中
    raw_messages: list = react_output.get("messages", [])
    normalised = _normalize_output_messages(raw_messages)
    new_messages = normalised[input_message_count:]
    new_records = _extract_execution_result(new_messages, current_step)
    new_records = _ensure_tool_summary(normalised, current_step, new_records)

    if not new_records:
        new_records = _recover_empty_response(
            react_input["messages"],
            current_step,
        )
        if not new_records:
            raise ExecutionError(
                f"Step '{current_step}' produced no tool result or final response "
                "after one tool-free recovery attempt."
            )
    _validate_required_tool_execution(current_step, new_records)

    state["execution_log"].extend(new_records)
    state["current_step_index"] += 1

    # If all steps are done, transition to reflecting
    if state["current_step_index"] >= len(plan_steps):
        state["status"] = "reflecting"

    return state
