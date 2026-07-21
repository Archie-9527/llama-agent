"""Inner ReAct subgraph — single-step tool-use loop.

Each outer ``executor_node`` invocation runs this inner agent once
for the current plan step.  The inner agent loops:
    LLM-decision → tool-execution → feed results back → LLM-decision
until the model produces a final text response (no more tool calls).

Two build paths exist:
    * **Main**: ``langchain.agents.create_agent`` (official wrapper).
    * **Fallback**: hand-rolled ``StateGraph`` — enabled by setting
      ``USE_OFFICIAL_CREATE_AGENT=false`` in the environment.

Both paths produce identically-shaped compiled graphs.
The inner graph deliberately does **not** receive a checkpointer
so that crash-recovery stays at the outer-graph node boundary.
"""

from __future__ import annotations

import os
from functools import lru_cache
from time import monotonic_ns
from typing import Annotated

from langchain.agents import create_agent  # type: ignore[import-untyped]
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool as langchain_tool_decorator
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent_core.capability_registry import Capability, get_capability, list_capabilities
from agent_core.llm_engine import get_engine


def _run_capability(cap: Capability, kwargs: dict) -> object:
    """Execute one capability and emit a stable tool event."""
    from agent_core.telemetry import get_telemetry

    telemetry = get_telemetry()
    started = monotonic_ns()
    try:
        # LangChain/Pydantic materialises omitted optional schema properties as
        # ``None``.  Dropping those values lets the provider's Python defaults
        # apply and avoids turning an omitted limit/offset into a runtime type
        # error (for example ``min(None, max_rows)``).
        normalized_kwargs = {
            key: value for key, value in kwargs.items() if value is not None
        }
        result = cap.handler(**normalized_kwargs)
    except Exception as exc:
        telemetry.record_event(
            "tool_events.jsonl",
            "tool_failed",
            tool_name=cap.name,
            duration_ms=(monotonic_ns() - started) / 1_000_000,
            input_bytes=len(str(kwargs).encode("utf-8")),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    telemetry.record_event(
        "tool_events.jsonl",
        "tool_completed",
        tool_name=cap.name,
        duration_ms=(monotonic_ns() - started) / 1_000_000,
        input_bytes=len(str(kwargs).encode("utf-8")),
        output_bytes=len(str(result).encode("utf-8")),
    )
    return result

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

USE_OFFICIAL_CREATE_AGENT = (
    os.environ.get("USE_OFFICIAL_CREATE_AGENT", "true").lower() == "true"
)

_AGENT_DEBUG = os.environ.get("AGENT_DEBUG", "false").lower() == "true"

_TOOL_RESULT_COMPACTION_MARKER = (
    "\n...[tool result compacted to fit the model context; "
    "the beginning and end are preserved]...\n"
)


def _compact_tool_messages_for_model(
    messages: list[BaseMessage],
    engine,
) -> list[BaseMessage]:
    """Bound the latest tool result before the follow-up summary inference.

    The complete ToolMessage remains in LangGraph state and telemetry; only the
    model-facing copy is compacted.  Keeping both the head and tail is
    important for file/log tools where final evidence commonly appears at EOF.
    This is a hard context-safety guard, independent of R1 artifact
    virtualization.
    """
    n_ctx = int(getattr(engine, "n_ctx", 8192))
    max_tokens = int(getattr(engine, "max_tokens", 512))
    budget = max(512, n_ctx - max_tokens - 512)

    def total_tokens(items: list[BaseMessage]) -> int:
        return sum(
            engine.get_num_tokens(
                item.content if isinstance(item.content, str) else str(item.content)
            )
            for item in items
        )

    copied = list(messages)
    if total_tokens(copied) <= budget:
        return copied

    tool_index = next(
        (
            index
            for index in range(len(copied) - 1, -1, -1)
            if isinstance(copied[index], ToolMessage)
        ),
        None,
    )
    if tool_index is None:
        return copied

    original = copied[tool_index]
    content = str(original.content or "")
    low, high = 0, len(content)
    best = _TOOL_RESULT_COMPACTION_MARKER
    while low <= high:
        retained = (low + high) // 2
        head_size = (retained + 1) // 2
        tail_size = retained // 2
        compacted = (
            content[:head_size]
            + _TOOL_RESULT_COMPACTION_MARKER
            + (content[-tail_size:] if tail_size else "")
        )
        candidate = list(copied)
        candidate[tool_index] = original.model_copy(
            update={"content": compacted}
        )
        if total_tokens(candidate) <= budget:
            best = compacted
            low = retained + 1
        else:
            high = retained - 1

    copied[tool_index] = original.model_copy(update={"content": best})
    return copied


# ---------------------------------------------------------------------------
# [INTERNAL] Capability → LangChain BaseTool conversion
# ---------------------------------------------------------------------------


def to_langchain_tool(cap: Capability) -> BaseTool:
    """Wrap a registered ``Capability`` as a LangChain ``BaseTool``.

    The returned tool delegates execution to ``cap.handler`` and
    carries the same ``name``, ``description``, and ``input_schema``
    so that ``create_agent`` / ``bind_tools`` receive consistent
    metadata.

    This is the **single** bridge between the capability registry and
    LangChain's tool abstraction.  Every other module consumes tools
    via ``list_capabilities()`` — only this function translates them.
    """
    # Build a dynamic schema model from the capability's JSON Schema
    # so that LangChain can generate the correct OpenAI-style tool def.
    from pydantic import create_model

    fields: dict[str, tuple[type, object]] = {}
    props = cap.input_schema.get("properties", {})
    required_set: set[str] = set(cap.input_schema.get("required", []))

    for param_name, param_schema in props.items():
        param_type = _json_type_to_python(param_schema.get("type", "string"))
        default = ... if param_name in required_set else None
        fields[param_name] = (param_type, default)

    # Wrap the raw handler so LangChain sees a proper callable signature.
    # We use **kwargs because create_model gives us keyword-only fields.
    def _handler(**kwargs):
        return _run_capability(cap, kwargs)

    # If the schema has no properties, create a simple no-arg tool.
    if not fields:
        return langchain_tool_decorator(
            name_or_callable=cap.name,
            description=cap.description,
        )(_handler)

    # Create a Pydantic model for structured input and decorate.
    ArgsModel = create_model(f"{cap.name}_args", **fields)  # type: ignore[call-overload]

    class _StructuredTool(BaseTool):
        name: str = cap.name
        description: str = cap.description
        args_schema: type = ArgsModel

        def _run(self, **kwargs):
            return _run_capability(cap, kwargs)

    return _StructuredTool()


def _json_type_to_python(json_type: str) -> type:
    """Map JSON Schema type names to Python types."""
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)


# ---------------------------------------------------------------------------
# [STABLE] Inner subgraph — main path (create_agent)
# ---------------------------------------------------------------------------


@wrap_model_call
def _summarize_after_tool(request, handler):
    """Switch from tool selection to a tool-free summary call.

    ``create_agent`` normally binds all tools again after every ToolMessage.
    Small local models can consequently repeat the same call or emit a raw
    protocol marker such as ``functions.count_lines:``.  Once a tool result is
    available for the current step, remove all tools and explicitly ask the
    model to answer from that real result.
    """
    if request.messages and isinstance(request.messages[-1], ToolMessage):
        summary_instruction = HumanMessage(
            content=(
                "工具已经执行完毕。请严格根据上面的真实工具结果，"
                "用自然语言总结当前步骤的最终答案。不要再次调用工具，"
                "不要输出 functions.<name>:、JSON 调用或调用说明。"
            )
        )
        model_messages = _compact_tool_messages_for_model(
            [*request.messages, summary_instruction],
            get_engine(),
        )
        summary_request = request.override(
            messages=model_messages,
            tools=[],
            tool_choice="none",
        )
        return handler(summary_request)
    return handler(request)


def _selected_capabilities(tool_names: tuple[str, ...] | None) -> list[Capability]:
    if tool_names is None:
        return list_capabilities()
    return [get_capability(name) for name in tool_names]


def _build_via_create_agent(tool_names: tuple[str, ...] | None = None):
    """Build the ReAct inner graph via ``langchain.agents.create_agent``.

    Design invariants:
        * ``system_prompt`` is ``None`` — dynamic instructions arrive
          via the ``messages`` input on each ``invoke()``.
        * ``checkpointer`` is ``None`` — crash recovery is handled at
          the outer-graph node boundary.
        * Tools come from ``list_capabilities()`` (single source of truth).
    """
    tools = [to_langchain_tool(cap) for cap in _selected_capabilities(tool_names)]
    engine = get_engine()

    return create_agent(
        model=engine,
        tools=tools,
        system_prompt=None,
        response_format=None,
        checkpointer=None,
        middleware=[_summarize_after_tool],
        debug=_AGENT_DEBUG,
        name="executor_inner_agent",
    )


# ---------------------------------------------------------------------------
# [INTERNAL] Fallback path — hand-rolled StateGraph
# ---------------------------------------------------------------------------


class _ReActState(TypedDict):
    """Minimal state for the hand-rolled ReAct inner loop.

    Only ``messages`` is needed — the inner loop has no concept of
    plan steps, iterations, or task goals (those live in AgentState)."""

    messages: Annotated[list[BaseMessage], add_messages]


def _agent_node(state: _ReActState) -> dict:
    """Select a tool first, then summarize without tools after ToolMessage."""
    messages = state["messages"]
    engine = get_engine()
    if messages and isinstance(messages[-1], ToolMessage):
        summary_instruction = HumanMessage(
            content=(
                "工具已经执行完毕。请严格根据真实工具结果总结当前步骤，"
                "不要再次调用工具或输出工具协议标记。"
            )
        )
        response: AIMessage = engine.invoke(  # type: ignore[assignment]
            _compact_tool_messages_for_model(
                [*messages, summary_instruction],
                engine,
            )
        )
    else:
        tools = [to_langchain_tool(cap) for cap in list_capabilities()]
        model_with_tools = engine.bind_tools(tools)
        response = model_with_tools.invoke(messages)  # type: ignore[assignment]
    return {"messages": [response]}


def _tool_node(state: _ReActState) -> dict:
    """Tool-execution node: run every tool_call in the last AIMessage."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    results: list[ToolMessage] = []
    for call in last_message.tool_calls:
        try:
            capability = get_capability(call["name"])
            result = _run_capability(capability, call["args"])
        except Exception as exc:
            result = f"Tool error: {exc}"
        results.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )
    return {"messages": results}


def _should_continue(state: _ReActState) -> str:
    """Route after the agent node: tools if tool_calls exist, else END."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def _build_via_self_made_stategraph(
    tool_names: tuple[str, ...] | None = None,
):
    """Build the ReAct inner graph from raw StateGraph primitives.

    The resulting graph has the same ``.invoke({"messages": [...]})``
    interface as the create_agent path, making the two paths drop-in
    interchangeable.

    Nodes:
        ``agent`` — LLM with bound tools.
        ``tools`` — execute every tool call and return ToolMessage results.

    Edges:
        entry → agent → (conditional) → tools → agent  …or…  → END
    """
    allowed = _selected_capabilities(tool_names)
    allowed_by_name = {cap.name: cap for cap in allowed}

    def scoped_agent_node(state: _ReActState) -> dict:
        messages = state["messages"]
        engine = get_engine()
        if messages and isinstance(messages[-1], ToolMessage):
            summary_instruction = HumanMessage(
                content=(
                    "工具已经执行完毕。请严格根据真实工具结果总结当前步骤，"
                    "不要再次调用工具或输出工具协议标记。"
                )
            )
            response: AIMessage = engine.invoke(  # type: ignore[assignment]
                _compact_tool_messages_for_model(
                    [*messages, summary_instruction],
                    engine,
                )
            )
        else:
            model_with_tools = engine.bind_tools(
                [to_langchain_tool(cap) for cap in allowed]
            )
            response = model_with_tools.invoke(messages)  # type: ignore[assignment]
        return {"messages": [response]}

    def scoped_tool_node(state: _ReActState) -> dict:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}
        results: list[ToolMessage] = []
        for call in last_message.tool_calls:
            try:
                capability = allowed_by_name[call["name"]]
                result = _run_capability(capability, call["args"])
            except Exception as exc:
                result = f"Tool error: {exc}"
            results.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )
        return {"messages": results}

    graph = StateGraph(_ReActState)
    graph.add_node("agent", scoped_agent_node)
    graph.add_node("tools", scoped_tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# [STABLE] Public entry-points
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def initialize_react_agent(tool_names: tuple[str, ...] | None = None):
    """Build and cache the compiled ReAct inner subgraph.

    Must be called **after** ``bootstrap_capabilities()`` so that
    ``list_capabilities()`` returns a non-empty list.

    The subgraph is built once and reused across every executor_node
    invocation.  Tools are snapshotted at build time.

    Route selection:
        * ``USE_OFFICIAL_CREATE_AGENT=true`` (default) → ``create_agent``.
        * ``USE_OFFICIAL_CREATE_AGENT=false`` → hand-rolled StateGraph.
    """
    if USE_OFFICIAL_CREATE_AGENT:
        return _build_via_create_agent(tool_names)
    return _build_via_self_made_stategraph(tool_names)


# Backward-compat alias — the V2/V3 naming transition.
get_react_agent = initialize_react_agent


def _reset_react_agent_for_testing() -> None:
    """[TEST-ONLY] Clear the cached ReAct subgraph so a fresh one can be
    built (e.g. after registering new capabilities in a test)."""
    initialize_react_agent.cache_clear()
