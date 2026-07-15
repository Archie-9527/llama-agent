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
from typing import Annotated

from langchain.agents import create_agent  # type: ignore[import-untyped]
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool as langchain_tool_decorator
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent_core.capability_registry import Capability, get_capability, list_capabilities
from agent_core.llm_engine import get_engine

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

USE_OFFICIAL_CREATE_AGENT = (
    os.environ.get("USE_OFFICIAL_CREATE_AGENT", "true").lower() == "true"
)

_AGENT_DEBUG = os.environ.get("AGENT_DEBUG", "false").lower() == "true"


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
        return cap.handler(**kwargs)

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
            return cap.handler(**kwargs)

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


def _build_via_create_agent():
    """Build the ReAct inner graph via ``langchain.agents.create_agent``.

    Design invariants:
        * ``system_prompt`` is ``None`` — dynamic instructions arrive
          via the ``messages`` input on each ``invoke()``.
        * ``checkpointer`` is ``None`` — crash recovery is handled at
          the outer-graph node boundary.
        * Tools come from ``list_capabilities()`` (single source of truth).
    """
    tools = [to_langchain_tool(cap) for cap in list_capabilities()]
    engine = get_engine()

    return create_agent(
        model=engine,
        tools=tools,
        system_prompt=None,
        response_format=None,
        checkpointer=None,
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
    """LLM-decision node: bind tools and call the model on current messages."""
    tools = [to_langchain_tool(cap) for cap in list_capabilities()]
    model_with_tools = get_engine().bind_tools(tools)
    response: AIMessage = model_with_tools.invoke(state["messages"])  # type: ignore[assignment]
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
            result = capability.handler(**call["args"])
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


def _build_via_self_made_stategraph():
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
    graph = StateGraph(_ReActState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# [STABLE] Public entry-point
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_react_agent():
    """Return the (cached) compiled ReAct inner subgraph.

    The subgraph is built once and reused across every executor_node
    invocation.  Tools are snapshotted at build time — if you register
    new capabilities after the first call you must clear the LRU cache
    (``get_react_agent.cache_clear()``) or restart the process.

    Route selection:
        * ``USE_OFFICIAL_CREATE_AGENT=true`` (default) → ``create_agent``.
        * ``USE_OFFICIAL_CREATE_AGENT=false`` → hand-rolled StateGraph.
    """
    if USE_OFFICIAL_CREATE_AGENT:
        return _build_via_create_agent()
    return _build_via_self_made_stategraph()
