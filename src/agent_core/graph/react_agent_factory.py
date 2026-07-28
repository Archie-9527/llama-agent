"""内部 ReAct 子图——单步骤工具使用循环。

外层每次调用 ``executor_node``，都会针对当前计划步骤运行一次内部 Agent。
内部 Agent 按以下流程循环：
    LLM 决策 → 工具执行 → 回传结果 → LLM 决策
直到模型生成最终文本回答，不再调用工具。

存在两条构建路径：
    * **主路径**：``langchain.agents.create_agent`` 官方包装。
    * **后备路径**：手工构建的 ``StateGraph``，通过环境变量
      ``USE_OFFICIAL_CREATE_AGENT=false`` 启用。

两条路径会生成形状相同的已编译图。内部图有意不接收 Checkpointer，使崩溃
恢复保持在外层图节点边界。
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
    """执行一个能力并发出稳定的工具事件。"""
    from agent_core.telemetry import get_telemetry

    telemetry = get_telemetry()
    started = monotonic_ns()
    try:
        # LangChain/Pydantic 会把省略的可选 Schema 属性具体化为 ``None``。
        # 删除这些值可让 Provider 的 Python 默认值生效，避免把省略的
        # limit/offset 变成运行时类型错误，例如 ``min(None, max_rows)``。
        normalized_kwargs = {
            key: value for key, value in kwargs.items() if value is not None
        }
        from agent_core.interactive.events import emit_interactive_event

        emit_interactive_event(
            "tool_started",
            tool_name=cap.name,
            arguments=normalized_kwargs,
        )
        result = cap.handler(**normalized_kwargs)
        # R1 拦截位于所有工具共享的唯一边界。策略关闭时返回完全相同的原始对象，
        # 从而保持 R0 行为。
        from agent_core.artifacts.virtualizer import maybe_virtualize_tool_result

        inline_result = maybe_virtualize_tool_result(cap.name, result)
    except Exception as exc:
        from agent_core.interactive.events import emit_interactive_event

        duration_ms = (monotonic_ns() - started) / 1_000_000
        emit_interactive_event(
            "tool_failed",
            tool_name=cap.name,
            duration_ms=duration_ms,
            error=f"{type(exc).__name__}: {exc}",
        )
        telemetry.record_event(
            "tool_events.jsonl",
            "tool_failed",
            tool_name=cap.name,
            duration_ms=duration_ms,
            input_bytes=len(str(kwargs).encode("utf-8")),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    duration_ms = (monotonic_ns() - started) / 1_000_000
    emit_interactive_event(
        "tool_completed",
        tool_name=cap.name,
        duration_ms=duration_ms,
        result_preview=str(inline_result)[:2000],
    )
    telemetry.record_event(
        "tool_events.jsonl",
        "tool_completed",
        tool_name=cap.name,
        duration_ms=duration_ms,
        input_bytes=len(str(kwargs).encode("utf-8")),
        output_bytes=len(str(result).encode("utf-8")),
    )
    return inline_result

# ---------------------------------------------------------------------------
# 功能开关
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
    """在后续摘要推理前限制最新工具结果的大小。

    完整 ToolMessage 仍保留在 LangGraph 状态和遥测中，只压缩面向模型的副本。
    同时保留头尾对于文件和日志工具很重要，因为最终证据通常出现在 EOF。
    这是独立于 R1 Artifact 虚拟化的强制上下文安全保护。
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
# [内部实现] Capability → LangChain BaseTool 转换
# ---------------------------------------------------------------------------


def to_langchain_tool(cap: Capability) -> BaseTool:
    """将已注册的 ``Capability`` 包装为 LangChain ``BaseTool``。

    返回的工具把执行委托给 ``cap.handler``，并携带相同的 ``name``、
    ``description`` 和 ``input_schema``，使 ``create_agent`` / ``bind_tools``
    接收到一致的元数据。

    这是能力注册表与 LangChain 工具抽象之间的**唯一**桥梁。其他所有模块都通过
    ``list_capabilities()`` 使用工具，只有本函数负责转换。
    """
    # 根据能力的 JSON Schema 构建动态 Schema 模型，使 LangChain 能生成正确的
    # OpenAI 风格工具定义。
    from pydantic import create_model

    fields: dict[str, tuple[type, object]] = {}
    props = cap.input_schema.get("properties", {})
    required_set: set[str] = set(cap.input_schema.get("required", []))

    for param_name, param_schema in props.items():
        param_type = _json_type_to_python(param_schema.get("type", "string"))
        default = ... if param_name in required_set else None
        fields[param_name] = (param_type, default)

    # 包装原始处理程序，使 LangChain 看到正确的可调用签名。由于 create_model
    # 生成仅限关键字的字段，此处使用 **kwargs。
    def _handler(**kwargs):
        return _run_capability(cap, kwargs)

    # 如果 Schema 不含属性，则创建简单的无参数工具。
    if not fields:
        return langchain_tool_decorator(
            name_or_callable=cap.name,
            description=cap.description,
        )(_handler)

    # 为结构化输入创建 Pydantic 模型并添加装饰。
    ArgsModel = create_model(f"{cap.name}_args", **fields)  # type: ignore[call-overload]

    class _StructuredTool(BaseTool):
        name: str = cap.name
        description: str = cap.description
        args_schema: type = ArgsModel

        def _run(self, **kwargs):
            return _run_capability(cap, kwargs)

    return _StructuredTool()


def _json_type_to_python(json_type: str) -> type:
    """将 JSON Schema 类型名称映射为 Python 类型。"""
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
# [稳定接口] 内部子图——主路径（create_agent）
# ---------------------------------------------------------------------------


@wrap_model_call
def _summarize_after_tool(request, handler):
    """从工具选择切换到不带工具的摘要调用。

    ``create_agent`` 通常会在每个 ToolMessage 后重新绑定全部工具。因此，小型
    本地模型可能重复同一调用，或输出 ``functions.count_lines:`` 之类的原始
    协议标记。当前步骤获得工具结果后，应移除所有工具，并明确要求模型根据该
    真实结果回答。
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
    """通过 ``langchain.agents.create_agent`` 构建 ReAct 内部图。

    设计不变量：
        * ``system_prompt`` 为 ``None``——动态指令在每次 ``invoke()`` 时通过
          ``messages`` 输入传入。
        * ``checkpointer`` 为 ``None``——崩溃恢复在外层图节点边界处理。
        * 工具来自唯一信息源 ``list_capabilities()``。
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
# [内部实现] 后备路径——手工构建的 StateGraph
# ---------------------------------------------------------------------------


class _ReActState(TypedDict):
    """手工构建的 ReAct 内部循环所需的最小状态。

    只需要 ``messages``；内部循环不了解计划步骤、迭代或任务目标，这些内容位于
    AgentState 中。
    """

    messages: Annotated[list[BaseMessage], add_messages]


def _agent_node(state: _ReActState) -> dict:
    """先选择工具，收到 ToolMessage 后再进行不带工具的总结。"""
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
    """工具执行节点：运行最后一个 AIMessage 中的每个 tool_call。"""
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
    """Agent 节点后的路由：存在 tool_calls 时进入工具，否则进入 END。"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def _build_via_self_made_stategraph(
    tool_names: tuple[str, ...] | None = None,
):
    """使用原始 StateGraph 基础组件构建 ReAct 内部图。

    生成的图与 create_agent 路径具有相同的
    ``.invoke({"messages": [...]})`` 接口，因此两条路径可以直接互换。

    节点：
        ``agent``——绑定工具的 LLM。
        ``tools``——执行所有工具调用并返回 ToolMessage 结果。

    边：
        入口 → agent →（条件）→ tools → agent，或者 → END
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
# [稳定接口] 公共入口
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def initialize_react_agent(tool_names: tuple[str, ...] | None = None):
    """构建并缓存已编译的 ReAct 内部子图。

    必须在 ``bootstrap_capabilities()`` **之后**调用，确保
    ``list_capabilities()`` 返回非空列表。

    子图只构建一次，并在每次 executor_node 调用时复用；工具在构建时生成快照。

    路由选择：
        * ``USE_OFFICIAL_CREATE_AGENT=true``（默认）→ ``create_agent``。
        * ``USE_OFFICIAL_CREATE_AGENT=false`` → 手工构建的 StateGraph。
    """
    if USE_OFFICIAL_CREATE_AGENT:
        return _build_via_create_agent(tool_names)
    return _build_via_self_made_stategraph(tool_names)


# 向后兼容别名——用于 V2/V3 命名过渡。
get_react_agent = initialize_react_agent


def _reset_react_agent_for_testing() -> None:
    """[仅测试] 清除已缓存的 ReAct 子图，以便重新构建，例如测试注册新能力后。"""
    initialize_react_agent.cache_clear()
