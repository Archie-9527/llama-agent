"""Executor 节点——在 ReAct 内部子图中运行当前计划步骤。

Executor 连接两种状态 Schema：
    * **外层** ``AgentState``——业务级字段，例如 plan_steps、execution_log。
    * **内层** ``{"messages": [...]}``——ReAct 子图所需的消息驱动状态。

所有桥接逻辑都封装在本文件中，因此 ``planner_node`` 和 ``reflector_node`` 无需
知道内部子图的存在。
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
# [内部实现] 状态桥接——外层 AgentState ↔ 内层 {"messages": [...]}
# ---------------------------------------------------------------------------


def _build_react_input(
    state: "AgentState",
    *,
    tools: list | None = None,
) -> dict:
    """将外层 ``AgentState`` 转换为内部子图的输入格式。

    调用 ``assemble_execution_prompt`` 生成完整组装、符合预算且结构有效的
    消息列表，再包装为内部子图需要的 ``{"messages": [...]}`` 字典。
    """
    engine = get_engine()
    messages = assemble_execution_prompt(
        state,
        engine,
        available_tools=tools,
    )
    return {"messages": messages}


def _required_tool_names_for_step(current_step: str) -> tuple[str, ...]:
    """返回一个原子计划步骤明确指定的工具。

    Planner 契约要求每个使用工具的步骤都给出工具名称并保持原子性。仅作为既有
    证据提到的工具名，例如“``line_number 为 search_log 返回的行号``”，不能
    变成另一次必需调用。优先识别命令式的“使用/调用/use/call <tool>”目标；
    为兼容旧格式，也接受步骤开头的工具名称。
    """
    from agent_core.capability_registry import list_capabilities

    names = {cap.name for cap in list_capabilities()}
    if not names:
        return ()

    alternation = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True)
    )
    imperative = re.search(
        rf"(?:使用|调用|通过|use|call)\s*(?:工具\s*)?({alternation})\b",
        current_step,
        re.IGNORECASE,
    )
    if imperative:
        matched = imperative.group(1)
        return (next(name for name in names if name.casefold() == matched.casefold()),)

    leading = re.search(
        rf"^\s*(?:\d+[.、)]\s*)?({alternation})\b",
        current_step,
        re.IGNORECASE,
    )
    if leading:
        matched = leading.group(1)
        return (next(name for name in names if name.casefold() == matched.casefold()),)
    return ()


def _is_tool_free_conversation_step(current_step: str) -> bool:
    """原子计划步骤未声明目标工具时返回真。"""
    return not _required_tool_names_for_step(current_step)


def _run_tool_free_step(
    state: "AgentState",
    current_step: str,
) -> list[dict]:
    """不绑定任何工具，执行已确认的纯会话步骤。"""
    messages = _build_react_input(state, tools=[])["messages"]
    response = get_engine().invoke(messages)
    content = str(response.content or "").strip()
    records = (
        [{"step": current_step, "result": content, "tool_used": None}]
        if content
        else _recover_empty_response(messages, current_step)
    )
    if not records:
        raise ExecutionError(
            f"Step '{current_step}' produced no final response after one "
            "tool-free recovery attempt."
        )
    _validate_required_tool_execution(current_step, records)
    return records


def _normalize_output_messages(raw_messages: list) -> list[BaseMessage]:
    """以防御方式规范化内部子图输出。

    某些 LangGraph 路径会在 ``messages`` 列表中返回普通 ``dict``，尤其是节点
    使用 ``add_messages`` 并返回字典时。本函数将所有内容转换为正确的
    ``BaseMessage`` 子类，使下游提取代码无需猜测格式。

    参数：
        raw_messages：原始 ``react_output["messages"]`` 列表，可能包含
            ``dict``、``BaseMessage`` 或二者混合。

    返回：
        每一项都是 ``BaseMessage`` 子类的列表。
    """
        # convert_to_messages 能处理 dict 与 BaseMessage 混合列表
    return list(convert_to_messages(raw_messages))


def _extract_execution_result(
    messages: list[BaseMessage],
    current_step: str,
) -> list[dict]:
    """将内部子图消息列表解析回执行日志记录。

    遍历规范化后的消息列表，把每个 ``AIMessage(tool_calls=…)`` 与其后的一个或
    多个 ``ToolMessage`` 配对。最后一个不含工具调用的 ``AIMessage``（步骤
    结论）也会生成记录。

    参数：
        messages：来自 ``_normalize_output_messages`` 的规范化消息列表。
        current_step：当前计划步骤文本，用于 ``step`` 字段。

    返回：
        符合 ``AgentState.execution_log`` Schema 的执行日志字典列表：
        ``{"step": str, "result": str, "tool_used": str | None}``。

    异常：
        ExecutionError：``AIMessage(tool_calls=…)`` 引用的 ``tool_call_id``
            在后续消息中没有匹配的 ``ToolMessage`` 时抛出。
    """

    records: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
            # 在剩余消息中查找匹配的 ToolMessage
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
            # 这是纯文本回答——当前步骤的最终结论
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
    required_tool_names: tuple[str, ...] | None = None,
) -> None:
    """拒绝从未真实执行的文本工具调用仿制品。

    真实工具执行一定会产生 ToolMessage，进而生成设置了 ``tool_used`` 的日志
    记录。小型本地模型有时知道需要工具，却只在普通 AIMessage 中打印
    ``tool_name(...)`` 或 JSON 片段。将其视为成功并不安全，会导致后续规划和
    反思轮次使用虚构的工具结果。
    """
    from agent_core.capability_registry import list_capabilities

    tool_names = [cap.name for cap in list_capabilities()]
    used_tools = {
        record["tool_used"] for record in records if record.get("tool_used")
    }

    required_by_step = list(
        _required_tool_names_for_step(current_step)
        if required_tool_names is None
        else required_tool_names
    )
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
        preview = re.sub(r"\s+", " ", plain_text).strip()[:240]
        raise ExecutionError(
            f"Step '{current_step}' returned textual pseudo tool call(s) "
            f"{pseudo_calls} instead of structured tool_calls; no tool was "
            f"executed. Response preview: {preview!r}"
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
    """对于空文本或泄漏的 llama.cpp 函数协议标记返回 False。"""
    stripped = content.strip()
    if not stripped:
        return False
    return re.fullmatch(r"functions\.[A-Za-z_][A-Za-z0-9_]*\s*:\s*", stripped) is None


def _recover_empty_response(
    messages: list[BaseMessage],
    current_step: str,
) -> list[dict]:
    """内部 Agent 没有返回可见输出时，不带工具重试一次。"""
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
    """内部 Agent 遗漏摘要时，不带工具重试一次模型调用。"""
    has_tool_result = any(record.get("tool_used") for record in records)
    if has_tool_result:
    # 仅包含协议的文本属于内部格式泄漏，不是面向用户的结论。判断是否需要恢复
    # 之前先将其丢弃。
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
# [稳定接口] executor_node
# ---------------------------------------------------------------------------


def executor_node(state: "AgentState", config: RunnableConfig) -> "AgentState":
    """通过 ReAct 内部子图执行当前计划步骤。

    1. 从 ``plan_steps[current_step_index]`` 取得当前步骤。
    2. 通过 ``_build_react_input`` 构建内部子图输入消息。
    3. 在 ``recursion_limit`` 安全上限下调用已缓存的内部子图。
    4. 将输出消息解析为 ``execution_log`` 记录。
    5. 推进 ``current_step_index``；全部步骤完成后，将 ``status`` 转换为
       ``"reflecting"``。

    返回：
        添加新执行日志并更新步骤索引后的 *state*。

    异常：
        ExecutionError：内部子图超过递归上限或输出消息格式错误时抛出。
    """
    plan_steps: list[str] = state.get("plan_steps", [])
    current_step_index: int = state.get("current_step_index", 0)

    if current_step_index >= len(plan_steps):
        raise ExecutionError(
            f"executor_node called with current_step_index={current_step_index} "
            f"but plan_steps has only {len(plan_steps)} steps."
        )

    current_step = plan_steps[current_step_index]

    required_tool_names = _required_tool_names_for_step(current_step)
    if not required_tool_names:
        new_records = _run_tool_free_step(state, current_step)
        state["execution_log"].extend(new_records)
        state["current_step_index"] += 1
        if state["current_step_index"] >= len(plan_steps):
            state["status"] = "reflecting"
        return state

    outer_thread_id = config["configurable"]["thread_id"]

    sub_config = {
        "configurable": {
            "thread_id": f"{outer_thread_id}::step-{current_step_index}"
        },
        "recursion_limit": 8
    }

    from agent_core.capability_registry import get_capability

    step_tools = [get_capability(name) for name in required_tool_names]

    # 只公开当前原子步骤的目标工具。这样既能减少 Prompt/Schema 开销，也能防止
    # 小型本地模型选择语义相关但不正确的能力。
    react_input = _build_react_input(state, tools=step_tools)

    # 记录输入消息的数量，以便我们稍后可以仅隔离由内部代理新生成的消息。
    # 内部子图可能会返回完整的对话历史记录（例如，当FakeReactAgent连接输入和输出时），
    # 我们不能重新从已经在先前步骤中记录的历史消息中提取记录。
    input_message_count = len(react_input["messages"])

    # 获取已缓存的内部 Agent
    from agent_core.graph.react_agent_factory import initialize_react_agent

    agent = initialize_react_agent(required_tool_names)

    # 在递归安全上限下运行内部循环
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
    _validate_required_tool_execution(
        current_step,
        new_records,
        required_tool_names=required_tool_names,
    )

    state["execution_log"].extend(new_records)
    state["current_step_index"] += 1

    # 所有步骤完成后转换到反思阶段
    if state["current_step_index"] >= len(plan_steps):
        state["status"] = "reflecting"

    return state
