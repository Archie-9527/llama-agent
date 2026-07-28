"""prompt组装器 — 将 ``AgentState`` 转换为 ``list[BaseMessage]``。

这是 ``graph/`` 节点（负责业务逻辑）与推理引擎（期望标准 LangChain 消息对象）之间的翻译层。
它保证其生成的每个消息列表具有以下四个属性：

1. **结构有效性** — 正确的角色排序，没有孤立的 ``ToolMessage``。
2. **预算有效性** — 总Tokens数 ≤ ``n_ctx - reserved_for_generation``。
3. **一致性** — 工具文本描述和语法约束来自相同的 ``list_capabilities()`` 调用。
4. **确定性** — 相同的输入总是产生相同的输出。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall

from agent_core.exceptions import (
    ContextBudgetExceededError,
    PromptAssemblyError,
)
from agent_core.knowledge_scope import TokenCounter, truncate_history

# Capability 前向引用——延迟导入以便测试进行 Mock。
# 实际类型位于 capability_registry.Capability。
Capability = Any  # 导入时替换，参见 _init_capability_type()


# ---------------------------------------------------------------------------
# [内部实现] Jinja2 环境
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_EVIDENCE_SYNTHESIS_RESULT_CHARS = 512

_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render_system_prompt(template_name: str, **context: Any) -> SystemMessage:
    """将指定的 Jinja2 模板渲染为 ``SystemMessage``。

    参数：
        template_name：模板基本名称，例如 ``"planning_system.jinja2"``。
        context：作为模板变量传入的关键字参数。

    返回：
        以渲染后的模板文本为内容的 ``SystemMessage``。

    异常：
        PromptAssemblyError：找不到模板文件时抛出。
    """
    try:
        template = _jinja_env.get_template(template_name)
    except TemplateNotFound:
        raise PromptAssemblyError(
            f"System prompt template '{template_name}' not found in {_TEMPLATES_DIR}"
        ) from None

    text = template.render(**context).strip()
    return SystemMessage(content=text)


# ---------------------------------------------------------------------------
# [内部实现] 工具描述渲染
# ---------------------------------------------------------------------------


def _format_schema_as_text(schema: dict) -> str:
    """将 JSON Schema 的 ``properties`` 字典转换为简洁的人类可读字符串。

    示例::

        {"query": {"type": "string", "description": "Search keyword"}}

    转换结果为：``"query (string, required): Search keyword"``。
    """
    properties = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    parts: list[str] = []
    for name, prop in properties.items():
        ptype = prop.get("type", "any")
        pdesc = prop.get("description", "")
        req = "required" if name in required else "optional"
        parts.append(f"- {name} ({ptype}, {req}){' — ' + pdesc if pdesc else ''}")
    return "\n".join(parts)


def _render_tools_section(tools: list[Capability]) -> str:
    """将能力列表渲染为 System Prompt 使用的人类可读文本。"""
    lines: list[str] = []
    for tool in tools:
        lines.append(f"- **{tool.name}**: {tool.description}")
        schema_text = _format_schema_as_text(tool.input_schema)
        if schema_text:
            lines.append(f"  参数:\n{schema_text}")
    return "\n".join(lines)


def _lifecycle_context_message(
    state: dict,
    engine: TokenCounter,
    *,
    phase: str,
) -> HumanMessage | None:
    """构建 R2 上下文投影，同时保持 R0/R1 Prompt 不变。"""
    from agent_core.memory import get_lifecycle_context_manager

    view = get_lifecycle_context_manager().prepare(
        state,
        phase=phase,
        engine=engine,
    )
    rendered = view.render().strip()
    if not rendered:
        return None
    return HumanMessage(
        content=(
            "以下内容是生命周期上下文管理器选择的可信历史。只在与当前"
            "任务有关时使用，不要把历史请求当成新的待执行指令。\n\n"
            + rendered
        )
    )


def _compact_record_text(value: object, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + "\n...[execution record compacted; head and tail preserved]...\n"
        + text[-tail:]
    )


def _project_execution_log(
    execution_log: list[dict],
    engine: TokenCounter,
    *,
    token_budget: int,
    max_result_chars: int = 4096,
) -> list[dict]:
    """为面向模型的 Prompt 构建有界且保留证据的视图。

    完整记录仍保存在 AgentState、Checkpoint 和遥测中。Planner、Reflector 与
    Finalizer 接收该投影，避免单个大型工具载荷占满其受保护的 SystemMessage。
    ``max_result_chars`` 还为证据综合阶段提供与表示方式无关的逐记录上限：
    R2 摘要不能仅因已比对应 R0/R1 记录更短，就获得更大的 Prompt 预算。
    """
    records = [dict(record) for record in execution_log[-32:]]
    max_chars = max(256, int(max_result_chars))
    while True:
        projected = [
            {
                "step": _compact_record_text(record.get("step", ""), 1024),
                "result": _compact_record_text(
                    record.get("result", ""), max_chars
                ),
                "tool_used": record.get("tool_used"),
                **(
                    {"tool_args": record.get("tool_args")}
                    if record.get("tool_args")
                    else {}
                ),
            }
            for record in records
        ]
        serialized = json.dumps(projected, ensure_ascii=False, default=str)
        if engine.get_num_tokens(serialized) <= token_budget:
            return projected
        if max_chars > 256:
            max_chars //= 2
            continue
        if len(records) > 1:
            records.pop(0)
            continue
        return projected


# ---------------------------------------------------------------------------
# [内部实现] 历史构建——AgentState → BaseMessage
# ---------------------------------------------------------------------------


def _build_history_messages(execution_log: list[dict]) -> list[BaseMessage]:
    """将 ``AgentState.execution_log`` 转换为 ``BaseMessage`` 序列。

    映射规则（依据设计文档第 5 节）：

    +------------------------------+--------------------------------------------------+
    | 日志条目                     | 生成内容                                         |
    +==============================+==================================================+
    | ``step`` (description)       | ``HumanMessage(step)`` — placed                   |
    |                              | 位于对应助手/工具消息**之前**                     |
    +------------------------------+--------------------------------------------------+
    | ``tool_used is not None``    | ``AIMessage(tool_calls=[…])`` +                   |
    |                              | ``ToolMessage(result, tool_call_id=…)``           |
    +------------------------------+--------------------------------------------------+
    | ``tool_used is None``        | ``AIMessage(content=result)``                     |
    +------------------------------+--------------------------------------------------+
    """
    messages: list[BaseMessage] = []
    previous_step: str | None = None
    for i, entry in enumerate(execution_log): 
        
        current_step = entry["step"]
        if current_step != previous_step:
            messages.append(HumanMessage(content=current_step))
            previous_step = current_step
        
        if entry["tool_used"] is not None:
            call_id = f"hist_call_{i}"
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": entry["tool_used"],
                            "args": dict(entry.get("tool_args") or {}),
                            "id": call_id,
                        }
                    ],
                )
            )
            messages.append(ToolMessage(content=entry["result"], tool_call_id=call_id))
        else:
            messages.append(AIMessage(content=entry["result"]))

    return messages


# ---------------------------------------------------------------------------
# [内部实现] 结构校验
# ---------------------------------------------------------------------------


def _validate_message_sequence(messages: list[BaseMessage]) -> None:
    """校验已组装消息列表的结构不变量。

    任一检查失败都会抛出 ``PromptAssemblyError``，错误消息会给出精确索引，
    便于操作人员定位问题。
    """
    if not messages:
        raise PromptAssemblyError(
            "Message list is empty — at least a SystemMessage is required."
        )

    # 1. 第一条消息必须是 SystemMessage
    if not isinstance(messages[0], SystemMessage):
        raise PromptAssemblyError(
            f"Message #0 must be SystemMessage, got {type(messages[0]).__name__}"
        )

    for idx, msg in enumerate(messages):
        msg_type = type(msg).__name__

        # 2. 不允许空内容消息，带 tool_calls 的 AIMessage 除外
        if isinstance(msg, (HumanMessage, SystemMessage, ToolMessage)):
            content = msg.content
            if isinstance(content, str) and content.strip() == "":
                raise PromptAssemblyError(
                    f"Message #{idx} ({msg_type}) has empty or whitespace-only content."
                )

        if isinstance(msg, AIMessage):
            has_tool_calls = bool(getattr(msg, "tool_calls", None))
            content = msg.content
            if (
                isinstance(content, str)
                and content.strip() == ""
                and not has_tool_calls
            ):
                raise PromptAssemblyError(
                    f"Message #{idx} (AIMessage) has empty content and no tool_calls."
                )

        # 3. ToolMessage 之前必须有匹配的 AIMessage(tool_calls)
        if isinstance(msg, ToolMessage):
            if idx == 0:
                raise PromptAssemblyError(
                    f"Message #0 is a ToolMessage (tool_call_id='{msg.tool_call_id}')"
                    f" — ToolMessage cannot appear as the first message."
                )

            prev = messages[idx - 1]
            if not isinstance(prev, AIMessage) or not prev.tool_calls:
                raise PromptAssemblyError(
                    f"Message #{idx} is a ToolMessage (tool_call_id='{msg.tool_call_id}') "
                    f"but message #{idx - 1} is {type(prev).__name__}, not an "
                    f"AIMessage with tool_calls."
                )

            tc_ids = {tc["id"] for tc in prev.tool_calls}
            if msg.tool_call_id not in tc_ids:
                raise PromptAssemblyError(
                    f"Message #{idx} ToolMessage tool_call_id='{msg.tool_call_id}' "
                    f"does not match any tool_call in message #{idx - 1} "
                    f"(ids: {tc_ids})."
                )


# ---------------------------------------------------------------------------
# [稳定接口] 公共入口
# ---------------------------------------------------------------------------


def assemble_planning_prompt(
    state: dict,
    engine: TokenCounter,
    reserved_for_generation: int = 512,
) -> list[BaseMessage]:
    """为 **Planner** 节点组装消息列表。

    内容：
      * **SystemMessage**——Planner 角色 + 任务目标 + 可用工具。
      * **HumanMessage**——上一轮迭代的反思记录（如有），添加前缀以避免与用户
        输入混淆。

    参数：
        state：形状符合 ``AgentState`` 的字典，参见 ``graph/state.py``。
        engine：Token 计数器，可以是 ``ChatLlamaCpp`` 或 Stub。
        reserved_for_generation：为模型回答预留的 Token 数。

    返回：
        结构有效且符合预算的消息列表。
    """
    from agent_core.capability_registry import list_capabilities  # 延迟导入

    tools = list_capabilities()
    tools_section = _render_tools_section(tools) if tools else ""

    task_goal = state.get("task_goal", "")
    plan_steps = state.get("plan_steps", [])
    reflection_notes: list[str] = state.get("reflection_notes", [])
    prior_execution_log = (
        _project_execution_log(
            state.get("execution_log", []),
            engine,
            token_budget=max(512, _context_window(state, engine) // 4),
        )
        if reflection_notes
        else []
    )

    sys_msg = _render_system_prompt(
        "planning_system.jinja2",
        task_goal=task_goal,
        tools_section=tools_section,
        plan_steps=plan_steps,
        execution_log=prior_execution_log,
    )

    messages: list[BaseMessage] = [sys_msg]
    lifecycle_message = _lifecycle_context_message(
        state, engine, phase="planner"
    )
    if lifecycle_message is not None:
        messages.append(lifecycle_message)

    # 追加上一循环的反思记录（如有）
    if reflection_notes:
        notes_text = "以下是上一轮的评估反馈：\n" + "\n".join(
            f"- {note}" for note in reflection_notes
        )
        messages.append(HumanMessage(content=notes_text))

    _validate_message_sequence(messages)

    # 预算管理
    n_ctx = _context_window(state, engine)
    budget = n_ctx - reserved_for_generation
    messages = _apply_budget(messages, budget, engine)

    return messages


def assemble_execution_prompt(
    state: dict,
    engine: TokenCounter,
    tool_result: Optional[str] = None,
    reserved_for_generation: int = 512,
    available_tools: Optional[list[Capability]] = None,
) -> list[BaseMessage]:
    """为 **Executor** 节点组装消息列表。

    内容：
      * **SystemMessage**——Executor 角色 + 当前步骤指令 + 工具。
      * **历史**：每个已记录步骤依次转换为 ``HumanMessage(step)`` →
        ``AIMessage(…/tool_calls)`` → ``ToolMessage(result)``。
      * 可选：将最新工具结果作为 ``ToolMessage`` 追加。

    参数：
        state：形状符合 ``AgentState`` 的字典。
        engine：Token 计数器。
        tool_result：最近一次工具调用的结果（如有）。
        reserved_for_generation：为生成预留的 Token 数。
        available_tools：当前步骤明确可用的工具。``None`` 表示使用所有已注册
            工具；空列表生成不带工具的 Prompt。

    返回：
        结构有效且符合预算的消息列表。
    """
    from agent_core.capability_registry import list_capabilities

    tools = list_capabilities() if available_tools is None else available_tools
    tools_section = _render_tools_section(tools) if tools else ""

    plan_steps: list[str] = state.get("plan_steps", [])
    current_step_index: int = state.get("current_step_index", 0)
    current_step = (
        plan_steps[current_step_index]
        if 0 <= current_step_index < len(plan_steps)
        else "(no step)"
    )

    sys_msg = _render_system_prompt(
        "execution_system.jinja2",
        task_goal=state.get("task_goal", ""),
        current_step=current_step,
        tools_section=tools_section,
        tool_free=available_tools is not None and not tools,
    )

    execution_log = _project_execution_log(
        state.get("execution_log", []),
        engine,
        token_budget=max(1024, min(6000, _context_window(state, engine) // 3)),
    )
    history_msgs = _build_history_messages(execution_log)

    # 初始输入始终以表示当前步骤的 HumanMessage 结束。这样既能明确指令，也能让
    # ReAct Middleware 区分历史中的旧 ToolMessage 与当前内部循环产生的
    # 工具消息。
    lifecycle_message = _lifecycle_context_message(
        state, engine, phase="executor"
    )
    raw: list[BaseMessage] = [sys_msg]
    if lifecycle_message is not None:
        raw.append(lifecycle_message)
    raw.extend(
        [
            *history_msgs,
            HumanMessage(content=f"请执行当前步骤：{current_step}"),
        ]
    )

    # 追加可选的最新工具结果
    if tool_result is not None:
        # 确保工具结果与之前的 AIMessage(tool_calls) 配对
        call_id = state.get("_last_tool_call_id", "unknown")
        last = raw[-1] if raw else None
        if (
            isinstance(last, AIMessage)
            and last.tool_calls
            and any(tc["id"] == call_id for tc in last.tool_calls)
        ):
            raw.append(ToolMessage(content=tool_result, tool_call_id=call_id))
        else:
            # 在结果之前合成匹配的 AIMessage(tool_calls)
            raw.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="(tool_result)", args={}, id=call_id)
                    ],
                )
            )
            raw.append(ToolMessage(content=tool_result, tool_call_id=call_id))

    _validate_message_sequence(raw)

    n_ctx = _context_window(state, engine)
    budget = n_ctx - reserved_for_generation
    return _apply_budget(raw, budget, engine)


def assemble_reflection_prompt(
    state: dict,
    engine: TokenCounter,
    reserved_for_generation: int = 512,
) -> list[BaseMessage]:
    """为 **Reflector** 节点组装消息列表。

    内容：
      * **SystemMessage**——评估者角色 + 任务目标 + 执行日志摘要。
      * 不包含工具描述——Reflector 永远不会调用工具。

    参数：
        state：形状符合 ``AgentState`` 的字典。
        engine：Token 计数器。
        reserved_for_generation：为生成预留的 Token 数。

    返回：
        结构有效且符合预算的消息列表。
    """
    task_goal = state.get("task_goal", "")
    plan_steps: list[str] = state.get("plan_steps", [])
    execution_log = _project_execution_log(
        state.get("execution_log", []),
        engine,
        token_budget=max(1024, min(6000, _context_window(state, engine) // 3)),
        max_result_chars=_EVIDENCE_SYNTHESIS_RESULT_CHARS,
    )

    sys_msg = _render_system_prompt(
        "reflection_system.jinja2",
        task_goal=task_goal,
        plan_steps=plan_steps,
        execution_log=execution_log,
    )

    raw: list[BaseMessage] = [sys_msg]
    lifecycle_message = _lifecycle_context_message(
        state, engine, phase="reflector"
    )
    if lifecycle_message is not None:
        raw.append(lifecycle_message)
    _validate_message_sequence(raw)

    n_ctx = _context_window(state, engine)
    budget = n_ctx - reserved_for_generation
    return _apply_budget(raw, budget, engine)


def assemble_finalization_prompt(
    state: dict,
    engine: TokenCounter,
    reserved_for_generation: int = 512,
) -> list[BaseMessage]:
    """构建不带工具、用于综合整个任务结果的 Prompt。"""
    projected = _project_execution_log(
        state.get("execution_log", []),
        engine,
        token_budget=max(
            1024,
            min(6000, _context_window(state, engine) // 3),
        ),
        max_result_chars=_EVIDENCE_SYNTHESIS_RESULT_CHARS,
    )
    sys_msg = _render_system_prompt(
        "finalization_system.jinja2",
        task_goal=state.get("task_goal", ""),
        model_summaries=[
            record for record in projected if not record.get("tool_used")
        ],
        tool_evidence=[
            record for record in projected if record.get("tool_used")
        ],
    )
    raw: list[BaseMessage] = [sys_msg]
    lifecycle_message = _lifecycle_context_message(
        state, engine, phase="finalizer"
    )
    if lifecycle_message is not None:
        raw.append(lifecycle_message)
    _validate_message_sequence(raw)
    budget = _context_window(state, engine) - reserved_for_generation
    return _apply_budget(raw, budget, engine)


# ---------------------------------------------------------------------------
# [内部实现] 预算辅助函数
# ---------------------------------------------------------------------------


def _apply_budget(
    messages: list[BaseMessage],
    budget: int,
    engine: TokenCounter,
) -> list[BaseMessage]:
    """应用 Token 预算裁剪并重新校验。"""
    try:
        trimmed = truncate_history(messages, max_tokens=budget, engine=engine)
    except ContextBudgetExceededError:
        raise  # 已经是正确类型
    _validate_message_sequence(trimmed)
    return trimmed


def _context_window(state: dict, engine: TokenCounter) -> int:
    """解析推理引擎实际使用的上下文窗口。

    旧 Checkpoint 可能带有显式 ``n_ctx`` 字段，存在时应保留。普通 AgentState
    不包含该字段，此时应使用引擎配置，而不是静默回退到硬编码的 4096 Token
    窗口。
    """
    state_n_ctx = state.get("n_ctx")
    if state_n_ctx is not None:
        return int(state_n_ctx)
    return int(getattr(engine, "n_ctx", 4096))
