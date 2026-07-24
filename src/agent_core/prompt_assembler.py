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

# Forward reference to Capability — imported lazily so tests can mock it.
# The actual type lives in capability_registry.Capability.
Capability = Any  # replaced at import time — see _init_capability_type()


# ---------------------------------------------------------------------------
# [INTERNAL] Jinja2 environment
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render_system_prompt(template_name: str, **context: Any) -> SystemMessage:
    """Render a named Jinja2 template into a ``SystemMessage``.

    Args:
        template_name: Base name of the template (e.g. ``"planning_system.jinja2"``).
        context: Keyword arguments passed as template variables.

    Returns:
        A ``SystemMessage`` with the rendered template text as content.

    Raises:
        PromptAssemblyError: If the template file is not found.
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
# [INTERNAL] Tool description rendering
# ---------------------------------------------------------------------------


def _format_schema_as_text(schema: dict) -> str:
    """Convert a JSON Schema ``properties`` dict into a concise human-readable string.

    Example::

        {"query": {"type": "string", "description": "Search keyword"}}

    becomes: ``"query (string, required): Search keyword"``.
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
    """Render a capability list as human-readable text for the system prompt."""
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
    """Build the R2 context projection without changing R0/R1 prompts."""
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
) -> list[dict]:
    """Build a bounded, evidence-preserving view for model-facing prompts.

    Full records remain in AgentState/checkpoints/telemetry.  Planner,
    Reflector and Finalizer receive this projection so one large tool payload
    cannot consume their entire protected SystemMessage.
    """
    records = [dict(record) for record in execution_log[-32:]]
    max_chars = 4096
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
# [INTERNAL] History construction — AgentState → BaseMessage
# ---------------------------------------------------------------------------


def _build_history_messages(execution_log: list[dict]) -> list[BaseMessage]:
    """Convert ``AgentState.execution_log`` into a ``BaseMessage`` sequence.

    Mapping rules (per design doc §5):

    +------------------------------+--------------------------------------------------+
    | Log entry                    | Produces                                         |
    +==============================+==================================================+
    | ``step`` (description)       | ``HumanMessage(step)`` — placed                   |
    |                              | **before** the corresponding assistant/tool msgs  |
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
# [INTERNAL] Structural validation
# ---------------------------------------------------------------------------


def _validate_message_sequence(messages: list[BaseMessage]) -> None:
    """Validate structural invariants on the assembled message list.

    Every check that fails raises ``PromptAssemblyError`` with an
    index-precise error message so the operator can locate the problem.
    """
    if not messages:
        raise PromptAssemblyError(
            "Message list is empty — at least a SystemMessage is required."
        )

    # 1. First message must be SystemMessage
    if not isinstance(messages[0], SystemMessage):
        raise PromptAssemblyError(
            f"Message #0 must be SystemMessage, got {type(messages[0]).__name__}"
        )

    for idx, msg in enumerate(messages):
        msg_type = type(msg).__name__

        # 2. No empty-content messages (except AIMessage with tool_calls)
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

        # 3. ToolMessage must be preceded by matching AIMessage(tool_calls)
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
# [STABLE] Public entry-points
# ---------------------------------------------------------------------------


def assemble_planning_prompt(
    state: dict,
    engine: TokenCounter,
    reserved_for_generation: int = 512,
) -> list[BaseMessage]:
    """Assemble the message list for the **Planner** node.

    Contents:
      * **SystemMessage** — planner role + task goal + available tools.
      * **HumanMessage** — reflection notes from the previous iteration
        (if any), prefixed so they do not collide with user input.

    Args:
        state: ``AgentState``-shaped dict (see ``graph/state.py``).
        engine: Token counter (``ChatLlamaCpp`` or stub).
        reserved_for_generation: Tokens reserved for the model's response.

    Returns:
        A structurally valid, budget-compliant message list.
    """
    from agent_core.capability_registry import list_capabilities  # deferred import

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

    # Append reflection notes from previous cycle (if any)
    if reflection_notes:
        notes_text = "以下是上一轮的评估反馈：\n" + "\n".join(
            f"- {note}" for note in reflection_notes
        )
        messages.append(HumanMessage(content=notes_text))

    _validate_message_sequence(messages)

    # Budget management
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
    """Assemble the message list for the **Executor** node.

    Contents:
      * **SystemMessage** — executor role + current step instruction + tools.
      * **History**: ``HumanMessage(step)`` → ``AIMessage(…/tool_calls)`` →
        ``ToolMessage(result)`` for each logged step.
      * (Optional) latest tool result appended as a ``ToolMessage``.

    Args:
        state: ``AgentState``-shaped dict.
        engine: Token counter.
        tool_result: Result from the most recent tool call, if any.
        reserved_for_generation: Tokens reserved for generation.
        available_tools: Explicit tools for this step. ``None`` uses every
            registered tool; an empty list produces a tool-free prompt.

    Returns:
        A structurally valid, budget-compliant message list.
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

    # Always end the initial input with the current step as a HumanMessage.
    # Besides making the instruction explicit, this lets the ReAct middleware
    # distinguish an old ToolMessage in history from a ToolMessage produced
    # during the current inner loop.
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

    # Append optional latest tool result
    if tool_result is not None:
        # Ensure the tool result is paired with a preceding AIMessage(tool_calls)
        call_id = state.get("_last_tool_call_id", "unknown")
        last = raw[-1] if raw else None
        if (
            isinstance(last, AIMessage)
            and last.tool_calls
            and any(tc["id"] == call_id for tc in last.tool_calls)
        ):
            raw.append(ToolMessage(content=tool_result, tool_call_id=call_id))
        else:
            # Synthesize a matching AIMessage(tool_calls) before the result
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
    """Assemble the message list for the **Reflector** node.

    Contents:
      * **SystemMessage** — evaluator role + task goal + execution log summary.
      * No tool descriptions — the reflector never calls tools.

    Args:
        state: ``AgentState``-shaped dict.
        engine: Token counter.
        reserved_for_generation: Tokens reserved for generation.

    Returns:
        A structurally valid, budget-compliant message list.
    """
    task_goal = state.get("task_goal", "")
    plan_steps: list[str] = state.get("plan_steps", [])
    execution_log = _project_execution_log(
        state.get("execution_log", []),
        engine,
        token_budget=max(1024, min(6000, _context_window(state, engine) // 3)),
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
    """Build a tool-free prompt that synthesizes the whole task result."""
    projected = _project_execution_log(
        state.get("execution_log", []),
        engine,
        token_budget=max(
            1024,
            min(6000, _context_window(state, engine) // 3),
        ),
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
# [INTERNAL] Budget helper
# ---------------------------------------------------------------------------


def _apply_budget(
    messages: list[BaseMessage],
    budget: int,
    engine: TokenCounter,
) -> list[BaseMessage]:
    """Apply token-budget trimming and re-validate."""
    try:
        trimmed = truncate_history(messages, max_tokens=budget, engine=engine)
    except ContextBudgetExceededError:
        raise  # already the right type
    _validate_message_sequence(trimmed)
    return trimmed


def _context_window(state: dict, engine: TokenCounter) -> int:
    """Resolve the real context window used by the inference engine.

    Older checkpoints may carry an explicit ``n_ctx`` field, so preserve it
    when present.  Normal AgentState does not contain that field; in that
    case use the engine configuration instead of silently falling back to a
    hard-coded 4096-token window.
    """
    state_n_ctx = state.get("n_ctx")
    if state_n_ctx is not None:
        return int(state_n_ctx)
    return int(getattr(engine, "n_ctx", 4096))
