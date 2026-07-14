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
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    ToolConsistencyError,
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
    for entry in execution_log:
        step_text = entry.get("step", "")
        result_text = entry.get("result", "")
        tool_used: Optional[str] = entry.get("tool_used")

        if step_text:
            messages.append(HumanMessage(content=step_text))

        if tool_used:
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name=tool_used,
                            args={},  # simplified — at assembly time we don't know args
                            id=call_id,
                        )
                    ],
                )
            )
            messages.append(
                ToolMessage(content=result_text, tool_call_id=call_id)
            )
        else:
            messages.append(AIMessage(content=result_text))
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

    sys_msg = _render_system_prompt(
        "planning_system.jinja2",
        task_goal=task_goal,
        tools_section=tools_section,
        plan_steps=plan_steps,
    )

    messages: list[BaseMessage] = [sys_msg]

    # Append reflection notes from previous cycle (if any)
    reflection_notes: list[str] = state.get("reflection_notes", [])
    if reflection_notes:
        notes_text = "以下是上一轮的评估反馈：\n" + "\n".join(
            f"- {note}" for note in reflection_notes
        )
        messages.append(HumanMessage(content=notes_text))

    _validate_message_sequence(messages)

    # Budget management
    n_ctx = state.get("n_ctx", 4096)
    budget = n_ctx - reserved_for_generation
    messages = _apply_budget(messages, budget, engine)

    return messages


def assemble_execution_prompt(
    state: dict,
    engine: TokenCounter,
    tool_result: Optional[str] = None,
    reserved_for_generation: int = 512,
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

    Returns:
        A structurally valid, budget-compliant message list.
    """
    from agent_core.capability_registry import list_capabilities

    tools = list_capabilities()
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
        current_step=current_step,
        tools_section=tools_section,
    )

    execution_log: list[dict] = state.get("execution_log", [])
    history_msgs = _build_history_messages(execution_log)

    raw: list[BaseMessage] = [sys_msg] + history_msgs

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

    n_ctx = state.get("n_ctx", 4096)
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
    execution_log: list[dict] = state.get("execution_log", [])

    sys_msg = _render_system_prompt(
        "reflection_system.jinja2",
        task_goal=task_goal,
        plan_steps=plan_steps,
        execution_log=execution_log,
    )

    raw: list[BaseMessage] = [sys_msg]
    _validate_message_sequence(raw)

    n_ctx = state.get("n_ctx", 4096)
    budget = n_ctx - reserved_for_generation
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
