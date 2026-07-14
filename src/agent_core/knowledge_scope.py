"""知识范围 — 上下文窗口预算管理。

当组合消息历史记录超过模型的上下文窗口时，代理无法生成有效的输出。此模块提供了一个函数，
``truncate_history``，用于优先修剪最旧的消息，同时保持结构不变（例如，``AIMessage(tool_calls)`` /
``ToolMessage`` 对永远不会被拆分）。

设计约束
    此模块仅依赖于 ``TokenCounter`` 协议，而不直接依赖于 ``ChatLlamaCpp``。这使得它在不加载真实模型的情况下可以进行测试。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent_core.exceptions import ContextBudgetExceededError


# ---------------------------------------------------------------------------
# [STABLE] Minimal interface for token counting
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for anything that can count tokens in a string.

    ``ChatLlamaCpp`` satisfies this via ``get_num_tokens()``, but unit
    tests can pass a lightweight stub instead.
    """

    def get_num_tokens(self, text: str) -> int:
        ...


# ---------------------------------------------------------------------------
# [STABLE] Public API
# ---------------------------------------------------------------------------


def truncate_history(
    messages: list[BaseMessage],
    max_tokens: int,
    engine: TokenCounter,
    protected_prefix: int = 1,
) -> list[BaseMessage]:
    """Trim *messages* to fit within *max_tokens* tokens.

    Trimming starts from the oldest messages (after *protected_prefix*)
    and prefers the most recent conversation turns.  When an
    ``AIMessage(tool_calls=…)`` is removed, the immediately following
    ``ToolMessage`` with a matching ``tool_call_id`` is removed as well
    (and vice-versa) — tool-call pairs are always deleted together.

    Args:
        messages: The full message list to trim.
        max_tokens: Hard token budget (must be ≥ 0).
        engine: Any object satisfying ``TokenCounter``.
        protected_prefix: Number of leading messages that are **never**
            trimmed (default 1 — protects the ``SystemMessage``).

    Returns:
        A (possibly shorter) message list whose total token count ≤
        *max_tokens*.

    Raises:
        ContextBudgetExceededError: If the budget cannot be met even
            after removing everything but the protected prefix — the
            remaining messages are simply too large.
        ValueError: If *max_tokens* is negative.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be ≥ 0")

    if not messages:
        return []

    # Fast path — already within budget
    if _total_tokens(messages, engine) <= max_tokens:
        return list(messages)

    # Split into protected head + trimmable tail
    protected = list(messages[:protected_prefix])
    trimmable = list(messages[protected_prefix:])

    if not trimmable:
        return _fail_if_over_budget(protected, max_tokens, engine)

    # Work from the oldest end (index 0 of trimmable) toward the newest
    idx = 0
    while idx < len(trimmable) and _total_tokens(protected + trimmable, engine) > max_tokens:
        removed = _pop_tool_call_pair(trimmable, idx)
        if removed is None:
            # No pair — just delete the single message
            del trimmable[idx]
        # When a pair is removed we stay at the same idx because the next
        # message shifted into this position.

    result = protected + trimmable
    return _fail_if_over_budget(result, max_tokens, engine)


# ---------------------------------------------------------------------------
# [INTERNAL] Helpers
# ---------------------------------------------------------------------------


def _total_tokens(messages: list[BaseMessage], engine: TokenCounter) -> int:
    """Sum the token count of every message's content."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else ""
        total += engine.get_num_tokens(content)
    return total


def _pop_tool_call_pair(
    messages: list[BaseMessage], idx: int
) -> tuple[BaseMessage, BaseMessage] | None:
    """If the message at *idx* is part of a tool-call pair with its neighbour,
    remove **both** and return them.  Otherwise return ``None``.

    Two cases are recognised:

    1. ``AIMessage(tool_calls=…)`` at *idx* followed by a ``ToolMessage``
       with a matching ``tool_call_id`` at *idx+1*.
    2. ``ToolMessage`` at *idx* preceded by an ``AIMessage(tool_calls=…)``
       at *idx-1* with a matching ``tool_call_id``.
    """
    if idx >= len(messages):
        return None

    msg = messages[idx]

    # Case 1 — AIMessage at idx, ToolMessage at idx+1
    if isinstance(msg, AIMessage) and msg.tool_calls:
        if idx + 1 < len(messages):
            next_msg = messages[idx + 1]
            if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id in {
                tc["id"] for tc in msg.tool_calls
            }:
                # Remove both — order matters: higher index first
                del messages[idx + 1]
                del messages[idx]
                return (msg, next_msg)

    # Case 2 — ToolMessage at idx, AIMessage at idx-1
    if isinstance(msg, ToolMessage) and idx > 0:
        prev_msg = messages[idx - 1]
        if (
            isinstance(prev_msg, AIMessage)
            and prev_msg.tool_calls
            and msg.tool_call_id in {tc["id"] for tc in prev_msg.tool_calls}
        ):
            del messages[idx]
            del messages[idx - 1]
            return (prev_msg, msg)

    return None


def _fail_if_over_budget(
    messages: list[BaseMessage], max_tokens: int, engine: TokenCounter
) -> list[BaseMessage]:
    """Raise ``ContextBudgetExceededError`` if *messages* still exceed the budget."""
    total = _total_tokens(messages, engine)
    if total > max_tokens:
        raise ContextBudgetExceededError(
            f"Context budget exceeded: {total} tokens needed but only "
            f"{max_tokens} available, even after maximum trimming. "
            f"{len(messages)} message(s) remain."
        )
    return messages
