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
# [稳定接口] Token 计数的最小接口
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenCounter(Protocol):
    """任何能够统计字符串 Token 数的对象所需实现的 Protocol。

    ``ChatLlamaCpp`` 通过 ``get_num_tokens()`` 满足该接口，单元测试也可传入
    轻量 Stub。
    """

    def get_num_tokens(self, text: str) -> int:
        ...


# ---------------------------------------------------------------------------
# [稳定接口] 公共 API
# ---------------------------------------------------------------------------


def truncate_history(
    messages: list[BaseMessage],
    max_tokens: int,
    engine: TokenCounter,
    protected_prefix: int = 1,
) -> list[BaseMessage]:
    """裁剪 *messages*，使其不超过 *max_tokens* 个 Token。

    从 *protected_prefix* 之后最旧的消息开始裁剪，优先保留最近的会话轮次。
    删除 ``AIMessage(tool_calls=…)`` 时，也会删除紧随其后且
    ``tool_call_id`` 匹配的 ``ToolMessage``，反之亦然；工具调用对始终一起删除。

    参数：
        messages：需要裁剪的完整消息列表。
        max_tokens：硬性 Token 预算，必须大于等于 0。
        engine：满足 ``TokenCounter`` 的任意对象。
        protected_prefix：永远不会被裁剪的前导消息数量，默认为 1，用于保护
            ``SystemMessage``。

    返回：
        总 Token 数不超过 *max_tokens* 的消息列表，长度可能缩短。

    异常：
        ContextBudgetExceededError：除受保护前缀外的内容全部删除后仍无法满足预算
            时抛出，说明剩余消息本身过大。
        ValueError：*max_tokens* 为负数时抛出。
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be ≥ 0")

    if not messages:
        return []

    # 快速路径——当前内容已在预算内
    if _total_tokens(messages, engine) <= max_tokens:
        return list(messages)

    # 拆分为受保护头部和可裁剪尾部
    protected = list(messages[:protected_prefix])
    trimmable = list(messages[protected_prefix:])

    if not trimmable:
        return _fail_if_over_budget(protected, max_tokens, engine)

    # 从最旧端（trimmable 的索引 0）向最新端处理
    idx = 0
    while idx < len(trimmable) and _total_tokens(protected + trimmable, engine) > max_tokens:
        removed = _pop_tool_call_pair(trimmable, idx)
        if removed is None:
            # 不属于配对，只删除当前单条消息
            del trimmable[idx]
        # 删除配对后保持同一 idx，因为下一条消息会移动到当前位置。

    result = protected + trimmable
    return _fail_if_over_budget(result, max_tokens, engine)


# ---------------------------------------------------------------------------
# [内部实现] 辅助函数
# ---------------------------------------------------------------------------


def _total_tokens(messages: list[BaseMessage], engine: TokenCounter) -> int:
    """汇总所有消息内容的 Token 数。"""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else ""
        total += engine.get_num_tokens(content)
    return total


def _pop_tool_call_pair(
    messages: list[BaseMessage], idx: int
) -> tuple[BaseMessage, BaseMessage] | None:
    """如果 *idx* 处消息与相邻消息构成工具调用对，则**同时**删除并返回二者；
    否则返回 ``None``。

    可识别两种情况：

    1. *idx* 处为 ``AIMessage(tool_calls=…)``，其后 *idx+1* 处为具有匹配
       ``tool_call_id`` 的 ``ToolMessage``。
    2. *idx* 处为 ``ToolMessage``，其前 *idx-1* 处为具有匹配
       ``tool_call_id`` 的 ``AIMessage(tool_calls=…)``。
    """
    if idx >= len(messages):
        return None

    msg = messages[idx]

    # 情况 1——idx 处是 AIMessage，idx+1 处是 ToolMessage
    if isinstance(msg, AIMessage) and msg.tool_calls:
        if idx + 1 < len(messages):
            next_msg = messages[idx + 1]
            if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id in {
                tc["id"] for tc in msg.tool_calls
            }:
            # 同时删除二者——顺序很重要，应先删除较大索引
                del messages[idx + 1]
                del messages[idx]
                return (msg, next_msg)

    # 情况 2——idx 处是 ToolMessage，idx-1 处是 AIMessage
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
    """*messages* 仍超过预算时抛出 ``ContextBudgetExceededError``。"""
    total = _total_tokens(messages, engine)
    if total > max_tokens:
        raise ContextBudgetExceededError(
            f"Context budget exceeded: {total} tokens needed but only "
            f"{max_tokens} available, even after maximum trimming. "
            f"{len(messages)} message(s) remain."
        )
    return messages
