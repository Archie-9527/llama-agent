"""测试 knowledge_scope.py。

覆盖范围：
  - 超出预算时移除最早的消息。
  - 受保护前缀始终保留。
  - 工具调用消息对（AIMessage+ToolMessage）会一起移除。
  - 预算无法满足时抛出 ContextBudgetExceededError。
  - 未超出预算时走快速路径，不做修改。
  - TokenCounter 协议兼容性。
"""

from __future__ import annotations

import os
import sys

import pytest

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall

from agent_core.knowledge_scope import (
    TokenCounter,
    truncate_history,
)
from agent_core.exceptions import ContextBudgetExceededError


# ── 轻量级桩 Token 计数器 ───────────────────────────────────────────────────


class _CharTokenCounter:
    """将每个字符视为一个 Token 的简单计数器。

    这种计算方式并不符合真实模型，但具有确定性，且适合快速单元测试。
    """

    def get_num_tokens(self, text: str) -> int:
        return len(text)


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


def _sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def _human(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def _ai(content: str = "", tool_calls: list | None = None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [])


def _tool(content: str, tool_call_id: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id)


def _token_count(msgs: list[BaseMessage], engine: TokenCounter) -> int:
    return sum(engine.get_num_tokens(m.content if isinstance(m.content, str) else "") for m in msgs)


# ============================================================================


class TestTokenCounterProtocol:
    """验证测试桩满足协议。"""

    def test_char_counter_satisfies_protocol(self):
        tc = _CharTokenCounter()
        assert isinstance(tc, TokenCounter)
        assert tc.get_num_tokens("hello") == 5


class TestTruncateHistory:
    """测试核心裁剪逻辑。"""

    def test_within_budget_returns_unchanged(self):
        engine = _CharTokenCounter()
        msgs = [_sys("You are helpful."), _human("Hi")]
        result = truncate_history(msgs, max_tokens=100, engine=engine)
        assert len(result) == 2
        assert result[0].content == "You are helpful."
        assert result[1].content == "Hi"

    def test_over_budget_trims_oldest(self):
        engine = _CharTokenCounter()
        msgs = [
            _sys("sys"),
            _human("old message"),
            _human("recent message"),
        ]
        # 预算为 sys（3）+ recent（14）= 17；加入旧消息后会超出预算。
        result = truncate_history(msgs, max_tokens=18, engine=engine)
        assert len(result) >= 2
        assert result[0].content == "sys"  # 受保护

    def test_protected_prefix_never_trimmed(self):
        engine = _CharTokenCounter()
        msgs = [
            _sys("system prompt here"),
            _human("x" * 100),  # 内容很长，应被裁剪
        ]
        result = truncate_history(
            msgs, max_tokens=len("system prompt here") + 10, engine=engine,
            protected_prefix=1,
        )
        assert result[0].content == "system prompt here"
        assert len(result) == 1  # 长用户消息已被移除

    def test_raises_when_budget_impossible(self):
        engine = _CharTokenCounter()
        msgs = [_sys("a" * 100)]  # 单独一条消息就已超出预算
        with pytest.raises(ContextBudgetExceededError):
            truncate_history(msgs, max_tokens=10, engine=engine, protected_prefix=1)

    def test_negative_budget_raises_value_error(self):
        engine = _CharTokenCounter()
        with pytest.raises(ValueError, match="must be"):
            truncate_history([], max_tokens=-1, engine=engine)

    def test_empty_input(self):
        engine = _CharTokenCounter()
        assert truncate_history([], max_tokens=10, engine=engine) == []


class TestToolCallPairDeletion:
    """AIMessage(tool_calls) 与 ToolMessage 消息对始终一起删除。"""

    def test_pair_deleted_when_aimessage_at_idx(self):
        """删除位置为 AIMessage 时，也移除后续 tool_call_id 匹配的 ToolMessage。"""
        engine = _CharTokenCounter()
        call_id = "call_abc"
        msgs = [
            _sys("sys"),
            _ai(tool_calls=[ToolCall(name="t", args={}, id=call_id)]),
            _tool("result", tool_call_id=call_id),
            _human("next step"),
        ]
        before_total = _token_count(msgs, engine)
        # 预算仅够容纳 sys 与 next step，因此必须裁剪消息对。
        budget = _token_count([msgs[0], msgs[3]], engine)
        result = truncate_history(msgs, max_tokens=budget, engine=engine)
        # 消息对应完全移除，不应遗留孤立的 ToolMessage。
        for msg in result:
            assert not isinstance(msg, ToolMessage)

    def test_pair_deleted_when_toolmessage_at_idx(self):
        """删除位置为 ToolMessage 时，也应移除前面的 AIMessage。"""
        engine = _CharTokenCounter()
        call_id = "call_xyz"
        msgs = [
            _sys("sys"),
            _human("do something"),
            _ai(tool_calls=[ToolCall(name="t", args={}, id=call_id)]),
            _tool("result", tool_call_id=call_id),
            _human("final"),
        ]
        # 预算很紧，应裁剪最早的用户消息和工具消息对。
        budget = 20
        result = truncate_history(msgs, max_tokens=budget, engine=engine)
        # 不应遗留孤立的 ToolMessage。
        for i, msg in enumerate(result):
            if isinstance(msg, ToolMessage):
                assert i > 0 and isinstance(result[i - 1], AIMessage)

    def test_unmatched_tool_message_trimmed_solo(self):
        """若对应的 AIMessage 已移除，则单独裁剪该 ToolMessage。"""
        engine = _CharTokenCounter()
        msgs = [
            _sys("sys"),
            _tool("orphan", tool_call_id="no_match"),
            _human("keep me"),
        ]
        budget = _token_count([msgs[0], msgs[2]], engine)
        result = truncate_history(msgs, max_tokens=budget, engine=engine)
        for msg in result:
            assert not isinstance(msg, ToolMessage)
