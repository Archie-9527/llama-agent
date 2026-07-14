"""Tests for knowledge_scope.py.

Covers:
  - Truncation when over budget removes oldest messages.
  - Protected prefix is never touched.
  - Tool-call pairs (AIMessage+ToolMessage) are removed together.
  - ContextBudgetExceededError when budget impossible.
  - Fast path: within budget → no-op.
  - TokenCounter protocol compatibility.
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


# ── Lightweight stub token counter ───────────────────────────────────────────


class _CharTokenCounter:
    """Simple token counter that treats each character as one 'token'.
    This is *not* realistic but is deterministic and fast for unit tests."""

    def get_num_tokens(self, text: str) -> int:
        return len(text)


# ── Helpers ──────────────────────────────────────────────────────────────────


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
    """Verify our stub satisfies the protocol."""

    def test_char_counter_satisfies_protocol(self):
        tc = _CharTokenCounter()
        assert isinstance(tc, TokenCounter)
        assert tc.get_num_tokens("hello") == 5


class TestTruncateHistory:
    """Core trimming logic."""

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
        # Budget = sys (3) + recent (14) = 17; old would push over
        result = truncate_history(msgs, max_tokens=18, engine=engine)
        assert len(result) >= 2
        assert result[0].content == "sys"  # protected

    def test_protected_prefix_never_trimmed(self):
        engine = _CharTokenCounter()
        msgs = [
            _sys("system prompt here"),
            _human("x" * 100),  # very long, should be trimmed
        ]
        result = truncate_history(
            msgs, max_tokens=len("system prompt here") + 10, engine=engine,
            protected_prefix=1,
        )
        assert result[0].content == "system prompt here"
        assert len(result) == 1  # the long human message was removed

    def test_raises_when_budget_impossible(self):
        engine = _CharTokenCounter()
        msgs = [_sys("a" * 100)]  # alone exceeds budget
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
    """AIMessage(tool_calls) + ToolMessage pairs are always deleted together."""

    def test_pair_deleted_when_aimessage_at_idx(self):
        """When the AIMessage is at the deletion index, the following
        ToolMessage with matching tool_call_id is also removed."""
        engine = _CharTokenCounter()
        call_id = "call_abc"
        msgs = [
            _sys("sys"),
            _ai(tool_calls=[ToolCall(name="t", args={}, id=call_id)]),
            _tool("result", tool_call_id=call_id),
            _human("next step"),
        ]
        before_total = _token_count(msgs, engine)
        # Budget just enough for sys + next step → must trim the pair
        budget = _token_count([msgs[0], msgs[3]], engine)
        result = truncate_history(msgs, max_tokens=budget, engine=engine)
        # The pair should be gone entirely — no orphan ToolMessage
        for msg in result:
            assert not isinstance(msg, ToolMessage)

    def test_pair_deleted_when_toolmessage_at_idx(self):
        """When the deletion falls on the ToolMessage, the preceding
        AIMessage should be removed as well."""
        engine = _CharTokenCounter()
        call_id = "call_xyz"
        msgs = [
            _sys("sys"),
            _human("do something"),
            _ai(tool_calls=[ToolCall(name="t", args={}, id=call_id)]),
            _tool("result", tool_call_id=call_id),
            _human("final"),
        ]
        # Tight budget — should trim the oldest human and the tool pair
        budget = 20
        result = truncate_history(msgs, max_tokens=budget, engine=engine)
        # No orphan ToolMessage
        for i, msg in enumerate(result):
            if isinstance(msg, ToolMessage):
                assert i > 0 and isinstance(result[i - 1], AIMessage)

    def test_unmatched_tool_message_trimmed_solo(self):
        """A ToolMessage whose AIMessage was already removed is trimmed alone."""
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
