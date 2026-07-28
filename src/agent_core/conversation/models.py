"""会话持久化模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Turn:
    turn_id: str
    conversation_id: str
    thread_id: str
    turn_index: int
    user_input: str
    assistant_output: str
    status: str
    created_at: str
    updated_at: str
    error: str | None = None
