"""供终端 UI 使用的同步会话服务。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent_core.conversation.manager import ConversationManager
from agent_core.conversation.models import Conversation, Turn
from agent_core.conversation.store import ConversationStore
from agent_core.interactive.events import (
    InteractiveEvent,
    capture_interactive_events,
)


@dataclass(frozen=True)
class ReasoningRecord:
    phase: str
    content: str
    input_tokens: int
    output_tokens: int
    duration_ms: float | None


@dataclass(frozen=True)
class TurnOutcome:
    conversation_id: str
    turn: Turn
    result: dict
    reasoning: tuple[ReasoningRecord, ...]


class InteractiveSession:
    """管理一个用户可见的会话选择。

    底层 ``TaskRunner`` 和 ``ConversationStore`` 由 CLI 提供，并各自负责自身
    生命周期。
    """

    def __init__(
        self,
        manager: ConversationManager,
        store: ConversationStore,
        *,
        conversation_id: str | None,
        last_conversation_file: Path,
    ) -> None:
        self.manager = manager
        self.store = store
        self.conversation_id = conversation_id
        self.last_conversation_file = last_conversation_file
        self._turn_lock = threading.Lock()

    def run_turn(
        self,
        text: str,
        *,
        on_event: Callable[[InteractiveEvent], None] | None = None,
    ) -> TurnOutcome:
        if not text.strip():
            raise ValueError("input must not be empty")

        reasoning: list[ReasoningRecord] = []

        def sink(event: InteractiveEvent) -> None:
            if event.kind == "inference_completed":
                content = str(event.data.get("reasoning_content") or "").strip()
                if content:
                    reasoning.append(
                        ReasoningRecord(
                            phase=str(event.data.get("phase") or "model"),
                            content=content,
                            input_tokens=int(event.data.get("input_tokens") or 0),
                            output_tokens=int(event.data.get("output_tokens") or 0),
                            duration_ms=(
                                float(event.data["duration_ms"])
                                if event.data.get("duration_ms") is not None
                                else None
                            ),
                        )
                    )
            if on_event is not None:
                on_event(event)

        with self._turn_lock, capture_interactive_events(sink):
            if self.conversation_id is None:
                conversation_id, turn, result = self.manager.start(text)
                self.conversation_id = conversation_id
                self._remember_conversation(conversation_id)
            else:
                turn, result = self.manager.continue_conversation(
                    self.conversation_id, text
                )
                conversation_id = self.conversation_id

        return TurnOutcome(
            conversation_id=conversation_id,
            turn=turn,
            result=result,
            reasoning=tuple(reasoning),
        )

    def new_conversation(self) -> None:
        if self._turn_lock.locked():
            raise RuntimeError("cannot change conversation while a turn is running")
        self.conversation_id = None
        # 如果用户尚未发送新会话的第一条消息就退出，不要恢复之前的会话。
        self.last_conversation_file.unlink(missing_ok=True)

    def resume_conversation(self, conversation_id: str) -> None:
        if self._turn_lock.locked():
            raise RuntimeError("cannot change conversation while a turn is running")
        if self.store.get_conversation(conversation_id) is None:
            raise ValueError(f"conversation does not exist: {conversation_id}")
        self.conversation_id = conversation_id
        self._remember_conversation(conversation_id)

    def history(self) -> list[Turn]:
        if self.conversation_id is None:
            return []
        return self.store.list_turns(self.conversation_id)

    def conversations(self) -> list[Conversation]:
        return self.store.list_conversations()

    def _remember_conversation(self, conversation_id: str) -> None:
        self.last_conversation_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_conversation_file.write_text(
            conversation_id, encoding="utf-8"
        )
