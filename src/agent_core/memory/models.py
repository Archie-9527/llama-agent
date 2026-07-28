"""R2 上下文策略与持久化层共享的带类型记录。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Lifecycle(StrEnum):
    PINNED = "pinned"
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DEAD = "dead"


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    owner_type: str
    owner_id: str
    kind: str
    content: str
    lifecycle: Lifecycle
    token_count: int
    source_type: str
    source_id: str | None = None
    importance: float = 0.5
    artifact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextView:
    conversation_context: str = ""
    pinned_context: str = ""
    summary_context: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    recalled_ids: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        sections: list[str] = []
        if self.pinned_context:
            sections.append("[固定事实与约束]\n" + self.pinned_context)
        if self.summary_context:
            sections.append("[历史执行摘要]\n" + self.summary_context)
        if self.conversation_context:
            sections.append("[相关会话历史]\n" + self.conversation_context)
        return "\n\n".join(sections)
