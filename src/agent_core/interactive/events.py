"""同步 Agent 运行时的线程局部事件捕获。

核心图保持同步运行。TUI 在一个工作线程中执行核心图，并为该线程安装事件接收
器。图节点、模型调用和工具随后可以发布轻量 UI 事件，而无需依赖 Textual，
也无需修改 AgentState 或 Checkpoint Schema。
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class InteractiveEvent:
    kind: str
    timestamp_utc: str
    data: dict[str, Any] = field(default_factory=dict)


EventSink = Callable[[InteractiveEvent], None]

_event_sink: contextvars.ContextVar[EventSink | None] = contextvars.ContextVar(
    "interactive_event_sink", default=None
)


def interactive_events_enabled() -> bool:
    return _event_sink.get() is not None


def emit_interactive_event(kind: str, **data: Any) -> None:
    """当交互式捕获作用域处于活动状态时发布事件。

    UI 诊断不能导致原本有效的 Agent 任务失败，因此会有意吞掉订阅者异常。
    """

    sink = _event_sink.get()
    if sink is None:
        return
    event = InteractiveEvent(
        kind=kind,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        data=data,
    )
    try:
        sink(event)
    except Exception:
        return


@contextlib.contextmanager
def capture_interactive_events(sink: EventSink) -> Iterator[None]:
    token = _event_sink.set(sink)
    try:
        yield
    finally:
        _event_sink.reset(token)
