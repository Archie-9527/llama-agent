"""Thread-local event capture for the synchronous Agent runtime.

The core graph remains synchronous.  A TUI runs it in one worker thread and
installs an event sink for that thread.  Graph nodes, model calls and tools can
then publish small UI events without depending on Textual or changing
AgentState/checkpoint schemas.
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
    """Publish an event when an interactive capture scope is active.

    UI diagnostics must never make an otherwise valid Agent task fail, so a
    subscriber exception is intentionally swallowed.
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
