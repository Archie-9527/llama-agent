"""Runtime telemetry collector and context propagation."""

from __future__ import annotations

import contextlib
import contextvars
import threading
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Any, Iterator

from agent_core.telemetry.kv_monitor import sample_kv
from agent_core.telemetry.accelerator_monitor import sample_accelerator
from agent_core.telemetry.models import TelemetryConfig
from agent_core.telemetry.process_monitor import sample_process
from agent_core.telemetry.sink import TelemetrySink

_task_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "telemetry_task_id", default=None
)
_phase_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "telemetry_phase", default="unknown"
)


class TelemetryCollector:
    def __init__(self, config: TelemetryConfig) -> None:
        config.validate()
        self.config = config
        self.sink = TelemetrySink(config.output_dir)
        self._stop_event = threading.Event()
        self._sampler: threading.Thread | None = None
        self._active_task_id: str | None = None
        self._active_phase: str = "unknown"
        self._context_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self) -> None:
        if not self.enabled or not self.config.collect_process or self._sampler:
            return
        self._stop_event.clear()
        self._sampler = threading.Thread(
            target=self._sampling_loop,
            name="agent-telemetry",
            daemon=True,
        )
        self._sampler.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._sampler is not None:
            self._sampler.join(timeout=max(1.0, self.config.sample_interval_ms / 500))
            self._sampler = None

    def _sampling_loop(self) -> None:
        interval = self.config.sample_interval_ms / 1000
        while not self._stop_event.wait(interval):
            self.record_process()

    def _context(self) -> dict[str, Any]:
        task_id = _task_id_var.get()
        phase = _phase_var.get()
        if task_id is None or phase == "unknown":
            with self._context_lock:
                task_id = task_id or self._active_task_id
                phase = (
                    self._active_phase if phase == "unknown" else phase
                )
        return {
            "task_id": task_id,
            "phase": phase,
        }

    def set_active_context(
        self, *, task_id: str | None = None, phase: str | None = None
    ) -> tuple[str | None, str]:
        with self._context_lock:
            previous = (self._active_task_id, self._active_phase)
            if task_id is not None:
                self._active_task_id = task_id
            if phase is not None:
                self._active_phase = phase
            return previous

    def restore_active_context(self, value: tuple[str | None, str]) -> None:
        with self._context_lock:
            self._active_task_id, self._active_phase = value

    def record_process(self) -> None:
        if not self.enabled or not self.config.collect_process:
            return
        accelerator = (
            sample_accelerator().to_dict()
            if self.config.collect_accelerator
            else {}
        )
        self.sink.append_csv(
            "system_memory.csv",
            {
                **sample_process().to_dict(),
                **accelerator,
                **self._context(),
            },
        )

    def record_kv(self, client: Any, event: str) -> None:
        if not self.enabled or not self.config.collect_kv:
            return
        self.sink.append_csv(
            "kv_metrics.csv",
            {
                **sample_kv(
                    client, collect_state_size=self.config.collect_state_size
                ).to_dict(),
                **self._context(),
                "event": event,
            },
        )

    def record_event(self, filename: str, event: str, **data: Any) -> None:
        if not self.enabled:
            return
        self.sink.append_jsonl(
            filename,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_ns": monotonic_ns(),
                **self._context(),
                "event": event,
                **data,
            },
        )


_collector = TelemetryCollector(TelemetryConfig())


def initialize_telemetry(config: TelemetryConfig) -> TelemetryCollector:
    global _collector
    _collector.close()
    _collector = TelemetryCollector(config)
    _collector.start()
    return _collector


def get_telemetry() -> TelemetryCollector:
    return _collector


def current_task_id() -> str | None:
    """Return the active task identity for task-scoped stores and tools."""
    task_id = _task_id_var.get()
    if task_id is not None:
        return task_id
    with _collector._context_lock:
        return _collector._active_task_id


@contextlib.contextmanager
def telemetry_task(task_id: str) -> Iterator[None]:
    collector = get_telemetry()
    previous = collector.set_active_context(task_id=task_id)
    token = _task_id_var.set(task_id)
    try:
        yield
    finally:
        _task_id_var.reset(token)
        collector.restore_active_context(previous)


@contextlib.contextmanager
def telemetry_phase(phase: str) -> Iterator[None]:
    collector = get_telemetry()
    previous = collector.set_active_context(phase=phase)
    token = _phase_var.set(phase)
    try:
        yield
    finally:
        _phase_var.reset(token)
        collector.restore_active_context(previous)
