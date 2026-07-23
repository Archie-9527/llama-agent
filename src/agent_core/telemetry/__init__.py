"""Non-invasive telemetry used by the agent and benchmark runner."""

from agent_core.telemetry.collector import (
    current_phase,
    current_task_id,
    TelemetryCollector,
    get_telemetry,
    initialize_telemetry,
    telemetry_phase,
    telemetry_task,
)
from agent_core.telemetry.models import TelemetryConfig

__all__ = [
    "TelemetryCollector",
    "TelemetryConfig",
    "get_telemetry",
    "current_task_id",
    "current_phase",
    "initialize_telemetry",
    "telemetry_phase",
    "telemetry_task",
]
