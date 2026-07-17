"""Typed telemetry configuration and snapshot models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = False
    output_dir: Path = Path("benchmark/results/runtime")
    sample_interval_ms: int = 200
    collect_process: bool = True
    collect_kv: bool = True
    collect_accelerator: bool = True
    collect_state_size: bool = False

    def validate(self) -> None:
        if self.sample_interval_ms < 10:
            raise ValueError("telemetry.sample_interval_ms must be >= 10")


@dataclass(frozen=True)
class ProcessSnapshot:
    timestamp_utc: str
    monotonic_ns: int
    rss_bytes: int | None
    vms_bytes: int | None
    uss_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KvSnapshot:
    timestamp_utc: str
    monotonic_ns: int
    logical_tokens: int | None
    capacity_tokens: int | None
    position_min: int | None
    position_max: int | None
    position_span_tokens: int | None
    state_size_bytes: int | None
    supported: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceleratorSnapshot:
    backend: str | None
    device_index: int | None
    process_used_bytes: int | None
    device_used_bytes: int | None
    device_total_bytes: int | None
    supported: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
