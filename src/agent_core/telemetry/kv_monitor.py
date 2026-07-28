"""兼容不同版本的 llama-cpp-python KV 状态诊断。

此处返回值会明确区分逻辑 Token 占用、位置范围与序列化状态大小，任何一个
指标都不会被标记为物理 KV 字节数。
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import monotonic_ns
from typing import Any

from agent_core.telemetry.models import KvSnapshot


def sample_kv(client: Any, *, collect_state_size: bool = False) -> KvSnapshot:
    now = datetime.now(timezone.utc).isoformat()
    tick = monotonic_ns()
    logical: int | None = None
    capacity: int | None = None
    pos_min: int | None = None
    pos_max: int | None = None
    state_size: int | None = None
    errors: list[str] = []

    try:
        logical = int(client.n_tokens)
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"n_tokens: {exc}")
    try:
        n_ctx = client.n_ctx
        capacity = int(n_ctx() if callable(n_ctx) else n_ctx)
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"n_ctx: {exc}")

    try:
        from llama_cpp import llama_cpp as low_level

        raw_ctx = client._ctx.ctx
        memory = low_level.llama_get_memory(raw_ctx)
        pos_min = int(low_level.llama_memory_seq_pos_min(memory, 0))
        pos_max = int(low_level.llama_memory_seq_pos_max(memory, 0))
        if collect_state_size and hasattr(low_level, "llama_state_seq_get_size"):
            state_size = int(low_level.llama_state_seq_get_size(raw_ctx, 0))
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"low_level: {exc}")

    span = None
    if pos_min is not None and pos_max is not None:
        span = 0 if pos_max < pos_min or pos_max < 0 else pos_max - pos_min + 1

    return KvSnapshot(
        timestamp_utc=now,
        monotonic_ns=tick,
        logical_tokens=logical,
        capacity_tokens=capacity,
        position_min=pos_min,
        position_max=pos_max,
        position_span_tokens=span,
        state_size_bytes=state_size,
        supported=logical is not None or pos_max is not None,
        error="; ".join(errors) if errors else None,
    )
