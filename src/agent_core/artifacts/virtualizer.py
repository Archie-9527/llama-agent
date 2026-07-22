"""R1 tool-output virtualization at the central capability boundary.

Large raw results are persisted in :class:`ArtifactStore`; the model and
LangGraph state receive only a compact descriptor, deterministic previews and
an opaque artifact id.  No LLM call is used to produce the descriptor, so the
optimization itself does not add inference latency or introduce hallucinated
facts.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from typing import Any

from agent_core.config import MemoryConfig

_ARTIFACT_ACCESS_TOOLS = frozenset(
    {"get_artifact_summary", "retrieve_artifact", "search_artifact"}
)
_LARGE_VALUE_KEYS = frozenset(
    {"content", "stdout", "stderr", "rows", "matches", "lines", "data"}
)
_lock = threading.Lock()
_config = MemoryConfig()


def initialize_artifact_virtualizer(config: MemoryConfig) -> None:
    """Install the process-local R1 policy after tool bootstrap."""
    config.validate()
    global _config
    with _lock:
        _config = config


def maybe_virtualize_tool_result(tool_name: str, result: object) -> object:
    """Return *result* unchanged unless the enabled R1 threshold is exceeded."""
    config = _config
    if not config.artifact_virtualization or tool_name in _ARTIFACT_ACCESS_TOOLS:
        return result

    text, content_type, structured = _serialize_result(result)
    original_bytes = len(text.encode("utf-8"))
    if original_bytes <= config.artifact_inline_max_bytes:
        return result

    from agent_core.artifacts.runtime import get_artifact_store
    from agent_core.telemetry import current_task_id, get_telemetry

    summary = _summarize(
        tool_name,
        text,
        structured,
        max_chars=config.artifact_summary_chars,
    )
    metadata = get_artifact_store().put(
        text,
        owner_id=current_task_id() or "__unscoped__",
        tool_name=tool_name,
        summary=summary,
        content_type=content_type,
    )
    head_preview, tail_preview = _previews(text, config.artifact_preview_chars)
    descriptor = {
        "success": _success_value(structured),
        "virtualized": True,
        "artifact_id": metadata.artifact_id,
        "tool_name": tool_name,
        "original_bytes": metadata.original_bytes,
        "content_hash": metadata.content_hash,
        "summary": summary,
        "head_preview": head_preview,
        "tail_preview": tail_preview,
        "available_operations": [
            "get_artifact_summary",
            "retrieve_artifact",
            "search_artifact",
        ],
    }
    inline_result = json.dumps(descriptor, ensure_ascii=False)
    inline_bytes = len(inline_result.encode("utf-8"))
    get_telemetry().record_event(
        "tool_events.jsonl",
        "artifact_virtualized",
        tool_name=tool_name,
        artifact_id=metadata.artifact_id,
        original_bytes=metadata.original_bytes,
        inline_bytes=inline_bytes,
        bytes_saved=max(0, metadata.original_bytes - inline_bytes),
        threshold_bytes=config.artifact_inline_max_bytes,
    )
    return inline_result


def _serialize_result(result: object) -> tuple[str, str, Any | None]:
    if isinstance(result, bytes):
        return (
            result.decode("utf-8", errors="replace"),
            "application/octet-stream",
            None,
        )
    if isinstance(result, str):
        try:
            structured = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            structured = None
        return (
            result,
            "application/json" if structured is not None else "text/plain",
            structured,
        )
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, default=str), "application/json", result
    return str(result), "text/plain", None


def _summarize(
    tool_name: str,
    text: str,
    structured: Any | None,
    *,
    max_chars: int,
) -> str:
    facts: dict[str, Any] = {}
    if isinstance(structured, dict):
        for key, value in structured.items():
            if key in _LARGE_VALUE_KEYS:
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                rendered = str(value)
                facts[key] = (
                    value if len(rendered) <= 160 else rendered[:157] + "..."
                )
    prefix = (
        f"{tool_name} returned {len(text.encode('utf-8'))} bytes "
        f"across {len(text.splitlines())} lines."
    )
    if facts:
        prefix += " Metadata: " + json.dumps(facts, ensure_ascii=False, default=str)
    return prefix[:max_chars]


def _previews(text: str, max_chars: int) -> tuple[str, str]:
    if max_chars <= 0:
        return "", ""
    if len(text) <= max_chars:
        return text, ""
    head_chars = (max_chars + 1) // 2
    tail_chars = max_chars // 2
    return text[:head_chars], text[-tail_chars:] if tail_chars else ""


def _success_value(structured: Any | None) -> bool:
    if isinstance(structured, dict) and isinstance(structured.get("success"), bool):
        return structured["success"]
    return True


def _reset_artifact_virtualizer_for_testing() -> None:
    initialize_artifact_virtualizer(MemoryConfig())


def artifact_virtualizer_config() -> dict[str, Any]:
    """Expose an immutable diagnostic snapshot for tests and diagnostics."""
    return asdict(_config)
