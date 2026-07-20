"""Deterministic, read-only local file capabilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.capabilities.base import (
    CapabilityProvider,
    ToolProviderConfigError,
    register_provider,
)
from agent_core.capabilities.providers._local_paths import (
    configured_roots,
    resolve_allowed_path,
)
from agent_core.capability_registry import Capability


@dataclass(frozen=True)
class FileToolConfig:
    allowed_roots: list[str] = field(default_factory=lambda: ["."])
    max_read_chars: int = 300_000
    max_matches: int = 20


def _json_error(exc: Exception) -> str:
    return json.dumps(
        {"success": False, "error": f"{type(exc).__name__}: {exc}"},
        ensure_ascii=False,
    )


@register_provider
class FileCapabilityProvider(CapabilityProvider):
    category = "file"

    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        try:
            config = FileToolConfig(**raw_config)
            roots = configured_roots(config.allowed_roots)
            if config.max_read_chars < 1 or config.max_matches < 1:
                raise ValueError("limits must be positive")
        except (TypeError, ValueError) as exc:
            raise ToolProviderConfigError(f"file provider config invalid: {exc}") from exc

        def _metadata(file_path: str) -> str:
            try:
                path = resolve_allowed_path(file_path, roots)
                stat = path.stat()
                return json.dumps(
                    {
                        "success": True,
                        "path": str(path),
                        "is_file": path.is_file(),
                        "is_directory": path.is_dir(),
                        "size_bytes": stat.st_size,
                    },
                    ensure_ascii=False,
                )
            except (OSError, ValueError) as exc:
                return _json_error(exc)

        def _read(file_path: str, offset: int = 0, length: int = 4096) -> str:
            try:
                if offset < 0 or length < 1:
                    raise ValueError("offset must be >= 0 and length must be >= 1")
                path = resolve_allowed_path(file_path, roots)
                if not path.is_file():
                    raise ValueError("path is not a file")
                requested = min(length, config.max_read_chars)
                text = path.read_text(encoding="utf-8", errors="replace")
                chunk = text[offset : offset + requested]
                return json.dumps(
                    {
                        "success": True,
                        "path": str(path),
                        "offset": offset,
                        "length": len(chunk),
                        "total_chars": len(text),
                        "truncated_by_limit": length > config.max_read_chars,
                        "eof": offset + len(chunk) >= len(text),
                        "content": chunk,
                    },
                    ensure_ascii=False,
                )
            except (OSError, ValueError) as exc:
                return _json_error(exc)

        def _search(
            file_path: str,
            query: str,
            max_matches: int = 10,
        ) -> str:
            try:
                if not query:
                    raise ValueError("query must not be empty")
                path = resolve_allowed_path(file_path, roots)
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                limit = max(1, min(max_matches, config.max_matches))
                matches = [
                    {"line_number": number, "line": line}
                    for number, line in enumerate(lines, start=1)
                    if query.casefold() in line.casefold()
                ][:limit]
                return json.dumps(
                    {
                        "success": True,
                        "path": str(path),
                        "query": query,
                        "match_count": len(matches),
                        "matches": matches,
                    },
                    ensure_ascii=False,
                )
            except (OSError, ValueError) as exc:
                return _json_error(exc)

        return [
            Capability(
                name="get_file_metadata",
                description="Return size and type metadata for one allowed local path.",
                input_schema={
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
                handler=_metadata,
            ),
            Capability(
                name="read_file",
                description=(
                    "Read a character range from an allowed UTF-8 local file. "
                    "Use offset and length to avoid loading unnecessary content."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "offset": {"type": "integer"},
                        "length": {"type": "integer"},
                    },
                    "required": ["file_path"],
                },
                handler=_read,
            ),
            Capability(
                name="search_file",
                description="Find text in an allowed local file and return matching lines.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "query": {"type": "string"},
                        "max_matches": {"type": "integer"},
                    },
                    "required": ["file_path", "query"],
                },
                handler=_search,
            ),
        ]
