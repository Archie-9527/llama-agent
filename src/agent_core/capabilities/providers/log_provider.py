"""Structured analysis tools for deterministic local log files."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
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

_LEVEL = re.compile(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)\b")
_FIELD = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


@dataclass(frozen=True)
class LogToolConfig:
    allowed_roots: list[str] = field(default_factory=lambda: ["."])
    max_matches: int = 50
    max_window_lines: int = 40


def _error(exc: Exception) -> str:
    return json.dumps(
        {"success": False, "error": f"{type(exc).__name__}: {exc}"},
        ensure_ascii=False,
    )


@register_provider
class LogCapabilityProvider(CapabilityProvider):
    category = "log"

    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        try:
            config = LogToolConfig(**raw_config)
            roots = configured_roots(config.allowed_roots)
            if config.max_matches < 1 or config.max_window_lines < 1:
                raise ValueError("limits must be positive")
        except (TypeError, ValueError) as exc:
            raise ToolProviderConfigError(f"log provider config invalid: {exc}") from exc

        def _lines(log_path: str) -> tuple[str, list[str]]:
            path = resolve_allowed_path(log_path, roots)
            if not path.is_file():
                raise ValueError("log_path is not a file")
            return log_path, path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()

        def _aggregate(log_path: str) -> str:
            try:
                path, lines = _lines(log_path)
                levels: Counter[str] = Counter()
                codes: Counter[str] = Counter()
                nodes: Counter[str] = Counter()
                for line in lines:
                    level_match = _LEVEL.search(line)
                    if level_match:
                        levels[level_match.group(1).replace("WARNING", "WARN")] += 1
                    fields = dict(_FIELD.findall(line))
                    if fields.get("code"):
                        codes[fields["code"]] += 1
                    if fields.get("node"):
                        nodes[fields["node"]] += 1
                return json.dumps(
                    {
                        "success": True,
                        "path": path,
                        "line_count": len(lines),
                        "levels": dict(levels),
                        "codes": dict(codes),
                        "nodes": dict(nodes),
                    },
                    ensure_ascii=False,
                )
            except (OSError, ValueError) as exc:
                return _error(exc)

        def _search(
            log_path: str,
            query: str,
            max_matches: int | None = 20,
        ) -> str:
            try:
                if not query:
                    raise ValueError("query must not be empty")
                path, lines = _lines(log_path)
                requested_matches = 20 if max_matches is None else max_matches
                limit = max(1, min(requested_matches, config.max_matches))
                matches = [
                    {"line_number": number, "line": line}
                    for number, line in enumerate(lines, start=1)
                    if query.casefold() in line.casefold()
                ][:limit]
                return json.dumps(
                    {
                        "success": True,
                        "path": path,
                        "query": query,
                        "match_count": len(matches),
                        "matches": matches,
                    },
                    ensure_ascii=False,
                )
            except (OSError, ValueError) as exc:
                return _error(exc)

        def _window(
            log_path: str,
            line_number: int,
            before: int | None = 2,
            after: int | None = 2,
        ) -> str:
            try:
                path, lines = _lines(log_path)
                if line_number < 1 or line_number > len(lines):
                    raise ValueError("line_number is outside the log")
                before = 2 if before is None else before
                after = 2 if after is None else after
                before = max(0, before)
                after = max(0, after)
                if before + after + 1 > config.max_window_lines:
                    raise ValueError("requested window exceeds configured limit")
                start = max(0, line_number - 1 - before)
                end = min(len(lines), line_number + after)
                return json.dumps(
                    {
                        "success": True,
                        "path": path,
                        "target_line": line_number,
                        "lines": [
                            {"line_number": index + 1, "line": lines[index]}
                            for index in range(start, end)
                        ],
                    },
                    ensure_ascii=False,
                )
            except (OSError, ValueError) as exc:
                return _error(exc)

        return [
            Capability(
                name="aggregate_log_errors",
                description=(
                    "Summarize a local log by severity, error code and node. "
                    "Use this before detailed searches."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"log_path": {"type": "string"}},
                    "required": ["log_path"],
                },
                handler=_aggregate,
            ),
            Capability(
                name="search_log",
                description="Search a local log and return matching line numbers.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "log_path": {"type": "string"},
                        "query": {"type": "string"},
                        "max_matches": {"type": "integer"},
                    },
                    "required": ["log_path", "query"],
                },
                handler=_search,
            ),
            Capability(
                name="get_log_window",
                description="Read lines before and after a known log line number.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "log_path": {"type": "string"},
                        "line_number": {"type": "integer"},
                        "before": {"type": "integer"},
                        "after": {"type": "integer"},
                    },
                    "required": ["log_path", "line_number"],
                },
                handler=_window,
            ),
        ]
