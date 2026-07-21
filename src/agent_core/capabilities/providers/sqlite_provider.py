"""Read-only SQLite inspection and query capabilities."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

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

_READ_ONLY_START = re.compile(r"^\s*(SELECT|WITH|PRAGMA)\b", re.IGNORECASE)
_MUTATING = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|VACUUM|ATTACH|DETACH)\b",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class SqliteToolConfig:
    allowed_roots: list[str] = field(default_factory=lambda: ["."])
    max_rows: int = 100


def _error(exc: Exception) -> str:
    return json.dumps(
        {"success": False, "error": f"{type(exc).__name__}: {exc}"},
        ensure_ascii=False,
    )


@register_provider
class SqliteCapabilityProvider(CapabilityProvider):
    category = "sqlite"

    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        try:
            config = SqliteToolConfig(**raw_config)
            roots = configured_roots(config.allowed_roots)
            if config.max_rows < 1:
                raise ValueError("max_rows must be positive")
        except (TypeError, ValueError) as exc:
            raise ToolProviderConfigError(
                f"sqlite provider config invalid: {exc}"
            ) from exc

        def _connect(db_path: str) -> tuple[str, sqlite3.Connection]:
            path = resolve_allowed_path(db_path, roots)
            if not path.is_file():
                raise ValueError("db_path is not a file")
            uri = f"file:{quote(str(path), safe='/')}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            return str(path), connection

        def _describe(db_path: str, table_name: str) -> str:
            try:
                if not _IDENTIFIER.fullmatch(table_name):
                    raise ValueError("invalid table_name")
                path, connection = _connect(db_path)
                try:
                    columns = [
                        dict(row)
                        for row in connection.execute(
                            f'PRAGMA table_info("{table_name}")'
                        ).fetchall()
                    ]
                finally:
                    connection.close()
                if not columns:
                    raise ValueError(f"table not found: {table_name}")
                return json.dumps(
                    {
                        "success": True,
                        "db_path": path,
                        "table": table_name,
                        "columns": columns,
                    },
                    ensure_ascii=False,
                )
            except (sqlite3.Error, OSError, ValueError) as exc:
                return _error(exc)

        def _query(
            db_path: str,
            query: str,
            max_rows: int | None = 50,
        ) -> str:
            try:
                statement = query.strip()
                if (
                    not _READ_ONLY_START.search(statement)
                    or _MUTATING.search(statement)
                    or ";" in statement.rstrip(";")
                ):
                    raise ValueError("only one read-only SELECT/WITH/PRAGMA is allowed")
                requested_rows = 50 if max_rows is None else max_rows
                limit = max(1, min(requested_rows, config.max_rows))
                path, connection = _connect(db_path)
                try:
                    cursor = connection.execute(statement)
                    rows = [dict(row) for row in cursor.fetchmany(limit + 1)]
                    columns = [
                        item[0] for item in cursor.description or ()
                    ]
                finally:
                    connection.close()
                truncated = len(rows) > limit
                rows = rows[:limit]
                return json.dumps(
                    {
                        "success": True,
                        "db_path": path,
                        "columns": columns,
                        "row_count": len(rows),
                        "truncated": truncated,
                        "rows": rows,
                    },
                    ensure_ascii=False,
                )
            except (sqlite3.Error, OSError, ValueError) as exc:
                return _error(exc)

        return [
            Capability(
                name="describe_sqlite_table",
                description="Return column metadata for one table in a local SQLite DB.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "table_name": {"type": "string"},
                    },
                    "required": ["db_path", "table_name"],
                },
                handler=_describe,
            ),
            Capability(
                name="query_sqlite",
                description=(
                    "Execute one read-only SELECT, WITH or PRAGMA query against "
                    "a local SQLite database and return structured rows."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "query": {"type": "string"},
                        "max_rows": {"type": "integer"},
                    },
                    "required": ["db_path", "query"],
                },
                handler=_query,
            ),
        ]
