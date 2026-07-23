"""SQLite persistence for compact task and conversation context."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_core.memory.models import ContextItem, Lifecycle


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContextStore:
    """Persist archived context with strict owner isolation."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS context_entries (
                    entry_id TEXT PRIMARY KEY,
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    lifecycle TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT,
                    importance REAL NOT NULL,
                    artifact_ids TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_context_owner
                    ON context_entries(owner_type, owner_id);
                CREATE INDEX IF NOT EXISTS idx_context_source
                    ON context_entries(owner_type, owner_id, source_type, source_id);
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def put(
        self,
        *,
        owner_type: str,
        owner_id: str,
        kind: str,
        lifecycle: Lifecycle,
        content: str,
        token_count: int,
        source_type: str,
        source_id: str | None = None,
        importance: float = 0.5,
        artifact_ids: tuple[str, ...] = (),
    ) -> ContextItem:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        # A deterministic source key makes lifecycle commits idempotent across
        # retries and checkpoint resume.
        stable = "|".join(
            [owner_type, owner_id, source_type, source_id or "", digest]
        )
        entry_id = "memory://" + uuid.uuid5(uuid.NAMESPACE_URL, stable).hex
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO context_entries (
                    entry_id, owner_type, owner_id, kind, lifecycle, content,
                    content_hash, token_count, source_type, source_id,
                    importance, artifact_ids, created_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(entry_id) DO UPDATE SET
                    lifecycle=excluded.lifecycle,
                    token_count=excluded.token_count,
                    importance=excluded.importance
                """,
                (
                    entry_id,
                    owner_type,
                    owner_id,
                    kind,
                    lifecycle.value,
                    content,
                    digest,
                    token_count,
                    source_type,
                    source_id,
                    importance,
                    "\n".join(artifact_ids),
                    _utc_now(),
                ),
            )
        return ContextItem(
            item_id=entry_id,
            owner_type=owner_type,
            owner_id=owner_id,
            kind=kind,
            content=content,
            lifecycle=lifecycle,
            token_count=token_count,
            source_type=source_type,
            source_id=source_id,
            importance=importance,
            artifact_ids=artifact_ids,
        )

    def list_owner(self, owner_type: str, owner_id: str) -> list[ContextItem]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM context_entries
                WHERE owner_type = ? AND owner_id = ?
                ORDER BY created_at, entry_id
                """,
                (owner_type, owner_id),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get(self, entry_id: str, *, owner_type: str, owner_id: str) -> ContextItem:
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT * FROM context_entries
                WHERE entry_id = ? AND owner_type = ? AND owner_id = ?
                """,
                (entry_id, owner_type, owner_id),
            ).fetchone()
            if row is None:
                raise KeyError("context entry not found or not owned by caller")
            self._connection.execute(
                "UPDATE context_entries SET last_accessed_at = ? WHERE entry_id = ?",
                (_utc_now(), entry_id),
            )
        return self._row_to_item(row)

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> ContextItem:
        return ContextItem(
            item_id=row["entry_id"],
            owner_type=row["owner_type"],
            owner_id=row["owner_id"],
            kind=row["kind"],
            content=row["content"],
            lifecycle=Lifecycle(row["lifecycle"]),
            token_count=int(row["token_count"]),
            source_type=row["source_type"],
            source_id=row["source_id"],
            importance=float(row["importance"]),
            artifact_ids=tuple(filter(None, str(row["artifact_ids"]).splitlines())),
        )
