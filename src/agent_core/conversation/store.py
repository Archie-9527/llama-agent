"""以 SQLite 为后端的 Conversation/Turn 持久化。"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from agent_core.conversation.models import Conversation, Turn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL UNIQUE,
                    turn_index INTEGER NOT NULL,
                    user_input TEXT NOT NULL,
                    assistant_output TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id),
                    UNIQUE(conversation_id, turn_index)
                );
                CREATE INDEX IF NOT EXISTS idx_turns_conversation
                    ON turns(conversation_id, turn_index);
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_conversation(self, conversation_id: str) -> Conversation:
        now = _utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO conversations VALUES (?, ?, ?)",
                (conversation_id, now, now),
            )
        return Conversation(conversation_id, now, now)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        row = self._connection.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return Conversation(**dict(row)) if row else None

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        rows = self._connection.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Conversation(**dict(row)) for row in rows]

    def next_turn_index(self, conversation_id: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_index "
            "FROM turns WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return int(row["next_index"])

    def create_running_turn(
        self,
        *,
        turn_id: str,
        conversation_id: str,
        thread_id: str,
        user_input: str,
    ) -> Turn:
        now = _utc_now()
        turn_index = self.next_turn_index(conversation_id)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO turns (
                    turn_id, conversation_id, thread_id, turn_index,
                    user_input, assistant_output, status, error,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, '', 'running', NULL, ?, ?)
                """,
                (
                    turn_id,
                    conversation_id,
                    thread_id,
                    turn_index,
                    user_input,
                    now,
                    now,
                ),
            )
            self._connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
        return self.get_turn(turn_id)  # type: ignore[return-value]

    def finish_turn(
        self,
        turn_id: str,
        *,
        status: str,
        assistant_output: str,
        error: str | None = None,
    ) -> Turn:
        now = _utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE turns
                SET status = ?, assistant_output = ?, error = ?, updated_at = ?
                WHERE turn_id = ?
                """,
                (status, assistant_output, error, now, turn_id),
            )
            self._connection.execute(
                """
                UPDATE conversations SET updated_at = ?
                WHERE conversation_id = (
                    SELECT conversation_id FROM turns WHERE turn_id = ?
                )
                """,
                (now, turn_id),
            )
        return self.get_turn(turn_id)  # type: ignore[return-value]

    def get_turn(self, turn_id: str) -> Turn | None:
        row = self._connection.execute(
            "SELECT * FROM turns WHERE turn_id = ?", (turn_id,)
        ).fetchone()
        return Turn(**dict(row)) if row else None

    def get_turn_by_thread(self, thread_id: str) -> Turn | None:
        row = self._connection.execute(
            "SELECT * FROM turns WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return Turn(**dict(row)) if row else None

    def list_turns(self, conversation_id: str) -> list[Turn]:
        rows = self._connection.execute(
            "SELECT * FROM turns WHERE conversation_id = ? ORDER BY turn_index",
            (conversation_id,),
        ).fetchall()
        return [Turn(**dict(row)) for row in rows]
