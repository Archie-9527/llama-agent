"""Task-isolated, file-backed storage for large tool outputs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_id: str
    owner_id: str
    tool_name: str
    content_type: str
    original_bytes: int
    content_hash: str
    summary: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class ArtifactStore:
    """Store content outside Agent messages and enforce owner isolation."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.content_dir = self.root / "content"
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "artifacts.sqlite"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    original_bytes INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_owner "
                "ON artifacts(owner_id)"
            )

    def put(
        self,
        content: str | bytes,
        *,
        owner_id: str,
        tool_name: str,
        summary: str,
        content_type: str = "text/plain",
    ) -> ArtifactMetadata:
        payload = content.encode("utf-8") if isinstance(content, str) else content
        opaque_id = uuid.uuid4().hex
        artifact_id = f"artifact://{opaque_id}"
        relative_path = f"{opaque_id}.bin"
        target = self.content_dir / relative_path
        target.write_bytes(payload)
        created_at = datetime.now(timezone.utc).isoformat()
        content_hash = hashlib.sha256(payload).hexdigest()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, owner_id, tool_name, content_type,
                    original_bytes, content_hash, summary, relative_path,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    owner_id,
                    tool_name,
                    content_type,
                    len(payload),
                    content_hash,
                    summary,
                    relative_path,
                    created_at,
                ),
            )
        return ArtifactMetadata(
            artifact_id=artifact_id,
            owner_id=owner_id,
            tool_name=tool_name,
            content_type=content_type,
            original_bytes=len(payload),
            content_hash=content_hash,
            summary=summary,
            created_at=created_at,
        )

    def metadata(self, artifact_id: str, *, owner_id: str) -> ArtifactMetadata:
        row = self._owned_row(artifact_id, owner_id)
        return ArtifactMetadata(
            artifact_id=row["artifact_id"],
            owner_id=row["owner_id"],
            tool_name=row["tool_name"],
            content_type=row["content_type"],
            original_bytes=int(row["original_bytes"]),
            content_hash=row["content_hash"],
            summary=row["summary"],
            created_at=row["created_at"],
        )

    def retrieve(
        self,
        artifact_id: str,
        *,
        owner_id: str,
        offset: int,
        length: int,
    ) -> dict:
        if offset < 0 or length < 1:
            raise ValueError("offset must be >= 0 and length must be >= 1")
        row = self._owned_row(artifact_id, owner_id)
        payload = self._content_path(row).read_bytes()
        chunk = payload[offset : offset + length]
        return {
            "artifact_id": artifact_id,
            "offset": offset,
            "length": len(chunk),
            "total_bytes": len(payload),
            "eof": offset + len(chunk) >= len(payload),
            "content": chunk.decode("utf-8", errors="replace"),
        }

    def search(
        self,
        artifact_id: str,
        *,
        owner_id: str,
        query: str,
        max_matches: int,
        context_chars: int = 120,
    ) -> dict:
        if not query:
            raise ValueError("query must not be empty")
        row = self._owned_row(artifact_id, owner_id)
        text = self._content_path(row).read_text(encoding="utf-8", errors="replace")
        folded_text = text.casefold()
        folded_query = query.casefold()
        matches = []
        cursor = 0
        while len(matches) < max_matches:
            index = folded_text.find(folded_query, cursor)
            if index < 0:
                break
            start = max(0, index - context_chars)
            end = min(len(text), index + len(query) + context_chars)
            matches.append(
                {
                    "offset": index,
                    "preview": text[start:end],
                }
            )
            cursor = index + max(1, len(query))
        return {
            "artifact_id": artifact_id,
            "query": query,
            "match_count": len(matches),
            "matches": matches,
        }

    def _owned_row(self, artifact_id: str, owner_id: str) -> sqlite3.Row:
        if not artifact_id.startswith("artifact://"):
            raise ValueError("invalid artifact_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ? AND owner_id = ?",
                (artifact_id, owner_id),
            ).fetchone()
        if row is None:
            raise KeyError("artifact not found or not accessible by this task")
        return row

    def _content_path(self, row: sqlite3.Row) -> Path:
        candidate = (self.content_dir / row["relative_path"]).resolve()
        if candidate.parent != self.content_dir:
            raise ValueError("invalid artifact storage path")
        return candidate

    @staticmethod
    def json_result(data: dict) -> str:
        return json.dumps(data, ensure_ascii=False)
