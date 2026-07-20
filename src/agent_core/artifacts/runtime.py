"""Process-local access to the configured ArtifactStore."""

from __future__ import annotations

import threading
from pathlib import Path

from agent_core.artifacts.store import ArtifactStore

_lock = threading.Lock()
_stores: dict[Path, ArtifactStore] = {}
_active_root: Path | None = None


def initialize_artifact_store(root: Path) -> ArtifactStore:
    global _active_root
    resolved = root.resolve()
    with _lock:
        store = _stores.get(resolved)
        if store is None:
            store = ArtifactStore(resolved)
            _stores[resolved] = store
        _active_root = resolved
        return store


def get_artifact_store() -> ArtifactStore:
    if _active_root is None:
        raise RuntimeError("ArtifactStore has not been initialized")
    return _stores[_active_root]


def _reset_artifact_stores_for_testing() -> None:
    global _active_root
    with _lock:
        _stores.clear()
        _active_root = None
