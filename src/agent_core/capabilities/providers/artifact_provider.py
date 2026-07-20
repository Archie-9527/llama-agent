"""Read-only Agent tools for task-owned externalized tool results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent_core.artifacts import initialize_artifact_store
from agent_core.capabilities.base import (
    CapabilityProvider,
    ToolProviderConfigError,
    register_provider,
)
from agent_core.capability_registry import Capability


@dataclass(frozen=True)
class ArtifactToolConfig:
    storage_dir: Path = Path("data/artifacts")
    max_retrieve_bytes: int = 8192
    max_search_matches: int = 10


def _current_owner_id() -> str:
    from agent_core.telemetry import current_task_id

    return current_task_id() or "__unscoped__"


def _safe(handler: Callable[..., dict]) -> Callable[..., str]:
    def wrapped(**kwargs: Any) -> str:
        try:
            return json.dumps(handler(**kwargs), ensure_ascii=False)
        except (KeyError, OSError, ValueError) as exc:
            return json.dumps(
                {"success": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )

    return wrapped


@register_provider
class ArtifactCapabilityProvider(CapabilityProvider):
    category = "artifact"

    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        try:
            normalized = dict(raw_config)
            environment_dir = os.environ.get("AGENT_ARTIFACT_STORAGE_DIR")
            if environment_dir:
                normalized["storage_dir"] = environment_dir
            if "storage_dir" in normalized:
                normalized["storage_dir"] = Path(normalized["storage_dir"])
            config = ArtifactToolConfig(**normalized)
            if config.max_retrieve_bytes < 1 or config.max_search_matches < 1:
                raise ValueError("limits must be positive")
            store = initialize_artifact_store(config.storage_dir)
        except (TypeError, ValueError, OSError) as exc:
            raise ToolProviderConfigError(
                f"artifact provider config invalid: {exc}"
            ) from exc

        def _summary(artifact_id: str) -> dict:
            return {
                "success": True,
                **store.metadata(
                    artifact_id, owner_id=_current_owner_id()
                ).to_dict(),
            }

        def _retrieve(
            artifact_id: str,
            offset: int = 0,
            length: int = 4096,
        ) -> dict:
            return {
                "success": True,
                **store.retrieve(
                    artifact_id,
                    owner_id=_current_owner_id(),
                    offset=offset,
                    length=min(length, config.max_retrieve_bytes),
                ),
            }

        def _search(
            artifact_id: str,
            query: str,
            max_matches: int = 5,
        ) -> dict:
            return {
                "success": True,
                **store.search(
                    artifact_id,
                    owner_id=_current_owner_id(),
                    query=query,
                    max_matches=min(max_matches, config.max_search_matches),
                ),
            }

        return [
            Capability(
                name="get_artifact_summary",
                description="Get metadata and summary for a task-owned artifact.",
                input_schema={
                    "type": "object",
                    "properties": {"artifact_id": {"type": "string"}},
                    "required": ["artifact_id"],
                },
                handler=_safe(_summary),
            ),
            Capability(
                name="retrieve_artifact",
                description="Read a bounded byte range from a task-owned artifact.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": "string"},
                        "offset": {"type": "integer"},
                        "length": {"type": "integer"},
                    },
                    "required": ["artifact_id"],
                },
                handler=_safe(_retrieve),
            ),
            Capability(
                name="search_artifact",
                description="Search text inside a task-owned artifact.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": "string"},
                        "query": {"type": "string"},
                        "max_matches": {"type": "integer"},
                    },
                    "required": ["artifact_id", "query"],
                },
                handler=_safe(_search),
            ),
        ]
