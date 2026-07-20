"""Shared local-path validation for deterministic file-based tools."""

from __future__ import annotations

from pathlib import Path


def configured_roots(raw_roots: list[str] | None) -> tuple[Path, ...]:
    roots = raw_roots or ["."]
    return tuple(Path(item).expanduser().resolve() for item in roots)


def resolve_allowed_path(
    raw_path: str,
    roots: tuple[Path, ...],
    *,
    must_exist: bool = True,
) -> Path:
    candidate = Path(raw_path).expanduser().resolve()
    if not any(candidate == root or root in candidate.parents for root in roots):
        raise ValueError(f"path is outside configured roots: {raw_path}")
    if must_exist and not candidate.exists():
        raise ValueError(f"path does not exist: {raw_path}")
    return candidate
