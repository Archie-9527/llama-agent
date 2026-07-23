"""Optional full-screen terminal front-end.

Textual is imported lazily so non-interactive commands and unit tests do not
need to initialise terminal UI machinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def discover_skill_names(tools_config: Any) -> set[str]:
    """Read configured YAML skill names for UI grouping and completion."""

    import yaml

    raw = tools_config.providers.get("skills", {})
    skills_dir = Path(raw.get("skills_dir", "src/agent_core/capabilities/skills"))
    names: set[str] = set()
    if not skills_dir.exists():
        return names
    for path in sorted(skills_dir.glob("*.yaml")):
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            name = str(value.get("name") or "").strip()
            if name:
                names.add(name)
        except (OSError, yaml.YAMLError):
            continue
    return names


def run_tui(**kwargs: Any) -> None:
    try:
        from agent_core.tui.app import LlamaAgentApp
    except ImportError as exc:
        if exc.name and (exc.name == "textual" or exc.name.startswith("textual.")):
            raise RuntimeError(
                "The interactive CLI requires Textual. Reinstall the project "
                "dependencies or run: pip install textual"
            ) from exc
        raise
    LlamaAgentApp(**kwargs).run()


__all__ = ["discover_skill_names", "run_tui"]
