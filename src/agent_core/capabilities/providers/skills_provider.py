"""Skills capability provider — data-driven, user-editable tools.

Scans a directory of YAML files, each defining a "skill" — a
higher-level, business-oriented tool that non-developers can create
and edit without touching Python code.

Supported kinds:
    * ``shell_template`` — a pre-built shell command with named
      parameter slots (e.g. ``pdftotext {file_path} -``).

Design:
    * Each ``kind`` maps to a builder function in ``_KIND_BUILDERS``.
    * A single malformed YAML file is logged and skipped; the rest load.
    * ``build()`` returns ``Capability`` instances — the conversion to
      Pydantic args schema happens later, in ``react_agent_factory.py``.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from agent_core.capability_registry import Capability
from agent_core.capabilities.base import (
    CapabilityProvider,
    ToolProviderConfigError,
    register_provider,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillsToolConfig:
    skills_dir: Path = Path("src/agent_core/capabilities/skills")
    default_timeout_seconds: float = 15.0

# ---------------------------------------------------------------------------
# Kind builders — one function per ``kind`` value
# ---------------------------------------------------------------------------


def _build_shell_template_handler(
    definition: dict[str, Any], config: SkillsToolConfig
) -> Callable[..., str]:
    """Construct a callable for ``kind: shell_template`` skills."""
    template: str = definition["command_template"]

    def _runner(**kwargs: Any) -> str:
        try:
            command_str = template.format(**kwargs)
            parts = shlex.split(command_str)
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=config.default_timeout_seconds,
                shell=False,
            )
            return json.dumps(
                {
                    "success": result.returncode == 0,
                    "exit_code": result.returncode,
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                    "error": (
                        None
                        if result.returncode == 0
                        else f"Skill command exited with status {result.returncode}."
                    ),
                },
                ensure_ascii=False,
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "success": False,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "error": (
                        f"Skill timed out after {config.default_timeout_seconds}s."
                    ),
                },
                ensure_ascii=False,
            )
        except (KeyError, ValueError, OSError) as exc:
            return json.dumps(
                {
                    "success": False,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "error": f"Skill execution failed: {exc}",
                },
                ensure_ascii=False,
            )

    return _runner


_KIND_BUILDERS: dict[str, Callable] = {
    "shell_template": _build_shell_template_handler,
}


# ---------------------------------------------------------------------------
# YAML file loader
# ---------------------------------------------------------------------------


def _load_skill_file(path: Path, config: SkillsToolConfig) -> Capability:
    """Parse one YAML skill file into a ``Capability``.

    Raises ``ToolProviderConfigError`` on parse / schema error."""
    with open(path, encoding="utf-8") as f:
        definition = yaml.safe_load(f)

    for required in ("name", "description", "kind"):
        if required not in definition:
            raise ToolProviderConfigError(
                f"Skill file {path} is missing required field '{required}'"
            )

    kind: str = definition["kind"]
    if kind not in _KIND_BUILDERS:
        raise ToolProviderConfigError(
            f"Skill file {path}: kind='{kind}' is not supported "
            f"(supported: {list(_KIND_BUILDERS.keys())})"
        )

    handler = _KIND_BUILDERS[kind](definition, config)

    # Build JSON Schema from args definition
    args_def: dict[str, Any] = definition.get("args", {})
    properties: dict[str, dict] = {
        name: {
            "type": spec.get("type", "string"),
            "description": spec.get("description", ""),
        }
        for name, spec in args_def.items()
    }
    required_args: list[str] = [
        name for name, spec in args_def.items() if spec.get("required", True)
    ]

    return Capability(
        name=definition["name"],
        description=definition["description"],
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required_args,
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


@register_provider
class SkillsCapabilityProvider(CapabilityProvider):
    """Data-driven skills.  TOML section: ``[tools.providers.skills]``.

    Scans ``skills_dir`` for ``*.yaml`` files and converts each into
    a ``Capability`` instance.
    """

    category = "skills"

    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        try:
            normalized_config = dict(raw_config)
            if "skills_dir" in normalized_config:
                raw_skills_dir = normalized_config["skills_dir"]
                if not isinstance(raw_skills_dir, (str, Path)):
                    raise TypeError("skills_dir must be a path string")
                normalized_config["skills_dir"] = Path(raw_skills_dir)
            config = SkillsToolConfig(**normalized_config)
        except (TypeError, ValueError) as exc:
            raise ToolProviderConfigError(
                f"skills provider config has invalid fields: {exc}"
            ) from exc

        if not config.skills_dir.exists():
            logger.warning(
                "Skills directory %s does not exist — skipping skills loading",
                config.skills_dir,
            )
            return []

        caps: list[Capability] = []
        for yaml_file in sorted(config.skills_dir.glob("*.yaml")):
            try:
                caps.append(_load_skill_file(yaml_file, config))
            except (ToolProviderConfigError, yaml.YAMLError) as exc:
                logger.warning(
                    "Skill file %s failed to load — skipped: %s", yaml_file, exc
                )
                continue

        return caps
