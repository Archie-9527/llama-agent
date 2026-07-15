"""Local shell-command execution capability provider.

Safety invariants:
    * Allowlist-only (not denylist) for command names.
    * ``subprocess.run`` with ``shell=False`` — no injection.
    * Mandatory timeout on every invocation.
    * Output truncated to ``max_output_chars``.
    * Internal errors return text — they never crash the graph.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any

from agent_core.capability_registry import Capability
from agent_core.capabilities.base import (
    CapabilityProvider,
    ToolProviderConfigError,
    register_provider,
)


@dataclass(frozen=True)
class ShellToolConfig:
    allowed_commands: list[str] = field(
        default_factory=lambda: ["ls", "cat", "pwd", "echo", "grep", "find", "wc"]
    )
    timeout_seconds: float = 10.0
    max_output_chars: int = 4000


@register_provider
class ShellCapabilityProvider(CapabilityProvider):
    """Local shell execution.  TOML section: ``[tools.providers.shell]``."""

    category = "shell"

    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        try:
            config = ShellToolConfig(**raw_config)
        except TypeError as exc:
            raise ToolProviderConfigError(
                f"shell provider config has invalid fields: {exc}"
            ) from exc

        def _execute_shell_command(command: str) -> str:
            try:
                parts = shlex.split(command)
            except ValueError as exc:
                return f"Command parse error: {exc}"

            if not parts:
                return "Empty command — nothing executed."

            if parts[0] not in config.allowed_commands:
                return (
                    f"Command '{parts[0]}' is not in the allowlist "
                    f"({', '.join(config.allowed_commands)})."
                )

            try:
                result = subprocess.run(
                    parts,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_seconds,
                    shell=False,
                )
                output = (result.stdout or "") + (result.stderr or "")
            except subprocess.TimeoutExpired:
                return f"Command timed out after {config.timeout_seconds}s."
            except OSError as exc:
                return f"Command failed: {exc}"

            if len(output) > config.max_output_chars:
                output = (
                    output[: config.max_output_chars]
                    + f"\n…(truncated, {len(output)} chars total)"
                )
            return output or "(command succeeded, no output)"

        return [
            Capability(
                name="execute_shell_command",
                description=(
                    "Execute a read-only shell command on the local machine. "
                    "Only commands in the configured allowlist are permitted. "
                    "Command and arguments should be space-separated, e.g. "
                    "'ls -la /tmp'.  No pipes, redirects, or shell syntax."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Space-separated command, e.g. 'ls -la /tmp'",
                        }
                    },
                    "required": ["command"],
                },
                handler=_execute_shell_command,
            )
        ]
