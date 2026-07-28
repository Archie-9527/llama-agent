"""本地 Shell 命令执行能力 Provider。

安全约束：
    * 命令名称只允许使用白名单，而不是黑名单。
    * 使用 ``shell=False`` 调用 ``subprocess.run``，避免命令注入。
    * 每次调用都必须设置超时。
    * 输出截断到 ``max_output_chars``。
    * 内部错误以文本形式返回，绝不能导致图崩溃。
"""

from __future__ import annotations

import json
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
    """本地 Shell 执行。对应 TOML 段：``[tools.providers.shell]``。"""

    category = "shell"

    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        try:
            config = ShellToolConfig(**raw_config)
        except TypeError as exc:
            raise ToolProviderConfigError(
                f"shell provider config has invalid fields: {exc}"
            ) from exc

        def _result(
            *,
            success: bool,
            command: str,
            exit_code: int | None = None,
            stdout: str = "",
            stderr: str = "",
            error: str | None = None,
        ) -> str:
            """向模型返回稳定且机器可读的结果。"""
            combined_length = len(stdout) + len(stderr)
            if combined_length > config.max_output_chars:
                remaining = config.max_output_chars
                stdout = stdout[:remaining]
                remaining -= len(stdout)
                stderr = stderr[:max(remaining, 0)]
                truncation = f"output truncated from {combined_length} characters"
                error = f"{error}; {truncation}" if error else truncation

            return json.dumps(
                {
                    "success": success,
                    "command": command,
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": error,
                },
                ensure_ascii=False,
            )

        def _execute_shell_command(command: str) -> str:
            try:
                parts = shlex.split(command)
            except ValueError as exc:
                return _result(
                    success=False,
                    command=command,
                    error=f"Command parse error: {exc}",
                )

            if not parts:
                return _result(
                    success=False,
                    command=command,
                    error="Empty command — nothing executed.",
                )

            if parts[0] not in config.allowed_commands:
                return _result(
                    success=False,
                    command=command,
                    error=(
                        f"Command '{parts[0]}' is not in the allowlist "
                        f"({', '.join(config.allowed_commands)})."
                    ),
                )

            try:
                result = subprocess.run(
                    parts,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_seconds,
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                return _result(
                    success=False,
                    command=command,
                    error=f"Command timed out after {config.timeout_seconds}s.",
                )
            except OSError as exc:
                return _result(
                    success=False,
                    command=command,
                    error=f"Command failed: {exc}",
                )

            return _result(
                success=result.returncode == 0,
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                error=(
                    None
                    if result.returncode == 0
                    else f"Command exited with status {result.returncode}."
                ),
            )

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
