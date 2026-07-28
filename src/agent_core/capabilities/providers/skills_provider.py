"""Skill 能力 Provider——数据驱动、用户可编辑的工具。

扫描目录中的 YAML 文件，每个文件定义一个“Skill”，即更高层、面向业务的
工具。非开发人员无需修改 Python 代码即可创建和编辑这些工具。

支持的类型：
    * ``shell_template``——带命名参数槽的预构建 Shell 命令，例如
      ``pdftotext {file_path} -``。

设计：
    * 每个 ``kind`` 都映射到 ``_KIND_BUILDERS`` 中的一个构建函数。
    * 单个格式错误的 YAML 文件会被记录并跳过，其余文件继续加载。
    * ``build()`` 返回 ``Capability`` 实例；Pydantic 参数 Schema 的转换稍后
      在 ``react_agent_factory.py`` 中完成。
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
# 类型构建器——每个 ``kind`` 值对应一个函数
# ---------------------------------------------------------------------------


def _build_shell_template_handler(
    definition: dict[str, Any], config: SkillsToolConfig
) -> Callable[..., str]:
    """为 ``kind: shell_template`` 的 Skill 构造可调用对象。"""
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
# YAML 文件加载器
# ---------------------------------------------------------------------------


def _load_skill_file(path: Path, config: SkillsToolConfig) -> Capability:
    """将一个 YAML Skill 文件解析为 ``Capability``。

    解析或 Schema 出错时抛出 ``ToolProviderConfigError``。
    """
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

    # 根据参数定义构建 JSON Schema
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
# Provider 实现
# ---------------------------------------------------------------------------


@register_provider
class SkillsCapabilityProvider(CapabilityProvider):
    """数据驱动的 Skill。对应 TOML 段：``[tools.providers.skills]``。

    扫描 ``skills_dir`` 中的 ``*.yaml`` 文件，并将每个文件转换为一个
    ``Capability`` 实例。
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
