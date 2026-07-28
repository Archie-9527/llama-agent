"""可选的全屏终端前端。

Textual 采用延迟导入，使非交互命令和单元测试无需初始化终端 UI 机制。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def discover_skill_names(tools_config: Any) -> set[str]:
    """读取配置的 YAML Skill 名称，供 UI 分组和补全使用。"""

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
