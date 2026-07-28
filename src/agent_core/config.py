"""配置层——合并默认值、TOML 文件、环境变量和 CLI 覆盖项。

两个配置域共享同一种合并算法：
    * ``AppConfig``——编排 / UI，例如 max_iterations、db_path。
    * ``EngineConfig``——LLM 推理内核，例如 model_path、n_ctx。

设计约束：
    1. ``AppConfig`` 是冻结的 DataClass。
    2. 两种配置都使用相同的 ``_cast_layer``、``_load_env_layer`` 和
       ``_resolve_config_file`` 辅助函数，不得为各配置域复制粘贴逻辑。
    3. 类型转换失败会抛出包含字段名称和原始值的 ``ValueError``。
    4. 显式指定的 ``config_file`` 不存在 → ``FileNotFoundError``；隐式搜索未
       找到 → 空字典，不报错。
    5. TOML 文件必须使用 ``[agent]`` / ``[engine]`` 段；平铺键会被静默忽略。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from agent_core.llm_engine import EngineConfig
from agent_core.telemetry.models import TelemetryConfig

if TYPE_CHECKING:
    from agent_core.capabilities.bootstrap import ToolsConfig

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

ENV_PREFIX: str = "AGENT_"

DEFAULT_CONFIG_SEARCH_PATHS: list[Path] = [
    Path("agent_config.toml"),
    Path.home() / ".config" / "llama-agent" / "config.toml",
]
_LOGGED_CONFIG_PATHS: set[Path] = set()


# ---------------------------------------------------------------------------
# [稳定接口] AppConfig——编排 / CLI 可调参数
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """编排层配置的唯一信息源。

    此处每个字段都有默认值，因此 ``AppConfig()`` 始终有效。
    """

    max_iterations: int = 6
    db_path: Path = Path("data/checkpoints.sqlite")
    last_thread_file: Path = Path("data/last_thread_id.txt")
    log_level: str = "INFO"
    conversation_db_path: Path = Path("data/conversations.sqlite")
    last_conversation_file: Path = Path("data/last_conversation_id.txt")
    conversation_history_turns: int = 8
    conversation_history_token_budget: int = 4096

    def to_run_config(self):
        """转换为 ``session.RunConfig``，保持两种数据结构解耦。"""
        from agent_core.session import RunConfig

        return RunConfig(
            max_iterations=self.max_iterations,
            db_path=self.db_path,
            last_thread_file=self.last_thread_file,
        )


@dataclass(frozen=True)
class TuiConfig:
    """交互式终端 UI 设置。

    Thinking 生成由 ``EngineConfig.disable_thinking`` 独立控制；
    ``show_thinking`` 只决定是否在对话记录中渲染模型返回的推理内容。
    """

    show_thinking: bool = False
    thinking_max_chars: int = 8000
    show_sidebar: bool = True
    refresh_interval_ms: int = 300
    tool_result_preview_chars: int = 500
    restore_last_conversation: bool = True
    log_file: Path = Path("data/llama-agent-tui.log")

    def validate(self) -> None:
        if self.thinking_max_chars < 0:
            raise ValueError("tui.thinking_max_chars must be >= 0")
        if self.refresh_interval_ms < 100:
            raise ValueError("tui.refresh_interval_ms must be >= 100")
        if self.tool_result_preview_chars < 0:
            raise ValueError("tui.tool_result_preview_chars must be >= 0")


@dataclass(frozen=True)
class MemoryConfig:
    """后续消融轮次需要的独立优化开关。"""

    artifact_virtualization: bool = False
    artifact_inline_max_bytes: int = 8192
    artifact_preview_chars: int = 1200
    artifact_summary_chars: int = 600
    lifecycle_context: bool = False
    context_store_path: Path = Path("data/context_memory.sqlite")
    context_budget_tokens: int = 12000
    context_activation_tokens: int = 2048
    context_trigger_ratio: float = 0.75
    context_reserved_generation_tokens: int = 1024
    context_min_compaction_bytes: int = 2048
    context_min_compaction_ratio: float = 0.30
    hot_execution_records: int = 4
    hot_conversation_turns: int = 4
    hot_reflection_notes: int = 2
    summary_mode: str = "deterministic"
    summary_trigger_tokens: int = 6000
    summary_target_chars: int = 800
    context_retrieval_top_k: int = 3
    context_retrieval_token_budget: int = 2000
    checkpoint_compaction: bool = True
    kv_lifecycle: bool = False
    branch_management: bool = False

    def validate(self) -> None:
        if self.artifact_inline_max_bytes < 1:
            raise ValueError("artifact_inline_max_bytes must be >= 1")
        if self.artifact_preview_chars < 0:
            raise ValueError("artifact_preview_chars must be >= 0")
        if self.artifact_summary_chars < 1:
            raise ValueError("artifact_summary_chars must be >= 1")
        if self.context_budget_tokens < 1:
            raise ValueError("context_budget_tokens must be >= 1")
        if self.context_activation_tokens < 1:
            raise ValueError("context_activation_tokens must be >= 1")
        if not 0 < self.context_trigger_ratio <= 1:
            raise ValueError("context_trigger_ratio must be in (0, 1]")
        if self.context_reserved_generation_tokens < 1:
            raise ValueError("context_reserved_generation_tokens must be >= 1")
        if self.context_min_compaction_bytes < 1:
            raise ValueError("context_min_compaction_bytes must be >= 1")
        if not 0 < self.context_min_compaction_ratio < 1:
            raise ValueError(
                "context_min_compaction_ratio must be in (0, 1)"
            )
        if self.hot_execution_records < 1:
            raise ValueError("hot_execution_records must be >= 1")
        if self.hot_conversation_turns < 1:
            raise ValueError("hot_conversation_turns must be >= 1")
        if self.hot_reflection_notes < 1:
            raise ValueError("hot_reflection_notes must be >= 1")
        if self.summary_mode != "deterministic":
            raise ValueError(
                "summary_mode currently supports only deterministic"
            )
        if self.summary_trigger_tokens < 1 or self.summary_target_chars < 1:
            raise ValueError("summary limits must be positive")
        if self.context_retrieval_top_k < 0:
            raise ValueError("context_retrieval_top_k must be >= 0")
        if self.context_retrieval_token_budget < 0:
            raise ValueError("context_retrieval_token_budget must be >= 0")


# ---------------------------------------------------------------------------
# 各配置域的类型转换表
# ---------------------------------------------------------------------------


_APP_FIELD_CASTERS: dict[str, Callable] = {
    "max_iterations": int,
    "db_path": Path,
    "last_thread_file": Path,
    "log_level": str,
    "conversation_db_path": Path,
    "last_conversation_file": Path,
    "conversation_history_turns": int,
    "conversation_history_token_budget": int,
}

_TELEMETRY_FIELD_CASTERS: dict[str, Callable] = {
    "enabled": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "output_dir": Path,
    "sample_interval_ms": int,
    "collect_process": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "collect_kv": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "collect_accelerator": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "collect_state_size": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
}

_TUI_FIELD_CASTERS: dict[str, Callable] = {
    "show_thinking": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "thinking_max_chars": int,
    "show_sidebar": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "refresh_interval_ms": int,
    "tool_result_preview_chars": int,
    "restore_last_conversation": lambda v: str(v).strip().lower()
    in ("1", "true", "yes"),
    "log_file": Path,
}

_MEMORY_FIELD_CASTERS: dict[str, Callable] = {
    **{
        name: (lambda v: str(v).strip().lower() in ("1", "true", "yes"))
        for name in (
            "artifact_virtualization",
            "lifecycle_context",
            "kv_lifecycle",
            "branch_management",
            "checkpoint_compaction",
        )
    },
    "artifact_inline_max_bytes": int,
    "artifact_preview_chars": int,
    "artifact_summary_chars": int,
    "context_store_path": Path,
    "context_budget_tokens": int,
    "context_activation_tokens": int,
    "context_trigger_ratio": float,
    "context_reserved_generation_tokens": int,
    "context_min_compaction_bytes": int,
    "context_min_compaction_ratio": float,
    "hot_execution_records": int,
    "hot_conversation_turns": int,
    "hot_reflection_notes": int,
    "summary_mode": str,
    "summary_trigger_tokens": int,
    "summary_target_chars": int,
    "context_retrieval_top_k": int,
    "context_retrieval_token_budget": int,
}

_ENGINE_FIELD_CASTERS: dict[str, Callable] = {
    "model_path": str,
    "n_ctx": int,
    "n_gpu_layers": int,
    "n_batch": int,
    "n_threads": int,
    "chat_format": str,
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "repeat_penalty": float,
    "max_tokens": int,
    "seed": int,
    "verbose": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "request_timeout": float,
    "disable_thinking": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
# 此处有意不包含 ``stop``——它是列表，无法表示为单个环境变量或 CLI 值，
# 只能通过 TOML 文件的数组语法设置。
}


# ---------------------------------------------------------------------------
# 共享辅助函数（AppConfig 与 EngineConfig 加载过程共同使用）
# ---------------------------------------------------------------------------


def _load_toml_file(path: Path) -> dict[str, Any]:
    """读取并解析 TOML 文件；``tomllib`` 不可用时返回 ``{}``。"""
    if tomllib is None:
        logger.warning("tomllib not available — skipping config file %s", path)
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_config_file(config_file: Optional[Path]) -> dict[str, Any]:
    """查找并加载 TOML 配置文件。

    * 显式路径 → 文件缺失时抛出 ``FileNotFoundError``。
    * 隐式搜索 → 使用第一个命中项；全部缺失时返回 ``{}``。
    """
    if config_file is not None:
        if not config_file.exists():
            raise FileNotFoundError(
                f"Specified config file does not exist: {config_file}"
            )
        return _load_toml_file(config_file)

    for candidate in DEFAULT_CONFIG_SEARCH_PATHS:
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in _LOGGED_CONFIG_PATHS:
                logger.info("Loaded config file: %s", candidate)
                _LOGGED_CONFIG_PATHS.add(resolved)
            return _load_toml_file(candidate)
    return {}


def _load_env_layer(prefix: str, casters: dict[str, Callable]) -> dict[str, Any]:
    """扫描 ``os.environ`` 中带 *prefix* 的键。

    ``AGENT_MAX_ITERATIONS=10`` → ``{"max_iterations": "10"}``.

    只收集映射后字段名称存在于 *casters* 的键；未知 ``AGENT_*`` 变量会被静默
    跳过。
    """
    result: dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            field_name = key[len(prefix):].lower()
            if field_name in casters:
                result[field_name] = value
    return result


def _cast_layer(
    raw: dict[str, Any], casters: dict[str, Callable]
) -> dict[str, Any]:
    """对 *raw* 中的每个值应用类型转换。

    * 未知键 → 记录 WARNING 并丢弃。
    * 转换失败 → 抛出包含字段名称和原始值的 ``ValueError``。
    """
    casted: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in casters:
            logger.warning("Unknown config key '%s' — ignored", key)
            continue
        try:
            casted[key] = casters[key](value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Config key '{key}' has value '{value}' which cannot be "
                f"converted to the expected type"
            ) from exc
    return casted


# ---------------------------------------------------------------------------
# [稳定接口] 公共入口
# ---------------------------------------------------------------------------


def load_app_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> AppConfig:
    """将四层配置合并为 ``AppConfig``。

    配置层（后者覆盖前者）：
        1. ``AppConfig()`` 默认值
        2. TOML ``[agent]`` 段
        3. ``AGENT_*`` 环境变量
        4. *cli_overrides* 中不为 ``None`` 的值
    """
    file_data = _resolve_config_file(config_file)

    merged: dict[str, Any] = {}
    merged.update(_cast_layer(file_data.get("agent", {}), _APP_FIELD_CASTERS))
    merged.update(
        _cast_layer(_load_env_layer(ENV_PREFIX, _APP_FIELD_CASTERS), _APP_FIELD_CASTERS)
    )
    if cli_overrides:
        merged.update({k: v for k, v in cli_overrides.items() if v is not None})

    return replace(AppConfig(), **merged)


def load_engine_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> EngineConfig:
    """将四层配置合并为 ``EngineConfig``。

    与 ``load_app_config`` 不同，本函数会在合并后显式检查 ``model_path``，
    因为它是 ``EngineConfig`` 中没有默认值的必填字段。

    异常：
        ValueError：合并所有配置层后仍缺少 ``model_path`` 时抛出。
    """
    file_data = _resolve_config_file(config_file)

    merged: dict[str, Any] = {}
    merged.update(_cast_layer(file_data.get("engine", {}), _ENGINE_FIELD_CASTERS))
    merged.update(
        _cast_layer(
            _load_env_layer(ENV_PREFIX, _ENGINE_FIELD_CASTERS), _ENGINE_FIELD_CASTERS
        )
    )
    if cli_overrides:
        merged.update({k: v for k, v in cli_overrides.items() if v is not None})

    if "model_path" not in merged:
        raise ValueError(
            "engine.model_path is not configured.  Provide it via the [engine] "
            "section of your TOML config file, the AGENT_MODEL_PATH environment "
            "variable, or the --model-path CLI argument."
        )

    return EngineConfig(**merged)


def config_field_names() -> set[str]:
    """返回 ``AppConfig`` 字段名称集合，供测试和校验使用。"""
    return {f.name for f in fields(AppConfig)}


def engine_config_field_names() -> set[str]:
    """返回 ``EngineConfig`` 字段名称集合，供测试和校验使用。"""
    return {f.name for f in fields(EngineConfig)}


# 向后兼容别名
load_config = load_app_config


def load_tools_config(config_file: Optional[Path] = None) -> "ToolsConfig":
    """从 TOML 的 ``[tools]`` 段加载工具配置。

    ``providers`` 子字典会原样传递，每个 Provider 自行负责解析与校验。
    """
    from agent_core.capabilities.bootstrap import ToolsConfig

    file_data = _resolve_config_file(config_file)
    tools_data: dict = file_data.get("tools", {})
    return ToolsConfig(
        enabled_tools=tools_data.get("enabled_tools", []),
        providers=tools_data.get("providers", {}),
    )


def load_telemetry_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> TelemetryConfig:
    """加载可选的 ``[telemetry]`` 段。

    环境变量使用 ``AGENT_TELEMETRY_*`` 命名，避免与应用字段冲突，例如
    ``AGENT_TELEMETRY_ENABLED=true``。
    """
    file_data = _resolve_config_file(config_file)
    merged = _cast_layer(
        file_data.get("telemetry", {}), _TELEMETRY_FIELD_CASTERS
    )
    merged.update(
        _cast_layer(
            _load_env_layer("AGENT_TELEMETRY_", _TELEMETRY_FIELD_CASTERS),
            _TELEMETRY_FIELD_CASTERS,
        )
    )
    if cli_overrides:
        merged.update({k: v for k, v in cli_overrides.items() if v is not None})
    config = replace(TelemetryConfig(), **merged)
    config.validate()
    return config


def load_tui_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> TuiConfig:
    """加载 ``[tui]``，并应用 ``AGENT_TUI_*`` 环境变量覆盖。"""

    file_data = _resolve_config_file(config_file)
    merged = _cast_layer(file_data.get("tui", {}), _TUI_FIELD_CASTERS)
    merged.update(
        _cast_layer(
            _load_env_layer("AGENT_TUI_", _TUI_FIELD_CASTERS),
            _TUI_FIELD_CASTERS,
        )
    )
    if cli_overrides:
        merged.update({k: v for k, v in cli_overrides.items() if v is not None})
    config = replace(TuiConfig(), **merged)
    config.validate()
    return config


def load_memory_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> MemoryConfig:
    file_data = _resolve_config_file(config_file)
    merged = _cast_layer(file_data.get("memory", {}), _MEMORY_FIELD_CASTERS)
    merged.update(
        _cast_layer(
            _load_env_layer("AGENT_MEMORY_", _MEMORY_FIELD_CASTERS),
            _MEMORY_FIELD_CASTERS,
        )
    )
    if cli_overrides:
        merged.update({k: v for k, v in cli_overrides.items() if v is not None})
    config = replace(MemoryConfig(), **merged)
    config.validate()
    return config
