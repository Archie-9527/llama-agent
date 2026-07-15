"""Configuration layer — merge defaults, TOML file, env vars, and CLI overrides.

Two config domains share one merge algorithm:
    * ``AppConfig``   — orchestration / UI (max_iterations, db_path, etc.)
    * ``EngineConfig`` — LLM inference kernel (model_path, n_ctx, etc.)

Design constraints:
    1. ``AppConfig`` is a frozen dataclass.
    2. Both configs use the same ``_cast_layer`` / ``_load_env_layer`` /
       ``_resolve_config_file`` helpers — never copy-paste logic per domain.
    3. Type-cast failures raise ``ValueError`` with field name + raw value.
    4. Explicit ``config_file`` missing → ``FileNotFoundError``; implicit
       search missing → empty dict (no error).
    5. TOML files must use ``[agent]`` / ``[engine]`` sections — flat keys
       are silently ignored.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Optional

from agent_core.llm_engine import EngineConfig

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_PREFIX: str = "AGENT_"

DEFAULT_CONFIG_SEARCH_PATHS: list[Path] = [
    Path("agent_config.toml"),
    Path.home() / ".config" / "llama-agent" / "config.toml",
]


# ---------------------------------------------------------------------------
# [STABLE] AppConfig — orchestration / CLI tunables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Single source of truth for orchestration-layer configuration.

    Every field here has a default — ``AppConfig()`` is always valid.
    """

    max_iterations: int = 6
    db_path: Path = Path("data/checkpoints.sqlite")
    last_thread_file: Path = Path("data/last_thread_id.txt")
    log_level: str = "INFO"

    def to_run_config(self):
        """Convert to ``session.RunConfig``, keeping the two data structures
        decoupled."""
        from agent_core.session import RunConfig

        return RunConfig(
            max_iterations=self.max_iterations,
            db_path=self.db_path,
            last_thread_file=self.last_thread_file,
        )


# ---------------------------------------------------------------------------
# Per-domain type-caster tables
# ---------------------------------------------------------------------------


_APP_FIELD_CASTERS: dict[str, Callable] = {
    "max_iterations": int,
    "db_path": Path,
    "last_thread_file": Path,
    "log_level": str,
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
    "verbose": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "request_timeout": float,
    # ``stop`` is intentionally absent — it is a list and cannot be
    # expressed as a single env-var / CLI value.  Only the TOML file
    # (array syntax) can set it.
}


# ---------------------------------------------------------------------------
# Shared helpers (used by both AppConfig and EngineConfig loading)
# ---------------------------------------------------------------------------


def _load_toml_file(path: Path) -> dict[str, Any]:
    """Read and parse a TOML file.  Returns ``{}`` if ``tomllib`` is unavailable."""
    if tomllib is None:
        logger.warning("tomllib not available — skipping config file %s", path)
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_config_file(config_file: Optional[Path]) -> dict[str, Any]:
    """Find and load the TOML config file.

    * Explicit path → missing raises ``FileNotFoundError``.
    * Implicit search → first hit wins; all missing returns ``{}``.
    """
    if config_file is not None:
        if not config_file.exists():
            raise FileNotFoundError(
                f"Specified config file does not exist: {config_file}"
            )
        return _load_toml_file(config_file)

    for candidate in DEFAULT_CONFIG_SEARCH_PATHS:
        if candidate.exists():
            logger.info("Loaded config file: %s", candidate)
            return _load_toml_file(candidate)
    return {}


def _load_env_layer(prefix: str, casters: dict[str, Callable]) -> dict[str, Any]:
    """Scan ``os.environ`` for keys with *prefix*.

    ``AGENT_MAX_ITERATIONS=10`` → ``{"max_iterations": "10"}``.

    Only keys whose mapped field name appears in *casters* are collected.
    Unknown ``AGENT_*`` vars are silently skipped.
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
    """Apply type conversion to every value in *raw*.

    * Unknown keys → logged as WARNING and dropped.
    * Conversion failure → ``ValueError`` with field name + raw value.
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
# [STABLE] Public entry points
# ---------------------------------------------------------------------------


def load_app_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> AppConfig:
    """Merge four layers into an ``AppConfig``.

    Layers (later overrides earlier):
        1. ``AppConfig()`` defaults
        2. TOML ``[agent]`` section
        3. ``AGENT_*`` environment variables
        4. *cli_overrides* (values that are not ``None``)
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
    """Merge four layers into an ``EngineConfig``.

    Unlike ``load_app_config``, this function explicitly checks for
    ``model_path`` after merging because ``EngineConfig`` has a required
    field without a default.

    Raises:
        ValueError: If ``model_path`` is absent after merging all layers.
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
    """Return the set of ``AppConfig`` field names (for tests / validation)."""
    return {f.name for f in fields(AppConfig)}


def engine_config_field_names() -> set[str]:
    """Return the set of ``EngineConfig`` field names (for tests / validation)."""
    return {f.name for f in fields(EngineConfig)}


# Backward-compat alias
load_config = load_app_config
