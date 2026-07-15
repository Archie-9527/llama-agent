"""Capability bootstrap — read ``ToolsConfig``, walk all registered
providers, filter by ``enabled_tools``, and write the final capability
set into ``capability_registry`` via ``register()``.

This module is the **single bridge** between the TOML config layer and
the global ``capability_registry``.  No other code may call
``capability_registry.register()`` directly — all tool registration
flows through ``bootstrap_capabilities()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_core.capability_registry import Capability, clear_registry, register

import agent_core.capabilities.providers  # noqa: F401  trigger auto-discovery

from agent_core.capabilities.base import (
    ToolProviderConfigError,
    all_registered_categories,
    get_provider_class,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# [STABLE] ToolsConfig — single config container for all tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolsConfig:
    """Top-level tool configuration.

    Instead of hard-coding one field per provider, we use a generic
    ``providers`` dict keyed by category name.  Adding a new provider
    needs zero changes here.

    Attributes:
        enabled_tools: Explicit allowlist of capability names.  An empty
            list means "all successfully built capabilities are enabled".
        providers: ``{category: raw_config_dict}``.  Each value is the
            raw sub-table under ``[tools.providers.<category>]`` in the
            TOML file.
    """

    enabled_tools: list[str] = field(default_factory=list)
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# [INTERNAL] Build capability map — walk providers, collect capabilities
# ---------------------------------------------------------------------------


def build_capability_map(config: ToolsConfig) -> dict[str, Capability]:
    """Walk every registered provider, call ``build()``, and collect
    capabilities into a ``{name: Capability}`` dict.

    Failure isolation: a single provider raising
    ``ToolProviderConfigError`` is logged and skipped.

    Name conflicts across providers: the later provider wins (logged).
    """
    result: dict[str, Capability] = {}

    for category in all_registered_categories():
        provider = get_provider_class(category)()
        raw_config = config.providers.get(category, {})

        try:
            caps = provider.build(raw_config)
        except ToolProviderConfigError as exc:
            logger.warning(
                "Tool category '%s' failed to load — skipped: %s", category, exc
            )
            continue

        for cap in caps:
            if cap.name in result:
                logger.warning(
                    "Capability name '%s' (from category '%s') conflicts — "
                    "the later provider's version will be used.",
                    cap.name,
                    category,
                )
            result[cap.name] = cap

    return result


# ---------------------------------------------------------------------------
# [STABLE] Public entry points
# ---------------------------------------------------------------------------


def get_enabled_capabilities(config: ToolsConfig) -> list[Capability]:
    """Build the full capability map, then filter by ``enabled_tools``.

    * ``enabled_tools`` is empty → return **all** capabilities.
    * Unknown names in ``enabled_tools`` are logged as WARNING and dropped.
    """
    cap_map = build_capability_map(config)

    if not config.enabled_tools:
        return list(cap_map.values())

    result: list[Capability] = []
    for name in config.enabled_tools:
        if name not in cap_map:
            logger.warning(
                "Unknown tool '%s' in enabled_tools — ignored "
                "(available: %s)",
                name,
                list(cap_map.keys()),
            )
            continue
        result.append(cap_map[name])

    return result


def bootstrap_capabilities(config: ToolsConfig) -> None:
    """The single public entry point: build, filter, and write every
    enabled capability into the global ``capability_registry``.

    Must be called exactly once per process lifetime, **before**
    ``react_agent_factory.initialize_react_agent()``, so that the
    ReAct inner subgraph captures a non-empty tool list.

    Raises:
        ValueError: If a capability name is already registered (usually
            means this function was called twice — Fail-Fast).
    """
    capabilities = get_enabled_capabilities(config)

    for cap in capabilities:
        register(
            name=cap.name,
            description=cap.description,
            input_schema=cap.input_schema,
        )(cap.handler)

    logger.info(
        "Bootstrapped %d capabilities: %s",
        len(capabilities),
        [c.name for c in capabilities],
    )


def _reset_capabilities_for_testing() -> None:
    """[TEST-ONLY] Clear the global ``capability_registry`` between tests."""
    clear_registry()
