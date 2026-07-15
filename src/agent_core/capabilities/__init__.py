"""Capability layer — plugin-based capability provider architecture.

Public API:
    * ``ToolsConfig`` — top-level configuration container.
    * ``bootstrap_capabilities`` — the single entry point for wiring
      config → providers → capability_registry.
    * ``CapabilityProvider`` / ``register_provider`` — base class and
      decorator for writing new providers.
"""

from agent_core.capabilities.bootstrap import (
    ToolsConfig,
    bootstrap_capabilities,
    build_capability_map,
    get_enabled_capabilities,
)
from agent_core.capabilities.base import (
    CapabilityProvider,
    ToolProviderConfigError,
    all_registered_categories,
    get_provider_class,
    register_provider,
)

__all__ = [
    "ToolsConfig",
    "CapabilityProvider",
    "ToolProviderConfigError",
    "all_registered_categories",
    "bootstrap_capabilities",
    "build_capability_map",
    "get_enabled_capabilities",
    "get_provider_class",
    "register_provider",
]
