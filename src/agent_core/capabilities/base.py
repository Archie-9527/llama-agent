"""Capability provider core abstraction — the plugin architecture.

Every category of capability (shell, web, skills, MCP, …) is a
``CapabilityProvider`` subclass that lives in its own file under
``providers/``.  Adding a new category means adding a single new file
— zero changes to any existing module.

Design principles:
    1. Each provider is self-contained — it owns its config parsing,
       validation, and ``Capability`` construction.
    2. A single provider failing must not crash the whole bootstrap —
       the error is logged and that provider's capabilities are skipped.
    3. Providers are discovered automatically via ``@register_provider``
       and ``providers/__init__.py``'s ``pkgutil`` scan.
    4. ``build()`` returns ``list[Capability]`` — framework-agnostic
       capability descriptors.  The conversion to LangChain
       ``BaseTool`` happens only in ``react_agent_factory.py``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from agent_core.capability_registry import Capability

logger = logging.getLogger(__name__)


class ToolProviderConfigError(Exception):
    """A provider's raw config is missing required fields or contains
    illegal values.  Caught by ``bootstrap.build_capability_map`` which
    logs a WARNING and skips the provider rather than crashing."""


# ---------------------------------------------------------------------------
# [STABLE] CapabilityProvider — the single interface every provider implements
# ---------------------------------------------------------------------------


class CapabilityProvider(ABC):
    """Every capability provider must subclass this and decorate the class
    with ``@register_provider``.

    Subclasses define two things:
        * ``category`` — a string key (e.g. ``"shell"``, ``"skills"``)
          matching the ``[tools.providers.<category>]`` TOML section.
        * ``build(raw_config)`` — take the raw dict from that section
          and return a list of ``Capability`` instances.
    """

    category: str  # must be set by subclass

    @abstractmethod
    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        """Construct ``Capability`` instances from *raw_config*.

        The bootstrap passes the raw TOML sub-table directly — each
        provider owns its own parsing and validation.

        Raises:
            ToolProviderConfigError: If required config is missing.
        """
        ...


# ---------------------------------------------------------------------------
# Self-registration — providers declare themselves via decorator
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, type[CapabilityProvider]] = {}


def register_provider(cls: type[CapabilityProvider]) -> type[CapabilityProvider]:
    """Class decorator: register a ``CapabilityProvider`` subclass.

    Duplicate ``category`` values are a programming error.
    """
    if cls.category in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Tool category '{cls.category}' is already registered by "
            f"{_PROVIDER_REGISTRY[cls.category].__name__} — cannot also "
            f"register {cls.__name__}"
        )
    _PROVIDER_REGISTRY[cls.category] = cls
    return cls


def all_registered_categories() -> list[str]:
    """Return every currently registered category name."""
    return list(_PROVIDER_REGISTRY.keys())


def get_provider_class(category: str) -> type[CapabilityProvider]:
    """Look up a provider class by category name.

    Raises:
        KeyError: If *category* is not registered.
    """
    if category not in _PROVIDER_REGISTRY:
        raise KeyError(
            f"Unknown tool category '{category}' "
            f"(registered: {all_registered_categories()})"
        )
    return _PROVIDER_REGISTRY[category]
