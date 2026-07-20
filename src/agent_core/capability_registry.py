"""功能注册表——可用工具的唯一信息源。

系统中的每个工具都在此注册一次。注册表被以下模块使用：

* ``prompt_assembler`` — 用于呈现人类可读的工具描述。
* ``grammar_builder``  — 用于构建工具选择的GBNF约束。
* ``graph/executor``   — 用于在运行时查找和调用工具处理程序。

架构约束
    除了``capability_registry``之外，任何模块都不得维护工具元数据的单独副本。
    添加工具只需用``@register(...)``装饰处理函数；其他任何地方都无需更改。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeAlias

#: Signature of a tool handler — callable that accepts keyword arguments
#: and returns a result (string, dict, number, …).
Handler: TypeAlias = Callable[..., Any]


@dataclass(frozen=True)
class Capability:
    """Immutable descriptor for a single registered tool.

    Attributes:
        name: Unique tool identifier (used in GBNF ``tool`` const).
        description: Human-readable explanation of what the tool does.
        handler: The callable that executes the tool.
        input_schema: JSON Schema dict describing the tool's parameters.
    """

    name: str
    description: str
    handler: Handler = field(compare=False, hash=False)
    input_schema: dict = field(compare=False, hash=False)


# ---------------------------------------------------------------------------
# Internal registry storage — guarded by a module-level lock so concurrent
# registration during startup (e.g. multi-threaded plugin loading) is safe.
# ---------------------------------------------------------------------------

_registry: Dict[str, Capability] = {}
_registry_lock = threading.Lock()


# ---------------------------------------------------------------------------
# [STABLE] Public API
# ---------------------------------------------------------------------------


def register(
    name: str,
    description: str,
    input_schema: Dict[str, Any],
) -> Callable[[Handler], Handler]:
    """Decorator that registers a function as an agent capability.

    Usage::

        @register(
            name="search_log",
            description="Search a local log for evidence.",
            input_schema={
                "type": "object",
                "properties": {
                    "log_path": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["log_path", "query"],
            },
        )
        def search_log(log_path: str, query: str) -> str:
            ...

    Args:
        name: Unique tool identifier.
        description: Human-readable summary of what the tool does.
        input_schema: JSON Schema for the tool's arguments.

    Returns:
        A decorator that wraps the handler function (identity — the
        original function is returned unchanged after registration).

    Raises:
        ValueError: If *name* is already registered.
    """

    def _decorator(handler: Handler) -> Handler:
        cap = Capability(
            name=name,
            description=description,
            handler=handler,
            input_schema=input_schema,
        )
        with _registry_lock:
            if name in _registry:
                raise ValueError(
                    f"Capability '{name}' is already registered — use a unique name."
                )
            _registry[name] = cap
        return handler

    return _decorator


def get_capability(name: str) -> Capability:
    """Look up a registered capability by name.

    Args:
        name: The unique tool identifier.

    Returns:
        The matching ``Capability`` descriptor.

    Raises:
        KeyError: If *name* is not registered.
    """
    try:
        return _registry[name]
    except KeyError:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise KeyError(
            f"Capability '{name}' not found.  Registered tools: {available}"
        ) from None


def list_capabilities() -> List[Capability]:
    """Return a snapshot of every currently registered capability.

    The returned list is a copy — mutating it does not affect the registry
    and racing registrations / deregistrations do not cause inconsistencies
    in callers that cached an earlier snapshot.
    """
    with _registry_lock:
        return list(_registry.values())


def clear_registry() -> None:
    """Remove **all** registered capabilities.

    Primarily intended for testing — call between test cases to avoid
    cross-test state leakage.

    WARNING: Do not call this in production code while the agent is running.
    """
    with _registry_lock:
        _registry.clear()
