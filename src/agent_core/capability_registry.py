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

#: 工具处理程序签名——接收关键字参数并返回结果（字符串、字典、数字等）的
#: 可调用对象。
Handler: TypeAlias = Callable[..., Any]


@dataclass(frozen=True)
class Capability:
    """单个已注册工具的不可变描述。

    属性：
        name：唯一工具标识符，用于 GBNF 的 ``tool`` 常量。
        description：对工具用途的人类可读说明。
        handler：执行工具的可调用对象。
        input_schema：描述工具参数的 JSON Schema 字典。
    """

    name: str
    description: str
    handler: Handler = field(compare=False, hash=False)
    input_schema: dict = field(compare=False, hash=False)


# ---------------------------------------------------------------------------
# 内部注册表存储——由模块级锁保护，确保启动期间的并发注册安全，例如多线程
# 加载插件。
# ---------------------------------------------------------------------------

_registry: Dict[str, Capability] = {}
_registry_lock = threading.Lock()


# ---------------------------------------------------------------------------
# [稳定接口] 公共 API
# ---------------------------------------------------------------------------


def register(
    name: str,
    description: str,
    input_schema: Dict[str, Any],
) -> Callable[[Handler], Handler]:
    """将函数注册为 Agent 能力的装饰器。

    用法::

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

    参数：
        name：唯一工具标识符。
        description：工具用途的人类可读摘要。
        input_schema：工具参数的 JSON Schema。

    返回：
        包装处理函数的装饰器。该包装保持恒等，注册后会原样返回原函数。

    异常：
        ValueError：*name* 已注册时抛出。
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
    """按名称查找已注册能力。

    参数：
        name：唯一工具标识符。

    返回：
        匹配的 ``Capability`` 描述。

    异常：
        KeyError：*name* 尚未注册时抛出。
    """
    try:
        return _registry[name]
    except KeyError:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise KeyError(
            f"Capability '{name}' not found.  Registered tools: {available}"
        ) from None


def list_capabilities() -> List[Capability]:
    """返回当前全部已注册能力的快照。

    返回的列表是副本，修改它不会影响注册表；并发注册或注销也不会导致缓存了
    较早快照的调用方出现不一致。
    """
    with _registry_lock:
        return list(_registry.values())


def clear_registry() -> None:
    """移除**所有**已注册能力。

    主要供测试使用，应在测试用例之间调用以避免跨测试状态泄漏。

    警告：Agent 运行期间不要在生产代码中调用本函数。
    """
    with _registry_lock:
        _registry.clear()
