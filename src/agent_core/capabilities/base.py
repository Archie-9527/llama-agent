"""能力 Provider 的核心抽象——插件架构。

每一类能力（文件、日志、SQLite、Artifact、Skill 等）都是一个
``CapabilityProvider`` 子类，并单独存放在 ``providers/`` 下的文件中。
添加新类别只需增加一个文件，无需修改任何现有模块。

设计原则：
    1. 每个 Provider 都是自包含的，独立负责配置解析、校验和 ``Capability``
       构造。
    2. 单个 Provider 失败不能导致整个引导过程崩溃；系统会记录错误并跳过该
       Provider 的能力。
    3. 通过 ``@register_provider`` 和 ``providers/__init__.py`` 中的
       ``pkgutil`` 扫描自动发现 Provider。
    4. ``build()`` 返回与框架无关的能力描述 ``list[Capability]``；只有
       ``react_agent_factory.py`` 负责将其转换为 LangChain ``BaseTool``。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from agent_core.capability_registry import Capability

logger = logging.getLogger(__name__)


class ToolProviderConfigError(Exception):
    """Provider 原始配置缺少必填字段或包含非法值。

    ``bootstrap.build_capability_map`` 会捕获该异常、记录 WARNING 并跳过对应
    Provider，而不是让程序崩溃。
    """


# ---------------------------------------------------------------------------
# [稳定接口] CapabilityProvider——所有 Provider 必须实现的统一接口
# ---------------------------------------------------------------------------


class CapabilityProvider(ABC):
    """所有能力 Provider 都必须继承本类，并使用 ``@register_provider`` 装饰。

    子类需要定义两项内容：
        * ``category``——字符串键（例如 ``"shell"``、``"skills"``），与
          TOML 的 ``[tools.providers.<category>]`` 段对应。
        * ``build(raw_config)``——接收该配置段的原始字典，返回
          ``Capability`` 实例列表。
    """

    category: str  # 必须由子类设置

    @abstractmethod
    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        """根据 *raw_config* 构造 ``Capability`` 实例。

        引导程序会直接传入原始 TOML 子表，每个 Provider 自行负责解析与校验。

        异常：
            ToolProviderConfigError：缺少必需配置时抛出。
        """
        ...


# ---------------------------------------------------------------------------
# 自注册机制——Provider 通过装饰器声明自身
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, type[CapabilityProvider]] = {}


def register_provider(cls: type[CapabilityProvider]) -> type[CapabilityProvider]:
    """用于注册 ``CapabilityProvider`` 子类的类装饰器。

    ``category`` 重复属于编程错误。
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
    """返回当前已注册的全部类别名称。"""
    return list(_PROVIDER_REGISTRY.keys())


def get_provider_class(category: str) -> type[CapabilityProvider]:
    """根据类别名称查找 Provider 类。

    异常：
        KeyError：*category* 尚未注册时抛出。
    """
    if category not in _PROVIDER_REGISTRY:
        raise KeyError(
            f"Unknown tool category '{category}' "
            f"(registered: {all_registered_categories()})"
        )
    return _PROVIDER_REGISTRY[category]
