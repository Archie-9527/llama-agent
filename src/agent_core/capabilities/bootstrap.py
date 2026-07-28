"""能力引导——读取 ``ToolsConfig``，遍历所有已注册的 Provider，按
``enabled_tools`` 过滤，并通过 ``register()`` 将最终能力集合写入
``capability_registry``。

本模块是 TOML 配置层与全局 ``capability_registry`` 之间的**唯一桥梁**。
其他代码不得直接调用 ``capability_registry.register()``，所有工具注册都必须
经过 ``bootstrap_capabilities()``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_core.capability_registry import Capability, clear_registry, register

import agent_core.capabilities.providers  # noqa: F401  触发自动发现

from agent_core.capabilities.base import (
    ToolProviderConfigError,
    all_registered_categories,
    get_provider_class,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# [稳定接口] ToolsConfig——所有工具的统一配置容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolsConfig:
    """顶层工具配置。

    这里不为每个 Provider 硬编码字段，而是使用以类别名称为键的通用
    ``providers`` 字典，因此添加新 Provider 无需修改本类。

    属性：
        enabled_tools：能力名称的显式白名单。空列表表示启用所有成功构建的能力。
        providers：``{category: raw_config_dict}``，每个值都是 TOML 文件中
            ``[tools.providers.<category>]`` 下的原始子表。
    """

    enabled_tools: list[str] = field(default_factory=list)
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# [内部实现] 构建能力映射——遍历 Provider 并收集能力
# ---------------------------------------------------------------------------


def build_capability_map(config: ToolsConfig) -> dict[str, Capability]:
    """遍历每个已注册的 Provider，调用 ``build()``，并将能力收集到
    ``{name: Capability}`` 字典中。

    故障隔离：单个 Provider 抛出 ``ToolProviderConfigError`` 时只记录并跳过。

    跨 Provider 名称冲突：使用后加载的 Provider，并记录日志。
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
# [稳定接口] 公共入口
# ---------------------------------------------------------------------------


def get_enabled_capabilities(config: ToolsConfig) -> list[Capability]:
    """构建完整能力映射，然后按 ``enabled_tools`` 过滤。

    * ``enabled_tools`` 为空 → 返回**全部**能力。
    * ``enabled_tools`` 中的未知名称会记录为 WARNING 并被丢弃。
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
    """唯一公共入口：构建、过滤并将所有启用能力写入全局
    ``capability_registry``。

    每个进程生命周期必须且只能调用一次，并且要在
    ``react_agent_factory.initialize_react_agent()`` **之前**调用，以确保
    ReAct 内部子图能获取非空工具列表。

    异常：
        ValueError：能力名称已注册时抛出，通常表示本函数被重复调用；系统会
            快速失败。
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
    """[仅测试] 在测试之间清空全局 ``capability_registry``。"""
    clear_registry()
