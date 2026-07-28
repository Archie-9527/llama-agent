"""能力层——基于插件的能力 Provider 架构。

公共 API：
    * ``ToolsConfig``——顶层配置容器。
    * ``bootstrap_capabilities``——连接
      配置 → Provider → capability_registry 的唯一入口。
    * ``CapabilityProvider`` / ``register_provider``——编写新 Provider 时使用的
      基类与装饰器。
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
