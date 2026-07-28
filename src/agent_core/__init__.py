"""Agent 核心——本地 LLM 推理、Grammar 约束与 Agent 编排。"""

from agent_core.llm_engine import (
    ChatLlamaCpp,
    EngineConfig,
    compile_json_schema_to_gbnf,
    get_engine,
    initialize_engine,
)
from agent_core.exceptions import (
    AgentCoreError,
    AgentEngineError,
    ContextBudgetExceededError,
    EngineAlreadyInitializedError,
    EngineConfigError,
    EngineNotInitializedError,
    ExecutionError,
    GrammarCompileError,
    GraphOrchestrationError,
    InferenceTimeoutError,
    ModelLoadError,
    PlanningError,
    PromptAssemblyError,
    ReflectionError,
    ToolConsistencyError,
)

from agent_core.config import AppConfig, load_app_config, load_engine_config
from agent_core.session import RunConfig, TaskRunner

__all__ = [
    # 推理引擎
    "ChatLlamaCpp",
    "EngineConfig",
    "compile_json_schema_to_gbnf",
    "get_engine",
    "initialize_engine",
    # 配置
    "AppConfig",
    "load_app_config",
    "load_engine_config",
    # 会话
    "RunConfig",
    "TaskRunner",
    # 异常
    "AgentCoreError",
    "AgentEngineError",
    "ContextBudgetExceededError",
    "EngineAlreadyInitializedError",
    "EngineConfigError",
    "EngineNotInitializedError",
    "ExecutionError",
    "GrammarCompileError",
    "GraphOrchestrationError",
    "InferenceTimeoutError",
    "ModelLoadError",
    "PlanningError",
    "PromptAssemblyError",
    "ReflectionError",
    "ToolConsistencyError",
]
