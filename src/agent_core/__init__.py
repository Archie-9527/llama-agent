"""Agent core — local LLM inference, grammar constraints, and agent orchestration."""

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
    # engine
    "ChatLlamaCpp",
    "EngineConfig",
    "compile_json_schema_to_gbnf",
    "get_engine",
    "initialize_engine",
    # config
    "AppConfig",
    "load_app_config",
    "load_engine_config",
    # session
    "RunConfig",
    "TaskRunner",
    # exceptions
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
