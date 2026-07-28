"""llama-agent 系统的自定义异常层次结构。

所有 agent-core 异常最终都继承自 ``AgentCoreError``，使图层错误处理代码无论
异常来自哪一层，都只需捕获一个根类型。
"""

from __future__ import annotations


# ── 统一项目根异常 ─────────────────────────────────────────────────────────


class AgentCoreError(Exception):
    """所有 agent-core 模块的根异常。

    图层代码应捕获这个唯一祖先类型，以便在引擎、Grammar、Prompt 组装、知识
    范围和能力注册表等所有内部子系统中统一记录错误并执行后备处理。
    """


# ── 推理引擎层（为向后兼容而保留的旧接口）────────────────────────────────


class AgentEngineError(AgentCoreError):
    """推理引擎内部异常的基类。"""


class ModelLoadError(AgentEngineError):
    """GGUF 模型文件缺失、损坏或因其他原因无法加载。"""


class GrammarCompileError(AgentCoreError):
    """GBNF Grammar 编译失败，或 JSON Schema 包含不支持的结构，例如未解析的
    ``$ref``。"""


class InferenceTimeoutError(AgentEngineError):
    """单次推理调用超过 ``request_timeout``，且没有产生输出。"""


# ── 标准接口（Prompt / 知识）层 ────────────────────────────────────────────


class PromptAssemblyError(AgentCoreError):
    """Prompt 组装失败的基类。"""


class ContextBudgetExceededError(PromptAssemblyError):
    """上下文窗口预算耗尽——即使积极裁剪，剩余内容仍超过 Token 上限。"""


class ToolConsistencyError(PromptAssemblyError):
    """嵌入 System Prompt 的人类可读工具描述与 Grammar 约束使用的工具集合不
    一致——模型看到的工具列表与 Grammar 实际允许的列表不同。"""


# ── 图编排层 ───────────────────────────────────────────────────────────────


class GraphOrchestrationError(AgentCoreError):
    """图编排（Planner / Executor / Reflector）异常的基类。"""


class PlanningError(GraphOrchestrationError):
    """Planner 输出无法解析为有效计划——JSON 格式错误、缺少 ``steps`` 键或
    步骤列表为空。"""


class ExecutionError(GraphOrchestrationError):
    """Executor 内部子图失败——超过递归上限、工具调用配对不一致，或步骤执行
    期间出现未处理的运行时错误。"""


class ReflectionError(GraphOrchestrationError):
    """Reflector 产生了有效枚举集合
    （``done`` / ``continue`` / ``failed``）之外的决策。"""


# ── 引擎生命周期（llm_engine.py v2 重构）──────────────────────────────────


class EngineNotInitializedError(AgentEngineError):
    """在 ``initialize_engine()`` 完成之前调用了 ``get_engine()``。"""


class EngineAlreadyInitializedError(AgentEngineError):
    """``initialize_engine()`` 被多次调用——单例已经存在，不得静默替换。"""


class EngineConfigError(AgentEngineError):
    """``EngineConfig`` 缺少必填字段（例如 ``model_path``）或包含非法值。"""
