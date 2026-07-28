"""测试 exceptions.py，验证统一的异常层级。

覆盖范围：
  - AgentCoreError 是所有内部异常的根类。
  - AgentEngineError、PromptAssemblyError、GrammarCompileError 均继承 AgentCoreError。
  - 所有旧异常类型仍保持向后兼容。
"""

from __future__ import annotations

import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from agent_core.exceptions import (
    AgentCoreError,
    AgentEngineError,
    ContextBudgetExceededError,
    GrammarCompileError,
    InferenceTimeoutError,
    ModelLoadError,
    PromptAssemblyError,
    ToolConsistencyError,
)


class TestExceptionHierarchy:
    """验证统一根类以及各异常类的正确方法解析顺序。"""

    def test_agent_core_error_is_root(self):
        assert issubclass(AgentEngineError, AgentCoreError)
        assert issubclass(PromptAssemblyError, AgentCoreError)
        assert issubclass(GrammarCompileError, AgentCoreError)

    def test_engine_errors_chain(self):
        assert issubclass(ModelLoadError, AgentEngineError)
        assert issubclass(InferenceTimeoutError, AgentEngineError)

    def test_prompt_errors_chain(self):
        assert issubclass(ContextBudgetExceededError, PromptAssemblyError)
        assert issubclass(ToolConsistencyError, PromptAssemblyError)

    def test_can_catch_all_via_agent_core_error(self):
        """单个 ``except AgentCoreError`` 应能捕获任意内部异常。"""
        for exc_cls in [
            AgentEngineError, ModelLoadError, InferenceTimeoutError,
            GrammarCompileError, PromptAssemblyError,
            ContextBudgetExceededError, ToolConsistencyError,
        ]:
            try:
                raise exc_cls("test")
            except AgentCoreError:
                pass  # 符合预期
            else:
                pytest.fail(f"{exc_cls.__name__} not caught by AgentCoreError")

    def test_builtin_exception_not_caught(self):
        """AgentCoreError 不应捕获普通的 Exception/ValueError。"""
        with pytest.raises(ValueError):
            try:
                raise ValueError("not ours")
            except AgentCoreError:
                pass  # 不应执行到此处


import pytest
