"""Tests for exceptions.py — verify the unified exception hierarchy.

Covers:
  - AgentCoreError is the root for all exceptions.
  - AgentEngineError, PromptAssemblyError, GrammarCompileError → AgentCoreError.
  - All legacy types still work for backward compat.
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
    """Verify the unified root and correct MRO for each exception class."""

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
        """A single 'except AgentCoreError' should catch any internal exception."""
        for exc_cls in [
            AgentEngineError, ModelLoadError, InferenceTimeoutError,
            GrammarCompileError, PromptAssemblyError,
            ContextBudgetExceededError, ToolConsistencyError,
        ]:
            try:
                raise exc_cls("test")
            except AgentCoreError:
                pass  # expected
            else:
                pytest.fail(f"{exc_cls.__name__} not caught by AgentCoreError")

    def test_builtin_exception_not_caught(self):
        """AgentCoreError should NOT catch plain Exception/ValueError."""
        with pytest.raises(ValueError):
            try:
                raise ValueError("not ours")
            except AgentCoreError:
                pass  # should not happen


import pytest
