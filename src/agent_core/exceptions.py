"""Custom exception hierarchy for the llama-agent system.

All agent-core exceptions ultimately inherit from ``AgentCoreError`` so that
graph-level error-handling code can catch a single root type regardless of
which layer the error came from.
"""

from __future__ import annotations


# ── Unified project root ────────────────────────────────────────────────────


class AgentCoreError(Exception):
    """Root exception for all agent-core modules.

    This is the single ancestor that graph-layer code should catch for
    unified error logging / fallback across all internal subsystems:
    engine, grammar, prompt assembly, knowledge scope, and capability registry.
    """


# ── Inference-engine layer (legacy — kept for backward compat) ───────────────


class AgentEngineError(AgentCoreError):
    """Base for exceptions originating inside the inference engine."""


class ModelLoadError(AgentEngineError):
    """GGUF model file is missing, corrupt, or otherwise unloadable."""


class GrammarCompileError(AgentCoreError):
    """GBNF grammar compilation failed, or the JSON Schema contains unsupported
    constructs (e.g. unresolved ``$ref``)."""


class InferenceTimeoutError(AgentEngineError):
    """Single inference call exceeded ``request_timeout`` without producing output."""


# ── Standardised interface (prompt / knowledge) layer ────────────────────────


class PromptAssemblyError(AgentCoreError):
    """Base exception for prompt-assembly failures."""


class ContextBudgetExceededError(PromptAssemblyError):
    """Context-window budget exhausted — even after aggressive trimming the
    remaining content still exceeds the token limit."""


class ToolConsistencyError(PromptAssemblyError):
    """The human-readable tool descriptions embedded in the system prompt do
    not match the tool set used for grammar constraint — the model sees a
    different tool list than what the grammar actually permits."""


# ── Graph orchestration layer ────────────────────────────────────────────────


class GraphOrchestrationError(AgentCoreError):
    """Base exception for graph orchestration (Planner / Executor / Reflector)."""


class PlanningError(GraphOrchestrationError):
    """Planner output could not be parsed into a valid plan — malformed JSON,
    missing ``steps`` key, or empty step list."""


class ExecutionError(GraphOrchestrationError):
    """Executor inner subgraph failed — recursion limit exceeded, tool-call
    pair mismatch, or unhandled runtime error during step execution."""


class ReflectionError(GraphOrchestrationError):
    """Reflector produced a decision outside the valid enum set
    (``done`` / ``continue`` / ``failed``)."""
