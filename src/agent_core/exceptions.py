"""Custom exception hierarchy for the llama-agent inference engine.

All exceptions that escape the inference kernel layer must be instances of
AgentEngineError or its subclasses. Raw llama_cpp exceptions must never leak
to upper modules.
"""


class AgentEngineError(Exception):
    """Base class for all inference-engine exceptions."""


class ModelLoadError(AgentEngineError):
    """Raised when the GGUF model file is missing, corrupt, or cannot be loaded
    (e.g. insufficient GPU memory)."""


class GrammarCompileError(AgentEngineError):
    """Raised when a GBNF grammar string fails to compile, or when a JSON Schema
    contains unsupported constructs (e.g. unresolved $ref)."""


class InferenceTimeoutError(AgentEngineError):
    """Raised when a single inference call exceeds ``request_timeout`` seconds
    without producing a result."""
