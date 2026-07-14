"""Agent factory: create a LangGraph agent backed by a local llama.cpp model."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent  # type: ignore[import-untyped]

from agent_core.llm_engine import ChatLlamaCpp


def create_llama_agent(
    model: Optional[BaseChatModel] = None,
    *,
    model_path: Optional[str] = None,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    system_prompt: Optional[str] = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
    temperature: float = 0.7,
    chat_format: str = "chatml",
    verbose: bool = False,
) -> CompiledStateGraph:
    """Create a LangGraph agent backed by a local llama.cpp model.

    This is a thin wrapper around langchain.agents.create_agent() that handles
    model setup automatically. All arguments after ``model`` are forwarded to
    ChatLlamaCpp when ``model`` is not provided directly.

    Args:
        model: An existing BaseChatModel instance. If None, a ChatLlamaCpp
            is created from the other parameters.
        model_path: Path to the GGUF model file (required if model is None).
        tools: Tools available to the agent.
        system_prompt: Optional system prompt for the agent.
        n_ctx: Context window size (default 4096).
        n_gpu_layers: GPU layers, 0 means CPU-only (default).
        temperature: Sampling temperature (default 0.7).
        chat_format: Chat template format (default 'chatml').
        verbose: Enable verbose logging.

    Returns:
        A compiled LangGraph StateGraph ready for .invoke() or .stream().
    """
    if model is None:
        if model_path is None:
            raise ValueError("model_path is required when model is not provided")
        model = ChatLlamaCpp(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            chat_format=chat_format,
            verbose=verbose,
        )

    # Convert system_prompt str -> SystemMessage if needed
    sys_msg: Optional[SystemMessage] = None
    if system_prompt is not None:
        sys_msg = SystemMessage(content=system_prompt)

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=sys_msg,
    )
