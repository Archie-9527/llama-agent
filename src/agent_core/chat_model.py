"""LlamaCppChatModel: BaseChatModel subclass wrapping a llama_cpp.Llama instance."""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

import llama_cpp
import llama_cpp.llama_chat_format as llama_chat_format


def _langchain_messages_to_llama(
    messages: List[BaseMessage],
) -> List[llama_cpp.llama_types.ChatCompletionRequestMessage]:
    """Convert LangChain message objects to llama-cpp-python dict format.

    LangChain uses Pydantic message types (SystemMessage, HumanMessage, AIMessage,
    ToolMessage) while llama-cpp-python expects plain dicts with role/content keys.
    This function bridges the two type systems.
    """
    result: List[llama_cpp.llama_types.ChatCompletionRequestMessage] = []
    for msg in messages:
        role = msg.type
        if role == "ai":
            role = "assistant"
        elif role == "human":
            role = "user"

        entry: llama_cpp.llama_types.ChatCompletionRequestMessage = {
            "role": role,  # type: ignore[typeddict-item]
            "content": msg.content,
        }

        # Forward tool_calls from assistant messages
        if role == "assistant" and isinstance(msg, AIMessage) and msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"], ensure_ascii=False),
                    },
                }
                for tc in msg.tool_calls
            ]  # type: ignore[typeddict-item]

        # Forward tool_call_id from tool messages
        if role == "tool" and isinstance(msg, ToolMessage):
            entry["tool_call_id"] = msg.tool_call_id  # type: ignore[typeddict-item]

        result.append(entry)
    return result


def _llama_chat_response_to_ai_message(
    response: llama_cpp.llama_types.CreateChatCompletionResponse,
) -> AIMessage:
    """Convert a llama-cpp-python chat completion response into an AIMessage.

    Handles both plain text responses and tool-call responses, extracting
    tool_calls into the AIMessage.tool_calls field that LangChain expects.
    """
    choice = response["choices"][0]
    msg_data = choice["message"]
    content = msg_data.get("content")
    tool_calls_data = msg_data.get("tool_calls") or []

    tool_calls: List[ToolCall] = [
        ToolCall(
            name=tc["function"]["name"],
            args=json.loads(tc["function"]["arguments"]),
            id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
        )
        for tc in tool_calls_data
    ]

    # Fallback: parse function_call if present but tool_calls is empty
    if not tool_calls and msg_data.get("function_call"):
        fc = msg_data["function_call"]
        tool_calls = [
            ToolCall(
                name=fc["name"],
                args=json.loads(fc["arguments"]),
                id=f"call_{uuid.uuid4().hex[:8]}",
            )
        ]

    return AIMessage(content=content or "", tool_calls=tool_calls)


def _convert_langchain_tools_to_llama(
    tools: List[Any],
) -> List[llama_cpp.llama_types.ChatCompletionTool]:
    """Convert LangChain tool objects to llama-cpp-python's tool dict format.

    Supports BaseTool, dict, and Callable tool types.
    """
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    result: List[llama_cpp.llama_types.ChatCompletionTool] = []
    for tool in tools:
        if isinstance(tool, BaseTool):
            schema = convert_to_openai_tool(tool)
            result.append(schema)  # type: ignore[arg-type]
        elif isinstance(tool, dict):
            result.append(tool)  # type: ignore[arg-type]
        elif callable(tool):
            schema = convert_to_openai_tool(tool)
            result.append(schema)  # type: ignore[arg-type]
    return result


class LlamaCppChatModel(BaseChatModel):
    """LangChain chat model backed by a local llama.cpp model via llama-cpp-python.

    Holds a llama_cpp.Llama instance internally. The model runs locally with no
    server process or network round-trips.

    Usage:
        model = LlamaCppChatModel(
            model_path="/path/to/model.gguf",
            n_ctx=4096,
            n_gpu_layers=-1,
            temperature=0.7,
            verbose=False,
        )
        response = model.invoke([HumanMessage(content="Hello!")])
    """

    model_path: str = Field(description="Path to the GGUF model file")
    n_gpu_layers: int = Field(default=-1, description="Number of GPU layers (-1 = all)")
    n_ctx: int = Field(default=4096, description="Context window size in tokens")
    n_batch: int = Field(default=512, description="Logical batch size for prompt processing")
    n_threads: int = Field(default=8, description="CPU thread count")
    chat_format: str = Field(default="chatml", description="Chat format (e.g. 'chatml')")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=40, description="Top-k sampling")
    repeat_penalty: float = Field(default=1.1, description="Repeat penalty")
    max_tokens: int = Field(default=512, description="Max tokens to generate per response")
    stop: Optional[List[str]] = Field(default=None, description="Additional stop tokens")
    verbose: bool = Field(default=False, description="Enable verbose llama.cpp output")
    streaming: bool = Field(default=False, description="Enable token-level streaming")

    _llm: Optional[llama_cpp.Llama] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "llama-cpp"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools to the model by storing them for use during _generate().

        LangChain's create_agent() calls bind_tools() to attach tools to the model.
        Since llama.cpp models handle tool calling via prompt formatting (not via a
        dedicated API), we store the tools and forward them to llama.chat_completion()
        when _generate() is called.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        stored_tools: list[dict[str, Any]] = list(kwargs.pop("_stored_tools", []))
        for t in tools:
            if isinstance(t, dict):
                stored_tools.append(t)
            elif isinstance(t, BaseTool):
                stored_tools.append(convert_to_openai_tool(t))  # type: ignore[arg-type]
            elif callable(t):
                stored_tools.append(convert_to_openai_tool(t))  # type: ignore[arg-type]

        return self.bind(tools=stored_tools, **kwargs)

    def _ensure_model_loaded(self) -> llama_cpp.Llama:
        """Lazy-load the llama.cpp model on first use."""
        if self._llm is None:
            self._llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                chat_format=self.chat_format,
                verbose=self.verbose,
            )
        return self._llm

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm = self._ensure_model_loaded()

        # Convert LangChain messages -> llama dict format
        llama_messages = _langchain_messages_to_llama(messages)

        # Extract tools from kwargs (passed by LangGraph ToolNode)
        langchain_tools = kwargs.get("tools") or []
        llama_tools = None
        if langchain_tools:
            llama_tools = _convert_langchain_tools_to_llama(langchain_tools)

        # Merge stop tokens
        combined_stop = list(self.stop or [])
        if stop:
            combined_stop.extend(stop)

        # Delegate to llama.chat_completion — its handler manages formatting,
        # grammar enforcement, and response parsing (including tool_calls).
        response = llm.create_chat_completion(
            messages=llama_messages,  # type: ignore[arg-type]
            tools=llama_tools,  # type: ignore[arg-type]
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            top_k=kwargs.get("top_k", self.top_k),
            repeat_penalty=kwargs.get("repeat_penalty", self.repeat_penalty),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=False,
            stop=combined_stop if combined_stop else None,
        )

        ai_message = _llama_chat_response_to_ai_message(response)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        llm = self._ensure_model_loaded()
        llama_messages = _langchain_messages_to_llama(messages)

        langchain_tools = kwargs.get("tools") or []
        llama_tools = None
        if langchain_tools:
            llama_tools = _convert_langchain_tools_to_llama(langchain_tools)

        combined_stop = list(self.stop or [])
        if stop:
            combined_stop.extend(stop)

        prev_text = ""
        for chunk in llm.create_chat_completion(
            messages=llama_messages,  # type: ignore[arg-type]
            tools=llama_tools,  # type: ignore[arg-type]
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            top_k=kwargs.get("top_k", self.top_k),
            repeat_penalty=kwargs.get("repeat_penalty", self.repeat_penalty),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
            stop=combined_stop if combined_stop else None,
        ):
            # llm returns CreateChatCompletionStreamResponse chunks
            choices = chunk.get("choices")  # type: ignore[union-attr]
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                delta_text = content[len(prev_text):] if content.startswith(prev_text) else content
                prev_text = content
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=delta_text)
                )
                if run_manager:
                    run_manager.on_llm_new_token(delta_text)
