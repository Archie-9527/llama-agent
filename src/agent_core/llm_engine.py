"""
LLM 接口引擎层
提供消息格式转换，GBNF 语法注入，Token计数，工具绑定，推理调用等功能
"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

import llama_cpp
import llama_cpp.llama_chat_format as llama_chat_format

from agent_core.exceptions import (
    AgentEngineError,
    InferenceTimeoutError,
    ModelLoadError,
)

# ---------------------------------------------------------------------------
# [INTERNAL] Message-format conversion helpers
# ---------------------------------------------------------------------------

# 将 LangChain 消息格式转换为 llama_cpp 消息格式
def _convert_messages_to_llama_format(
    messages: List[BaseMessage],
) -> List[llama_cpp.llama_types.ChatCompletionRequestMessage]:
    """将 LangChain ``BaseMessage`` 列表通过 ``llama_cpp.Llama.create_chat_completion()`` 转换为 OpenAI 风格的字典列表。
    角色映射:

    * ``SystemMessage`` → ``{"role": "system", "content": ...}``
    * ``HumanMessage``  → ``{"role": "user", "content": ...}``
    * ``AIMessage``     → ``{"role": "assistant", "content": ...}``
      — 如果存在 ``tool_calls``，它们会被序列化到消息中。
    * ``ToolMessage``   → ``{"role": "tool", "content": ..., "tool_call_id": ...}``
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

def _convert_llama_response_to_aimessage(
    response: llama_cpp.llama_types.CreateChatCompletionResponse,
) -> AIMessage:
    """将原始的 llama.cpp 聊天完成响应转换为 ``AIMessage``。

    当响应包含 ``tool_calls`` 时，结构化数据会被解析到 ``AIMessage.tool_calls`` 中，
    以便调用者（例如 LangGraph ToolNode）能够看到标准的 LangChain 结构，
    无需在上游手动解析 JSON。
    """
    choice = response["choices"][0]
    msg_data = choice["message"]
    content = msg_data.get("content") or ""

    tool_calls_data = msg_data.get("tool_calls") or []
    tool_calls: List[ToolCall] = []
    for tc in tool_calls_data:
        func = tc["function"]
        try:
            args = json.loads(func["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}
        tool_calls.append(
            ToolCall(
                name=func["name"],
                args=args,
                id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            )
        )

    # Fallback: parse legacy function_call if present but tool_calls is empty
    if not tool_calls and msg_data.get("function_call"):
        fc = msg_data["function_call"]
        try:
            fc_args = json.loads(fc.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            fc_args = {}
        tool_calls = [
            ToolCall(
                name=fc.get("name", ""),
                args=fc_args,
                id=f"call_{uuid.uuid4().hex[:8]}",
            )
        ]

    return AIMessage(content=content, tool_calls=tool_calls)


def _convert_langchain_tools_to_llama(
    tools: List[Any],
) -> List[llama_cpp.llama_types.ChatCompletionTool]:
    """Convert LangChain tool objects into llama.cpp tool-schema dicts.

    Accepts ``BaseTool`` instances, plain dicts, and callables.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    result: List[llama_cpp.llama_types.ChatCompletionTool] = []
    for tool in tools:
        if isinstance(tool, BaseTool):
            result.append(convert_to_openai_tool(tool))  # type: ignore[arg-type]
        elif isinstance(tool, dict):
            result.append(tool)  # type: ignore[arg-type]
        elif callable(tool):
            result.append(convert_to_openai_tool(tool))  # type: ignore[arg-type]
    return result


# ---------------------------------------------------------------------------
# [STABLE] JSON Schema → GBNF conversion (sole bridge for grammar_builder)
# ---------------------------------------------------------------------------


def compile_json_schema_to_gbnf(schema: dict) -> str:
    """将 JSON Schema 字典转换为 GBNF 语法字符串。
    """
    from llama_cpp.llama_grammar import json_schema_to_gbnf

    return json_schema_to_gbnf(json.dumps(schema))


# ---------------------------------------------------------------------------
# [STABLE] ChatLlamaCpp — the core inference model
# ---------------------------------------------------------------------------


class ChatLlamaCpp(BaseChatModel):

    """ 
    基于LangChain的ChatLlamaCpp类，封装了llama_cpp.Llama实例，提供统一的推理接口。
    该类继承自BaseChatModel，支持消息格式转换、工具绑定、
    这是整个代理系统的唯一推理入口点，使用单例模式，保证线程安全
    """

    # ---- Pydantic fields (construct-time, config-file friendly) ----------
    model_path: str = Field(description="Absolute path to the GGUF model file")
    n_ctx: int = Field(default=4096, description="Context window size in tokens")
    n_gpu_layers: int = Field(
        default=0, description="Number of GPU layers (0 = CPU-only, -1 = all)"
    )
    n_batch: int = Field(default=512, description="Logical batch size for prompt processing")
    n_threads: int = Field(default=8, description="CPU thread count")
    chat_format: str = Field(default="chatml", description="Chat template format")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=40, description="Top-k sampling")
    repeat_penalty: float = Field(default=1.1, description="Repeat penalty")
    max_tokens: int = Field(default=512, description="Max tokens to generate per response")
    stop: Optional[List[str]] = Field(default=None, description="Additional stop tokens")
    verbose: bool = Field(default=False, description="Enable verbose llama.cpp output")
    request_timeout: float = Field(
        default=60.0, description="Max seconds for a single inference call"
    )

    # ---- Private internal state (excluded from pydantic serialisation) ----
    _client: Optional[llama_cpp.Llama] = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Construction — eager model loading (fail-fast)
    # ------------------------------------------------------------------

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._client = self._load_model()  # [INTERNAL] load now, don't defer

    def _load_model(self) -> llama_cpp.Llama:
        """Instantiate the underlying ``llama_cpp.Llama`` object.

        All ``llama_cpp``-originated exceptions are caught and re-raised as
        ``ModelLoadError`` so that callers never see raw library types.
        """
        try:
            return llama_cpp.Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                chat_format=self.chat_format,
                verbose=self.verbose,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from '{self.model_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # BaseChatModel required properties
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-agent"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
        }

    # ------------------------------------------------------------------
    # [STABLE] Token counting
    # ------------------------------------------------------------------

    def get_num_tokens(self, text: str) -> int:
        """Return the exact token count for *text* using the native tokenizer.

        This delegates to ``llama_cpp.Llama.tokenize()`` — no character/4
        approximations.  Used by ``knowledge_scope`` for context-window
        trimming decisions.
        """
        assert self._client is not None
        tokens = self._client.tokenize(text.encode("utf-8"))
        return len(tokens)

    # ------------------------------------------------------------------
    # [STABLE] Tool binding
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable[..., Any], BaseTool]],
        *,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools to the model so they are available during inference.

        LangChain's ``create_agent()`` (and ``ToolNode``) call this to inform
        the model which tools exist.  We store the tool schemas so
        ``_generate`` can forward them to ``llama_cpp``'s native tool-calling
        support.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        stored_tools: List[Dict[str, Any]] = list(kwargs.pop("_stored_tools", []))
        for t in tools:
            if isinstance(t, dict):
                stored_tools.append(t)
            elif isinstance(t, BaseTool):
                stored_tools.append(convert_to_openai_tool(t))  # type: ignore[arg-type]
            elif callable(t):
                stored_tools.append(convert_to_openai_tool(t))  # type: ignore[arg-type]

        return self.bind(tools=stored_tools, **kwargs)

    # ------------------------------------------------------------------
    # [STABLE] Synchronous inference
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Core synchronous inference — called indirectly via ``.invoke()``.

        The *grammar* kwarg (if present) is forwarded to llama.cpp to enable
        constrained decoding (GBNF).  All exceptions from llama.cpp are
        caught and translated into ``AgentEngineError`` subclasses.
        """
        assert self._client is not None

        llama_messages = _convert_messages_to_llama_format(messages)

        # Tools — may come from bind_tools() or be passed at call-time
        langchain_tools = kwargs.get("tools") or []
        llama_tools = (
            _convert_langchain_tools_to_llama(langchain_tools)
            if langchain_tools
            else None
        )

        # Stop tokens
        combined_stop = list(self.stop or [])
        if stop:
            combined_stop.extend(stop)

        # Grammar for constrained decoding
        grammar = kwargs.get("grammar", None)

        # ---- Critical section: llama_cpp.Llama is not thread-safe ----
        try:
            with self._lock:
                response = self._client.create_chat_completion(
                    messages=llama_messages,  # type: ignore[arg-type]
                    tools=llama_tools,  # type: ignore[arg-type]
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=kwargs.get("top_p", self.top_p),
                    top_k=kwargs.get("top_k", self.top_k),
                    repeat_penalty=kwargs.get("repeat_penalty", self.repeat_penalty),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    stream=False,
                    stop=combined_stop if combined_stop else None,
                    grammar=grammar,
                )
        except AgentEngineError:
            raise  # already our type — don't double-wrap
        except Exception as exc:
            raise AgentEngineError(
                f"Inference failed: {exc}"
            ) from exc

        ai_message = _convert_llama_response_to_aimessage(response)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ------------------------------------------------------------------
    # [STABLE] Streaming inference
    # ------------------------------------------------------------------

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Token-level streaming — called indirectly via ``.stream()``."""
        assert self._client is not None

        llama_messages = _convert_messages_to_llama_format(messages)

        langchain_tools = kwargs.get("tools") or []
        llama_tools = (
            _convert_langchain_tools_to_llama(langchain_tools)
            if langchain_tools
            else None
        )

        combined_stop = list(self.stop or [])
        if stop:
            combined_stop.extend(stop)

        grammar = kwargs.get("grammar", None)

        prev_text = ""
        try:
            with self._lock:
                stream = self._client.create_chat_completion(
                    messages=llama_messages,  # type: ignore[arg-type]
                    tools=llama_tools,  # type: ignore[arg-type]
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=kwargs.get("top_p", self.top_p),
                    top_k=kwargs.get("top_k", self.top_k),
                    repeat_penalty=kwargs.get("repeat_penalty", self.repeat_penalty),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    stream=True,
                    stop=combined_stop if combined_stop else None,
                    grammar=grammar,
                )
                for chunk in stream:
                    choices = chunk.get("choices")  # type: ignore[union-attr]
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        # llama.cpp streaming accumulates content — emit deltas
                        delta_text = (
                            content[len(prev_text):]
                            if content.startswith(prev_text)
                            else content
                        )
                        prev_text = content
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content=delta_text)
                        )
                        if run_manager:
                            run_manager.on_llm_new_token(delta_text)
        except AgentEngineError:
            raise
        except Exception as exc:
            raise AgentEngineError(
                f"Streaming inference failed: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# [STABLE] Global singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[ChatLlamaCpp] = None
_engine_lock = threading.Lock()


def get_engine(config: Optional[Dict[str, Any]] = None) -> ChatLlamaCpp:
    """Return (or create) the process-wide ``ChatLlamaCpp`` singleton.

    Loading a GGUF model is expensive (seconds to tens of seconds); every
    ``graph/`` node shares the same instance.  The internal lock guarantees
    that multiple racing callers cannot accidentally load the model twice.
    """
    global _engine_instance

    if _engine_instance is not None:
        return _engine_instance

    with _engine_lock:
        # Double-checked locking — another thread may have created it while we waited
        if _engine_instance is not None:
            return _engine_instance

        cfg = config or {}
        _engine_instance = ChatLlamaCpp(**cfg)
        return _engine_instance
