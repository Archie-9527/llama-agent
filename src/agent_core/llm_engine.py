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
    EngineAlreadyInitializedError,
    EngineConfigError,
    EngineNotInitializedError,
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

        content: str
        if isinstance(msg.content, str):
            content = msg.content
        else:
            # 某些消息类型 content 可能是 list[dict]（多模态），此处转为 JSON 字符串兜底
            content = json.dumps(msg.content, ensure_ascii=False)

        entry: llama_cpp.llama_types.ChatCompletionRequestMessage = {
            "role": role,  # type: ignore[typeddict-item]
            "content": content,
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


def _ensure_tool_messages_visible(
    messages: List[llama_cpp.llama_types.ChatCompletionRequestMessage],
) -> List[llama_cpp.llama_types.ChatCompletionRequestMessage]:
    """将 ``role="tool"`` 的消息改写为 ``role="user"``，防止简单 chat format
    处理器（如 ``format_qwen``）在 ``tools=None`` 时静默丢弃工具结果。

    当传入 ``tools`` 参数时不应调用此函数——llama-cpp-python 会走 Jinja2 模板
    路径，原生支持 tool role。
    """
    rewritten: List[llama_cpp.llama_types.ChatCompletionRequestMessage] = []
    for msg in messages:
        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            tool_content = msg.get("content", "")
            rewritten.append({
                "role": "user",
                "content": (
                    f"[工具执行结果 | tool_call_id={tool_call_id}]\n{tool_content}"
                ),
            })  # type: ignore[arg-type]
        else:
            rewritten.append(msg)
    return rewritten

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

        stored_tools: List[Dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                stored_tools.append(t)
            elif isinstance(t, BaseTool):
                stored_tools.append(convert_to_openai_tool(t))  # type: ignore[arg-type]
            elif callable(t):
                stored_tools.append(convert_to_openai_tool(t))  # type: ignore[arg-type]
        bind_kwargs = dict(kwargs)
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice

        return self.bind(tools=stored_tools, **bind_kwargs)

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

        # 当 tools=None 时，简单 chat format 处理器会丢弃 role="tool" 的消息。
        # 将 tool 消息改写为 user 消息，确保工具执行结果不被静默丢失。
        if llama_tools is None:
            llama_messages = _ensure_tool_messages_visible(llama_messages)

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
                    tool_choice=kwargs.get("tool_choice"),
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

        # 当 tools=None 时，将 tool 消息改写为 user 消息（同 _generate 的逻辑）
        if llama_tools is None:
            llama_messages = _ensure_tool_messages_visible(llama_messages)

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
# [STABLE] EngineConfig — typed contract for ChatLlamaCpp construction
# ---------------------------------------------------------------------------

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class EngineConfig:
    """Typed, immutable configuration for ``ChatLlamaCpp``.

    This is the single place that declares which parameters the engine
    needs.  Field names must stay in sync with ``ChatLlamaCpp``'s
    pydantic fields.

    ``model_path`` is the only field without a default — it must be
    provided by the caller.
    """

    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    n_batch: int = 512
    n_threads: int = 8
    chat_format: str = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    verbose: bool = False
    request_timeout: float = 60.0

    def validate(self) -> None:
        """Fail-fast check before constructing the expensive ``ChatLlamaCpp``.

        Raises:
            EngineConfigError: If ``model_path`` is empty or whitespace-only.
        """
        if not self.model_path or not self.model_path.strip():
            raise EngineConfigError(
                "EngineConfig.model_path is missing or empty. "
                "Provide it via the [engine] section of your TOML config file, "
                "the AGENT_MODEL_PATH environment variable, or the "
                "--model-path CLI argument."
            )


# ---------------------------------------------------------------------------
# [STABLE] Global singleton — initialise once, get many times
# ---------------------------------------------------------------------------

_engine_instance: Optional[ChatLlamaCpp] = None
_engine_lock = threading.Lock()


def initialize_engine(config: EngineConfig) -> ChatLlamaCpp:
    """One-shot, process-level engine initialisation.

    Must be called by the CLI layer (``cli.py``) during startup, before
    any ``graph/`` node attempts to call ``get_engine()``.

    Args:
        config: Fully merged ``EngineConfig`` (all four layers resolved).

    Returns:
        The newly created ``ChatLlamaCpp`` singleton instance.

    Raises:
        EngineConfigError: ``config.model_path`` is missing or empty.
        EngineAlreadyInitializedError: The singleton was already created.
        ModelLoadError: The underlying ``llama_cpp.Llama`` constructor failed.
    """
    global _engine_instance

    config.validate()

    with _engine_lock:
        if _engine_instance is not None:
            raise EngineAlreadyInitializedError(
                "initialize_engine() has already been called — the engine "
                "singleton exists and must not be silently replaced.  "
                "Use _reset_engine_for_testing() in test fixtures."
            )
        _engine_instance = ChatLlamaCpp(**asdict(config))
        return _engine_instance


def get_engine() -> ChatLlamaCpp:
    """Return the previously initialised engine singleton.

    This function accepts **no arguments**.  If the engine has not been
    initialised yet it raises ``EngineNotInitializedError`` rather than
    silently constructing one with default (empty) parameters.

    Raises:
        EngineNotInitializedError: ``initialize_engine()`` has not been
            called yet.
    """
    if _engine_instance is None:
        raise EngineNotInitializedError(
            "get_engine() was called before initialize_engine() completed.  "
            "Make sure cli.py calls initialize_engine(EngineConfig(...)) "
            "during startup before any graph node tries to invoke the LLM."
        )
    return _engine_instance


def _reset_engine_for_testing() -> None:
    """[TEST-ONLY] Clear the singleton so a fresh engine can be created.

    Do **not** call this from production code.  The leading underscore
    and the name suffix are deliberate signals that this is a test hook.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
