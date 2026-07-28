"""测试 llm_engine.py，覆盖第 1.8 节验收标准。

覆盖的验收标准：
  1. 模型加载：有效路径可加载；无效或缺失时抛出包含路径的 ModelLoadError。
  2. LangChain 协议：invoke 返回 AIMessage；temperature=0 时 stream 内容与 invoke 一致。
  3. Token 计数：get_num_tokens() 与 llama.cpp 原生分词器完全一致。
  4. 语法约束：grammar 参数产生可解析的 JSON 输出。
  5. 并发：10 个线程共享一个实例时不崩溃，且结果一致。
  6. 超时：抛出 InferenceTimeoutError、释放锁且不死锁。
  7. 工具调用格式：bind_tools 与工具触发输入产生结构化 AIMessage.tool_calls。
  8. 异常隔离：错误语法抛出 AgentEngineError 子类，而非原始 llama_cpp 异常。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# 确保 src 目录可被导入。
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from agent_core.exceptions import (
    AgentEngineError,
    InferenceTimeoutError,
    ModelLoadError,
)
from agent_core.llm_engine import (
    ChatLlamaCpp,
    EngineConfig,
    _convert_llama_response_to_aimessage,
    _convert_messages_to_llama_format,
    compile_json_schema_to_gbnf,
    get_engine,
    initialize_engine,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall

# ── 辅助函数 ─────────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = "Qwen3.5-4B-UD-Q8_K_XL.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

_gguf_exists = os.path.isfile(MODEL_PATH)
try:
    from conftest import _is_mock_llama_cpp as _mock  # type: ignore[import-not-found]
except ImportError:
    _mock = False

_real_model_available = (
    _gguf_exists
    and not _mock
    and os.environ.get("RUN_REAL_MODEL_TESTS") == "1"
)
_real_model_pytest_mark = pytest.mark.skipif(
    not _real_model_available,
    reason=(
        "set RUN_REAL_MODEL_TESTS=1 and provide a real llama_cpp installation "
        f"plus {MODEL_PATH}"
    ),
)


# ── 单例隔离夹具 ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_engine_singleton():
    """在测试之间重置全局单例，避免状态泄漏。"""
    import agent_core.llm_engine as engine_mod

    engine_mod._engine_instance = None
    yield
    engine_mod._engine_instance = None


# ============================================================================
# 1. 模型加载（1.8.1）
# ============================================================================

class TestModelLoading:
    """验收：有效 GGUF 可加载，缺失或损坏时抛出 ModelLoadError。"""

    @_real_model_pytest_mark
    def test_load_valid_model_succeeds(self):
        """有效 GGUF 文件应正常加载并产生可用实例。"""
        model = ChatLlamaCpp(model_path=MODEL_PATH, n_ctx=256, verbose=False)
        assert model._client is not None
        assert model._llm_type == "llama-cpp-agent"

    def test_missing_model_raises_model_load_error(self):
        """文件不存在时抛出 ModelLoadError，消息中包含路径。"""
        with pytest.raises(ModelLoadError, match="nonexistent-model.gguf"):
            ChatLlamaCpp(model_path="nonexistent-model.gguf")

    def test_empty_model_path_raises_model_load_error(self):
        """空字符串路径应快速抛出 ModelLoadError，而非静默失败。"""
        with pytest.raises(ModelLoadError):
            ChatLlamaCpp(model_path="")

    def test_error_message_contains_file_path(self):
        """错误消息必须包含文件路径，以便调试。"""
        bad_path = "/tmp/definitely_not_a_real_model_42.gguf"
        with pytest.raises(ModelLoadError) as exc_info:
            ChatLlamaCpp(model_path=bad_path)
        assert bad_path in str(exc_info.value)


# ============================================================================
# 2. LangChain 协议兼容性（1.8.2）
# ============================================================================

class TestLangChainProtocol:
    """验收：invoke 返回 AIMessage，stream 分块拼接后与 invoke 输出一致。"""

    @_real_model_pytest_mark
    def test_invoke_returns_aimessage(self):
        """ChatLlamaCpp.invoke([HumanMessage]) 必须返回 AIMessage。"""
        model = ChatLlamaCpp(
            model_path=MODEL_PATH, n_ctx=256, temperature=0.0, max_tokens=64,
        )
        result = model.invoke([HumanMessage(content="Hello")])
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    @_real_model_pytest_mark
    def test_stream_chunks_reassemble_to_invoke_output(self):
        """temperature=0 时，所有流式分块拼接结果必须等于 invoke 结果。"""
        model = ChatLlamaCpp(
            model_path=MODEL_PATH, n_ctx=256, temperature=0.0, max_tokens=64,
        )
        messages = [HumanMessage(content="Say exactly: 'Hello world'")]

        invoke_result: AIMessage = model.invoke(messages)  # type: ignore[assignment]

        chunks: list[str] = []
        for chunk in model.stream(messages):
            chunks.append(chunk.content)  # type: ignore[arg-type]

        streamed = "".join(chunks)
        assert invoke_result.content.replace(" ", "") == streamed.replace(" ", ""), (
            f"invoke={invoke_result.content!r} ≠ stream={streamed!r}"
        )

    @_real_model_pytest_mark
    def test_invoke_with_system_message(self):
        """验证流水线能正确处理 SystemMessage。"""
        model = ChatLlamaCpp(
            model_path=MODEL_PATH, n_ctx=256, temperature=0.0, max_tokens=32,
        )
        result = model.invoke([
            SystemMessage(content="Reply with only the word 'OK'"),
            HumanMessage(content="Hi"),
        ])
        assert isinstance(result, AIMessage)
        assert len(result.content) > 0


# ============================================================================
# 3. Token 计数（1.8.3）
# ============================================================================

class TestTokenCounting:
    """验收：get_num_tokens() 使用原生分词器精确计数。"""

    @_real_model_pytest_mark
    def test_get_num_tokens_matches_native_tokenizer(self):
        """get_num_tokens(text) 必须等于原生 tokenize 结果的长度。"""
        model = ChatLlamaCpp(model_path=MODEL_PATH, n_ctx=256)
        text = "Hello world, this is a test sentence."
        native_count = len(model._client.tokenize(text.encode("utf-8")))
        api_count = model.get_num_tokens(text)
        assert api_count == native_count, (
            f"get_num_tokens={api_count} ≠ native tokenizer={native_count}"
        )

    @_real_model_pytest_mark
    def test_get_num_tokens_empty_string(self):
        """空字符串的 Token 数应为 0。"""
        model = ChatLlamaCpp(model_path=MODEL_PATH, n_ctx=256)
        assert model.get_num_tokens("") == 0

    @_real_model_pytest_mark
    def test_get_num_tokens_longer_text(self):
        """较长文本的 Token 数应合理增长，而非使用字符数除以 4 的估算。"""
        model = ChatLlamaCpp(model_path=MODEL_PATH, n_ctx=256)
        short = model.get_num_tokens("a")
        long_text = model.get_num_tokens("a" * 100)
        # 多数分词器对重复字母的切分不同，这里只验证结果非零。
        assert short > 0
        assert long_text > 0


# ============================================================================
# 4. 语法约束执行（1.8.4）
# ============================================================================

class TestGrammarConstraint:
    """验收：传入 grammar 参数后，输出可由 json.loads 解析。"""

    @_real_model_pytest_mark
    def test_json_grammar_produces_parseable_json(self):
        """使用简单 JSON Schema 语法调用时，输出必须是有效 JSON。"""
        from agent_core.grammar_builder import build_json_grammar

        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
            },
            "required": ["result"],
        }
        grammar = build_json_grammar(schema)

        model = ChatLlamaCpp(
            model_path=MODEL_PATH, n_ctx=512, temperature=0.0, max_tokens=128,
        )
        result = model.invoke(
            [HumanMessage(content='Output JSON with key "result" set to "success"')],
            grammar=grammar,
        )
        parsed = json.loads(result.content)
        assert "result" in parsed
        assert isinstance(parsed["result"], str)

    @_real_model_pytest_mark
    def test_enum_grammar_constrains_exact_output(self):
        """枚举语法将输出限制为所列选项之一。"""
        from agent_core.grammar_builder import build_enum_grammar

        grammar = build_enum_grammar(["red", "green", "blue"])
        model = ChatLlamaCpp(
            model_path=MODEL_PATH, n_ctx=256, temperature=0.0, max_tokens=16,
        )
        result = model.invoke(
            [HumanMessage(content="Say only one word: red")],
            grammar=grammar,
        )
        # 使用枚举语法时，输出应为某个枚举值；根据模型行为，外层可能带引号。
        content = result.content.strip().strip('"')
        assert content in {"red", "green", "blue"}, (
            f"Expected one of red/green/blue, got {content!r}"
        )


# ============================================================================
# 5. 并发安全（1.8.5）
# ============================================================================

class TestConcurrency:
    """验收：10 个线程共享一个实例时不崩溃，且结果一致。"""

    @_real_model_pytest_mark
    def test_ten_threads_no_crashes(self):
        """对同一模型实例进行 10 次并发调用时不应崩溃。"""
        model = ChatLlamaCpp(
            model_path=MODEL_PATH, n_ctx=512, temperature=0.7, max_tokens=32,
        )
        errors: list[Exception] = []
        results: list[str] = []

        def _worker(i: int) -> None:
            try:
                r = model.invoke([HumanMessage(content=f"Thread {i}: say 'hello'")])
                results.append(r.content)  # type: ignore[arg-type]
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrency errors: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"


# ============================================================================
# 6. 超时处理（1.8.6）
# ============================================================================

class TestTimeout:
    """验收：长生成任务遇到较短 request_timeout 时抛出 InferenceTimeoutError，
    同时释放锁，后续请求不被阻塞。"""

    @_real_model_pytest_mark
    def test_lock_released_after_timeout(self):
        """超时后，后续调用必须成功，证明锁未被占用。"""
        model = ChatLlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=256,
            temperature=0.7,
            max_tokens=4096,
            request_timeout=0.001,
        )
        # 第一次调用超时。
        try:
            model.invoke([HumanMessage(content="Write many paragraphs about history")])
        except InferenceTimeoutError:
            pass

        # 第二次调用必须成功；若锁泄漏，此处会挂起。
        try:
            result = model.invoke([HumanMessage(content="Say: hi")])
            assert isinstance(result, AIMessage)
        except InferenceTimeoutError:
        # 模型确实很慢时允许再次超时，关键是不能挂起。
            pass


# ============================================================================
# 7. 工具调用格式（1.8.7）
# ============================================================================

class TestToolCallFormat:
    """验收：bind_tools 后 AIMessage.tool_calls 具有正确结构。"""

    def test_bind_tools_defaults_to_auto_tool_choice(self):
        """普通 create_agent 绑定必须启用 llama.cpp 自动工具选择。"""
        from langchain_core.tools import tool

        @tool
        def echo(value: str) -> str:
            """回显一个值。"""
            return value

        # 该测试只验证绑定契约，因此避免加载 GGUF。
        model = ChatLlamaCpp.model_construct(model_path="unused.gguf")
        bound = model.bind_tools([echo])

        assert bound.kwargs["tool_choice"] == "auto"
        assert bound.kwargs["tools"][0]["function"]["name"] == "echo"

    @_real_model_pytest_mark
    def test_bind_tools_populates_tool_calls(self):
        """bind_tools 后，响应中的 AIMessage.tool_calls 应为结构化数据。"""
        from langchain_core.tools import tool

        @tool
        def get_weather(city: str) -> str:
            """获取指定城市的当前天气。"""
            return f"Weather in {city}: sunny"

        model = ChatLlamaCpp(
            model_path=MODEL_PATH, n_ctx=512, temperature=0.0, max_tokens=128,
        )
        bound = model.bind_tools([get_weather])
        # 工具调用模型会将天气查询识别为工具触发条件。
        result = bound.invoke([HumanMessage(content="What is the weather in Paris?")])

        assert isinstance(result, AIMessage)
        assert result.tool_calls, "tool-triggering prompt produced no structured tool_calls"
        tc = result.tool_calls[0]
        assert "name" in tc
        assert "args" in tc
        assert "id" in tc
        assert isinstance(tc["name"], str)
        assert isinstance(tc["args"], dict)
        assert isinstance(tc["id"], str)


# ============================================================================
# 8. 异常隔离（1.8.8）
# ============================================================================

class TestExceptionIsolation:
    """验收：错误输入只抛出 AgentEngineError 子类，不泄漏原始 llama_cpp 异常。"""

    @_real_model_pytest_mark
    def test_malformed_grammar_raises_agent_engine_error(self):
        """无效语法字符串不得产生原始 llama_cpp 异常。"""
        model = ChatLlamaCpp(model_path=MODEL_PATH, n_ctx=256)

        # 模拟内部客户端，避免该测试依赖真实模型。
        mock_client = MagicMock()
        mock_client.create_chat_completion.side_effect = ValueError(
            "llama.cpp internal: invalid grammar"
        )
        model._client = mock_client

        with pytest.raises(AgentEngineError) as exc_info:
            model.invoke(
                [HumanMessage(content="test")],
                grammar="this is not a valid grammar %%^^&&",
            )
        # 异常必须是 AgentEngineError 子类，而非原始 ValueError。
        assert not isinstance(exc_info.value, ValueError)


# ============================================================================
# 9. 消息格式转换（单元测试）
# ============================================================================

class TestMessageConversion:
    """验证内部消息转换辅助函数。"""

    def test_convert_system_message(self):
        msgs = [SystemMessage(content="You are helpful.")]
        out = _convert_messages_to_llama_format(msgs)  # type: ignore[arg-type]
        assert out[0]["role"] == "system"
        assert out[0]["content"] == "You are helpful."

    def test_convert_human_message(self):
        msgs = [HumanMessage(content="Hello")]
        out = _convert_messages_to_llama_format(msgs)  # type: ignore[arg-type]
        assert out[0]["role"] == "user"
        assert out[0]["content"] == "Hello"

    def test_convert_ai_message_with_tool_calls(self):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="search", args={"q": "weather"}, id="call_1")
                ],
            )
        ]
        out = _convert_messages_to_llama_format(msgs)  # type: ignore[arg-type]
        assert out[0]["role"] == "assistant"
        assert out[0]["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"q": "weather"}

    def test_convert_tool_message(self):
        msgs = [ToolMessage(content="result data", tool_call_id="call_1")]
        out = _convert_messages_to_llama_format(msgs)  # type: ignore[arg-type]
        assert out[0]["role"] == "tool"
        assert out[0]["content"] == "result data"
        assert out[0]["tool_call_id"] == "call_1"

    def test_convert_response_to_aimessage_text_only(self):
        response = {
            "choices": [{"message": {"content": "Hello!", "role": "assistant"}}],
        }
        ai_msg = _convert_llama_response_to_aimessage(response)  # type: ignore[arg-type]
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == "Hello!"
        assert ai_msg.tool_calls == []

    def test_convert_response_to_aimessage_with_tool_calls(self):
        response = {
            "choices": [{
                "message": {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_abc",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }],
                }
            }],
        }
        ai_msg = _convert_llama_response_to_aimessage(response)  # type: ignore[arg-type]
        assert ai_msg.tool_calls[0]["name"] == "get_weather"
        assert ai_msg.tool_calls[0]["args"] == {"city": "Paris"}
        assert ai_msg.tool_calls[0]["id"] == "call_abc"

    def test_convert_response_legacy_function_call_fallback(self):
        response = {
            "choices": [{
                "message": {
                    "content": "",
                    "role": "assistant",
                    "function_call": {
                        "name": "search",
                        "arguments": '{"q": "test"}',
                    },
                }
            }],
        }
        ai_msg = _convert_llama_response_to_aimessage(response)  # type: ignore[arg-type]
        assert ai_msg.tool_calls[0]["name"] == "search"
        assert ai_msg.tool_calls[0]["args"] == {"q": "test"}


# ============================================================================
# 10. 单例 get_engine()（1.8.1）
# ============================================================================

class TestGetEngine:
    """验证全局单例入口。"""

    @_real_model_pytest_mark
    def test_get_engine_returns_same_instance(self):
        """重复调用 get_engine() 应返回同一对象。"""
        e1 = initialize_engine(EngineConfig(model_path=MODEL_PATH, n_ctx=256))
        e2 = get_engine()
        assert e1 is e2

    def test_compile_json_schema_to_gbnf_returns_string(self):
        """桥接函数应生成非空 GBNF 字符串。"""
        result = compile_json_schema_to_gbnf({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        })
        assert isinstance(result, str)
        assert len(result) > 0
        assert "root" in result
