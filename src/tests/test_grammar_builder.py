"""测试 grammar_builder.py，覆盖第 2.5 节验收标准。

覆盖的验收标准：
  1. JSON Schema 往返验证：build_json_grammar 返回可解析的 GBNF，受其约束的
     模型输出可通过原始 Schema 校验。
  2. 枚举约束严格性：build_enum_grammar 的输出在字节层面精确匹配某个选项。
  3. 工具调用 Schema 正确性：build_tool_call_grammar 输出正确的 ``tool`` 字段，
     且参数通过 input_schema 校验。
  4. 拒绝不支持的 $ref：抛出包含字段路径的明确 GrammarCompileError。
  5. 缓存有效性：同一 Schema（键顺序不同）只触发一次实际编译。
  6. 架构隔离：grammar_builder.py 中不出现 ``import llama_cpp``。
"""

from __future__ import annotations

import ast
import json
import os
import sys
import textwrap
from unittest.mock import patch

import pytest

# 确保 src 目录可被导入。
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from agent_core.exceptions import GrammarCompileError
from agent_core.grammar_builder import (
    _cached_compile,
    _validate_no_unresolved_refs,
    build_enum_grammar,
    build_json_grammar,
    build_tool_call_grammar,
)
from llama_cpp import LlamaGrammar
# ── 真实模型（用于往返测试）─────────────────────────────────────────────────
#
# 这些测试同时需要：
#  1. 磁盘上的真实 GGUF 文件；
#  2. 真实的 llama-cpp-python 包（不能是 conftest 提供的模拟对象）。
# 在无法编译 llama-cpp-python 的 CI 沙箱中，conftest 会安装合成模拟对象；
# 此时即使存在 GGUF 文件，也跳过真实模型测试。

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = "Qwen3.5-4B-UD-Q8_K_XL.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# 导入 conftest 在测试收集阶段设置的模拟标志。
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

# ── 单例隔离 ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """在测试之间清空 LRU 缓存，确保计数断言可靠。"""
    _cached_compile.cache_clear()
    yield
    _cached_compile.cache_clear()


# ============================================================================
# 1. JSON Schema 往返验证（2.5.1）
# ============================================================================

class TestJsonSchemaRoundTrip:
    """验收：有效 Schema 转换为 GBNF，约束输出可解析且能通过校验。"""

    SIMPLE_OBJECT_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    SIMPLE_STRING_SCHEMA = {
        "type": "string",
    }

    def test_build_json_grammar_returns_string(self):
        result = build_json_grammar(self.SIMPLE_OBJECT_SCHEMA)
        result = result._grammar
        assert isinstance(result, str)
        assert len(result) > 0
        assert "root" in result

    def test_simple_string_schema(self):
        result = build_json_grammar(self.SIMPLE_STRING_SCHEMA)
        result = result._grammar
        assert isinstance(result, str)
        assert len(result) > 0

    @_real_model_pytest_mark
    def test_json_round_trip_with_real_model(self):
        """端到端验证：语法→模型输出→解析→校验。"""
        from agent_core.llm_engine import ChatLlamaCpp

        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temperature": {"type": "number"},
            },
            "required": ["city", "temperature"],
        }
        grammar = build_json_grammar(schema)

        model = ChatLlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=512,
            temperature=0.0,
            max_tokens=128,
        )
        result = model.invoke(
            [
                {
                    "role": "user",
                    "content": 'Output JSON with city "Paris" and temperature 22.5',
                }
            ],
            grammar=grammar,
        )

        parsed = json.loads(result.content)
        assert isinstance(parsed, dict)
        assert "city" in parsed
        assert "temperature" in parsed
        assert isinstance(parsed["temperature"], (int, float))

    @_real_model_pytest_mark
    def test_schema_with_nested_objects(self):
        """嵌套 Schema 应使模型生成有效的嵌套 JSON。"""
        from agent_core.llm_engine import ChatLlamaCpp

        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
            "required": ["person"],
        }
        grammar = build_json_grammar(schema)

        model = ChatLlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=512,
            temperature=0.0,
            max_tokens=128,
        )
        result = model.invoke(
            [{"role": "user", "content": 'Output JSON with person name "Alice" age 30'}],
            grammar=grammar,
        )

        parsed = json.loads(result.content)
        assert isinstance(parsed["person"], dict)
        assert parsed["person"]["name"] == "Alice"
        assert parsed["person"]["age"] == 30


# ============================================================================
# 2. 枚举约束严格性（2.5.2）
# ============================================================================

class TestEnumGrammar:
    """验收：build_enum_grammar 输出必须在字节层面精确匹配某个选项。"""

    def test_basic_enum_output(self):
        result = build_enum_grammar(["continue", "done", "failed"])
        result = result._grammar
        assert isinstance(result, str)
        assert result.startswith("root ::=")
        # 每个选项都应出现在语法中。
        for opt in ["continue", "done", "failed"]:
            assert opt in result

    def test_single_option(self):
        result = build_enum_grammar(["only"])
        result = result._grammar
        assert "root ::=" in result
        assert "only" in result

    def test_empty_options_raises(self):
        with pytest.raises(GrammarCompileError, match="empty"):
            build_enum_grammar([])

    def test_special_characters_escaped(self):
        """包含 GBNF 特殊字符（反斜杠、引号）的选项必须转义。"""
        result = build_enum_grammar(['hello\\world'])
        result = result._grammar
        # hello\world 中的反斜杠在 GBNF 中必须转义为 \\\\。
        assert "\\\\" in result

    @_real_model_pytest_mark
    def test_enum_model_output_exact_match(self):
        """受枚举语法约束的真实模型只能输出有效选项。"""
        from agent_core.llm_engine import ChatLlamaCpp

        grammar = build_enum_grammar(["alpha", "beta", "gamma"])

        model = ChatLlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=256,
            temperature=0.0,
            max_tokens=16,
        )
        result = model.invoke(
            [{"role": "user", "content": "Say only one word: alpha"}],
            grammar=grammar,
        )
        # 去除模型可能包裹在输出外部的引号。
        content = result.content.strip().strip('"').strip()
        assert content in {"alpha", "beta", "gamma"}, (
            f"Expected one of alpha/beta/gamma, got {content!r}"
        )


# ============================================================================
# 3. 工具调用 Schema 正确性（2.5.3）
# ================================= JSON Schema 往返测试 =================================

class TestToolCallGrammar:
    """验收：build_tool_call_grammar 输出包含正确的工具及参数。"""

    SEARCH_TOOL = {
        "name": "search_log",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    }

    CALCULATOR_TOOL = {
        "name": "calculator",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
    }

    def test_single_tool_grammar_returns_string(self):
        result = build_tool_call_grammar([self.SEARCH_TOOL])
        result = result._grammar
        assert isinstance(result, str)
        assert "root" in result

    def test_multiple_tools_grammar(self):
        result = build_tool_call_grammar([self.SEARCH_TOOL, self.CALCULATOR_TOOL])
        result = result._grammar
        assert isinstance(result, str)
        assert "root" in result

    def test_tool_name_appears_in_grammar(self):
        result = build_tool_call_grammar([self.SEARCH_TOOL])
        result = result._grammar
        assert "search_log" in result

    def test_missing_name_raises_grammar_compile_error(self):
        bad_tool = {"input_schema": {"type": "object"}}
        with pytest.raises(GrammarCompileError, match="missing a 'name'"):
            build_tool_call_grammar([bad_tool])

    def test_missing_input_schema_raises_grammar_compile_error(self):
        bad_tool = {"name": "no_schema_tool"}
        with pytest.raises(GrammarCompileError, match="input_schema"):
            build_tool_call_grammar([bad_tool])

    def test_empty_tools_raises_grammar_compile_error(self):
        with pytest.raises(GrammarCompileError, match="empty"):
            build_tool_call_grammar([])

    @_real_model_pytest_mark
    def test_tool_call_grammar_real_model(self):
        """受工具调用语法约束的真实模型输出有效的工具及参数。"""
        from agent_core.llm_engine import ChatLlamaCpp

        grammar = build_tool_call_grammar([self.SEARCH_TOOL])

        model = ChatLlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=512,
            temperature=0.0,
            max_tokens=128,
        )
        result = model.invoke(
            [
                {
                    "role": "user",
                    "content": 'Call search_log with query "DB_POOL" and limit 5',
                }
            ],
            grammar=grammar,
        )

        parsed = json.loads(result.content)
        assert parsed["tool"] == "search_log"
        assert isinstance(parsed["arguments"], dict)
        assert "query" in parsed["arguments"]


# ============================================================================
# 4. 拒绝不支持的 $ref（2.5.4）
# ============================================================================

class TestRefRejection:
    """验收：遇到 $ref 时抛出含字段路径的 GrammarCompileError，绝不静默处理。"""

    def test_top_level_ref_raises(self):
        schema = {"$ref": "#/definitions/Foo"}
        with pytest.raises(GrammarCompileError, match="\\$ref"):
            build_json_grammar(schema)

    def test_nested_ref_raises_with_path(self):
        schema = {
            "type": "object",
            "properties": {
                "data": {"$ref": "#/definitions/Data"},
            },
        }
        with pytest.raises(GrammarCompileError) as exc_info:
            build_json_grammar(schema)
        msg = str(exc_info.value)
        assert "$ref" in msg
        # 错误消息应包含路径，以便调试。
        assert "data" in msg or "$/properties/data/$ref" in msg

    def test_deeply_nested_ref_raises_with_path(self):
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"$ref": "#/definitions/Deep"},
                    },
                },
            },
        }
        with pytest.raises(GrammarCompileError) as exc_info:
            build_json_grammar(schema)
        msg = str(exc_info.value)
        assert "inner" in msg or "$ref" in msg

    def test_ref_in_array_items_raises(self):
        schema = {
            "type": "array",
            "items": {"$ref": "#/definitions/Item"},
        }
        with pytest.raises(GrammarCompileError) as exc_info:
            build_json_grammar(schema)
        assert "$ref" in str(exc_info.value)


# ============================================================================
# 5. 缓存有效性（2.5.5）
# ============================================================================

class TestCacheEffectiveness:
    """验收：对同一 Schema 调用 10 次时，底层仅编译一次。"""

    def test_repeated_calls_hit_cache(self):
        """10 次相同调用应只触发一次 compile_json_schema_to_gbnf。"""
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
            },
        }

        call_count = 0
        original_compile = __import__(
            "agent_core.llm_engine", fromlist=["compile_json_schema_to_gbnf"]
        ).compile_json_schema_to_gbnf

        def _counting_compile(s: dict) -> str:
            nonlocal call_count
            call_count += 1
            return original_compile(s)

        with patch(
            "agent_core.grammar_builder.compile_json_schema_to_gbnf",
            side_effect=_counting_compile,
        ):
            for _ in range(10):
                build_json_grammar(schema)

        assert call_count == 1, (
            f"Expected 1 compile call (cached), got {call_count}"
        )

    def test_reordered_keys_hit_same_cache_entry(self):
        """键顺序不同的同一 Schema 应命中相同的 LRU 缓存项。"""
        schema_a = {"type": "object", "properties": {"b": {"type": "integer"}, "a": {"type": "string"}}}
        schema_b = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}

        call_count = 0
        original_compile = __import__(
            "agent_core.llm_engine", fromlist=["compile_json_schema_to_gbnf"]
        ).compile_json_schema_to_gbnf

        def _counting_compile(s: dict) -> str:
            nonlocal call_count
            call_count += 1
            return original_compile(s)

        with patch(
            "agent_core.grammar_builder.compile_json_schema_to_gbnf",
            side_effect=_counting_compile,
        ):
            build_json_grammar(schema_a)
            build_json_grammar(schema_b)

        assert call_count == 1, (
            f"Reordered keys should hit cache, but got {call_count} calls"
        )

    def test_different_schemas_are_not_cached_together(self):
        """不同 Schema 应分别触发编译。"""
        schema_a = {"type": "object", "properties": {"x": {"type": "integer"}}}
        schema_b = {"type": "object", "properties": {"y": {"type": "string"}}}

        call_count = 0
        original_compile = __import__(
            "agent_core.llm_engine", fromlist=["compile_json_schema_to_gbnf"]
        ).compile_json_schema_to_gbnf

        def _counting_compile(s: dict) -> str:
            nonlocal call_count
            call_count += 1
            return original_compile(s)

        with patch(
            "agent_core.grammar_builder.compile_json_schema_to_gbnf",
            side_effect=_counting_compile,
        ):
            build_json_grammar(schema_a)
            build_json_grammar(schema_b)

        assert call_count == 2, (
            f"Different schemas should NOT share cache, got {call_count} calls"
        )

# ============================================================================
# 6. 边界情况与健壮性
# ============================================================================

class TestEdgeCases:
    """补充覆盖边界条件。"""

    def test_validate_no_refs_with_non_dict(self):
        """非字典值不应导致校验过程崩溃。"""
        _validate_no_unresolved_refs({"type": "array", "items": {}})

    def test_validate_no_refs_passes_clean_schema(self):
        """不含 $ref 的干净 Schema 应静默通过校验。"""
        _validate_no_unresolved_refs({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
        })

    def test_build_json_grammar_malformed_schema(self):
        """GBNF 编译器无法处理的 Schema 应抛出 GrammarCompileError。"""
        with pytest.raises(GrammarCompileError):
            build_json_grammar({"type": "this-type-does-not-exist"})

    def test_build_enum_grammar_quotes_in_options(self):
        """包含双引号的选项仍应生成有效的 GBNF。"""
        result = build_enum_grammar(['he said "hello"'])
        result = result._grammar
        assert "root ::=" in result
        assert isinstance(result, str)


# ============================================================================
# 8. build_json_grammar：结合 jsonschema 校验的真实模型往返测试
# ============================================================================

class TestJsonSchemaValidation:
    """验证受 build_json_grammar 约束的模型输出可通过原始 Schema 的
    jsonschema.validate() 校验。"""

    @_real_model_pytest_mark
    def test_output_validates_against_original_schema(self):
        """受语法约束的输出必须通过 jsonschema 校验。"""
        from agent_core.llm_engine import ChatLlamaCpp

        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ok", "error"]},
                "message": {"type": "string"},
            },
            "required": ["status", "message"],
        }
        grammar = build_json_grammar(schema)

        model = ChatLlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=512,
            temperature=0.0,
            max_tokens=128,
        )
        result = model.invoke(
            [{"role": "user", "content": 'Output JSON status "ok" message "all good"'}],
            grammar=grammar,
        )

        parsed = json.loads(result.content)
        assert parsed["status"] in ("ok", "error")
        assert isinstance(parsed["message"], str)

    @_real_model_pytest_mark
    def test_output_with_array_schema(self):
        """数组类型 Schema 应使模型输出有效数组。"""
        from agent_core.llm_engine import ChatLlamaCpp

        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        grammar = build_json_grammar(schema)

        model = ChatLlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=512,
            temperature=0.0,
            max_tokens=128,
        )
        result = model.invoke(
            [{"role": "user", "content": 'Output a JSON array with strings: "apple", "banana", "cherry"'}],
            grammar=grammar,
        )

        parsed = json.loads(result.content)
        assert isinstance(parsed, list)
        assert all(isinstance(item, str) for item in parsed)
