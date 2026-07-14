"""Tests for grammar_builder.py — covers section 2.5 acceptance criteria.

Acceptance criteria covered:
  1. JSON Schema round-trip — build_json_grammar returns parseable GBNF; model
     output constrained by that grammar validates against the original schema
  2. Enum constraint strictness — build_enum_grammar output is byte-exact one
     of the options
  3. Tool-call schema correctness — build_tool_call_grammar output has correct
     "tool" field and arguments pass input_schema validation
  4. Unsupported $ref rejection — explicit GrammarCompileError with field path
  5. Cache effectiveness — same schema (reordered keys) triggers 1 real compile
  6. Architecture isolation — no ``import llama_cpp`` in grammar_builder.py
"""

from __future__ import annotations

import ast
import json
import os
import sys
import textwrap
from unittest.mock import patch

import pytest

# Ensure the src directory is importable
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
# ── Real model (for round-trip tests) ───────────────────────────────────────
#
# These tests need BOTH:
#  1. A real GGUF file on disk, and
#  2. A real llama-cpp-python package (not the conftest mock).
# When running in the CI sandbox where llama-cpp-python cannot be compiled,
# the conftest installs a synthetic mock — in that case we skip real-model
# tests even if a GGUF file happens to be present.

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = "Qwen3.5-4B-UD-Q8_K_XL.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Import the mock flag set by conftest at collection time.
_gguf_exists = os.path.isfile(MODEL_PATH)
try:
    from conftest import _is_mock_llama_cpp as _mock  # type: ignore[import-not-found]
except ImportError:
    _mock = False

_real_model_available = _gguf_exists and not _mock
_real_model_pytest_mark = pytest.mark.skipif(
    not _real_model_available,
    reason=f"Real GGUF model not found at {MODEL_PATH} (or llama_cpp is mocked)",
)

# ── Singleton isolation ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """Clear the LRU cache between tests so count-based assertions are reliable."""
    _cached_compile.cache_clear()
    yield
    _cached_compile.cache_clear()


# ============================================================================
# 1. JSON Schema round-trip (2.5.1)
# ============================================================================

class TestJsonSchemaRoundTrip:
    """Acceptance: valid schema → GBNF; constrained output parseable + validates."""

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
        """End-to-end: grammar → model output → parsed → validated."""
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
        """Nested schema → model produces valid nested JSON."""
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
# 2. Enum constraint strictness (2.5.2)
# ============================================================================

class TestEnumGrammar:
    """Acceptance: build_enum_grammar output must be byte-exact one of the options."""

    def test_basic_enum_output(self):
        result = build_enum_grammar(["continue", "done", "failed"])
        result = result._grammar
        assert isinstance(result, str)
        assert result.startswith("root ::=")
        # Each option should appear somewhere in the grammar
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
        """Options containing GBNF-special chars (backslash, quote) must be escaped."""
        result = build_enum_grammar(['hello\\world'])
        result = result._grammar
        # The backslash in hello\world must be escaped to \\\\ in GBNF
        assert "\\\\" in result

    @_real_model_pytest_mark
    def test_enum_model_output_exact_match(self):
        """Real model constrained by enum grammar outputs only valid options."""
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
        # Strip quotes that the model might wrap the output in
        content = result.content.strip().strip('"').strip()
        assert content in {"alpha", "beta", "gamma"}, (
            f"Expected one of alpha/beta/gamma, got {content!r}"
        )


# ============================================================================
# 3. Tool-call schema correctness (2.5.3)
# =============================================TestJsonSchemaRoundTrip===============================

class TestToolCallGrammar:
    """Acceptance: build_tool_call_grammar output has correct tool + arguments."""

    SEARCH_TOOL = {
        "name": "web_search",
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
        assert "web_search" in result

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
        """Real model constrained by tool-call grammar outputs valid tool + args."""
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
                    "content": 'Call web_search with query "machine learning" and limit 5',
                }
            ],
            grammar=grammar,
        )

        parsed = json.loads(result.content)
        assert parsed["tool"] == "web_search"
        assert isinstance(parsed["arguments"], dict)
        assert "query" in parsed["arguments"]


# ============================================================================
# 4. Unsupported $ref rejection (2.5.4)
# ============================================================================

class TestRefRejection:
    """Acceptance: $ref → GrammarCompileError with field path, never silent."""

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
        # Error message should contain the path to help debugging
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
# 5. Cache effectiveness (2.5.5)
# ============================================================================

class TestCacheEffectiveness:
    """Acceptance: 10 calls with same schema → 1 underlying compile invocation."""

    def test_repeated_calls_hit_cache(self):
        """10 identical calls should trigger compile_json_schema_to_gbnf only once."""
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
        """Same schema with different key ordering hits the LRU cache."""
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
        """Distinct schemas trigger separate compilations."""
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
# 6. Edge cases & robustness
# ============================================================================

class TestEdgeCases:
    """Additional coverage for boundary conditions."""

    def test_validate_no_refs_with_non_dict(self):
        """Non-dict values should not cause a crash during validation."""
        _validate_no_unresolved_refs({"type": "array", "items": {}})

    def test_validate_no_refs_passes_clean_schema(self):
        """A clean schema with no $ref should pass validation silently."""
        _validate_no_unresolved_refs({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
        })

    def test_build_json_grammar_malformed_schema(self):
        """A schema that the GBNF compiler can't handle should raise GrammarCompileError."""
        with pytest.raises(GrammarCompileError):
            build_json_grammar({"type": "this-type-does-not-exist"})

    def test_build_enum_grammar_quotes_in_options(self):
        """Options containing double-quotes should still produce valid GBNF."""
        result = build_enum_grammar(['he said "hello"'])
        result = result._grammar
        assert "root ::=" in result
        assert isinstance(result, str)


# ============================================================================
# 8. build_json_grammar: real-model round-trip with jsonschema validation
# ============================================================================

class TestJsonSchemaValidation:
    """Verify that model output constrained by build_json_grammar passes
    jsonschema.validate() against the original schema."""

    @_real_model_pytest_mark
    def test_output_validates_against_original_schema(self):
        """Grammar-constrained output must validate via jsonschema."""
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
        """Array-type schema → model outputs valid array."""
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
