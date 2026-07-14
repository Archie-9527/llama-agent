"""Shared pytest fixtures and module-level mocks for the llama-agent test suite.

The VM sandbox may not have llama-cpp-python (no pre-built wheel, source
build timeout).  This conftest provides a synthetic ``llama_cpp`` module
so that pure unit tests (grammar_builder, message conversion, exception
isolation) can run without the real native library.

Tests that need a real GGUF model are guarded by ``@pytest.mark.skipif``
with a ``_real_model_available`` check and are auto-skipped when the
model or the native library is missing.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Exposed so test modules can check whether llama_cpp is real or mocked.
_is_mock_llama_cpp = False


def _is_existing_path(path: str) -> bool:
    """Return True if *path* is an actual readable file on disk."""
    import os as _os
    return bool(path) and _os.path.isfile(path)


def _install_llama_cpp_mock():
    """Install a minimal ``llama_cpp`` mock if the real package is absent."""
    global _is_mock_llama_cpp

    if "llama_cpp" in sys.modules:
        return

    # Try to import the real thing — if it works, great.
    try:
        import llama_cpp  # noqa: F401
        return
    except ImportError:
        pass

    _is_mock_llama_cpp = True

    # Build a synthetic module just capable enough for import-level code paths.
    mock_llama_cpp = MagicMock()
    mock_llama_cpp.__version__ = "0.0.0-mock"

    # llama_cpp.Llama — used by ChatLlamaCpp._load_model.
    # The real constructor checks for file existence; our mock simulates that.
    class _MockLlama:
        """A callable mock class that raises on non-existent model paths."""

        def __init__(self, *, model_path: str, **kwargs):
            if not model_path or not _is_existing_path(model_path):
                raise ValueError(f"File not found: {model_path}")
            self.model_path = model_path
            self.n_ctx = kwargs.get("n_ctx")
            for k, v in kwargs.items():
                setattr(self, k, v)

        def tokenize(self, data: bytes) -> list:
            """Return a plausible list of token IDs (length depends on input)."""
            return list(range(len(data)))

    mock_llama_cpp.Llama = _MockLlama

    # llama_cpp.LlamaGrammar — used by grammar_builder.py which wraps
    # GBNF strings in LlamaGrammar objects before returning them.
    class _MockLlamaGrammar:
        """Minimal LlamaGrammar stand-in with a ``_grammar`` string attr."""

        def __init__(self, _grammar: str):
            self._grammar = _grammar

        def __str__(self) -> str:
            return self._grammar

    mock_llama_cpp.LlamaGrammar = _MockLlamaGrammar

    # llama_cpp.llama_types — used by type annotations in llm_engine.py
    mock_llama_cpp.llama_types = MagicMock()
    mock_llama_cpp.llama_types.ChatCompletionRequestMessage = dict
    mock_llama_cpp.llama_types.CreateChatCompletionResponse = dict
    mock_llama_cpp.llama_types.ChatCompletionTool = dict

    # llama_cpp.llama_chat_format — imported as llama_chat_format
    mock_llama_cpp.llama_chat_format = MagicMock()

    # llama_cpp.llama_grammar — used by compile_json_schema_to_gbnf
    mock_llama_grammar = MagicMock()

    # Known valid JSON Schema types for the mock to accept
    _VALID_SCHEMA_TYPES = {
        "string", "integer", "number", "boolean", "null",
        "object", "array",
    }

    class _MockGrammarCompileError(Exception):
        """Simulates llama_cpp.llama_grammar raising on broken schemas."""

    def _mock_json_schema_to_gbnf(schema_json_str: str) -> str:
        """Produce a plausible GBNF snippet for testing.  This is NOT a real
        GBNF compiler — it only needs to produce a non-empty string with
        ``root ::=`` so that grammar_builder unit tests pass.

        It supports basic ``object`` (with property names), ``oneOf``,
        and ``const`` so that tool-call-grammar and structured-output
        unit tests receive realistic-enough output.  Unrecognised schema
        types deliberately raise so the error-wrapping path is exercised."""
        import json as _json
        schema = _json.loads(schema_json_str) if isinstance(schema_json_str, str) else schema_json_str

        # oneOf → include the const values from each variant
        if "oneOf" in schema:
            parts: list[str] = []
            for variant in schema["oneOf"]:
                props = variant.get("properties", {})
                tool_prop = props.get("tool", {})
                const_val = tool_prop.get("const", "?")
                parts.append(f'"\\\\"{const_val}\\\\"')
            schema_type = schema.get("type", "object")
            extras = f' {"|".join(parts)}' if parts else ""
            return f'root ::= "mocked-gbnf-{schema_type}-oneOf{extras}"'

        schema_type = schema.get("type", "string")

        if schema_type not in _VALID_SCHEMA_TYPES:
            raise _MockGrammarCompileError(
                f"Unsupported schema type: {schema_type}"
            )

        # For object schemas, include property names in the mock output
        # so that tests checking for tool names / property keys can pass.
        if schema_type == "object":
            props = schema.get("properties", {})
            prop_names = "|".join(sorted(props.keys())) if props else "none"
            return f'root ::= "mocked-gbnf-object-props:{prop_names}"'

        return f'root ::= "mocked-gbnf-for-{schema_type}"'

    mock_llama_grammar.json_schema_to_gbnf = _mock_json_schema_to_gbnf
    mock_llama_cpp.llama_grammar = mock_llama_grammar

    sys.modules["llama_cpp"] = mock_llama_cpp
    sys.modules["llama_cpp.llama_types"] = mock_llama_cpp.llama_types
    sys.modules["llama_cpp.llama_chat_format"] = mock_llama_cpp.llama_chat_format
    sys.modules["llama_cpp.llama_grammar"] = mock_llama_grammar


_install_llama_cpp_mock()
