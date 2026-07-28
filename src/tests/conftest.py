"""llama-agent 测试套件共享的 pytest Fixture 与模块级 Mock。

虚拟机沙箱中可能没有 llama-cpp-python，例如缺少预编译 Wheel 或源码构建超时。
本 conftest 提供合成的 ``llama_cpp`` 模块，使 Grammar Builder、消息转换和异常
隔离等纯单元测试无需真实原生库即可运行。

需要真实 GGUF 模型的测试使用带 ``_real_model_available`` 检查的
``@pytest.mark.skipif`` 保护；缺少模型或原生库时会自动跳过。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# 对外公开，使测试模块可以检查 llama_cpp 是真实模块还是 Mock。
_is_mock_llama_cpp = False


def _is_existing_path(path: str) -> bool:
    """*path* 是磁盘上实际可读的文件时返回 True。"""
    import os as _os
    return bool(path) and _os.path.isfile(path)


def _install_llama_cpp_mock():
    """真实包不存在时安装最小化的 ``llama_cpp`` Mock。"""
    global _is_mock_llama_cpp

    if "llama_cpp" in sys.modules:
        return

    # 尝试导入真实模块；导入成功则直接使用。
    try:
        import llama_cpp  # noqa: F401
        return
    except ImportError:
        pass

    _is_mock_llama_cpp = True

    # 构建刚好满足导入级代码路径所需能力的合成模块。
    mock_llama_cpp = MagicMock()
    mock_llama_cpp.__version__ = "0.0.0-mock"

    # ChatLlamaCpp._load_model 使用 llama_cpp.Llama。真实构造函数会检查文件
    # 是否存在，此 Mock 会模拟该行为。
    class _MockLlama:
        """模型路径不存在时抛出异常的可调用 Mock 类。"""

        def __init__(self, *, model_path: str, **kwargs):
            if not model_path or not _is_existing_path(model_path):
                raise ValueError(f"File not found: {model_path}")
            self.model_path = model_path
            self.n_ctx = kwargs.get("n_ctx")
            for k, v in kwargs.items():
                setattr(self, k, v)

        def tokenize(self, data: bytes) -> list:
            """返回合理的 Token ID 列表，长度取决于输入。"""
            return list(range(len(data)))

    mock_llama_cpp.Llama = _MockLlama

    # grammar_builder.py 使用 llama_cpp.LlamaGrammar，在返回前用 LlamaGrammar
    # 对象包装 GBNF 字符串。
    class _MockLlamaGrammar:
        """具有 ``_grammar`` 字符串属性的最小 LlamaGrammar 替代实现。"""

        def __init__(self, _grammar: str):
            self._grammar = _grammar

        def __str__(self) -> str:
            return self._grammar

    mock_llama_cpp.LlamaGrammar = _MockLlamaGrammar

    # llm_engine.py 的类型注解使用 llama_cpp.llama_types
    mock_llama_cpp.llama_types = MagicMock()
    mock_llama_cpp.llama_types.ChatCompletionRequestMessage = dict
    mock_llama_cpp.llama_types.CreateChatCompletionResponse = dict
    mock_llama_cpp.llama_types.ChatCompletionTool = dict

    # llama_cpp.llama_chat_format 会以 llama_chat_format 名称导入
    mock_llama_cpp.llama_chat_format = MagicMock()

    # compile_json_schema_to_gbnf 使用 llama_cpp.llama_grammar
    mock_llama_grammar = MagicMock()

    # Mock 可接受的已知有效 JSON Schema 类型
    _VALID_SCHEMA_TYPES = {
        "string", "integer", "number", "boolean", "null",
        "object", "array",
    }

    class _MockGrammarCompileError(Exception):
        """模拟 llama_cpp.llama_grammar 遇到损坏 Schema 时抛出异常。"""

    def _mock_json_schema_to_gbnf(schema_json_str: str) -> str:
        """生成供测试使用的合理 GBNF 片段。

        这不是真正的 GBNF 编译器；它只需生成包含 ``root ::=`` 的非空字符串，
        使 grammar_builder 单元测试能够通过。

        它支持基础 ``object``（含属性名）、``oneOf`` 和 ``const``，使工具调用
        Grammar 与结构化输出单元测试获得足够真实的结果。遇到无法识别的
        Schema 类型时会有意抛出异常，以覆盖错误包装路径。
        """
        import json as _json
        schema = _json.loads(schema_json_str) if isinstance(schema_json_str, str) else schema_json_str

        # oneOf → 包含各变体的 const 值
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

        # 对于对象 Schema，在 Mock 输出中包含属性名称，使检查工具名和属性键的
        # 测试能够通过。
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
