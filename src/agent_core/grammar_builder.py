"""
结构化输出约束构建器 — 将业务级别的模式需求转换为 ``llm_engine.ChatLlamaCpp`` 使用的 GBNF (GGML BNF) 语法字符串。
GBNF 语法允许**约束解码**：在每个Token生成步骤中，模型仅限于符合语法的Token.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List

from agent_core.exceptions import GrammarCompileError
from agent_core.llm_engine import compile_json_schema_to_gbnf

from llama_cpp import LlamaGrammar

# ---------------------------------------------------------------------------
# [内部实现] Schema 校验
# ---------------------------------------------------------------------------

# 需要在字符串字面量中转义的 GBNF 特殊字符。
_GBNF_ESCAPE_TABLE = {
    '"': '\\"',
    "\\": "\\\\",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _gbnf_escape(value: str) -> str:
    """转义 *value* 中的特殊字符，使其能安全渲染到 GBNF 字符串中。"""
    return "".join(_GBNF_ESCAPE_TABLE.get(ch, ch) for ch in value)


def _validate_no_unresolved_refs(schema: Dict[str, Any], path: str = "$") -> None:
    """
    校验schema是否包含未解析的$ref（本系统不支持跨文件引用，
    若检测到需直接抛出GrammarCompileError，而不是让转换过程静默产生错误结果）；
    """
    if not isinstance(schema, dict):
        return

    if "$ref" in schema:
        raise GrammarCompileError(
            f"Unresolved $ref at '{path}/$ref': this system does not support "
            f"JSON Schema $ref resolution.  Inline or expand the reference "
            f"before passing the schema to grammar_builder."
        )

    for key, value in schema.items():
        child_path = f"{path}/{key}"
        if isinstance(value, dict):
            _validate_no_unresolved_refs(value, child_path)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    _validate_no_unresolved_refs(item, f"{child_path}[{idx}]")


# ---------------------------------------------------------------------------
# [内部实现] 编译缓存
# ---------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _cached_compile(schema_json_str: str) -> str:
    """
    缓存层
    以排序后的JSON字符串作为缓存key，保证语义相同的schema即便字段顺序不同也能命中缓存。
    """
    return compile_json_schema_to_gbnf(json.loads(schema_json_str))


# ---------------------------------------------------------------------------
# [稳定接口] 公共 API
# ---------------------------------------------------------------------------


def build_json_grammar(schema: Dict[str, Any]) -> LlamaGrammar:
    """
    将JSON Schema转换为GBNF语法字符串
    调用compile_json_schema_to_gbnf完成实际转换，结果按schema内容做缓存
    """
    _validate_no_unresolved_refs(schema)
    try:
        return LlamaGrammar(_grammar = _cached_compile(json.dumps(schema, sort_keys=True)))
    except GrammarCompileError:
        raise
    except Exception as exc:
        raise GrammarCompileError(
            f"Failed to compile JSON Schema to GBNF: {exc}"
        ) from exc


def build_enum_grammar(options: List[str]) -> LlamaGrammar:
    """构建将输出严格限制为 *options* 其中一项的 GBNF Grammar。

    每个选项都由双引号包裹（GBNF 字符串字面量），并转义特殊字符。生成的
    Grammar 如下::

        root ::= "continue" | "done" | "failed"

    Planner 节点使用它约束路由决策，例如
    ``["continue", "done", "failed"]``。

    异常：
        GrammarCompileError：*options* 为空时抛出。此时 Grammar 必然不可满足，
            会导致约束解码锁死。
    """
    if not options:
        raise GrammarCompileError(
            "build_enum_grammar: options list must not be empty — an empty "
            "enum grammar would prevent the model from generating any token."
        )

    escaped = [_gbnf_escape(opt) for opt in options]
    alternatives = " | ".join(f'"\\"{e}\\""' for e in escaped)
    return LlamaGrammar(_grammar=f"root ::= {alternatives}")


def build_tool_call_grammar(tools: List[Any]) -> str:
    """构建约束模型输出有效工具调用的 GBNF Grammar。

    每个工具至少应具有 ``name`` 属性和 ``input_schema`` 字典，后者是其参数的
    JSON Schema。生成的 Grammar 会把所有工具包装进 ``oneOf`` JSON Schema::

        {
          "oneOf": [
            {
              "properties": {
                "tool":      {"const": "<tool-A-name>"},
                "arguments": <tool-A-input_schema>
              }
            },
            ...
          ]
        }

    然后委托 ``build_json_grammar`` 完成实际转换。

    异常：
        GrammarCompileError：任一工具缺少 ``name`` 或 ``input_schema`` 时抛出。
    """
    variants: List[Dict[str, Any]] = []
    for tool in tools:
        # 同时接受具有 .name / .input_schema 属性的对象（Capability）和普通字典。
        if isinstance(tool, dict):
            name = tool.get("name") or tool.get("function", {}).get("name")
            input_schema = tool.get("input_schema") or tool.get(
                "function", {}
            ).get("parameters")
        else:
            name = getattr(tool, "name", None)
            input_schema = getattr(tool, "input_schema", None)

        if not name:
            raise GrammarCompileError(
                f"Tool {tool!r} is missing a 'name' — every tool must have a "
                f"unique name for the grammar to constrain tool selection."
            )
        if not input_schema:
            raise GrammarCompileError(
                f"Tool '{name}' is missing an 'input_schema' — a JSON Schema "
                f"for its arguments is required to build the tool-call grammar."
            )

        variants.append({
            "type": "object",
            "properties": {
                "tool": {"type": "string", "const": name},
                "arguments": input_schema,
            },
            "required": ["tool", "arguments"],
        })

    if not variants:
        raise GrammarCompileError(
            "build_tool_call_grammar: tools list is empty — cannot build a "
            "grammar with no tools."
        )

    return build_json_grammar({"oneOf": variants})
