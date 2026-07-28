"""Reflector 节点——评估执行进度并决定下一步操作。

Reflector 是外层图的决策门。它检查任务目标、计划和执行日志，然后通过受枚举
约束的 LLM 调用产生三种决策之一：

    * ``done``——任务目标已经完全实现。
    * ``continue``——仍需继续工作，返回 Planner。
    * ``failed``——遇到不可恢复的错误。

执行该决策的路由逻辑位于 ``build_graph.py``，不在本文件中。Reflector 只负责
**产生**决策，从而将“如何路由”的逻辑集中在一个文件中。
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from agent_core.exceptions import ReflectionError
from agent_core.grammar_builder import build_json_grammar
from agent_core.llm_engine import get_engine
from agent_core.prompt_assembler import assemble_reflection_prompt

if TYPE_CHECKING:
    from agent_core.graph.state import AgentState

# ---------------------------------------------------------------------------
# Reflector 允许输出的决策集合。
# ---------------------------------------------------------------------------

_VALID_DECISIONS = {"done", "continue", "failed"}

REFLECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": sorted(_VALID_DECISIONS),
        },
        "reason": {
            "type": "string",
            "description": "Concrete evidence for the decision and retry advice.",
        },
    },
    "required": ["decision", "reason"],
}


def _parse_reflection_output(raw: str) -> object:
    """解析一个受约束的值，同时容忍无害的尾随文本。

    某些本地模型即使受到 Grammar/Prompt 约束，仍会在有效 JSON 对象后附加
    Markdown 说明。``raw_decode`` 会严格校验第一个 JSON 值，同时避免从任意
    文本中接受虚构决策。
    """
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidate_offsets = [0]
        candidate_offsets.extend(
            index for index, char in enumerate(stripped) if char == "{"
        )
        seen: set[int] = set()
        for offset in candidate_offsets:
            if offset in seen:
                continue
            seen.add(offset)
            try:
                value, _ = decoder.raw_decode(stripped[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "decision" in value:
                return value
    # Qwen 偶尔会用排版引号（”）结束最后一个 JSON 字符串，再附加说明文本。
    # 此处只恢复精确的受约束反思结构，随后仍在 ``reflector_node`` 中校验枚举。
        near_json = re.search(
            r'\{\s*"decision"\s*:\s*"(?P<decision>done|continue|failed)"\s*,'
            r'\s*"reason"\s*:\s*"(?P<reason>.*?)(?:"|”)\s*\}',
            stripped,
            re.DOTALL,
        )
        if near_json:
            return near_json.groupdict()
    # 向后兼容旧 Checkpoint 和测试：它们可能返回不带结构、可选引号包裹的枚举，
    # 而不是结构化 Schema。
        return stripped.strip('"').strip("'")


# ---------------------------------------------------------------------------
# [稳定接口] reflector_node
# ---------------------------------------------------------------------------


def reflector_node(state: "AgentState") -> "AgentState":
    """评估执行进度并输出路由决策。

    步骤：
        1. 组装反思 Prompt（任务目标 + 计划 + 执行日志）。
        2. 构建包含 ``decision`` 和诊断字段 ``reason`` 的 JSON Grammar。
        3. 调用引擎并提取结构化决策。
        4. 向 ``reflection_notes`` 追加记录，递增 ``current_iteration``，
           并将 ``status`` 设置为该决策。

    ``status`` 字段是**临时**值；``build_graph.py`` 的条件边函数是最终裁决者，
    负责决定接受该值还是强制终止，例如达到 ``max_iterations`` 时。

    返回：
        写入新反思记录并更新状态后的 *state*。

    异常：
        ReflectionError：模型输出不属于
            ``{"done", "continue", "failed"}`` 时抛出。
    """
    engine = get_engine()
    messages = assemble_reflection_prompt(state, engine)
    grammar = build_json_grammar(REFLECTION_SCHEMA)

    response = engine.invoke(messages, grammar=grammar)
    raw = str(response.content).strip()
    reason = ""
    parsed = _parse_reflection_output(raw)
    if isinstance(parsed, dict):
        decision = str(parsed.get("decision", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
    else:
    # 向后兼容旧 Checkpoint 和轻量测试。
        decision = str(parsed).strip()

    if decision not in _VALID_DECISIONS:
        raise ReflectionError(
            f"Reflector output '{decision}' is not a valid decision — "
            f"expected one of {sorted(_VALID_DECISIONS)}."
        )

    # 记录决策
    current_iteration: int = state.get("current_iteration", 0)
    note = f"[iteration {current_iteration}] decision={decision}"
    if reason:
        note += f" reason={reason}"
    reflection_notes: list[str] = state.get("reflection_notes", [])
    reflection_notes.append(note)

    next_iteration = current_iteration + 1
    state["reflection_notes"] = reflection_notes
    state["current_iteration"] = next_iteration

    # 安全上限已经耗尽时，不要返回具有误导性的非终止 ``continue`` 状态。
    # 将任务标记为失败，并为 CLI/API 调用方留下明确诊断。
    max_iterations: int = state.get("max_iterations", 10)
    if decision == "continue" and next_iteration >= max_iterations:
        state["reflection_notes"].append(
            f"[iteration limit] max_iterations={max_iterations} reached "
            "before the task was completed"
        )
        state["status"] = "failed"
        state["error"] = {
            "node": "reflector",
            "type": "IterationLimitExceeded",
            "message": f"max_iterations={max_iterations} reached",
        }
    elif decision == "failed":
        state["status"] = "failed"
        state.setdefault(
            "error",
            {
                "node": "reflector",
                "type": "ReflectionRejected",
                "message": (
                    "Reflector rejected the execution result at "
                    f"iteration {current_iteration}."
                ),
            },
        )
    else:
        state["status"] = decision

    return state
