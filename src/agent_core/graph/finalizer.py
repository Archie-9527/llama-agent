"""Finalizer 节点——生成稳定、面向用户的任务结果。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agent_core.exceptions import ReflectionError
from agent_core.llm_engine import get_engine
from agent_core.prompt_assembler import assemble_finalization_prompt

if TYPE_CHECKING:
    from agent_core.graph.state import AgentState


def finalizer_node(state: "AgentState") -> "AgentState":
    summaries = _usable_step_summaries(state)
    # Executor 已执行不带工具的总结调用。对于单步骤任务，该结论就是完整答案，
    # 再调用一次 LLM 只会增加延迟，并可能暴露或重复模型推理。
    if len(state.get("plan_steps", [])) <= 1 and summaries:
        state["final_answer"] = summaries[-1]
        state["status"] = "done"
        return state

    engine = get_engine()
    response = engine.invoke(assemble_finalization_prompt(state, engine))
    answer = str(response.content).strip()
    finish_reason = str(
        getattr(response, "response_metadata", {}).get("finish_reason") or ""
    ).casefold()
    unusable = (
        not answer
        or finish_reason in {"length", "max_tokens"}
        or bool(
            re.fullmatch(
                r"functions\.[A-Za-z_][A-Za-z0-9_]*\s*:\s*",
                answer,
            )
        )
    )
    if unusable:
        if summaries:
            answer = "\n".join(summaries)
        else:
            raise ReflectionError(
                "Finalizer produced an empty, truncated or protocol-only answer"
            )
    state["final_answer"] = answer
    state["status"] = "done"
    return state


def _usable_step_summaries(state: "AgentState") -> list[str]:
    summaries: list[str] = []
    for record in state.get("execution_log", []):
        if record.get("tool_used") is not None:
            continue
        content = str(record.get("result", "")).strip()
        if not content or re.fullmatch(
            r"functions\.[A-Za-z_][A-Za-z0-9_]*\s*:\s*", content
        ):
            continue
        if content not in summaries:
            summaries.append(content)
    return summaries
