"""Planner 节点——将用户任务目标拆分为有序步骤。

Planner 是外层图的入口。它使用 GBNF Grammar 调用 LLM，将输出约束为包含
``steps`` 数组的 JSON 对象，从而保证结果可解析。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from typing import TYPE_CHECKING

from agent_core.exceptions import PlanningError
from agent_core.grammar_builder import build_json_grammar
from agent_core.llm_engine import get_engine
from agent_core.prompt_assembler import assemble_planning_prompt

if TYPE_CHECKING:
    from agent_core.graph.state import AgentState

# ---------------------------------------------------------------------------
# 约束 Planner LLM 输出的 JSON Schema。
# 模型必须输出 ``{"steps": ["step 1", "step 2", ...]}``。
# ---------------------------------------------------------------------------

PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Ordered list of executable steps.",
        }
    },
    "required": ["steps"],
}


def _explicit_tool_sequence(task_goal: str) -> list[str]:
    """按用户顺序提取未被否定的命令式工具要求。"""
    from agent_core.capability_registry import list_capabilities

    names = {cap.name for cap in list_capabilities()}
    if not names:
        return []
    alternatives = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?:使用|调用|通过|用|use|call)\s*(?:工具\s*)?({alternatives})\b",
        re.IGNORECASE,
    )
    required: list[str] = []
    for match in pattern.finditer(task_goal):
        prefix = task_goal[max(0, match.start() - 8) : match.start()]
        if re.search(
            r"(?:不要|无需|禁止|避免|不必)(?:再|再次)?\s*$",
            prefix,
        ):
            continue
        matched = match.group(1)
        required.append(
            next(name for name in names if name.casefold() == matched.casefold())
        )
    return required


def _step_target_tool(step: str, tool_names: set[str]) -> str | None:
    if not tool_names:
        return None
    alternatives = "|".join(
        re.escape(name) for name in sorted(tool_names, key=len, reverse=True)
    )
    match = re.search(
        rf"(?:使用|调用|通过|用|use|call)\s*(?:工具\s*)?({alternatives})\b",
        step,
        re.IGNORECASE,
    ) or re.search(
        rf"^\s*(?:\d+[.、)]\s*)?({alternatives})\b",
        step,
        re.IGNORECASE,
    )
    if not match:
        return None
    matched = match.group(1)
    return next(
        name for name in tool_names if name.casefold() == matched.casefold()
    )


def _enforce_explicit_tool_sequence(
    task_goal: str,
    steps: list[str],
    executed_tools: list[str] | None = None,
) -> list[str]:
    """将每个显式要求的工具调用保留为原子计划步骤。

    小型本地模型有时会把强制的双工具工作流压缩成第一次调用加文本回答。这个
    确定性后置条件不会虚构工具，只会按用户顺序恢复其明确指定的调用，包括
    重复调用。
    """
    required = _explicit_tool_sequence(task_goal)
    explicitly_named_tools = set(required)
    completed = Counter(executed_tools or [])
    remaining: list[str] = []
    for tool_name in required:
        if completed[tool_name] > 0:
            completed[tool_name] -= 1
        else:
            remaining.append(tool_name)
    required = remaining
    if not required:
        return steps

    queues: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
    tool_names = set(required)
    for index, step in enumerate(steps):
        target = _step_target_tool(step, tool_names)
        if target is not None:
            queues[target].append((index, step))

    consumed: set[int] = set()
    ordered: list[str] = []
    for operation_index, tool_name in enumerate(required, start=1):
        if queues[tool_name]:
            index, step = queues[tool_name].popleft()
            consumed.add(index)
            ordered.append(step)
        else:
            ordered.append(
                f"使用 {tool_name} 完成用户明确要求的第 {operation_index} 个"
                "工具操作；参数必须来自原始任务目标和前序真实执行结果"
            )

    for index, step in enumerate(steps):
        if index in consumed:
            continue
        # 显式强制调用的必要实例已经放入计划后，不再保留其重复或重放版本。
        if _step_target_tool(step, explicitly_named_tools) is not None:
            continue
        ordered.append(step)
    return ordered


_CONVERSATION_ONLY_PATTERN = re.compile(
    r"(?:"
    r"记住|不要忘记|更正|修正为|改为|不再有效|"
    r"最初.{0,20}(?:是|为)|上一轮|此前|之前告诉|"
    r"综合此前对话|直接回答|只回复|不要再次调用工具|不要调用工具"
    r")"
)
_EXPLICIT_EXTERNAL_OPERATION_PATTERN = re.compile(
    r"(?:"
    r"(?:读取|搜索|查询|修改|写入|删除|执行|运行|重启|启动|检查)"
    r".{0,16}(?:文件|目录|路径|日志|数据库|SQL|命令|脚本|服务|URL)|"
    r"(?:文件|目录|路径|日志|数据库|SQL|命令|脚本|服务|URL)"
    r".{0,16}(?:读取|搜索|查询|修改|写入|删除|执行|运行|重启|启动|检查)|"
    r"(?:/|\.{1,2}/)[A-Za-z0-9_.\-/]+"
    r")",
    re.IGNORECASE,
)
_NO_TOOL_PATTERN = re.compile(
    r"(?:不要|无需|禁止|避免|不必)(?:再|再次)?调用(?:任何|外部)?工具"
)


def _is_tool_free_conversation_intent(state: "AgentState") -> bool:
    """识别不得引入工具的会话记忆轮次。

    显式工具请求始终优先。本保护只处理声明、修正、召回或总结会话事实的轮次，
    防止本地模型把事实修正变成虚构的文件编辑或服务重启。
    """
    if not state.get("conversation_id"):
        return False
    current_input = str(
        state.get("current_user_input") or state.get("task_goal", "")
    ).strip()
    if not current_input or _explicit_tool_sequence(current_input):
        return False
    no_tool_requested = bool(_NO_TOOL_PATTERN.search(current_input))
    if (
        _EXPLICIT_EXTERNAL_OPERATION_PATTERN.search(current_input)
        and not no_tool_requested
    ):
        return False
    return no_tool_requested or bool(
        _CONVERSATION_ONLY_PATTERN.search(current_input)
    )


# ---------------------------------------------------------------------------
# [稳定接口] planner_node
# ---------------------------------------------------------------------------


def planner_node(state: "AgentState") -> "AgentState":
    """生成或更新当前任务的计划。

    步骤：
        1. 组装规划 Prompt（任务目标 + 反思记录 + 工具）。
        2. 构建强制输出 JSON ``{"steps": [...]}`` 的 GBNF Grammar。
        3. 调用引擎并解析结果。
        4. 写入 ``plan_steps``，将 ``current_step_index`` 重置为 0，并把
           ``status`` 转换为 ``"executing"``。

    返回：
        写入新计划后的 *state*。

    异常：
        PlanningError：LLM 输出无法解析为有效计划时抛出，例如 JSON 格式错误、
            缺少 ``steps`` 键或步骤列表为空。
    """
    current_input = str(
        state.get("current_user_input") or state.get("task_goal", "")
    )
    # 纯会话轮次不需要模型生成执行计划。在 get_engine()/invoke 之前短路，
    # 避免 R2 记忆会话承担一次随后会被丢弃的冗余 Planner 推理。
    if _is_tool_free_conversation_intent(state):
        state["plan_steps"] = [f"直接处理当前会话请求：{current_input}"]
        state["current_step_index"] = 0
        state["status"] = "executing"
        return state

    engine = get_engine()
    messages = assemble_planning_prompt(state, engine)
    grammar = build_json_grammar(PLAN_SCHEMA)

    response = engine.invoke(messages, grammar=grammar)

    # 解析受约束的 JSON 输出
    try:
        parsed = json.loads(response.content)
        steps: list[str] = parsed["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps field is empty or not a list")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise PlanningError(
            f"Planner output could not be parsed as a valid plan: "
            f"{response.content!r}"
        ) from exc

    # 校验每个步骤都是非空字符串
    for i, step in enumerate(steps):
        if not isinstance(step, str) or not step.strip():
            raise PlanningError(
                f"Plan step {i} is empty or not a string: {step!r}"
            )

    state["plan_steps"] = _enforce_explicit_tool_sequence(
        current_input,
        steps,
        [
            str(record["tool_used"])
            for record in state.get("execution_log", [])
            if record.get("tool_used")
        ],
    )
    state["current_step_index"] = 0
    state["status"] = "executing"

    return state
