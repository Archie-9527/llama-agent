"""Planner node — decompose the user's task goal into ordered steps.

The Planner is the entry point of the outer graph.  It calls the LLM
with a GBNF grammar that constrains the output to a JSON object
containing a ``steps`` array, guaranteeing parseable output.
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
# JSON Schema that constrains the Planner's LLM output.
# The model MUST emit ``{"steps": ["step 1", "step 2", ...]}``.
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
    """Extract non-negated, imperative tool requirements in user order."""
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
    """Preserve every explicitly requested tool call as an atomic plan step.

    Small local models sometimes collapse a mandated two-tool workflow into
    the first call plus a textual answer.  This deterministic post-condition
    does not invent tools: it only restores calls explicitly named by the
    user, including repeated calls, in the user's order.
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
        # Do not retain duplicate/replayed versions of an explicitly mandated
        # call after its required occurrence has already been placed.
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
    """Identify conversation-memory turns that must not acquire tools.

    Explicit tool requests always win.  This guard only handles turns whose
    language declares, corrects, recalls or summarizes conversation facts.
    It prevents a local model from turning a fact correction into an invented
    file edit or service restart.
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
# [STABLE] planner_node
# ---------------------------------------------------------------------------


def planner_node(state: "AgentState") -> "AgentState":
    """Generate or update the plan for the current task.

    Steps:
        1. Assemble the planning prompt (task goal + reflection notes + tools).
        2. Build a GBNF grammar that forces JSON ``{"steps": [...]}`` output.
        3. Invoke the engine and parse the result.
        4. Write ``plan_steps``, reset ``current_step_index`` to 0,
           and transition ``status`` to ``"executing"``.

    Returns:
        *state* mutated with the new plan.

    Raises:
        PlanningError: If the LLM output cannot be parsed as a valid plan
            (malformed JSON, missing ``steps`` key, or empty step list).
    """
    current_input = str(
        state.get("current_user_input") or state.get("task_goal", "")
    )
    # Conversation-only turns do not need a model-generated execution plan.
    # Short-circuit before get_engine()/invoke so R2 memory conversations do
    # not pay for a redundant Planner inference that is discarded below.
    if _is_tool_free_conversation_intent(state):
        state["plan_steps"] = [f"直接处理当前会话请求：{current_input}"]
        state["current_step_index"] = 0
        state["status"] = "executing"
        return state

    engine = get_engine()
    messages = assemble_planning_prompt(state, engine)
    grammar = build_json_grammar(PLAN_SCHEMA)

    response = engine.invoke(messages, grammar=grammar)

    # Parse the constrained JSON output
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

    # Validate each step is a non-empty string
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
