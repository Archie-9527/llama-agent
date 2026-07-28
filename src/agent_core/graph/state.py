"""AgentState——外层图唯一且规范的 TypedDict。

``graph/`` 中的每个节点都必须将输入和输出声明为 ``AgentState``，并且**不得**
添加未声明的键。需要新字段时应先扩展本文件。
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    """Planner、Executor 和 Reflector 节点共享的完整状态 Schema。

    属性：
        task_goal：用户最初的任务描述。
        plan_steps：Planner 生成的有序步骤列表。
        current_step_index：Executor 下一步应处理步骤的从零开始索引。
        execution_log：逐步骤记录，每个字典包含 ``step``（str）、
            ``result``（str）和 ``tool_used``（str | None）。
        reflection_notes：Reflector 评估记录历史。
        status：当前生命周期阶段，可以是 ``planning``、``executing``、
            ``reflecting``、``done`` 或 ``failed``。
        max_iterations：计划—反思循环的安全上限。
        current_iteration：目前已经执行的计划→反思循环次数。
        final_answer：Finalizer 生成的稳定、面向用户的结果。
        error：结构化终止错误，或 ``None``。
    """

    task_goal: str
    plan_steps: list[str]
    current_step_index: int
    execution_log: list[dict]
    reflection_notes: list[str]
    status: str  # 可取 planning、executing、reflecting、done 或 failed
    max_iterations: int
    current_iteration: int
    final_answer: str
    error: dict | None
    conversation_id: str | None
    current_user_input: str
    conversation_context: str
    pinned_facts: list[dict]
    context_summary: dict
    archived_context_ids: list[str]
    context_version: int
    lifecycle_stats: dict
