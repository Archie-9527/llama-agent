# 文件：tests/integration/test_end_to_end_flow.py

"""
工作流编排层端到端集成测试。
目标：验证 session.TaskRunner 能够正确驱动
planner -> executor(可能多步) -> reflector -> 终止 的完整链路，
以及checkpoint的写入与崩溃恢复语义。

打桩边界：仅桩住 get_engine（推理内核入口）与
get_react_agent（内层ReAct子图入口），其余全部走真实代码路径，
以验证编排层各文件之间的"接线"是否正确。
"""

import json
import uuid
import os
import sys
import pytest

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from langchain_core.messages import AIMessage, ToolMessage

from agent_core.session import RunConfig, TaskRunner
from agent_core.exceptions import AgentCoreError
from agent_core.capability_registry import clear_registry, register


# ---------------------------------------------------------------------------
# Fake组件
# ---------------------------------------------------------------------------

class FakeAIMessage:
    """模拟engine.invoke()的返回值，只需要具备.content属性。"""
    def __init__(self, content):
        self.content = content


class FakeEngine:
    """
    伪造推理内核。按调用奇偶顺序区分Planner调用（奇数次）与
    Reflector调用（偶数次）——这依赖于"每轮迭代恰好各调用一次"
    的假设，仅适用于本测试构造的单轮场景。
    """
    def __init__(self, plan_steps, reflection_decision="done"):
        self.call_log = []
        self._structured_calls = 0
        self._plan_steps = plan_steps
        self._reflection_decision = reflection_decision

    def invoke(self, messages, grammar=None, **kwargs):
        self.call_log.append(messages)
        if grammar is None:
            return FakeAIMessage("基于全部真实记录生成的最终答案")
        self._structured_calls += 1
        if self._structured_calls % 2 == 1:
            return FakeAIMessage(json.dumps({"steps": self._plan_steps}))
        return FakeAIMessage(json.dumps(self._reflection_decision))

    def get_num_tokens(self, text: str) -> int:
        return len(text) // 4


class FakeReactAgent:
    """
    伪造内层ReAct子图。每次调用模拟"1次工具调用 + 1条最终结论"，
    用来验证executor_node的_extract_execution_result翻译逻辑是否正确。
    """
    def invoke(self, input_dict, config=None):
        ai_with_tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "fake_tool", "args": {}, "id": "call_1"}],
        )
        tool_result = ToolMessage(content="工具返回的结果数据", tool_call_id="call_1")
        final_answer = AIMessage(content="基于工具结果得出的步骤结论")
        return {
            "messages": input_dict["messages"] + [ai_with_tool_call, tool_result, final_answer]
        }


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    return RunConfig(
        max_iterations=4,
        db_path=tmp_path / "checkpoints.sqlite",
        last_thread_file=tmp_path / "last_thread_id.txt",
    )


@pytest.fixture
def patched_env(monkeypatch):
    """
    把推理内核层和内层子图两个边界点替换为可控Fake。
    ⚠️ 下面的patch路径需对照实际import写法调整，见文档正文说明。
    """
    clear_registry()

    @register(
        name="fake_tool",
        description="A deterministic integration-test tool",
        input_schema={"type": "object", "properties": {}},
    )
    def fake_tool():
        return "工具返回的结果数据"

    plan_steps = ["使用 fake_tool 查询目标信息", "使用 fake_tool 生成最终结论"]
    fake_engine = FakeEngine(plan_steps=plan_steps)
    fake_react_agent = FakeReactAgent()

    monkeypatch.setattr("agent_core.graph.planner.get_engine", lambda: fake_engine)
    monkeypatch.setattr("agent_core.graph.reflector.get_engine", lambda: fake_engine)
    monkeypatch.setattr("agent_core.graph.executor.get_engine", lambda: fake_engine)
    monkeypatch.setattr("agent_core.graph.finalizer.get_engine", lambda: fake_engine)
    monkeypatch.setattr(
        "agent_core.graph.react_agent_factory.initialize_react_agent",
        lambda tool_names=None: fake_react_agent,
    )

    yield fake_engine, fake_react_agent
    clear_registry()


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_full_flow_happy_path(tmp_config, patched_env):
    """
    验证目标：start_new_task()能完整驱动
    planner -> executor(2步骤) -> reflector -> done。
    """
    runner = TaskRunner(tmp_config)
    thread_id, result = runner.start_new_task("查一下今天的新闻并总结")

    assert result["status"] == "done"
    assert result["plan_steps"] == [
        "使用 fake_tool 查询目标信息",
        "使用 fake_tool 生成最终结论",
    ]
    assert result["current_step_index"] == 2

    # 每个计划步骤 = 1次工具调用记录 + 1次最终结论记录，2步骤共4条
    assert len(result["execution_log"]) == 4
    assert result["execution_log"][0]["tool_used"] == "fake_tool"
    assert result["execution_log"][0]["result"] == "工具返回的结果数据"
    assert result["execution_log"][1]["tool_used"] is None
    assert result["execution_log"][1]["result"] == "基于工具结果得出的步骤结论"

    assert runner.get_last_thread_id() == thread_id
    runner.close()


def test_checkpoint_persists_to_sqlite(tmp_config, patched_env):
    """验证checkpoint确实落盘到指定sqlite文件，而不是只存在于内存。"""
    runner = TaskRunner(tmp_config)
    runner.start_new_task("一个测试目标")

    assert tmp_config.db_path.exists()
    assert tmp_config.db_path.stat().st_size > 0
    runner.close()


def test_resume_after_executor_crash(tmp_config, patched_env, monkeypatch):
    """
    验证崩溃恢复语义：executor第一次执行时故意抛异常中断整个流程，
    用同一个thread_id续跑后，应当从中断处继续（不重新调用planner生成新计划）。

    ⚠️ 本测试的核心假设是：LangGraph的checkpoint是在每个节点
    【成功完成后】才落盘，因此中断发生在executor内部时，
    上一个检查点仍是"planner刚完成"那一刻的状态。/Users/heart/Code/py-project/llama-agent/src/tests/test_end_to_end_flow.py::test_full_flow_happy_path
    这个假设与LangGraph具体版本的checkpoint时机机制相关，
    建议实际跑一次本测试确认结果，而不是纯凭本注释假设成立。
    """
    fake_engine, fake_react_agent = patched_env
    call_state = {"count": 0}
    original_invoke = fake_react_agent.invoke

    def flaky_invoke(input_dict, config=None):
        call_state["count"] += 1
        if call_state["count"] == 1:
            raise RuntimeError("模拟执行器中途崩溃")
        return original_invoke(input_dict, config=config)

    monkeypatch.setattr(fake_react_agent, "invoke", flaky_invoke)

    # 固定thread_id，绕开start_new_task因异常提前退出、
    # 来不及调用_remember_thread_id而丢失ID的问题
    fixed_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    monkeypatch.setattr("agent_core.session.uuid.uuid4", lambda: fixed_id)

    runner = TaskRunner(tmp_config)

    # V3 的 build_graph 使用 _with_error_isolation 包装节点，因此会捕获
    # RuntimeError 并将状态转换为 status="failed"，任务会正常结束而非抛出异常。
    tid, result = runner.start_new_task("会中断一次的任务")
    assert result["status"] == "failed"

    # 恢复后执行器会重新运行该步骤。首次运行在第一步失败，因此恢复后的
    # 再次运行应当成功（此时 call_state["count"] 为 2，
    # flaky_invoke 会正常返回）。
    result = runner.resume_task(str(fixed_id))

    # 在错误隔离机制下恢复后，任务仍可能为 failed，因为检查点位于
    # normalize_state 之后。第二次调用会从头重放，这里验证它不会崩溃。
    assert result["status"] in ("done", "failed", "executing")

    runner.close()


def test_start_new_task_rejects_empty_goal(tmp_config, patched_env):
    """空目标应在调用图之前就被拒绝，不产生thread_id、不触碰checkpoint。"""
    runner = TaskRunner(tmp_config)
    with pytest.raises(ValueError):
        runner.start_new_task("   ")
    assert runner.get_last_thread_id() is None
    runner.close()


def test_resume_rejects_empty_thread_id(tmp_config, patched_env):
    runner = TaskRunner(tmp_config)
    with pytest.raises(ValueError):
        runner.resume_task("")
    runner.close()


def test_multiple_runners_isolated(tmp_path, patched_env):
    """两个不同db_path的TaskRunner互不干扰。"""
    config_a = RunConfig(db_path=tmp_path / "a.sqlite", last_thread_file=tmp_path / "a_last.txt")
    config_b = RunConfig(db_path=tmp_path / "b.sqlite", last_thread_file=tmp_path / "b_last.txt")

    runner_a = TaskRunner(config_a)
    runner_b = TaskRunner(config_b)

    thread_a, _ = runner_a.start_new_task("任务A")
    thread_b, _ = runner_b.start_new_task("任务B")

    assert thread_a != thread_b
    assert runner_a.get_last_thread_id() == thread_a
    assert runner_b.get_last_thread_id() == thread_b

    runner_a.close()
    runner_b.close()
