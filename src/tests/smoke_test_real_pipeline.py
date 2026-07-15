"""
真实模型端到端烟雾测试——手动运行
"""

import sys
import logging
import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from agent_core.session import TaskRunner, RunConfig

logging.basicConfig(level=logging.INFO)


def main():
    runner = TaskRunner(RunConfig(max_iterations=3))
    goal = sys.argv[1] if len(sys.argv) > 1 else "用一句话介绍今天是星期几"

    print(f"任务目标: {goal}")
    thread_id, result = runner.start_new_task(goal)

    print(f"\nthread_id = {thread_id}")
    print(f"最终状态 = {result['status']}")
    print(f"计划步骤 = {result['plan_steps']}")
    print("\n执行日志:")
    for i, record in enumerate(result["execution_log"]):
        print(f"  [{i}] step={record['step']!r} tool={record['tool_used']} result={record['result'][:80]!r}")
    print("\n反思记录:")
    for note in result["reflection_notes"]:
        print(f"  {note}")

    runner.close()


if __name__ == "__main__":
    main()
