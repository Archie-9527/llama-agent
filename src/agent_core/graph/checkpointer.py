"""Checkpointer 生命周期——基于 SQLite 的外层图持久化。

每个外层图节点完成后，Checkpointer 都会写入完整 ``AgentState`` 快照。
崩溃恢复由 ``thread_id`` 驱动：使用同一 ``thread_id`` 再次调用时，会从最后
完成的节点继续，而不是从 ``planner_node`` 重新开始。

设计说明：
    只有**外层**图接收 Checkpointer。内部 ReAct 子图有意不设置 Checkpoint，
    相关权衡分析见编排层设计文档第 6.2 节。

用法（位于 cli.py）::

    with get_checkpointer("data/checkpoints.sqlite") as cp:
        graph = build_graph(checkpointer=cp)
        graph.invoke(state, config={"configurable": {"thread_id": "..."}})
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# 默认位置——相对于项目根目录。
# 调用方（build_graph / cli）可以覆盖该路径。
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path("data/checkpoints.sqlite")


# ---------------------------------------------------------------------------
# [稳定接口] get_checkpointer
# ---------------------------------------------------------------------------


@contextmanager
def get_checkpointer(
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Iterator[SqliteSaver]:
    """生成用于外层图持久化的 ``SqliteSaver`` 的上下文管理器。

    父目录不存在时会自动创建；退出上下文时会清理 SQLite 连接。

    参数：
        db_path：SQLite 数据库文件路径，默认为当前工作目录下的
            ``data/checkpoints.sqlite``。

    生成：
        可直接作为 ``checkpointer`` 参数传给 ``graph.compile()`` 的
        ``SqliteSaver`` 实例。
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver
