"""Checkpointer lifecycle — SQLite-based persistence for the outer graph.

The checkpointer writes a snapshot of the full ``AgentState`` after
each outer-graph node completes.  Crash recovery is driven by
``thread_id``: re-invoking with the same ``thread_id`` resumes from
the last completed node rather than restarting from ``planner_node``.

Design note:
    Only the **outer** graph receives the checkpointer.  The inner
    ReAct subgraph is deliberately not checkpointed — see §6.2 of
    the orchestration-layer design doc for the trade-off analysis.

Usage (in cli.py)::

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
# Default location — relative to the project root.
# The caller (build_graph / cli) may override this.
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path("data/checkpoints.sqlite")


# ---------------------------------------------------------------------------
# [STABLE] get_checkpointer
# ---------------------------------------------------------------------------


@contextmanager
def get_checkpointer(
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Iterator[SqliteSaver]:
    """Context manager that yields a ``SqliteSaver`` for outer-graph persistence.

    The parent directory is created automatically if it doesn't exist.
    The SQLite connection is cleaned up when the context exits.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``data/checkpoints.sqlite`` in the current working directory.

    Yields:
        A ``SqliteSaver`` instance ready to be passed as the
        ``checkpointer`` argument to ``graph.compile()``.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver
