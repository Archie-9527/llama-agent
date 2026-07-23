"""Application service layer — bridges the graph orchestration layer to CLI/API consumers.

``session.py`` is the single place that holds a compiled graph and a
checkpointer connection, manages ``thread_id`` lifecycle, constructs
initial ``AgentState``, and normalises all exceptions so that callers
(``cli.py``, web handlers) only deal with ``AgentCoreError``.

Public API:
    * ``RunConfig`` — immutable-ish configuration dataclass.
    * ``TaskRunner`` — start new tasks, resume interrupted ones,
      remember the last ``thread_id``, and release resources cleanly.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from agent_core.exceptions import AgentCoreError
from agent_core.graph.build_graph import build_graph
from agent_core.graph.checkpointer import get_checkpointer
from agent_core.graph.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# [STABLE] RunConfig — single place for all tunables
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """All tuneable parameters for a ``TaskRunner`` instance.

    New knobs should be added here with a sensible default so existing
    callers are not affected.

    Attributes:
        max_iterations: Safety cap on plan→reflect loops (default 6).
        db_path: Where the SQLite checkpoint database lives.
        last_thread_file: Tiny file that remembers the most recent
            ``thread_id`` for parameter-less resume.
    """

    max_iterations: int = 6
    db_path: Path = field(default_factory=lambda: Path("data/checkpoints.sqlite"))
    last_thread_file: Path = field(
        default_factory=lambda: Path("data/last_thread_id.txt")
    )


# ---------------------------------------------------------------------------
# [STABLE] TaskRunner — the application-level task executor
# ---------------------------------------------------------------------------


class TaskRunner:
    """Holds a compiled LangGraph graph + checkpoint connection and exposes
    a simple start / resume / close interface.

    Typical usage::

        runner = TaskRunner()
        tid, result = runner.start_new_task("Summarise the meeting notes.")
        runner.close()

        # Or as a context manager:
        with TaskRunner() as runner:
            tid, result = runner.start_new_task("Query tomorrow's weather")

        # Resume a crashed task:
        with TaskRunner() as runner:
            result = runner.resume_task(tid)
    """

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config or RunConfig()
        self._exit_stack = ExitStack()
        self._checkpointer = self._init_checkpointer()
        self._graph = build_graph(checkpointer=self._checkpointer)
        logger.info("TaskRunner initialised  db_path=%s", self.config.db_path)

    # ------------------------------------------------------------------
    # Resource initialisation
    # ------------------------------------------------------------------

    def _init_checkpointer(self):
        """Defensively initialise the ``SqliteSaver``.

        ``get_checkpointer`` is a ``@contextmanager`` that yields a
        saver.  We use ``ExitStack.enter_context`` so that the saver's
        cleanup is guaranteed regardless of which path we take, and so
        that ``close()`` can release everything in one call.
        """
        cm = get_checkpointer(self.config.db_path)
        return self._exit_stack.enter_context(cm)

    def close(self) -> None:
        """Release all underlying resources (DB connection, etc.).

        Safe to call multiple times — ``ExitStack`` is idempotent after
        the first close.
        """
        self._exit_stack.close()
        logger.info("TaskRunner resources released")

    def __enter__(self) -> "TaskRunner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Initial state construction
    # ------------------------------------------------------------------

    def _build_initial_state(
        self,
        task_goal: str,
        *,
        conversation_id: str | None = None,
        conversation_context: str = "",
    ) -> AgentState:
        """Construct a valid ``AgentState`` from a bare task-goal string.

        Every default value is defined here and *only* here — callers
        never assemble ``AgentState`` dicts by hand.
        """
        return AgentState(
            task_goal=task_goal,
            plan_steps=[],
            current_step_index=0,
            execution_log=[],
            reflection_notes=[],
            current_iteration=0,
            max_iterations=self.config.max_iterations,
            status="planning",
            final_answer="",
            error=None,
            conversation_id=conversation_id,
            current_user_input=task_goal,
            conversation_context=conversation_context,
            pinned_facts=[],
            context_summary={},
            archived_context_ids=[],
            context_version=0,
            lifecycle_stats={},
        )

    # ------------------------------------------------------------------
    # [STABLE] Public entry points
    # ------------------------------------------------------------------

    def start_new_task(
        self,
        task_goal: str,
        *,
        thread_id: str | None = None,
        conversation_id: str | None = None,
        conversation_context: str = "",
    ) -> tuple[str, AgentState]:
        """Start a brand-new task.

        Args:
            task_goal: A non-empty description of what the agent should do.

        Returns:
            A ``(thread_id, final_state)`` tuple.  The caller should save
            ``thread_id`` — it is needed for ``resume_task``.

        Raises:
            ValueError: If *task_goal* is empty or whitespace-only.
            AgentCoreError: If any orchestration-layer error occurs.
        """
        if not task_goal or not task_goal.strip():
            raise ValueError("task_goal must not be empty")

        thread_id = thread_id or str(uuid.uuid4())
        initial_state = self._build_initial_state(
            task_goal,
            conversation_id=conversation_id,
            conversation_context=conversation_context,
        )
        logger.info("Starting new task  thread_id=%s  goal=%r", thread_id, task_goal)

        from agent_core.telemetry import get_telemetry, telemetry_task

        telemetry = get_telemetry()
        with telemetry_task(thread_id):
            telemetry.record_event("lifecycle_events.jsonl", "task_started")
            result = self._invoke(initial_state, thread_id)
            telemetry.record_event(
                "lifecycle_events.jsonl",
                "task_finished",
                status=result.get("status"),
                final_answer_bytes=len(
                    str(result.get("final_answer", "")).encode("utf-8")
                ),
            )
        self._remember_thread_id(thread_id)
        return thread_id, result

    def resume_task(self, thread_id: str) -> AgentState:
        """Resume a task that was interrupted (crash / manual kill).

        **This method deliberately does not accept a ``task_goal``
        argument.**  LangGraph detects the existing checkpoint for the
        given ``thread_id`` and resumes from the last completed node —
        any newly supplied initial state would be silently ignored.
        Passing it would create a misleading API surface.

        Args:
            thread_id: The identifier returned by a previous
                ``start_new_task`` call.

        Returns:
            The final ``AgentState`` after the resumed run completes.

        Raises:
            ValueError: If *thread_id* is empty or ``None``.
            AgentCoreError: If any orchestration-layer error occurs.
        """
        if not thread_id:
            raise ValueError("thread_id must not be empty")

        logger.info("Resuming task  thread_id=%s", thread_id)
        from agent_core.telemetry import telemetry_task

        with telemetry_task(thread_id):
            return self._invoke(None, thread_id)

    # ------------------------------------------------------------------
    # Internal invoke — single choke-point for graph calls
    # ------------------------------------------------------------------

    def _invoke(
        self, state_or_none: AgentState | None, thread_id: str
    ) -> AgentState:
        """Unified graph-invoke entry point with exception normalisation.

        Every call to ``self._graph.invoke`` must go through here so
        that exception handling is applied consistently.  Callers above
        this layer only need to catch ``AgentCoreError``.
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            return self._graph.invoke(state_or_none, config=config)  # type: ignore[arg-type]
        except AgentCoreError:
            # Already a semantic exception — re-raise unchanged.
            raise
        except Exception as exc:
            # Defensive catch-all: any raw exception that slipped through
            # the graph layer (e.g. an unexpected LangGraph internal error)
            # gets wrapped so cli.py sees only AgentCoreError.
            logger.exception(
                "Unexpected error during task execution  thread_id=%s", thread_id
            )
            raise AgentCoreError(
                f"Unexpected error during task execution: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # thread_id persistence (app-layer concern — not in SqliteSaver)
    # ------------------------------------------------------------------

    def _remember_thread_id(self, thread_id: str) -> None:
        """Persist the most recent ``thread_id`` so ``get_last_thread_id``
        can retrieve it later."""
        path = self.config.last_thread_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(thread_id)

    def get_last_thread_id(self) -> str | None:
        """Return the last persisted ``thread_id``, or ``None``.

        Callers use this to support "resume the last task" without the
        user having to remember and re-type a UUID.
        """
        path = self.config.last_thread_file
        if not path.exists():
            return None
        content = path.read_text().strip()
        return content or None
