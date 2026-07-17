"""Application service for user-visible multi-turn conversations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from agent_core.conversation.models import Turn
from agent_core.conversation.store import ConversationStore
from agent_core.llm_engine import ChatLlamaCpp
from agent_core.session import TaskRunner


@dataclass(frozen=True)
class ConversationConfig:
    history_turns: int = 8
    history_token_budget: int = 4096


class ConversationManager:
    def __init__(
        self,
        runner: TaskRunner,
        store: ConversationStore,
        engine: ChatLlamaCpp,
        config: ConversationConfig | None = None,
    ) -> None:
        self.runner = runner
        self.store = store
        self.engine = engine
        self.config = config or ConversationConfig()

    def start(self, user_input: str) -> tuple[str, Turn, dict]:
        conversation_id = str(uuid.uuid4())
        self.store.create_conversation(conversation_id)
        turn, result = self.continue_conversation(conversation_id, user_input)
        return conversation_id, turn, result

    def continue_conversation(
        self, conversation_id: str, user_input: str
    ) -> tuple[Turn, dict]:
        if not user_input or not user_input.strip():
            raise ValueError("user_input must not be empty")
        if self.store.get_conversation(conversation_id) is None:
            raise ValueError(f"conversation does not exist: {conversation_id}")

        thread_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        history = self._render_history(conversation_id)
        task_goal = self._compose_goal(history, user_input)
        self.store.create_running_turn(
            turn_id=turn_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            user_input=user_input,
        )

        try:
            _, result = self.runner.start_new_task(task_goal, thread_id=thread_id)
            answer = str(result.get("final_answer", "")).strip()
            status = str(result.get("status", "failed"))
            error = None if status == "done" else self._result_error(result)
            turn = self.store.finish_turn(
                turn_id,
                status=status,
                assistant_output=answer,
                error=error,
            )
            return turn, result
        except Exception as exc:
            self.store.finish_turn(
                turn_id,
                status="failed",
                assistant_output="",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def reconcile_resumed_turn(self, thread_id: str, result: dict) -> Turn | None:
        turn = self.store.get_turn_by_thread(thread_id)
        if turn is None:
            return None
        return self.store.finish_turn(
            turn.turn_id,
            status=str(result.get("status", "failed")),
            assistant_output=str(result.get("final_answer", "")),
            error=None if result.get("status") == "done" else self._result_error(result),
        )

    def _render_history(self, conversation_id: str) -> str:
        completed = [
            turn
            for turn in self.store.list_turns(conversation_id)
            if turn.status == "done" and turn.assistant_output
        ][-self.config.history_turns :]
        selected: list[str] = []
        used = 0
        for turn in reversed(completed):
            block = (
                f"[历史第 {turn.turn_index + 1} 轮]\n"
                f"用户：{turn.user_input}\n助手：{turn.assistant_output}"
            )
            tokens = self.engine.get_num_tokens(block)
            if selected and used + tokens > self.config.history_token_budget:
                break
            if tokens > self.config.history_token_budget and not selected:
                continue
            selected.append(block)
            used += tokens
        return "\n\n".join(reversed(selected))

    @staticmethod
    def _compose_goal(history: str, user_input: str) -> str:
        if not history:
            return user_input
        return (
            "以下是同一会话中已经完成的历史轮次。回答当前请求时可以引用这些"
            "事实，但不要把历史请求误当成当前任务重新执行。\n\n"
            f"{history}\n\n[当前用户请求]\n{user_input}"
        )

    @staticmethod
    def _result_error(result: dict) -> str:
        if result.get("error"):
            return str(result["error"])
        notes = result.get("reflection_notes") or []
        return str(notes[-1]) if notes else "task did not complete"
