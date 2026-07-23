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
        from agent_core.memory import get_lifecycle_context_manager

        lifecycle = get_lifecycle_context_manager()
        historical_turns = [
            turn
            for turn in self.store.list_turns(conversation_id)
            if turn.status != "running" and turn.user_input.strip()
        ]
        if lifecycle.enabled:
            history = lifecycle.select_conversation_context(
                conversation_id=conversation_id,
                turns=historical_turns,
                current_input=user_input,
                engine=self.engine,
            )
            task_goal = user_input
        else:
            history = self._render_history(conversation_id)
            task_goal = self._compose_goal(history, user_input)
        self.store.create_running_turn(
            turn_id=turn_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            user_input=user_input,
        )

        try:
            _, result = self.runner.start_new_task(
                task_goal,
                thread_id=thread_id,
                conversation_id=conversation_id if lifecycle.enabled else None,
                conversation_context=history if lifecycle.enabled else "",
            )
            answer = str(result.get("final_answer", "")).strip()
            status = str(result.get("status", "failed"))
            error = None if status == "done" else self._result_error(result)
            turn = self.store.finish_turn(
                turn_id,
                status=status,
                assistant_output=answer,
                error=error,
            )
            lifecycle.record_conversation_turn(
                conversation_id, turn, self.engine
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
        # A failed turn's user input is still a valid conversation fact.  Its
        # assistant output is not trusted and is therefore never replayed.
        historical = [
            turn
            for turn in self.store.list_turns(conversation_id)
            if turn.status != "running" and turn.user_input.strip()
        ][-self.config.history_turns :]
        selected: list[str] = []
        used = 0
        for turn in reversed(historical):
            assistant = (
                turn.assistant_output
                if turn.status == "done" and turn.assistant_output
                else "<该轮执行失败，没有可依赖的助手回答>"
            )
            block = (
                f"[历史第 {turn.turn_index + 1} 轮 | 状态={turn.status}]\n"
                f"用户：{turn.user_input}\n助手：{assistant}"
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
            "以下是同一会话的历史轮次。历史中的用户消息是可信的会话事实；"
            "失败轮次的助手回答不可用。回答当前请求时应优先使用历史中已经"
            "给出的事实，不要把历史请求重新执行。若当前问题询问“上一轮”"
            "或“此前告诉你的内容”，并且答案已在历史中，直接回答，不要调用"
            "网页、Shell 或其他外部工具。\n\n"
            f"{history}\n\n[当前用户请求]\n{user_input}"
        )

    @staticmethod
    def _result_error(result: dict) -> str:
        if result.get("error"):
            return str(result["error"])
        notes = result.get("reflection_notes") or []
        return str(notes[-1]) if notes else "task did not complete"
