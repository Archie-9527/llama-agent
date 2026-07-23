"""R2 lifecycle policy, deterministic compaction and conversation recall."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from agent_core.config import MemoryConfig
from agent_core.memory.models import ContextView, Lifecycle
from agent_core.memory.store import ContextStore

_PIN_MARKERS = ("记住", "不要忘记", "以后", "代号是", "校验码是", "偏好是")
_ARTIFACT_PATTERN = re.compile(r"artifact://[0-9a-fA-F]+")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]")


def _terms(text: str) -> set[str]:
    return {item.casefold() for item in _TOKEN_PATTERN.findall(text)}


def _approx_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)


def _clip(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n...[R2 compacted]...\n" + text[-tail:]


class LifecycleContextManager:
    def __init__(self, config: MemoryConfig, store: ContextStore | None) -> None:
        self.config = config
        self.store = store

    @property
    def enabled(self) -> bool:
        return self.config.lifecycle_context

    def extract_pinned_facts(self, text: str, source_id: str | None = None) -> list[dict]:
        facts: list[dict] = []
        for sentence in re.split(r"[。！？!?\n]+", str(text)):
            cleaned = sentence.strip()
            if cleaned and any(marker in cleaned for marker in _PIN_MARKERS):
                facts.append(
                    {
                        "text": _clip(cleaned, 500),
                        "source_type": "user",
                        "source_id": source_id,
                        "confidence": "confirmed",
                    }
                )
        return facts

    def prepare(self, state: dict, *, phase: str, engine: Any) -> ContextView:
        if not self.enabled:
            return ContextView()
        before_parts = [
            str(state.get("conversation_context", "")),
            json.dumps(state.get("pinned_facts", []), ensure_ascii=False),
            json.dumps(state.get("context_summary", {}), ensure_ascii=False),
        ]
        before = sum(engine.get_num_tokens(part) for part in before_parts if part)
        pinned = "\n".join(
            f"- {item.get('text', '')}"
            for item in state.get("pinned_facts", [])
            if item.get("text")
        )
        summary = self._render_summary(state.get("context_summary", {}))
        conversation = str(state.get("conversation_context", "")).strip()
        rendered = "\n\n".join(filter(None, (pinned, summary, conversation)))
        budget = min(
            self.config.context_budget_tokens,
            max(
                1,
                int(getattr(engine, "n_ctx", self.config.context_budget_tokens))
                - self.config.context_reserved_generation_tokens,
            ),
        )
        if engine.get_num_tokens(rendered) > budget:
            # Conversation context is the only fully discardable part here;
            # pinned facts and structured summaries remain protected.
            conversation = self._fit_text(
                conversation,
                max(0, budget - engine.get_num_tokens(pinned + summary)),
                engine,
            )
        view = ContextView(
            conversation_context=conversation,
            pinned_context=pinned,
            summary_context=summary,
            tokens_before=before,
            tokens_after=engine.get_num_tokens(
                "\n\n".join(filter(None, (pinned, summary, conversation)))
            ),
        )
        self._event(
            "context_budget_applied",
            phase=phase,
            tokens_before=view.tokens_before,
            tokens_after=view.tokens_after,
            budget_tokens=budget,
            reduction_tokens=max(0, view.tokens_before - view.tokens_after),
        )
        return view

    def commit(self, state: dict, *, event: str) -> dict:
        if not self.enabled:
            return state
        owner_id = self._task_owner()
        existing = {item.get("text") for item in state.get("pinned_facts", [])}
        for fact in self.extract_pinned_facts(
            str(state.get("current_user_input") or state.get("task_goal", "")),
            source_id=owner_id,
        ):
            if fact["text"] not in existing:
                state.setdefault("pinned_facts", []).append(fact)
                existing.add(fact["text"])

        if self.config.checkpoint_compaction:
            self._compact_execution_log(state, owner_id, event)
            self._compact_reflections(state, owner_id)
        state["context_version"] = int(state.get("context_version", 0)) + 1
        stats = state.setdefault("lifecycle_stats", {})
        stats["commits"] = int(stats.get("commits", 0)) + 1
        return state

    def select_conversation_context(
        self,
        *,
        conversation_id: str,
        turns: Iterable[Any],
        current_input: str,
        engine: Any,
    ) -> str:
        turns = list(turns)
        if not self.enabled:
            return ""
        query_terms = _terms(current_input)
        recent = turns[-self.config.hot_conversation_turns :]
        selected_ids = {turn.turn_id for turn in recent}
        candidates: list[tuple[float, Any]] = []
        all_blocks: list[str] = []
        for turn in turns:
            assistant = (
                turn.assistant_output
                if turn.status == "done" and turn.assistant_output
                else "<该轮执行失败，没有可依赖的助手回答>"
            )
            block = self._turn_block(turn, assistant)
            all_blocks.append(block)
            pinned = any(marker in turn.user_input for marker in _PIN_MARKERS)
            overlap = len(query_terms & _terms(block))
            score = overlap + (100.0 if pinned else 0.0)
            if turn.turn_id not in selected_ids and score > 0:
                candidates.append((score, turn))
            if self.store is not None:
                try:
                    self.store.put(
                        owner_type="conversation",
                        owner_id=conversation_id,
                        kind="conversation_turn",
                        lifecycle=(
                            Lifecycle.PINNED
                            if pinned
                            else Lifecycle.HOT
                            if turn.turn_id in selected_ids
                            else Lifecycle.WARM
                            if overlap
                            else Lifecycle.COLD
                        ),
                        content=block,
                        token_count=engine.get_num_tokens(block),
                        source_type="turn",
                        source_id=turn.turn_id,
                        importance=1.0 if pinned else 0.5,
                    )
                except (OSError, sqlite3.Error) as exc:
                    self._event(
                        "context_store_error",
                        operation="archive_conversation",
                        error=f"{type(exc).__name__}: {exc}",
                    )
        candidates.sort(key=lambda item: item[0], reverse=True)
        recalled = [
            turn
            for _, turn in candidates[: self.config.context_retrieval_top_k]
        ]
        combined = sorted(
            [*recalled, *recent], key=lambda turn: int(turn.turn_index)
        )
        unique: list[Any] = []
        seen: set[str] = set()
        for turn in combined:
            if turn.turn_id not in seen:
                unique.append(turn)
                seen.add(turn.turn_id)
        blocks = []
        for turn in unique:
            assistant = (
                turn.assistant_output
                if turn.status == "done" and turn.assistant_output
                else "<该轮执行失败，没有可依赖的助手回答>"
            )
            blocks.append(self._turn_block(turn, assistant))
        rendered = "\n\n".join(blocks)
        rendered = self._fit_text(
            rendered,
            self.config.context_retrieval_token_budget,
            engine,
            keep_tail=False,
        )
        self._event(
            "context_recalled",
            owner_type="conversation",
            owner_id=conversation_id,
            candidate_count=len(candidates),
            recalled_count=len(recalled),
            selected_turn_count=len(unique),
            recalled_tokens=engine.get_num_tokens(rendered),
        )
        tokens_before = engine.get_num_tokens("\n\n".join(all_blocks))
        tokens_after = engine.get_num_tokens(rendered)
        self._event(
            "context_budget_applied",
            phase="conversation",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            budget_tokens=self.config.context_retrieval_token_budget,
            reduction_tokens=max(0, tokens_before - tokens_after),
        )
        return rendered

    def record_conversation_turn(self, conversation_id: str, turn: Any, engine: Any) -> None:
        if not self.enabled or self.store is None:
            return
        assistant = (
            turn.assistant_output
            if turn.status == "done" and turn.assistant_output
            else "<该轮执行失败，没有可依赖的助手回答>"
        )
        block = self._turn_block(turn, assistant)
        pinned = any(marker in turn.user_input for marker in _PIN_MARKERS)
        try:
            self.store.put(
                owner_type="conversation",
                owner_id=conversation_id,
                kind="conversation_turn",
                lifecycle=Lifecycle.PINNED if pinned else Lifecycle.HOT,
                content=block,
                token_count=engine.get_num_tokens(block),
                source_type="turn",
                source_id=turn.turn_id,
                importance=1.0 if pinned else 0.7,
            )
        except (OSError, sqlite3.Error) as exc:
            # Persistence telemetry must never turn an otherwise successful
            # conversation turn into a failed user-visible turn.
            self._event(
                "context_store_error",
                operation="record_conversation",
                error=f"{type(exc).__name__}: {exc}",
            )

    def diagnostic_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def _compact_execution_log(self, state: dict, owner_id: str, event: str) -> None:
        records = state.get("execution_log", [])
        hot = self.config.hot_execution_records
        compact_until = max(0, len(records) - hot)
        pressure_threshold = min(
            self.config.summary_trigger_tokens,
            max(
                1,
                int(
                    self.config.context_budget_tokens
                    * self.config.context_trigger_ratio
                ),
            ),
        )
        active_tokens = sum(
            _approx_tokens(str(record.get("result", ""))) for record in records
        )
        if active_tokens > pressure_threshold and len(records) > 1:
            # Under token pressure retain only the newest record verbatim;
            # every older record remains recoverable through ContextStore.
            compact_until = max(compact_until, len(records) - 1)
        warm_from = max(0, len(records) - hot * 2)
        archived = state.setdefault("archived_context_ids", [])
        summary = state.setdefault("context_summary", {})
        completed = summary.setdefault("completed_steps", [])
        compacted_count = 0
        source_bytes = 0
        compacted_bytes = 0
        for record in records[compact_until:]:
            record["lifecycle"] = Lifecycle.HOT.value
        for index, record in enumerate(records[:compact_until]):
            if record.get("_r2_compacted"):
                continue
            original = str(record.get("result", ""))
            compacted = self._summarize_record(record)
            artifacts = tuple(_ARTIFACT_PATTERN.findall(original))
            lifecycle = Lifecycle.WARM if index >= warm_from else Lifecycle.COLD
            memory_id: str | None = None
            if self.store is not None:
                try:
                    item = self.store.put(
                        owner_type="task",
                        owner_id=owner_id,
                        kind="execution_record",
                        lifecycle=lifecycle,
                        content=original,
                        token_count=_approx_tokens(original),
                        source_type="execution_log",
                        source_id=str(index),
                        importance=0.8 if record.get("tool_used") else 0.6,
                        artifact_ids=artifacts,
                    )
                except (OSError, sqlite3.Error) as exc:
                    self._event(
                        "context_store_error",
                        operation="archive_execution",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    # Do not compact unless the recoverable raw record was
                    # durably written first.
                    continue
                memory_id = item.item_id
                if memory_id not in archived:
                    archived.append(memory_id)
            record["result"] = compacted
            record["memory_ref"] = memory_id
            record["lifecycle"] = lifecycle.value
            record["_r2_compacted"] = True
            step = str(record.get("step", "")).strip()
            if step and step not in completed:
                completed.append(step)
            compacted_count += 1
            source_bytes += len(original.encode("utf-8"))
            compacted_bytes += len(compacted.encode("utf-8"))
        if compacted_count:
            self._event(
                "context_compacted",
                trigger=event,
                record_count=compacted_count,
                source_bytes=source_bytes,
                compacted_bytes=compacted_bytes,
                bytes_saved=max(0, source_bytes - compacted_bytes),
                warm_records=sum(
                    1
                    for record in records
                    if record.get("lifecycle") == Lifecycle.WARM.value
                ),
                cold_records=sum(
                    1
                    for record in records
                    if record.get("lifecycle") == Lifecycle.COLD.value
                ),
            )

    def _compact_reflections(self, state: dict, owner_id: str) -> None:
        notes = state.get("reflection_notes", [])
        keep = self.config.hot_reflection_notes
        if len(notes) <= keep:
            return
        old = notes[:-keep]
        if self.store is not None:
            for index, note in enumerate(old):
                try:
                    item = self.store.put(
                        owner_type="task",
                        owner_id=owner_id,
                        kind="reflection_note",
                        lifecycle=Lifecycle.COLD,
                        content=str(note),
                        token_count=_approx_tokens(str(note)),
                        source_type="reflection",
                        source_id=str(index),
                    )
                except (OSError, sqlite3.Error) as exc:
                    self._event(
                        "context_store_error",
                        operation="archive_reflection",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    return
                if item.item_id not in state.setdefault("archived_context_ids", []):
                    state["archived_context_ids"].append(item.item_id)
        state["reflection_notes"] = notes[-keep:]

    def _summarize_record(self, record: dict) -> str:
        result = str(record.get("result", ""))
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            if parsed.get("virtualized"):
                parts = [str(parsed.get("summary", "")).strip()]
                if parsed.get("artifact_id"):
                    parts.append(f"artifact={parsed['artifact_id']}")
                return _clip("; ".join(filter(None, parts)), self.config.summary_target_chars)
            small = {
                key: value
                for key, value in parsed.items()
                if key not in {"content", "stdout", "stderr", "rows", "matches", "data"}
                and (value is None or isinstance(value, (str, int, float, bool)))
            }
            if small:
                return _clip(
                    json.dumps(small, ensure_ascii=False, default=str),
                    self.config.summary_target_chars,
                )
        return _clip(result, self.config.summary_target_chars)

    @staticmethod
    def _render_summary(summary: dict) -> str:
        lines: list[str] = []
        for fact in summary.get("confirmed_facts", []):
            text = fact.get("text") if isinstance(fact, dict) else fact
            if text:
                lines.append(f"- 已确认：{text}")
        for step in summary.get("completed_steps", [])[-8:]:
            lines.append(f"- 已完成：{step}")
        return "\n".join(lines)

    @staticmethod
    def _turn_block(turn: Any, assistant: str) -> str:
        return (
            f"[历史第 {int(turn.turn_index) + 1} 轮 | 状态={turn.status}]\n"
            f"用户：{turn.user_input}\n助手：{assistant}"
        )

    @staticmethod
    def _fit_text(
        text: str,
        budget: int,
        engine: Any,
        *,
        keep_tail: bool = True,
    ) -> str:
        if budget <= 0:
            return ""
        if engine.get_num_tokens(text) <= budget:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = text[-middle:] if keep_tail else text[:middle]
            if engine.get_num_tokens(candidate) <= budget:
                low = middle
            else:
                high = middle - 1
        return text[-low:] if keep_tail and low else text[:low]

    @staticmethod
    def _task_owner() -> str:
        from agent_core.telemetry import current_task_id

        return current_task_id() or "__unscoped__"

    @staticmethod
    def _event(event: str, **data: Any) -> None:
        from agent_core.telemetry import get_telemetry

        get_telemetry().record_event("lifecycle_events.jsonl", event, **data)


_lock = threading.Lock()
_manager = LifecycleContextManager(MemoryConfig(), None)


def initialize_lifecycle_context(config: MemoryConfig) -> LifecycleContextManager:
    config.validate()
    store = ContextStore(Path(config.context_store_path)) if config.lifecycle_context else None
    global _manager
    with _lock:
        if _manager.store is not None:
            _manager.store.close()
        _manager = LifecycleContextManager(config, store)
    return _manager


def get_lifecycle_context_manager() -> LifecycleContextManager:
    return _manager


def _reset_lifecycle_context_for_testing() -> None:
    initialize_lifecycle_context(MemoryConfig())
