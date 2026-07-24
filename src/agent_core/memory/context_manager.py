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

_PIN_MARKERS = (
    "记住",
    "不要忘记",
    "以后",
    "代号是",
    "校验码是",
    "偏好是",
    "最初",
    "更正",
    "修正为",
    "改为",
    "不再有效",
)
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
        self._store_path = Path(config.context_store_path)
        self._skipped_compaction_sources: set[tuple[str, int, str]] = set()

    @property
    def enabled(self) -> bool:
        return self.config.lifecycle_context

    def should_manage_conversation(
        self,
        *,
        turns: Iterable[Any],
        current_input: str,
        engine: Any,
        baseline_history_turns: int,
    ) -> bool:
        """Activate R2 only when it can reduce or recover real context."""
        if not self.enabled:
            return False
        turns = list(turns)
        if not turns:
            return False
        blocks = [
            self._turn_block(
                turn,
                turn.assistant_output
                if turn.status == "done" and turn.assistant_output
                else "<该轮执行失败，没有可依赖的助手回答>",
            )
            for turn in turns
        ]
        if (
            engine.get_num_tokens("\n\n".join(blocks))
            >= self.config.context_activation_tokens
        ):
            return True

        recent_ids = {
            turn.turn_id for turn in turns[-max(1, baseline_history_turns):]
        }
        query_terms = _terms(current_input)
        active_pinned = self._active_pinned_turn_ids(turns)
        return any(
            turn.turn_id not in recent_ids
            and turn.turn_id in active_pinned
            and bool(query_terms & _terms(self._turn_block(turn, "")))
            for turn in turns
        )

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
        if phase not in {"planner", "executor"}:
            return ContextView()
        pinned_items = [
            item
            for item in state.get("pinned_facts", [])
            if item.get("text")
        ]
        summary_data = state.get("context_summary", {})
        conversation = str(state.get("conversation_context", "")).strip()
        pinned = "\n".join(
            f"- {item.get('text', '')}"
            for item in pinned_items
        )
        summary = self._render_summary(summary_data)
        # ``completed_steps`` remains in state/checkpoints for diagnostics, but
        # it is already represented by the plan and execution history.  Only
        # model-facing semantic content should activate lifecycle injection.
        rendered_before = "\n\n".join(
            filter(None, (pinned, summary, conversation))
        )
        if not rendered_before:
            return ContextView()
        before = engine.get_num_tokens(rendered_before)
        budget = min(
            self.config.context_budget_tokens,
            max(
                1,
                int(getattr(engine, "n_ctx", self.config.context_budget_tokens))
                - self.config.context_reserved_generation_tokens,
            ),
        )
        if before > budget:
            # Conversation context is the only fully discardable part here;
            # pinned facts and structured summaries remain protected.
            protected = "\n\n".join(filter(None, (pinned, summary)))
            protected_tokens = (
                engine.get_num_tokens(protected) if protected else 0
            )
            conversation = self._fit_text(
                conversation,
                max(0, budget - protected_tokens),
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
        changed = False
        if self.config.checkpoint_compaction:
            execution_changed = self._compact_execution_log(
                state, owner_id, event
            )
            reflection_changed = self._compact_reflections(state, owner_id)
            changed = execution_changed or reflection_changed
        if not changed:
            return state
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
        latest_facts = self._latest_facts(turns)
        active_fact_turns = {
            key: item["turn_id"] for key, item in latest_facts.items()
        }
        active_pinned_ids = self._active_pinned_turn_ids(turns)
        unkeyed_pinned_ids = active_pinned_ids - set(
            active_fact_turns.values()
        )
        query_fact_keys = self._fact_keys(current_input)
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
            pinned = turn.turn_id in active_pinned_ids
            overlap = len(query_terms & _terms(block))
            turn_fact_keys = self._fact_keys(turn.user_input)
            relevant_pin = turn.turn_id in unkeyed_pinned_ids or any(
                active_fact_turns.get(key) == turn.turn_id
                for key in query_fact_keys
            )
            if not query_fact_keys:
                relevant_pin = pinned
            # Known keyed facts are rendered below as a deduplicated
            # authoritative block. Do not recall their old full turns, which
            # may also contain a superseded value for another fact key.
            score = (
                0.0
                if turn_fact_keys and turn.turn_id not in selected_ids
                else overlap + (100.0 if relevant_pin else 0.0)
            )
            if turn.turn_id not in selected_ids and score > 0:
                candidates.append((score, turn))
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
        fact_lines = [
            f"- {key}: {item['value']}"
            for key, item in latest_facts.items()
        ]
        blocks = (
            ["[最新会话事实（后写覆盖前写）]\n" + "\n".join(fact_lines)]
            if fact_lines
            else []
        )
        for turn in unique:
            assistant = (
                turn.assistant_output
                if turn.status == "done" and turn.assistant_output
                else "<该轮执行失败，没有可依赖的助手回答>"
            )
            blocks.append(self._turn_block(turn, assistant))
        # ConversationStore already owns the complete durable transcript.
        # Do not duplicate omitted turns into ContextStore.
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

    def record_conversation_turn(
        self,
        conversation_id: str,
        turn: Any,
        engine: Any,
        *,
        active: bool = True,
    ) -> None:
        # Active turns are already durable in ConversationStore. ContextStore
        # is intentionally a cold archive, not a duplicate conversation log.
        return

    def diagnostic_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def _compact_execution_log(
        self,
        state: dict,
        owner_id: str,
        event: str,
    ) -> bool:
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
        under_pressure = active_tokens > pressure_threshold
        if under_pressure and len(records) > 1:
            # Under token pressure retain only the newest record verbatim;
            # every older record remains recoverable through ContextStore.
            compact_until = max(compact_until, len(records) - 1)
        else:
            # Record count alone is not memory pressure. Keep small tasks on
            # the R1 representation even when they contain many tool calls.
            return False
        warm_from = max(0, len(records) - hot * 2)
        archived = state.setdefault("archived_context_ids", [])
        summary = state.setdefault("context_summary", {})
        completed = summary.setdefault("completed_steps", [])
        compacted_count = 0
        source_bytes = 0
        compacted_bytes = 0
        skipped_count = 0
        for index, record in enumerate(records[:compact_until]):
            if record.get("_r2_compacted"):
                continue
            original = str(record.get("result", ""))
            compacted = self._summarize_record(record)
            original_bytes = len(original.encode("utf-8"))
            candidate_bytes = len(compacted.encode("utf-8"))
            reduction_ratio = (
                max(0, original_bytes - candidate_bytes) / original_bytes
                if original_bytes
                else 0.0
            )
            if (
                original_bytes < self.config.context_min_compaction_bytes
                or reduction_ratio < self.config.context_min_compaction_ratio
            ):
                digest = str(hash(original))
                skip_key = (owner_id, index, digest)
                if skip_key not in self._skipped_compaction_sources:
                    self._skipped_compaction_sources.add(skip_key)
                    skipped_count += 1
                continue
            artifacts = tuple(_ARTIFACT_PATTERN.findall(original))
            lifecycle = Lifecycle.WARM if index >= warm_from else Lifecycle.COLD
            try:
                item = self._ensure_store().put(
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
            source_bytes += original_bytes
            compacted_bytes += candidate_bytes
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
        if skipped_count:
            self._event(
                "context_compaction_skipped",
                trigger=event,
                record_count=skipped_count,
                reason="insufficient_roi",
                min_source_bytes=self.config.context_min_compaction_bytes,
                min_reduction_ratio=self.config.context_min_compaction_ratio,
            )
        return bool(compacted_count)

    def _compact_reflections(self, state: dict, owner_id: str) -> bool:
        notes = state.get("reflection_notes", [])
        keep = self.config.hot_reflection_notes
        if len(notes) <= keep:
            return False
        old = notes[:-keep]
        if sum(len(str(note).encode("utf-8")) for note in old) < (
            self.config.context_min_compaction_bytes
        ):
            return False
        for index, note in enumerate(old):
            try:
                item = self._ensure_store().put(
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
                return False
            if item.item_id not in state.setdefault("archived_context_ids", []):
                state["archived_context_ids"].append(item.item_id)
        state["reflection_notes"] = notes[-keep:]
        return True

    def _ensure_store(self) -> ContextStore:
        if self.store is None:
            self.store = ContextStore(self._store_path)
        return self.store

    @staticmethod
    def _fact_keys(text: str) -> set[str]:
        patterns = (
            ("project_code", r"项目代号"),
            ("checksum", r"校验码"),
            ("deployment", r"部署环境"),
            ("service", r"服务(?:名称)?"),
            ("port", r"端口"),
            ("request_id", r"(?:故障请求号|request[_ -]?id)"),
        )
        return {
            key
            for key, pattern in patterns
            if re.search(pattern, text, re.IGNORECASE)
        }

    def _active_fact_turns(self, turns: Iterable[Any]) -> dict[str, str]:
        return {
            key: item["turn_id"]
            for key, item in self._latest_facts(turns).items()
        }

    def _latest_facts(
        self,
        turns: Iterable[Any],
    ) -> dict[str, dict[str, str]]:
        keyed: dict[str, dict[str, str]] = {}
        for turn in turns:
            if not any(marker in turn.user_input for marker in _PIN_MARKERS):
                continue
            for key in self._fact_keys(turn.user_input):
                value = self._extract_fact_value(turn.user_input, key)
                if value:
                    keyed[key] = {
                        "turn_id": turn.turn_id,
                        "value": value,
                    }
        return keyed

    @staticmethod
    def _extract_fact_value(text: str, key: str) -> str:
        patterns = {
            "project_code": r"项目代号\s*(?:是|为|改为|修正为|更正为)?\s*[:：]?\s*([A-Za-z0-9_.-]+)",
            "checksum": r"校验码\s*(?:是|为|改为|修正为|更正为)?\s*[:：]?\s*([A-Za-z0-9_.-]+)",
            "deployment": r"部署环境\s*(?:是|为|改为|修正为|更正为)?\s*[:：]?\s*([A-Za-z0-9_.-]+)",
            "service": r"服务(?:名称)?\s*(?:是|为|改为|修正为|更正为)?\s*[:：]?\s*([A-Za-z0-9_.-]+)",
            "port": r"端口\s*(?:是|为|改为|修正为|更正为)?\s*[:：]?\s*(\d+)",
            "request_id": r"(?:故障请求号|request[_ -]?id)\s*(?:是|为|改为|修正为|更正为)?\s*[:：=]?\s*([A-Za-z0-9_.-]+)",
        }
        match = re.search(patterns[key], text, re.IGNORECASE)
        return match.group(1) if match else ""

    def _active_pinned_turn_ids(self, turns: Iterable[Any]) -> set[str]:
        turns = list(turns)
        unkeyed: set[str] = set()
        for turn in turns:
            if not any(marker in turn.user_input for marker in _PIN_MARKERS):
                continue
            keys = self._fact_keys(turn.user_input)
            if not keys or not any(
                self._extract_fact_value(turn.user_input, key)
                for key in keys
            ):
                unkeyed.add(turn.turn_id)
        return unkeyed | set(self._active_fact_turns(turns).values())

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
            evidence: dict[str, Any] = {}
            for key in ("matches", "rows", "data"):
                value = parsed.get(key)
                if value not in (None, "", [], {}):
                    evidence[key] = self._bounded_json_value(value)
            preview_chars = max(128, self.config.summary_target_chars // 2)
            for key in ("content", "stdout", "stderr"):
                value = str(parsed.get(key) or "").strip()
                if value:
                    evidence[f"{key}_preview"] = _clip(value, preview_chars)
            if small or evidence:
                return _clip(
                    json.dumps(
                        {**small, **evidence},
                        ensure_ascii=False,
                        default=str,
                    ),
                    self.config.summary_target_chars,
                )
        return _clip(result, self.config.summary_target_chars)

    @staticmethod
    def _bounded_json_value(value: Any) -> Any:
        """Retain semantic evidence while bounding large collections."""
        if isinstance(value, list):
            if len(value) <= 4:
                return value
            return {
                "count": len(value),
                "head": value[:2],
                "tail": value[-2:],
            }
        if isinstance(value, dict):
            scalar = {
                key: item
                for key, item in value.items()
                if item is None or isinstance(item, (str, int, float, bool))
            }
            return scalar or {"keys": list(value)[:20]}
        return value

    @staticmethod
    def _render_summary(summary: dict) -> str:
        lines: list[str] = []
        for fact in summary.get("confirmed_facts", []):
            text = fact.get("text") if isinstance(fact, dict) else fact
            if text:
                lines.append(f"- 已确认：{text}")
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
    global _manager
    with _lock:
        if _manager.store is not None:
            _manager.store.close()
        # Open SQLite only after the first archive passes the ROI policy.
        _manager = LifecycleContextManager(config, None)
    return _manager


def get_lifecycle_context_manager() -> LifecycleContextManager:
    return _manager


def _reset_lifecycle_context_for_testing() -> None:
    initialize_lifecycle_context(MemoryConfig())
