"""Fresh-process execution worker for one benchmark sample."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic_ns

from agent_core.benchmark.evaluator import evaluate
from agent_core.benchmark.models import BenchmarkCase
from agent_core.capabilities.bootstrap import bootstrap_capabilities
from agent_core.config import (
    load_app_config,
    load_engine_config,
    load_memory_config,
    load_tools_config,
    load_telemetry_config,
)
from agent_core.conversation.manager import ConversationConfig, ConversationManager
from agent_core.conversation.store import ConversationStore
from agent_core.graph.react_agent_factory import initialize_react_agent
from agent_core.llm_engine import get_engine, initialize_engine
from agent_core.session import TaskRunner
from agent_core.telemetry import initialize_telemetry


def _turn_result_payload(turn, result: dict) -> dict:
    """Return the minimal per-turn diagnostic persisted by a benchmark run."""
    return {
        "turn_index": turn.turn_index,
        "turn_id": turn.turn_id,
        "thread_id": turn.thread_id,
        "user_input": turn.user_input,
        "status": turn.status,
        "final_answer": result.get("final_answer", ""),
        "error": result.get("error") or turn.error,
        "reflection_notes": result.get("reflection_notes", []),
        "execution_log": result.get("execution_log", []),
    }


def _run_conversation_case(
    manager: ConversationManager,
    turns: tuple[str, ...],
) -> tuple[dict, list[dict], int | None]:
    """Run turns in order and stop after the first failed turn.

    Continuing after a failure creates misleading follow-up errors because the
    prior assistant answer was never produced.  Remaining turns are retained as
    explicit skipped records so the report still shows the full case shape.
    """
    conversation_id, turn, result = manager.start(turns[0])
    turn_results = [_turn_result_payload(turn, result)]
    failed_turn_index = 0 if result.get("status") != "done" else None

    for turn_index, turn_text in enumerate(turns[1:], start=1):
        if failed_turn_index is not None:
            turn_results.append(
                {
                    "turn_index": turn_index,
                    "user_input": turn_text,
                    "status": "skipped_due_to_previous_failure",
                }
            )
            continue
        turn, result = manager.continue_conversation(conversation_id, turn_text)
        turn_results.append(_turn_result_payload(turn, result))
        if result.get("status") != "done":
            failed_turn_index = turn_index

    combined_result = dict(result)
    combined_result["execution_log"] = [
        record
        for turn_result in turn_results
        for record in turn_result.get("execution_log", [])
    ]
    if failed_turn_index is not None and not combined_result.get("error"):
        combined_result["error"] = turn_results[failed_turn_index].get("error")
    return combined_result, turn_results, failed_turn_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    case = BenchmarkCase.from_dict(json.loads(args.case_json))
    started = monotonic_ns()
    result: dict
    conversation_store: ConversationStore | None = None
    turn_results: list[dict] = []
    failed_turn_index: int | None = None
    collector = None
    try:
        app = load_app_config(args.config)
        telemetry_config = load_telemetry_config(
            args.config,
            {
                "enabled": True,
                "output_dir": args.output_dir,
            },
        )
        collector = initialize_telemetry(telemetry_config)
        collector.record_process()
        initialize_engine(
            load_engine_config(args.config, {"seed": args.seed})
        )
        collector.record_process()
        os.environ["AGENT_ARTIFACT_STORAGE_DIR"] = str(
            args.output_dir / "artifacts"
        )
        os.environ["AGENT_MEMORY_CONTEXT_STORE_PATH"] = str(
            args.output_dir / "context_memory.sqlite"
        )
        bootstrap_capabilities(load_tools_config(args.config))
        from agent_core.artifacts.virtualizer import (
            initialize_artifact_virtualizer,
        )

        memory_config = load_memory_config(args.config)
        initialize_artifact_virtualizer(memory_config)
        from agent_core.memory import initialize_lifecycle_context

        initialize_lifecycle_context(memory_config)
        initialize_react_agent()

        runner_config = app.to_run_config()
        runner_config.db_path = args.output_dir / "checkpoints.sqlite"
        runner_config.last_thread_file = args.output_dir / "last_thread_id.txt"
        with TaskRunner(runner_config) as runner:
            if case.tasks:
                with ThreadPoolExecutor(
                    max_workers=max(1, min(case.concurrency, len(case.tasks)))
                ) as pool:
                    futures = [
                        pool.submit(runner.start_new_task, task)
                        for task in case.tasks
                    ]
                    subresults = [future.result()[1] for future in futures]
                result = {
                    "status": (
                        "done"
                        if all(item.get("status") == "done" for item in subresults)
                        else "failed"
                    ),
                    "final_answer": "\n".join(
                        str(item.get("final_answer", "")) for item in subresults
                    ),
                    "execution_log": [
                        record
                        for item in subresults
                        for record in item.get("execution_log", [])
                    ],
                    "subtask_count": len(subresults),
                    "subtask_statuses": [
                        item.get("status") for item in subresults
                    ],
                }
            elif case.turns:
                conversation_store = ConversationStore(
                    args.output_dir / "conversations.sqlite"
                )
                manager = ConversationManager(
                    runner,
                    conversation_store,
                    get_engine(),
                    ConversationConfig(
                        history_turns=app.conversation_history_turns,
                        history_token_budget=app.conversation_history_token_budget,
                    ),
                )
                result, turn_results, failed_turn_index = _run_conversation_case(
                    manager, case.turns
                )
                result["turn_results"] = turn_results
            else:
                _, result = runner.start_new_task(case.goal)
        duration_ms = (monotonic_ns() - started) / 1_000_000
        payload = {
            "case_id": case.case_id,
            "category": case.category,
            "seed": args.seed,
            "status": result.get("status"),
            "final_answer": result.get("final_answer", ""),
            "duration_ms": duration_ms,
            "evaluation": evaluate(case, result),
            "execution_log": result.get("execution_log", []),
            "error": result.get("error"),
            "reflection_notes": result.get("reflection_notes", []),
            "turn_results": turn_results,
            "failed_turn_index": failed_turn_index,
        }
        args.result_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return 0 if payload["evaluation"]["passed"] else 1
    except Exception as exc:
        payload = {
            "case_id": case.case_id,
            "category": case.category,
            "seed": args.seed,
            "status": "worker_failed",
            "final_answer": "",
            "duration_ms": (monotonic_ns() - started) / 1_000_000,
            "evaluation": {
                "passed": False,
                "checks": {},
                "error": f"{type(exc).__name__}: {exc}",
            },
        }
        args.result_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 2
    finally:
        if conversation_store is not None:
            conversation_store.close()
        if collector is not None:
            collector.record_process()
            collector.close()


if __name__ == "__main__":
    sys.exit(main())
