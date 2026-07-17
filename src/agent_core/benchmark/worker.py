"""Fresh-process execution worker for one benchmark sample."""

from __future__ import annotations

import argparse
import json
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
    load_tools_config,
    load_telemetry_config,
)
from agent_core.conversation.manager import ConversationConfig, ConversationManager
from agent_core.conversation.store import ConversationStore
from agent_core.graph.react_agent_factory import initialize_react_agent
from agent_core.llm_engine import get_engine, initialize_engine
from agent_core.session import TaskRunner
from agent_core.telemetry import initialize_telemetry


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
        initialize_engine(load_engine_config(args.config))
        collector.record_process()
        bootstrap_capabilities(load_tools_config(args.config))
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
                conversation_id, _, result = manager.start(case.turns[0])
                for turn_text in case.turns[1:]:
                    _, result = manager.continue_conversation(
                        conversation_id, turn_text
                    )
            else:
                _, result = runner.start_new_task(case.goal)
        duration_ms = (monotonic_ns() - started) / 1_000_000
        payload = {
            "case_id": case.case_id,
            "category": case.category,
            "status": result.get("status"),
            "final_answer": result.get("final_answer", ""),
            "duration_ms": duration_ms,
            "evaluation": evaluate(case, result),
            "execution_log": result.get("execution_log", []),
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
