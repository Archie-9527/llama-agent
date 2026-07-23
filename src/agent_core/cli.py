"""CLI layer — the single user-facing entry point.

Responsibilities (and what is deliberately excluded):
    * Parse command-line arguments (argparse).
    * Load AppConfig + EngineConfig via config.py.
    * Call ``initialize_engine`` exactly once (Fail-Fast).
    * Drive ``TaskRunner`` for ``run`` / ``resume`` subcommands.
    * Print human-readable results.

Explicitly NOT:
    * Importing anything from the graph package.
    * Calling ``get_engine()`` directly.
    * Exiting the process (only ``main()`` returns an exit code).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agent_core.config import (
    AppConfig,
    load_app_config,
    load_engine_config,
    load_telemetry_config,
    load_tui_config,
)
from agent_core.exceptions import AgentCoreError, AgentEngineError
from agent_core.llm_engine import EngineConfig, initialize_engine
from agent_core.session import TaskRunner

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_BUSINESS_ERROR = 1
# 2 is reserved by argparse for invalid arguments
EXIT_NO_RESUMABLE_TASK = 3
EXIT_ENGINE_INIT_ERROR = 4


# ---------------------------------------------------------------------------
# [INTERNAL] Helpers
# ---------------------------------------------------------------------------


def _setup_logging(level: str, *, log_file: Path | None = None) -> None:
    """Configure the root logger.  Invalid level strings silently fall back
    to ``INFO``.  Full-screen TUI mode writes to a file so log lines cannot
    corrupt the terminal layout."""
    resolved = getattr(logging, level.upper(), logging.INFO)
    handlers = None
    force = False
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(log_file, encoding="utf-8")]
        force = True
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=force,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llama-agent", description="Local LLM Agent — CLI"
    )

    # --- global flags (apply to all subcommands) ---------------------------
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to TOML config file"
    )
    parser.add_argument(
        "--log-level", type=str, default=None, help="Override log level"
    )
    parser.add_argument(
        "--db-path", type=Path, default=None, help="Override checkpoint DB path"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Override GGUF model path"
    )
    parser.add_argument(
        "--n-gpu-layers", type=int, default=None, help="Override GPU layer count"
    )
    parser.add_argument(
        "--n-ctx", type=int, default=None, help="Override context window size"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---------------------------------------------------------------
    run_p = subparsers.add_parser("run", help="Start a new task")
    run_p.add_argument("goal", type=str, help="Task description")
    run_p.add_argument(
        "--max-iterations", type=int, default=None,
        help="Override max planning iterations",
    )

    # --- resume ------------------------------------------------------------
    resume_p = subparsers.add_parser("resume", help="Resume an interrupted task")
    resume_p.add_argument(
        "thread_id", type=str, nargs="?", default=None,
        help="Thread ID to resume (omit to resume the most recent task)",
    )

    # --- show-config -------------------------------------------------------
    subparsers.add_parser("show-config", help="Print merged config and exit")

    # --- user-visible multi-turn conversation -----------------------------
    continue_p = subparsers.add_parser(
        "continue", help="Continue or start a persistent conversation"
    )
    continue_p.add_argument("goal", type=str)
    continue_p.add_argument("--conversation-id", type=str, default=None)

    chat_p = subparsers.add_parser("chat", help="Interactive persistent chat")
    chat_p.add_argument("--conversation-id", type=str, default=None)

    tui_p = subparsers.add_parser(
        "cli", help="Full-screen interactive Agent terminal"
    )
    tui_p.add_argument("--conversation-id", type=str, default=None)
    tui_p.add_argument(
        "--show-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show or hide raw model thinking returned after each inference",
    )

    subparsers.add_parser(
        "list-conversations", help="List persistent conversations"
    )

    # --- benchmark parent process (does not load the model itself) --------
    benchmark_p = subparsers.add_parser(
        "benchmark", help="Run an isolated R0 benchmark suite"
    )
    benchmark_p.add_argument("--suite", type=Path, required=True)
    benchmark_p.add_argument(
        "--output-root", type=Path, default=Path("benchmark/results")
    )
    benchmark_p.add_argument("--round", dest="round_name", default="R0")
    benchmark_p.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        metavar="CASE_ID",
        help=(
            "Run only the selected benchmark case_id. Repeat --case to "
            "select multiple cases."
        ),
    )

    return parser


def _collect_app_cli_overrides(args: argparse.Namespace) -> dict:
    """Extract AppConfig overrides from parsed CLI args."""
    overrides: dict = {}
    if args.db_path is not None:
        overrides["db_path"] = args.db_path
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if getattr(args, "max_iterations", None) is not None:
        overrides["max_iterations"] = args.max_iterations
    return overrides


def _collect_engine_cli_overrides(args: argparse.Namespace) -> dict:
    """Extract EngineConfig overrides from parsed CLI args."""
    overrides: dict = {}
    if args.model_path is not None:
        overrides["model_path"] = args.model_path
    if args.n_gpu_layers is not None:
        overrides["n_gpu_layers"] = args.n_gpu_layers
    if args.n_ctx is not None:
        overrides["n_ctx"] = args.n_ctx
    return overrides


def _print_config(config: AppConfig) -> None:
    print("Active application config:")
    for field_name in (
        "max_iterations",
        "db_path",
        "last_thread_file",
        "log_level",
        "conversation_db_path",
        "last_conversation_file",
        "conversation_history_turns",
        "conversation_history_token_budget",
    ):
        print(f"  {field_name} = {getattr(config, field_name)}")


def _print_engine_config(config: EngineConfig) -> None:
    print("\nActive engine config:")
    for field_name in (
        "model_path",
        "n_ctx",
        "n_gpu_layers",
        "chat_format",
        "temperature",
        "disable_thinking",
    ):
        print(f"  {field_name} = {getattr(config, field_name)}")


def _print_tui_config(config) -> None:
    print("\nActive interactive CLI config:")
    for field_name in (
        "show_thinking",
        "thinking_max_chars",
        "show_sidebar",
        "refresh_interval_ms",
        "tool_result_preview_chars",
        "restore_last_conversation",
        "log_file",
    ):
        print(f"  {field_name} = {getattr(config, field_name)}")


def _print_result(result: dict) -> None:
    print(f"Final status: {result['status']}")
    print(f"Plan steps: {result['plan_steps']}")
    print("Execution log:")
    for i, record in enumerate(result["execution_log"]):
        tag = f"[tool:{record['tool_used']}]" if record["tool_used"] else "[result]"
        print(f"  {i}. {tag} {record['result']}")
    final_answer = str(result.get("final_answer", "")).strip()
    if final_answer:
        print("Final answer:")
        print(f"  {final_answer}")
    if result.get("reflection_notes"):
        print("Reflection notes:")
        for note in result["reflection_notes"]:
            print(f"  {note}")
    if result.get("error"):
        print(f"Error: {result['error']}")


# ---------------------------------------------------------------------------
# [STABLE] main — the CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, load config, initialise the engine, and dispatch.

    Returns an exit code (0–4).  This function never calls ``sys.exit()``
    itself — that is the caller's responsibility.
    """
    # Step 1 — argparse (SystemExit(2) for invalid args, not caught here)
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Step 2 — application config
    try:
        app_config = load_app_config(
            config_file=args.config,
            cli_overrides=_collect_app_cli_overrides(args),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Application config load failed: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR

    tui_config = None
    if args.command == "cli":
        try:
            tui_config = load_tui_config(
                config_file=args.config,
                cli_overrides={"show_thinking": args.show_thinking},
            )
        except ValueError as exc:
            print(f"Interactive CLI config error: {exc}", file=sys.stderr)
            return EXIT_BUSINESS_ERROR

    # Step 3 — logging
    _setup_logging(
        app_config.log_level,
        log_file=tui_config.log_file if tui_config is not None else None,
    )
    logger = logging.getLogger("agent_core.cli")

    # Benchmark is a parent-only command.  Each sample loads its own model in
    # a fresh worker process, preventing allocator/KV state contamination.
    if args.command == "benchmark":
        from agent_core.benchmark.runner import BenchmarkRunner

        if args.config is not None:
            config_path = args.config
        else:
            from agent_core.config import DEFAULT_CONFIG_SEARCH_PATHS

            config_path = next(
                (path for path in DEFAULT_CONFIG_SEARCH_PATHS if path.exists()),
                Path("agent_config.toml"),
            )
        try:
            run_dir = BenchmarkRunner(
                config_file=config_path,
                suite_file=args.suite,
                output_root=args.output_root,
                round_name=args.round_name,
                engine_overrides=_collect_engine_cli_overrides(args),
                case_ids=tuple(args.case_ids),
            ).run()
        except Exception as exc:
            logger.exception("Benchmark failed")
            print(f"Benchmark failed: {exc}", file=sys.stderr)
            return EXIT_BUSINESS_ERROR
        print(f"Benchmark completed: {run_dir}")
        print(f"Report: {run_dir / 'report.md'}")
        return EXIT_OK

    # Step 4 — show-config (must NOT initialise engine or create TaskRunner)
    if args.command == "show-config":
        _print_config(app_config)
        try:
            engine_config = load_engine_config(
                config_file=args.config,
                cli_overrides=_collect_engine_cli_overrides(args),
            )
            _print_engine_config(engine_config)
            _print_tui_config(load_tui_config(config_file=args.config))
        except ValueError as exc:
            print(f"\n[Engine config] Load failed: {exc}")
        return EXIT_OK

    if args.command == "list-conversations":
        from agent_core.conversation.store import ConversationStore

        store = ConversationStore(app_config.conversation_db_path)
        try:
            conversations = store.list_conversations()
            if not conversations:
                print("No conversations found.")
            for conversation in conversations:
                turns = store.list_turns(conversation.conversation_id)
                print(
                    f"{conversation.conversation_id}  turns={len(turns)}  "
                    f"updated={conversation.updated_at}"
                )
        finally:
            store.close()
        return EXIT_OK

    try:
        telemetry_config = load_telemetry_config(config_file=args.config)
        from agent_core.telemetry import initialize_telemetry

        telemetry_collector = initialize_telemetry(telemetry_config)
    except ValueError as exc:
        print(f"Telemetry config error: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR

    # Step 5 — engine initialisation (one-shot, must happen before TaskRunner)
    try:
        engine_config = load_engine_config(
            config_file=args.config,
            cli_overrides=_collect_engine_cli_overrides(args),
        )
        initialize_engine(engine_config)
    except ValueError as exc:
        telemetry_collector.close()
        print(f"Engine config error: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR
    except AgentEngineError as exc:
        telemetry_collector.close()
        logger.error("Model load failed: %s", exc)
        print(f"Model load failed: {exc}", file=sys.stderr)
        return EXIT_ENGINE_INIT_ERROR

    # Step 6 — tool bootstrap (must happen BEFORE ReAct subgraph construction)
    try:
        from agent_core.config import load_tools_config
        from agent_core.capabilities.bootstrap import bootstrap_capabilities

        tools_config = load_tools_config(config_file=args.config)
        bootstrap_capabilities(tools_config)
    except Exception as exc:
        telemetry_collector.close()
        logger.error("Tool bootstrap failed: %s", exc)
        print(f"Tool bootstrap failed: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR

    # Step 7 — ReAct inner subgraph construction (reads from capability_registry)
    try:
        from agent_core.artifacts.virtualizer import (
            initialize_artifact_virtualizer,
        )
        from agent_core.config import load_memory_config
        from agent_core.graph.react_agent_factory import initialize_react_agent

        memory_config = load_memory_config(args.config)
        initialize_artifact_virtualizer(memory_config)
        from agent_core.memory import initialize_lifecycle_context

        initialize_lifecycle_context(memory_config)
        initialize_react_agent()
    except Exception as exc:
        telemetry_collector.close()
        logger.error("ReAct agent init failed: %s", exc)
        print(f"ReAct agent init failed: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR

    # Step 8 — TaskRunner
    try:
        runner = TaskRunner(app_config.to_run_config())
    except Exception as exc:
        telemetry_collector.close()
        logger.error("Task runner init failed: %s", exc)
        print(f"Task runner init failed: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR
    conversation_store = None
    try:
        # Step 9 — dispatch
        if args.command == "run":
            thread_id, result = runner.start_new_task(args.goal)
            print(f"Task ID: {thread_id}")
            _print_result(result)
            return EXIT_OK

        if args.command == "resume":
            thread_id = args.thread_id or runner.get_last_thread_id()
            if thread_id is None:
                print(
                    "No resumable task found.  Use 'run' to start a new task.",
                    file=sys.stderr,
                )
                return EXIT_NO_RESUMABLE_TASK
            result = runner.resume_task(thread_id)
            from agent_core.conversation.manager import (
                ConversationConfig,
                ConversationManager,
            )
            from agent_core.conversation.store import ConversationStore

            conversation_store = ConversationStore(app_config.conversation_db_path)
            if conversation_store.get_turn_by_thread(thread_id) is not None:
                from agent_core.llm_engine import get_engine

                ConversationManager(
                    runner,
                    conversation_store,
                    get_engine(),
                    ConversationConfig(
                        history_turns=app_config.conversation_history_turns,
                        history_token_budget=app_config.conversation_history_token_budget,
                    ),
                ).reconcile_resumed_turn(thread_id, result)
            print(f"Resume task ID: {thread_id}")
            _print_result(result)
            return EXIT_OK

        if args.command in ("continue", "chat", "cli"):
            from agent_core.conversation.manager import (
                ConversationConfig,
                ConversationManager,
            )
            from agent_core.conversation.store import ConversationStore
            from agent_core.llm_engine import get_engine

            conversation_store = ConversationStore(app_config.conversation_db_path)
            manager = ConversationManager(
                runner,
                conversation_store,
                get_engine(),
                ConversationConfig(
                    history_turns=app_config.conversation_history_turns,
                    history_token_budget=app_config.conversation_history_token_budget,
                ),
            )
            restore_last = (
                tui_config.restore_last_conversation
                if args.command == "cli" and tui_config is not None
                else True
            )
            conversation_id = args.conversation_id or (
                _read_last_conversation(app_config.last_conversation_file)
                if restore_last
                else None
            )

            if args.command == "cli":
                from agent_core.capability_registry import list_capabilities
                from agent_core.interactive.session import InteractiveSession
                from agent_core.tui import discover_skill_names, run_tui

                assert tui_config is not None
                interactive_session = InteractiveSession(
                    manager,
                    conversation_store,
                    conversation_id=conversation_id,
                    last_conversation_file=app_config.last_conversation_file,
                )
                try:
                    run_tui(
                        session=interactive_session,
                        tui_config=tui_config,
                        engine_config=engine_config,
                        memory_config=memory_config,
                        capabilities=list_capabilities(),
                        skill_names=discover_skill_names(tools_config),
                    )
                except RuntimeError as exc:
                    print(f"Interactive CLI failed: {exc}", file=sys.stderr)
                    return EXIT_BUSINESS_ERROR
                return EXIT_OK

            def run_turn(text: str) -> None:
                nonlocal conversation_id
                if conversation_id is None:
                    conversation_id, _, turn_result = manager.start(text)
                    _remember_conversation(
                        app_config.last_conversation_file, conversation_id
                    )
                else:
                    _, turn_result = manager.continue_conversation(
                        conversation_id, text
                    )
                print(f"Conversation ID: {conversation_id}")
                _print_result(turn_result)

            if args.command == "continue":
                run_turn(args.goal)
                return EXIT_OK

            print("Interactive chat. Type /exit to leave.")
            while True:
                try:
                    text = input("You> ").strip()
                except EOFError:
                    break
                if text in ("/exit", "/quit"):
                    break
                if text:
                    run_turn(text)
            return EXIT_OK

    # Step 10 — exception normalisation
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR
    except AgentCoreError as exc:
        logger.error("Task execution failed: %s", exc)
        print(f"Task execution failed: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR
    finally:
        if conversation_store is not None:
            conversation_store.close()
        runner.close()
        telemetry_collector.close()

    return EXIT_OK


def _read_last_conversation(path: Path) -> str | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _remember_conversation(path: Path, conversation_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(conversation_id, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
