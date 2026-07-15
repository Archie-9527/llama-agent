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

from agent_core.config import AppConfig, load_app_config, load_engine_config
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


def _setup_logging(level: str) -> None:
    """Configure the root logger.  Invalid level strings silently fall back
    to ``INFO``."""
    resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
    for field_name in ("max_iterations", "db_path", "last_thread_file", "log_level"):
        print(f"  {field_name} = {getattr(config, field_name)}")


def _print_engine_config(config: EngineConfig) -> None:
    print("\nActive engine config:")
    for field_name in (
        "model_path", "n_ctx", "n_gpu_layers", "chat_format", "temperature",
    ):
        print(f"  {field_name} = {getattr(config, field_name)}")


def _print_result(result: dict) -> None:
    print(f"Final status: {result['status']}")
    print(f"Plan steps: {result['plan_steps']}")
    print("Execution log:")
    for i, record in enumerate(result["execution_log"]):
        tag = f"[tool:{record['tool_used']}]" if record["tool_used"] else "[result]"
        print(f"  {i}. {tag} {record['result']}")
    if result.get("reflection_notes"):
        print("Reflection notes:")
        for note in result["reflection_notes"]:
            print(f"  {note}")


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

    # Step 3 — logging
    _setup_logging(app_config.log_level)
    logger = logging.getLogger("agent_core.cli")

    # Step 4 — show-config (must NOT initialise engine or create TaskRunner)
    if args.command == "show-config":
        _print_config(app_config)
        try:
            engine_config = load_engine_config(
                config_file=args.config,
                cli_overrides=_collect_engine_cli_overrides(args),
            )
            _print_engine_config(engine_config)
        except ValueError as exc:
            print(f"\n[Engine config] Load failed: {exc}")
        return EXIT_OK

    # Step 5 — engine initialisation (one-shot, must happen before TaskRunner)
    try:
        engine_config = load_engine_config(
            config_file=args.config,
            cli_overrides=_collect_engine_cli_overrides(args),
        )
        initialize_engine(engine_config)
    except ValueError as exc:
        print(f"Engine config error: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR
    except AgentEngineError as exc:
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
        logger.error("Tool bootstrap failed: %s", exc)
        print(f"Tool bootstrap failed: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR

    # Step 7 — ReAct inner subgraph construction (reads from capability_registry)
    try:
        from agent_core.graph.react_agent_factory import initialize_react_agent

        initialize_react_agent()
    except Exception as exc:
        logger.error("ReAct agent init failed: %s", exc)
        print(f"ReAct agent init failed: {exc}", file=sys.stderr)
        return EXIT_BUSINESS_ERROR

    # Step 8 — TaskRunner
    runner = TaskRunner(app_config.to_run_config())
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
            print(f"Resume task ID: {thread_id}")
            _print_result(result)
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
        runner.close()

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
