from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from agent_core.config import load_tui_config
from agent_core.conversation.models import Conversation, Turn
from agent_core.interactive.commands import parse_input
from agent_core.interactive.events import emit_interactive_event
from agent_core.interactive.session import InteractiveSession


def _turn(*, conversation_id: str = "conversation-1", index: int = 0) -> Turn:
    return Turn(
        turn_id=f"turn-{index}",
        conversation_id=conversation_id,
        thread_id=f"thread-{index}",
        turn_index=index,
        user_input="hello",
        assistant_output="world",
        status="done",
        created_at="now",
        updated_at="now",
        error=None,
    )


def test_cli_parser_accepts_interactive_command():
    from agent_core.cli import _build_parser

    args = _build_parser().parse_args(
        ["cli", "--conversation-id", "c1", "--show-thinking"]
    )
    assert args.command == "cli"
    assert args.conversation_id == "c1"
    assert args.show_thinking is True


def test_tui_config_file_and_environment_override(
    tmp_path: Path, monkeypatch
):
    config_file = tmp_path / "agent.toml"
    config_file.write_text(
        "\n".join(
            (
                "[tui]",
                "show_thinking = false",
                "thinking_max_chars = 1234",
                'log_file = "runtime/tui.log"',
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_TUI_SHOW_THINKING", "true")

    config = load_tui_config(config_file)

    assert config.show_thinking is True
    assert config.thinking_max_chars == 1234
    assert config.log_file == Path("runtime/tui.log")


def test_parse_plain_and_escaped_input():
    plain = parse_input("你好")
    escaped = parse_input("//literal")

    assert plain.kind == "message"
    assert plain.message == "你好"
    assert escaped.kind == "message"
    assert escaped.message == "/literal"


def test_parse_capability_short_form_constrains_agent():
    parsed = parse_input(
        "/count_lines dir_path=src pattern=*.py",
        tool_names=("count_lines", "read_file"),
        skill_names=("count_lines",),
    )

    assert parsed.kind == "message"
    assert parsed.name == "count_lines"
    assert "必须使用 count_lines" in parsed.message
    assert "dir_path=src" in parsed.message
    assert parsed.display_text.startswith("/count_lines")


def test_parse_tool_and_skill_directives():
    tool = parse_input(
        "/tool read_file path=README.md",
        tool_names=("read_file", "count_lines"),
        skill_names=("count_lines",),
    )
    skill = parse_input(
        "/skill count_lines 统计 src",
        tool_names=("read_file", "count_lines"),
        skill_names=("count_lines",),
    )

    assert tool.kind == "message" and tool.name == "read_file"
    assert skill.kind == "message" and skill.name == "count_lines"


def test_unknown_slash_command_is_not_sent_to_model():
    parsed = parse_input("/does-not-exist", tool_names=("read_file",))
    assert parsed.kind == "error"
    assert "未知命令" in str(parsed.error)


def test_interactive_session_collects_reasoning_without_persisting_it(
    tmp_path: Path,
):
    manager = MagicMock()
    turn = _turn()

    def start(_text: str):
        emit_interactive_event(
            "inference_completed",
            phase="planner",
            reasoning_content="raw planner thought",
            input_tokens=10,
            output_tokens=4,
            duration_ms=25.0,
        )
        return (
            "conversation-1",
            turn,
            {"status": "done", "final_answer": "world"},
        )

    manager.start.side_effect = start
    store = MagicMock()
    session = InteractiveSession(
        manager,
        store,
        conversation_id=None,
        last_conversation_file=tmp_path / "last.txt",
    )
    observed = []

    outcome = session.run_turn("hello", on_event=observed.append)

    assert outcome.result["final_answer"] == "world"
    assert len(outcome.reasoning) == 1
    assert outcome.reasoning[0].phase == "planner"
    assert outcome.reasoning[0].content == "raw planner thought"
    assert observed[0].kind == "inference_completed"
    assert (tmp_path / "last.txt").read_text() == "conversation-1"


def test_interactive_session_can_resume_existing_conversation(tmp_path: Path):
    manager = MagicMock()
    store = MagicMock()
    store.get_conversation.return_value = Conversation("c2", "now", "now")
    store.list_turns.return_value = [_turn(conversation_id="c2")]
    session = InteractiveSession(
        manager,
        store,
        conversation_id=None,
        last_conversation_file=tmp_path / "last.txt",
    )

    session.resume_conversation("c2")

    assert session.conversation_id == "c2"
    assert session.history()[0].conversation_id == "c2"
    assert (tmp_path / "last.txt").read_text() == "c2"


def test_new_conversation_clears_persisted_resume_pointer(tmp_path: Path):
    last_file = tmp_path / "last.txt"
    last_file.write_text("old-conversation", encoding="utf-8")
    session = InteractiveSession(
        MagicMock(),
        MagicMock(),
        conversation_id="old-conversation",
        last_conversation_file=last_file,
    )

    session.new_conversation()

    assert session.conversation_id is None
    assert not last_file.exists()


def test_textual_app_mounts_with_status_and_prompt():
    from agent_core.config import MemoryConfig, TuiConfig
    from agent_core.llm_engine import EngineConfig
    from agent_core.tui.app import LlamaAgentApp

    async def exercise() -> None:
        session = MagicMock()
        session.conversation_id = None
        session.history.return_value = []
        app = LlamaAgentApp(
            session=session,
            tui_config=TuiConfig(refresh_interval_ms=1000),
            engine_config=EngineConfig(model_path="/tmp/model.gguf"),
            memory_config=MemoryConfig(),
            capabilities=[],
            skill_names=set(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.query_one("#prompt") is not None
            assert app.query_one("#status") is not None

    asyncio.run(exercise())
