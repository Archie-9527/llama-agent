"""用于非流式交互 Agent CLI 的 Textual 应用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from agent_core.capability_registry import Capability
from agent_core.config import MemoryConfig, TuiConfig
from agent_core.interactive.commands import ParsedInput, parse_input
from agent_core.interactive.events import InteractiveEvent
from agent_core.interactive.session import InteractiveSession, TurnOutcome
from agent_core.llm_engine import EngineConfig
from agent_core.telemetry.accelerator_monitor import sample_accelerator
from agent_core.telemetry.process_monitor import sample_process


_PHASE_LABELS = {
    "planner": "正在规划",
    "executor": "正在执行",
    "reflector": "正在检查结果",
    "finalizer": "正在整理最终回答",
    "unknown": "正在处理",
}


@dataclass
class _UiRuntime:
    phase: str = "idle"
    active_tool: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    inference_calls: int = 0
    kv_tokens: int | None = None
    kv_capacity: int | None = None
    turn_index: int = 0

    def reset_turn(self) -> None:
        self.phase = "planning"
        self.active_tool = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.inference_calls = 0


class LlamaAgentApp(App[None]):
    TITLE = "llama-agent"
    SUB_TITLE = "Local Agent"

    CSS = """
    Screen {
        layout: vertical;
        background: #111318;
    }

    Header, Footer {
        background: #1b1f2a;
        color: #d7dae0;
    }

    #main {
        height: 1fr;
    }

    #left {
        width: 1fr;
        height: 100%;
    }

    #transcript {
        height: 1fr;
        border: round #3b4252;
        padding: 0 1;
        background: #111318;
    }

    #activity {
        height: 3;
        padding: 1 2;
        color: #a6adc8;
        background: #181b22;
    }

    #prompt {
        height: 3;
        border: round #5e81ac;
        margin: 0 0 1 0;
    }

    #status {
        width: 34;
        min-width: 28;
        height: 100%;
        border: round #3b4252;
        padding: 1 1;
        color: #cdd6f4;
        background: #181b22;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "退出"),
        ("ctrl+l", "clear_transcript", "清屏"),
        ("f2", "toggle_thinking", "Thinking"),
    ]

    def __init__(
        self,
        *,
        session: InteractiveSession,
        tui_config: TuiConfig,
        engine_config: EngineConfig,
        memory_config: MemoryConfig,
        capabilities: list[Capability],
        skill_names: set[str],
    ) -> None:
        super().__init__()
        self.session = session
        self.tui_config = tui_config
        self.engine_config = engine_config
        self.memory_config = memory_config
        self.capabilities = sorted(capabilities, key=lambda item: item.name)
        self.skill_names = set(skill_names)
        self.show_thinking = tui_config.show_thinking
        self.runtime = _UiRuntime()
        self._busy = False
        self._spinner_index = 0
        self._last_process: Any = None
        self._last_accelerator: Any = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield RichLog(
                    id="transcript",
                    wrap=True,
                    markup=False,
                    highlight=False,
                )
                yield Static("就绪", id="activity")
                yield Input(
                    placeholder="输入消息，或输入 /help 查看命令",
                    id="prompt",
                )
            if self.tui_config.show_sidebar:
                yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        transcript = self.query_one("#transcript", RichLog)
        transcript.write(
            Panel(
                Text(
                    "本地 Agent 已就绪。输入 /help 查看命令，/quit 退出。",
                    style="bold #88c0d0",
                ),
                title="llama-agent",
                border_style="#5e81ac",
            )
        )
        if self.session.conversation_id:
            transcript.write(
                Text(
                    f"已恢复会话：{self.session.conversation_id}",
                    style="dim",
                )
            )
            self.runtime.turn_index = len(self.session.history())
        self.set_interval(0.35, self._tick_activity)
        self.set_interval(
            self.tui_config.refresh_interval_ms / 1000,
            self._refresh_status,
        )
        self._refresh_status()
        self.query_one("#prompt", Input).focus()

    @on(Input.Submitted)
    def _on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value
        event.input.value = ""
        parsed = parse_input(
            raw,
            tool_names=(cap.name for cap in self.capabilities),
            skill_names=self.skill_names,
        )
        if parsed.kind == "empty":
            return
        if parsed.kind == "error":
            self._write_notice(parsed.error or "输入无效", error=True)
            return
        if parsed.kind == "command":
            self._handle_command(parsed)
            return
        if self._busy:
            self._write_notice("上一轮仍在运行，请等待完成。", error=True)
            return

        self._write_user(parsed.display_text or parsed.message)
        self._busy = True
        self.runtime.reset_turn()
        event.input.disabled = True
        self._set_activity("正在思考")
        self._execute_turn(parsed.message)

    @work(thread=True, exclusive=True, group="agent-turn")
    def _execute_turn(self, message: str) -> None:
        try:
            outcome = self.session.run_turn(
                message,
                on_event=lambda item: self.call_from_thread(
                    self._handle_runtime_event, item
                ),
            )
        except Exception as exc:
            self.call_from_thread(self._finish_error, exc)
            return
        self.call_from_thread(self._finish_turn, outcome)

    def _handle_runtime_event(self, event: InteractiveEvent) -> None:
        data = event.data
        if event.kind == "phase_changed":
            phase = str(data.get("phase") or "unknown")
            if data.get("status") == "started":
                self.runtime.phase = phase
                self._set_activity(_PHASE_LABELS.get(phase, f"正在执行 {phase}"))
            return
        if event.kind == "tool_started":
            tool = str(data.get("tool_name") or "tool")
            self.runtime.active_tool = tool
            self.runtime.phase = "tool"
            self._set_activity(f"正在执行工具 {tool}")
            return
        if event.kind == "tool_completed":
            tool = str(data.get("tool_name") or "tool")
            duration = float(data.get("duration_ms") or 0)
            preview = str(data.get("result_preview") or "")
            self.runtime.active_tool = None
            body = f"完成，用时 {duration:.1f} ms"
            limit = self.tui_config.tool_result_preview_chars
            if limit and preview:
                clipped = preview[:limit]
                if len(preview) > limit:
                    clipped += "…"
                body += f"\n{clipped}"
            self.query_one("#transcript", RichLog).write(
                Panel(
                    Text(body, style="#a3be8c"),
                    title=f"Tool · {tool}",
                    border_style="#a3be8c",
                )
            )
            return
        if event.kind == "tool_failed":
            tool = str(data.get("tool_name") or "tool")
            self.runtime.active_tool = None
            self._write_notice(
                f"工具 {tool} 执行失败：{data.get('error', 'unknown error')}",
                error=True,
            )
            return
        if event.kind == "inference_completed":
            self.runtime.inference_calls += 1
            self.runtime.input_tokens += int(data.get("input_tokens") or 0)
            self.runtime.output_tokens += int(data.get("output_tokens") or 0)
            kv = data.get("kv") or {}
            if kv.get("logical_tokens") is not None:
                self.runtime.kv_tokens = int(kv["logical_tokens"])
            if kv.get("capacity_tokens") is not None:
                self.runtime.kv_capacity = int(kv["capacity_tokens"])

    def _finish_turn(self, outcome: TurnOutcome) -> None:
        self._busy = False
        self.runtime.phase = "idle"
        self.runtime.turn_index = outcome.turn.turn_index + 1
        self._set_activity("就绪")

        if self.show_thinking:
            for record in outcome.reasoning:
                content = record.content
                limit = self.tui_config.thinking_max_chars
                if limit and len(content) > limit:
                    content = (
                        content[:limit]
                        + f"\n… Thinking 已截断，原始长度 {len(record.content)} 字符"
                    )
                self.query_one("#transcript", RichLog).write(
                    Panel(
                        Text(content, style="dim italic #7f849c"),
                        title=f"Thinking · {record.phase}",
                        border_style="#45475a",
                    )
                )

        answer = str(outcome.result.get("final_answer") or "").strip()
        if answer:
            self.query_one("#transcript", RichLog).write(
                Panel(
                    Markdown(answer),
                    title="Assistant",
                    border_style="#89b4fa",
                )
            )
        else:
            error = outcome.result.get("error") or "任务没有生成最终回答"
            self._write_notice(str(error), error=True)
        self._enable_prompt()

    def _finish_error(self, exc: Exception) -> None:
        self._busy = False
        self.runtime.phase = "failed"
        self._set_activity("执行失败")
        self._write_notice(f"{type(exc).__name__}: {exc}", error=True)
        self._enable_prompt()

    def _enable_prompt(self) -> None:
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self._refresh_status()

    def _handle_command(self, command: ParsedInput) -> None:
        name = command.name
        argument = command.argument.strip()
        if name == "quit":
            if self._busy:
                self._write_notice("任务运行中，暂不能退出。", error=True)
            else:
                self.exit()
            return
        if name == "help":
            self._write_help()
            return
        if name == "clear":
            self.action_clear_transcript()
            return
        if name == "new":
            self.session.new_conversation()
            self.runtime.turn_index = 0
            self._write_notice("已创建新的会话；下一条消息将生成新会话 ID。")
            return
        if name == "resume":
            if not argument:
                self._write_notice("用法：/resume <conversation_id>", error=True)
                return
            try:
                self.session.resume_conversation(argument)
                self.runtime.turn_index = len(self.session.history())
                self._write_notice(f"已切换到会话：{argument}")
            except ValueError as exc:
                self._write_notice(str(exc), error=True)
            return
        if name == "history":
            self._write_history()
            return
        if name == "conversations":
            values = self.session.conversations()
            if not values:
                self._write_notice("没有已保存的会话。")
            else:
                text = "\n".join(
                    f"{item.conversation_id}  updated={item.updated_at}"
                    for item in values
                )
                self._write_block("Conversations", text)
            return
        if name == "tools":
            text = "\n".join(
                f"/{cap.name} — {cap.description}" for cap in self.capabilities
            )
            self._write_block("Tools", text or "没有可用工具")
            return
        if name == "skills":
            caps = [
                cap for cap in self.capabilities if cap.name in self.skill_names
            ]
            text = "\n".join(
                f"/{cap.name} — {cap.description}" for cap in caps
            )
            self._write_block("Skills", text or "没有启用的 Skill")
            return
        if name == "thinking":
            self._set_thinking(argument)
            return
        if name == "stats":
            self._write_block("Runtime Stats", self._status_text())
            return
        if name == "config":
            self._write_block("Configuration", self._config_text())
            return
        self._write_notice(f"尚未实现命令：/{name}", error=True)

    def _set_thinking(self, argument: str) -> None:
        normalized = argument.casefold()
        if normalized in ("on", "show"):
            self.show_thinking = True
        elif normalized in ("off", "hide"):
            self.show_thinking = False
        elif normalized in ("", "toggle"):
            self.show_thinking = not self.show_thinking
        else:
            self._write_notice(
                "用法：/thinking on|off（也可按 F2）", error=True
            )
            return
        note = "显示" if self.show_thinking else "隐藏"
        suffix = (
            "；但 engine.disable_thinking=true，模型通常不会生成 Thinking"
            if self.show_thinking and self.engine_config.disable_thinking
            else ""
        )
        self._write_notice(f"原始 Thinking 已设为{note}{suffix}。")
        self._refresh_status()

    def _write_history(self) -> None:
        turns = self.session.history()
        if not turns:
            self._write_notice("当前会话没有历史轮次。")
            return
        lines: list[str] = []
        for turn in turns[-20:]:
            answer = turn.assistant_output or f"<{turn.status}>"
            lines.append(
                f"[{turn.turn_index + 1}] You: {turn.user_input}\n"
                f"    Assistant: {answer}"
            )
        self._write_block("History", "\n\n".join(lines))

    def _write_help(self) -> None:
        self._write_block(
            "Commands",
            "\n".join(
                (
                    "/help                         查看帮助",
                    "/quit                         退出",
                    "/new                          新建会话",
                    "/resume <conversation_id>     切换会话",
                    "/history                      查看当前会话历史",
                    "/conversations                查看会话列表",
                    "/tools                        查看工具",
                    "/skills                       查看 Skill",
                    "/tool <name> <request>         指定工具完成本轮",
                    "/skill <name> <request>        指定 Skill 完成本轮",
                    "/<capability> <request>        指定能力的短格式",
                    "/thinking on|off               显示或隐藏原始 Thinking",
                    "/stats                         显示运行状态",
                    "/config                        显示主要配置",
                    "/clear                         清空界面",
                    "//text                         发送以 / 开头的普通消息",
                )
            ),
        )

    def _write_user(self, value: str) -> None:
        self.query_one("#transcript", RichLog).write(
            Panel(Text(value), title="You", border_style="#88c0d0")
        )

    def _write_notice(self, value: str, *, error: bool = False) -> None:
        style = "#f38ba8" if error else "dim #a6adc8"
        self.query_one("#transcript", RichLog).write(Text(value, style=style))

    def _write_block(self, title: str, value: str) -> None:
        self.query_one("#transcript", RichLog).write(
            Panel(Text(value), title=title, border_style="#585b70")
        )

    def _set_activity(self, value: str) -> None:
        self.query_one("#activity", Static).update(value)

    def _tick_activity(self) -> None:
        if not self._busy:
            return
        dots = ("", ".", "..", "...")
        self._spinner_index = (self._spinner_index + 1) % len(dots)
        phase = self.runtime.phase
        if self.runtime.active_tool:
            label = f"正在执行工具 {self.runtime.active_tool}"
        else:
            label = _PHASE_LABELS.get(phase, "正在思考")
        self._set_activity(label + dots[self._spinner_index])

    def _refresh_status(self) -> None:
        self._last_process = sample_process()
        self._last_accelerator = sample_accelerator()
        if self.tui_config.show_sidebar:
            self.query_one("#status", Static).update(
                Panel(
                    Text(self._status_text()),
                    title="Runtime",
                    border_style="#585b70",
                )
            )

    def _status_text(self) -> str:
        process = self._last_process
        accelerator = self._last_accelerator
        model_name = Path(self.engine_config.model_path).name
        kv = (
            f"{self.runtime.kv_tokens}/{self.runtime.kv_capacity}"
            if self.runtime.kv_tokens is not None
            and self.runtime.kv_capacity is not None
            else "N/A"
        )
        gpu = "N/A"
        if accelerator and accelerator.supported:
            used = accelerator.process_used_bytes or accelerator.device_used_bytes
            total = accelerator.device_total_bytes
            gpu = f"{_mib(used)}/{_mib(total)} MiB"
        rss = _mib(process.rss_bytes) if process else "N/A"
        conversation = self.session.conversation_id or "<new>"
        return "\n".join(
            (
                f"Model\n  {model_name}",
                f"Phase\n  {self.runtime.phase}",
                f"Conversation\n  {conversation[:18]}",
                f"Turns\n  {self.runtime.turn_index}",
                "Tokens · current turn",
                f"  input  {self.runtime.input_tokens}",
                f"  output {self.runtime.output_tokens}",
                f"  calls  {self.runtime.inference_calls}",
                f"KV logical\n  {kv}",
                f"Process RSS\n  {rss} MiB",
                f"GPU memory\n  {gpu}",
                f"Thinking\n  {'shown' if self.show_thinking else 'hidden'}",
                "Optimizations",
                f"  R1 {'on' if self.memory_config.artifact_virtualization else 'off'}",
                f"  R2 {'on' if self.memory_config.lifecycle_context else 'off'}",
                f"  R3 {'on' if self.memory_config.kv_lifecycle else 'off'}",
                f"Tools\n  {len(self.capabilities)}",
            )
        )

    def _config_text(self) -> str:
        return "\n".join(
            (
                f"model_path = {self.engine_config.model_path}",
                f"n_ctx = {self.engine_config.n_ctx}",
                f"n_gpu_layers = {self.engine_config.n_gpu_layers}",
                f"chat_format = {self.engine_config.chat_format}",
                f"temperature = {self.engine_config.temperature}",
                f"max_tokens = {self.engine_config.max_tokens}",
                f"disable_thinking = {self.engine_config.disable_thinking}",
                f"show_thinking = {self.show_thinking}",
            )
        )

    def action_clear_transcript(self) -> None:
        self.query_one("#transcript", RichLog).clear()

    def action_toggle_thinking(self) -> None:
        self._set_thinking("toggle")


def _mib(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / 1024**2:.1f}"
