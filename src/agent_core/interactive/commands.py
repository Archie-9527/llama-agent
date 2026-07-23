"""Slash-command parser for the interactive terminal."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ParsedInput:
    kind: str
    name: str = ""
    argument: str = ""
    message: str = ""
    display_text: str = ""
    error: str | None = None


_LOCAL_COMMANDS = {
    "help",
    "quit",
    "exit",
    "new",
    "resume",
    "history",
    "conversations",
    "tools",
    "skills",
    "thinking",
    "stats",
    "config",
    "clear",
}


def parse_input(
    text: str,
    *,
    tool_names: Iterable[str] = (),
    skill_names: Iterable[str] = (),
) -> ParsedInput:
    """Parse user input without executing anything.

    ``/<capability> request`` and ``/tool <capability> request`` constrain the
    normal Agent turn to that capability; they do not bypass the Agent's tool
    validation/execution pipeline.
    """

    stripped = text.strip()
    if not stripped:
        return ParsedInput(kind="empty")
    if stripped.startswith("//"):
        return ParsedInput(
            kind="message",
            message=stripped[1:],
            display_text=stripped[1:],
        )
    if not stripped.startswith("/"):
        return ParsedInput(
            kind="message", message=stripped, display_text=stripped
        )

    raw = stripped[1:]
    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        return ParsedInput(kind="error", error=f"命令解析失败：{exc}")
    if not parts:
        return ParsedInput(kind="command", name="help")

    command = parts[0]
    argument = raw[len(parts[0]) :].strip()
    available_tools = set(tool_names)
    available_skills = set(skill_names)

    if command in ("quit", "exit"):
        return ParsedInput(kind="command", name="quit")
    if command in _LOCAL_COMMANDS:
        return ParsedInput(kind="command", name=command, argument=argument)

    if command in ("tool", "skill"):
        return _parse_capability_directive(
            command,
            argument,
            available_tools=available_tools,
            available_skills=available_skills,
            display_text=stripped,
        )

    if command in available_tools:
        return _capability_message(
            command,
            argument,
            kind="skill" if command in available_skills else "tool",
            display_text=stripped,
        )

    return ParsedInput(
        kind="error",
        error=(
            f"未知命令或能力：/{command}。输入 /help 查看命令，"
            "输入 /tools 查看可用工具。"
        ),
    )


def _parse_capability_directive(
    directive: str,
    argument: str,
    *,
    available_tools: set[str],
    available_skills: set[str],
    display_text: str,
) -> ParsedInput:
    try:
        parts = shlex.split(argument)
    except ValueError as exc:
        return ParsedInput(kind="error", error=f"参数解析失败：{exc}")
    if not parts:
        return ParsedInput(
            kind="error",
            error=f"用法：/{directive} <name> <request-or-arguments>",
        )
    name = parts[0]
    remainder = argument[len(name) :].strip()
    allowed = available_skills if directive == "skill" else available_tools
    if name not in allowed:
        return ParsedInput(
            kind="error",
            error=f"未找到{directive}：{name}",
        )
    return _capability_message(
        name,
        remainder,
        kind=directive,
        display_text=display_text,
    )


def _capability_message(
    name: str,
    request: str,
    *,
    kind: str,
    display_text: str,
) -> ParsedInput:
    detail = request or "根据当前会话和该能力的参数定义完成操作"
    label = "技能" if kind == "skill" else "工具"
    message = (
        f"本轮必须使用 {name} {label}完成请求，并且只在确有必要时使用该能力。"
        f"用户提供的请求或参数如下：\n{detail}"
    )
    return ParsedInput(
        kind="message",
        name=name,
        argument=request,
        message=message,
        display_text=display_text,
    )
