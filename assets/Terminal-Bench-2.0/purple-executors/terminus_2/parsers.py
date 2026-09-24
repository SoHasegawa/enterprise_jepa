"""Terminus-2 response parsers (vendored from terminal-bench terminus_2 agent)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from terminus_json_plain_parser import ParseResult as JsonParseResult
from terminus_json_plain_parser import TerminusJSONPlainParser
from terminus_xml_plain_parser import TerminusXMLPlainParser


@dataclass
class ParsedCommand:
    command: str
    timeout: int


@dataclass
class AgentParseResult:
    commands: list[ParsedCommand]
    is_task_complete: bool
    error: str
    warning: str


DEFAULT_EXEC_TIMEOUT = 60
MIN_EXEC_TIMEOUT = 30
MAX_EXEC_TIMEOUT = 300

_SLOW_COMMAND_RE = re.compile(
    r"(?i)\b("
    r"make|cmake|ninja|meson|autoconf|configure|"
    r"apt-get|apt\s+install|yum|dnf|apk\s+add|pip3?\s+install|"
    r"wget|curl\b|git\s+clone|"
    r"gcc|g\+\+|clang|rustc|cargo|go\s+build|npm\s+(install|ci)|yarn\s+install|"
    r"python3?\s+|mvn|gradle|"
    r"docker\s+build"
    r")\b"
)


class TerminusParser(Protocol):
    def parse_response(self, response: str) -> JsonParseResult: ...


def _is_slow_command(command: str) -> bool:
    return bool(_SLOW_COMMAND_RE.search(command))


def duration_to_exec_timeout(duration: float | int | None, command: str) -> int:
    """Map Terminus poll/wait duration to a Harbor shell exec timeout.

    Terminus ``duration`` is how long tmux waits before reading output (often 0.1s
    for ``ls``/``cd``). Green uses ``timeout`` as a hard kill limit, so we apply a
    floor and command heuristics instead of ``int(0.1) == 1``.
    """
    timeout = DEFAULT_EXEC_TIMEOUT
    if duration is not None:
        try:
            wait_sec = float(duration)
        except (TypeError, ValueError):
            wait_sec = 0.0
        if wait_sec > 0:
            timeout = max(MIN_EXEC_TIMEOUT, int(math.ceil(wait_sec * 30)))
    if _is_slow_command(command):
        timeout = max(timeout, 180)
    return max(MIN_EXEC_TIMEOUT, min(timeout, MAX_EXEC_TIMEOUT))


def get_parser(parser_name: str) -> TerminusParser:
    if parser_name == "json":
        return TerminusJSONPlainParser()
    if parser_name == "xml":
        return TerminusXMLPlainParser()
    raise ValueError(f"Unknown TERMINUS_2_PARSER: {parser_name!r}. Use 'json' or 'xml'.")


def keystrokes_to_exec(command_keystrokes: str) -> ParsedCommand | None:
    """Map Terminus keystrokes to a Harbor shell exec_request."""
    keystrokes = command_keystrokes
    stripped = keystrokes.strip()
    if not stripped:
        return ParsedCommand(command="sleep 1", timeout=5)

    if stripped in {"C-c", "C-d", "^C", "^D", "\x03", "\x04"}:
        return ParsedCommand(command="true", timeout=5)

    shell_command = keystrokes.rstrip("\n").strip()
    if not shell_command:
        return ParsedCommand(command="sleep 1", timeout=5)

    return ParsedCommand(command=shell_command, timeout=30)


def convert_parse_result(result: JsonParseResult) -> AgentParseResult:
    commands: list[ParsedCommand] = []
    for parsed in result.commands:
        converted = keystrokes_to_exec(parsed.keystrokes)
        if converted is not None:
            timeout = duration_to_exec_timeout(parsed.duration, converted.command)
            commands.append(
                ParsedCommand(command=converted.command, timeout=timeout)
            )

    return AgentParseResult(
        commands=commands,
        is_task_complete=bool(result.is_task_complete),
        error=str(result.error or ""),
        warning=str(result.warning or ""),
    )
