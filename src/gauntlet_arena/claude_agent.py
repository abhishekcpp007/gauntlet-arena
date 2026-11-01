"""A real coding agent: Claude with a bash tool and a file editor, both executed inside the sandbox.

The sandbox is the security boundary. It has no network, dropped capabilities and capped
resources, so commands run as-is instead of going through an allowlist. File paths are still
confined to the repo so the editor can't wander around the container.
"""

from __future__ import annotations

import os
import posixpath
import shlex
from typing import Any

import anthropic
from gauntlet_repo2env import Task
from gauntlet_sandbox import Sandbox

from .agents import Agent, Log

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOOL_OUTPUT = 20_000

TOOLS: list[dict[str, Any]] = [
    {"type": "bash_20250124", "name": "bash"},
    {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
        "max_characters": MAX_TOOL_OUTPUT,
    },
]

SYSTEM = """\
You're a software engineer working in a checkout of a Python repository at {workdir}. \
The task below comes from a commit message in the project's own history, so read the code \
to work out exactly what behavior it expects.

You have a bash tool and a file editor. There's no network, but the project and pytest are \
already installed. Run the relevant existing tests as you go.

Hidden tests will grade your change. Fix the actual behavior. Don't edit, skip or delete tests, \
and don't special-case test inputs, since the hidden tests replace the test files and a \
separate checker flags tampering.

When you're finished, reply with two or three sentences on what you changed and stop calling tools."""


class ToolError(Exception):
    pass


class ClaudeAgent(Agent):
    name = "claude"
    description = "Claude with bash and a file editor, running inside the task sandbox."

    def __init__(
        self,
        model: str | None = None,
        *,
        effort: str = "high",
        max_turns: int = 40,
        command_timeout: float = 120,
        client: Any = None,
    ) -> None:
        self.model = model or os.environ.get("GAUNTLET_CLAUDE_MODEL", DEFAULT_MODEL)
        self.effort = effort
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self._client = client
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    # ------------------------------------------------------------------ loop

    def solve(self, task: Task, sb: Sandbox, log: Log) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"<task>\n{task.problem_statement}\n</task>"}
        ]
        system = SYSTEM.format(workdir=task.workdir)

        for _ in range(self.max_turns):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            self.usage["input_tokens"] += response.usage.input_tokens
            self.usage["output_tokens"] += response.usage.output_tokens

            if response.stop_reason == "refusal":
                log("the model declined this task")
                return

            # append the whole turn untouched, thinking blocks included
            messages.append({"role": "assistant", "content": response.content})
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    log(block.text.strip())

            if response.stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if response.stop_reason == "max_tokens":
                    log("ran out of output tokens mid-turn")
                return

            results = [self._tool_result(task, sb, block) for block in tool_uses]
            messages.append({"role": "user", "content": results})
        log(f"stopped after {self.max_turns} turns")

    def _tool_result(self, task: Task, sb: Sandbox, block: Any) -> dict[str, Any]:
        try:
            if block.name == "bash":
                content = self._bash(task, sb, block.input)
            elif block.name == "str_replace_based_edit_tool":
                content = self._edit(task, sb, block.input)
            else:
                raise ToolError(f"unknown tool {block.name}")
            return {"type": "tool_result", "tool_use_id": block.id, "content": content}
        except ToolError as e:
            return {"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True}

    # ------------------------------------------------------------------ tools

    def _bash(self, task: Task, sb: Sandbox, args: dict[str, Any]) -> str:
        if args.get("restart"):
            return "Shell restarted. Every command already runs in a fresh shell from the repo root."
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError("bash needs a non-empty `command`")
        res = sb.exec(command, workdir=task.workdir, timeout=self.command_timeout)
        out = (res.stdout + res.stderr)[-MAX_TOOL_OUTPUT:]
        if res.timed_out:
            raise ToolError(f"{out}\n[killed after {self.command_timeout:g}s timeout]")
        return f"{out}\n[exit code {res.exit_code}]" if res.exit_code else (out or "[no output]")

    def _path(self, task: Task, raw: Any) -> str:
        if not isinstance(raw, str) or not raw:
            raise ToolError("a `path` is required")
        root = task.workdir.rstrip("/")
        full = posixpath.normpath(raw if raw.startswith("/") else posixpath.join(root, raw))
        if full != root and not full.startswith(root + "/"):
            raise ToolError(f"{raw} is outside the repo at {root}")
        return full

    def _edit(self, task: Task, sb: Sandbox, args: dict[str, Any]) -> str:
        command = args.get("command")
        path = self._path(task, args.get("path"))

        if command == "view":
            quoted = shlex.quote(path)
            if sb.exec(f"test -d {quoted}").ok:
                res = sb.exec(f"find {quoted} -maxdepth 2 -not -path '*/.git*' | sort | head -400")
                return res.stdout
            lines = self._read(sb, path).splitlines()
            start, end = 1, len(lines)
            if args.get("view_range"):
                start, end = args["view_range"]
                end = len(lines) if end == -1 else end
            numbered = [f"{n:6}\t{lines[n - 1]}" for n in range(max(1, start), min(end, len(lines)) + 1)]
            return "\n".join(numbered)[:MAX_TOOL_OUTPUT] or "[empty file]"

        if command == "create":
            text = args.get("file_text")
            if not isinstance(text, str):
                raise ToolError("create needs `file_text`")
            sb.write_file(path, text)
            return f"wrote {path}"

        if command == "str_replace":
            old, new = args.get("old_str"), args.get("new_str", "")
            if not isinstance(old, str) or not old:
                raise ToolError("str_replace needs a non-empty `old_str`")
            text = self._read(sb, path)
            count = text.count(old)
            if count != 1:
                raise ToolError(f"old_str matched {count} times in {path}, it has to match exactly once")
            sb.write_file(path, text.replace(old, new or "", 1))
            return f"edited {path}"

        if command == "insert":
            line = args.get("insert_line")
            new_text = args.get("insert_text", args.get("new_str"))
            if not isinstance(line, int) or not isinstance(new_text, str):
                raise ToolError("insert needs `insert_line` and `insert_text`")
            lines = self._read(sb, path).splitlines(keepends=True)
            if not 0 <= line <= len(lines):
                raise ToolError(f"insert_line must be between 0 and {len(lines)}")
            lines.insert(line, new_text if new_text.endswith("\n") else new_text + "\n")
            sb.write_file(path, "".join(lines))
            return f"inserted into {path}"

        raise ToolError(f"unknown editor command {command!r}")

    @staticmethod
    def _read(sb: Sandbox, path: str) -> str:
        try:
            return sb.read_text(path)
        except FileNotFoundError as e:
            raise ToolError(f"{path} doesn't exist") from e
        except (IsADirectoryError, UnicodeDecodeError) as e:
            raise ToolError(f"can't open {path} as text") from e
