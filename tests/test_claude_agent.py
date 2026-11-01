"""The Claude agent's tool loop, driven by a scripted fake client. No API calls, no Docker."""

from types import SimpleNamespace as NS

import pytest
from gauntlet_repo2env import Task
from gauntlet_sandbox import ExecResult

from gauntlet_arena.claude_agent import ClaudeAgent

TASK = Task(
    instance_id="demo-1",
    repo="local",
    image="img",
    workdir="/workspace/repo",
    base_commit="a",
    solution_commit="b",
    problem_statement="Add mul",
    gold_patch="",
    test_patch="",
    test_files=["tests/test_calc.py"],
    fail_to_pass=["tests/test_calc.py::test_mul"],
    pass_to_pass=[],
)


class FakeSandbox:
    def __init__(self, files):
        self.files = dict(files)
        self.commands = []

    def exec(self, command, *, workdir=None, timeout=None, env=None):
        self.commands.append(command)
        if command.startswith("test -d"):
            return ExecResult(exit_code=1, stdout="", stderr="", duration=0)
        if command == "slow":
            return ExecResult(exit_code=137, stdout="", stderr="", duration=9, timed_out=True)
        return ExecResult(exit_code=0, stdout=f"ran: {command}\n", stderr="", duration=0)

    def read_text(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def write_file(self, path, content):
        self.files[path] = content
        return path


def tool_use(id, name, **input):
    return NS(type="tool_use", id=id, name=name, input=input)


def text(t):
    return NS(type="text", text=t)


def reply(*blocks, stop="tool_use"):
    return NS(content=list(blocks), stop_reason=stop, usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        # snapshot the messages, the agent keeps appending to the same list
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


def test_full_loop_edits_and_runs():
    sb = FakeSandbox({"/workspace/repo/calc/__init__.py": "def add(a, b):\n    return a + b\n"})
    client = FakeClient(
        [
            reply(
                text("Let me look."),
                tool_use("t1", "str_replace_based_edit_tool", command="view", path="calc/__init__.py"),
                tool_use("t2", "bash", command="python -m pytest -q"),
            ),
            reply(
                tool_use(
                    "t3",
                    "str_replace_based_edit_tool",
                    command="insert",
                    path="calc/__init__.py",
                    insert_line=2,
                    insert_text="\n\ndef mul(a, b):\n    return a * b",
                )
            ),
            reply(text("Added mul."), stop="end_turn"),
        ]
    )
    notes = []
    agent = ClaudeAgent(client=client)
    agent.solve(TASK, sb, notes.append)

    assert "def mul(a, b):\n    return a * b\n" in sb.files["/workspace/repo/calc/__init__.py"]
    assert sb.commands == ["test -d /workspace/repo/calc/__init__.py", "python -m pytest -q"]
    assert notes == ["Let me look.", "Added mul."]
    assert agent.usage == {"input_tokens": 30, "output_tokens": 15}

    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["thinking"] == {"type": "adaptive"}
    assert first["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in first["betas"]
    assert {t["name"] for t in first["tools"]} == {"bash", "str_replace_based_edit_tool"}

    # both results go back together in one user turn, right after the assistant turn
    second = client.calls[1]["messages"]
    assert second[1]["role"] == "assistant"
    results = second[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert "1\tdef add(a, b):" in results[0]["content"]
    assert "ran: python -m pytest -q" in results[1]["content"]


@pytest.mark.parametrize(
    "call,needle",
    [
        (dict(command="view", path="/etc/passwd"), "outside the repo"),
        (dict(command="view", path="../../etc/passwd"), "outside the repo"),
        (dict(command="view", path="missing.py"), "doesn't exist"),
        (dict(command="str_replace", path="a.py", old_str="x", new_str="y"), "matched 2 times"),
        (dict(command="explode", path="a.py"), "unknown editor command"),
    ],
)
def test_editor_errors_go_back_as_tool_errors(call, needle):
    sb = FakeSandbox({"/workspace/repo/a.py": "x = x\n"})
    client = FakeClient(
        [reply(tool_use("t1", "str_replace_based_edit_tool", **call)), reply(text("ok"), stop="end_turn")]
    )
    ClaudeAgent(client=client).solve(TASK, sb, lambda _: None)
    result = client.calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert needle in result["content"]
    assert sb.files["/workspace/repo/a.py"] == "x = x\n"


def test_timeout_is_an_error():
    sb = FakeSandbox({})
    client = FakeClient([reply(tool_use("t1", "bash", command="slow")), reply(text("hm"), stop="end_turn")])
    ClaudeAgent(client=client).solve(TASK, sb, lambda _: None)
    result = client.calls[1]["messages"][2]["content"][0]
    assert result["is_error"] and "timeout" in result["content"]


def test_refusal_stops_cleanly():
    notes = []
    client = FakeClient([reply(stop="refusal")])
    ClaudeAgent(client=client).solve(TASK, FakeSandbox({}), notes.append)
    assert notes == ["the model declined this task"]
    assert len(client.calls) == 1


def test_turn_limit():
    replies = [reply(tool_use(f"t{i}", "bash", command="ls")) for i in range(3)]
    notes = []
    ClaudeAgent(client=FakeClient(replies), max_turns=3).solve(TASK, FakeSandbox({}), notes.append)
    assert notes == ["stopped after 3 turns"]
