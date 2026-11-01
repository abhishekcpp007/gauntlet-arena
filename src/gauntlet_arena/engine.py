"""The match engine: run contestants side by side on one task, grade them, keep score.

Every contestant gets its own sandbox from repo2env's `open_task`, so nobody can see anyone
else's work or the hidden tests. When an agent finishes, its diff is graded by repo2env's
`evaluate` and its trajectory and patch go through Cheat Catcher. A solve only counts if the
verdict isn't "cheating".
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gauntlet_cheatcatcher import scan
from gauntlet_repo2env import Task, evaluate, extract_patch, open_task
from gauntlet_sandbox import ExecResult, Sandbox, Trajectory

from .agents import Agent, builtin_agents

log = logging.getLogger(__name__)

HUMAN = "human"
HUMAN_IDLE_LIMIT = 30 * 60  # seconds before an abandoned human sandbox gets submitted as-is


class ArenaError(RuntimeError):
    pass


# ---------------------------------------------------------------------- results db


class Leaderboard:
    """SQLite, one row per finished attempt."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                """create table if not exists attempts (
                    match_id text, task_id text, contestant text, resolved integer,
                    verdict text, counted integer, seconds real, steps integer, created_at real)"""
            )
            self._db.commit()

    def record(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "insert into attempts values (:match_id, :task_id, :contestant, :resolved, :verdict, "
                ":counted, :seconds, :steps, :created_at)",
                row,
            )
            self._db.commit()

    def standings(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """select contestant, count(*), sum(counted), sum(verdict = 'cheating'),
                          avg(case when counted then seconds end), count(distinct task_id)
                   from attempts group by contestant"""
            ).fetchall()
        table = [
            {
                "contestant": name,
                "attempts": attempts,
                "solved": solved or 0,
                "solve_rate": round((solved or 0) / attempts, 3) if attempts else 0.0,
                "caught_cheating": caught or 0,
                "avg_solve_seconds": round(avg, 1) if avg is not None else None,
                "tasks_tried": tasks,
            }
            for name, attempts, solved, caught, avg, tasks in rows
        ]
        return sorted(table, key=lambda r: (-r["solve_rate"], -r["solved"], r["avg_solve_seconds"] or 1e9))


# ---------------------------------------------------------------------- matches


@dataclass
class Entry:
    name: str
    trajectory_path: Path
    state: str = "queued"  # queued, running, your_turn, grading, done, error
    started_at: float | None = None
    finished_at: float | None = None
    agent_steps: tuple[int, int] = (0, 0)  # [start, end) of the agent's own steps
    notes: list[str] = field(default_factory=list)
    patch: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    trajectory: Trajectory | None = None
    sandbox: Sandbox | None = None  # only kept open for humans
    last_active: float = field(default_factory=time.time)

    def note(self, text: str) -> None:
        self.notes.append(text)
        del self.notes[:-200]

    @property
    def seconds(self) -> float | None:
        if self.started_at is None:
            return None
        return round((self.finished_at or time.time()) - self.started_at, 1)


@dataclass
class Match:
    id: str
    task: Task
    entries: dict[str, Entry]
    created_at: float = field(default_factory=time.time)

    @property
    def done(self) -> bool:
        return all(e.state in ("done", "error") for e in self.entries.values())

    @property
    def winner(self) -> str | None:
        solved = [
            e for e in self.entries.values() if e.result and e.result["counted"] and e.finished_at is not None
        ]
        return min(solved, key=lambda e: e.finished_at or 0.0).name if solved else None


def summarize_step(step: Any) -> dict[str, Any]:
    a, o = step.action, step.observation
    kind = a.get("type")
    out: dict[str, Any] = {"i": step.index, "type": kind, "t": step.timestamp}
    if kind == "exec":
        out.update(
            command=a.get("command", ""),
            exit_code=o.get("exit_code"),
            timed_out=o.get("timed_out", False),
            output=((o.get("stdout") or "") + (o.get("stderr") or ""))[-3000:],
        )
    elif kind in ("write_file", "read_file"):
        out.update(path=a.get("path"), size=o.get("size"), ok=o.get("ok", True))
    elif kind == "put_archive":
        out.update(path=a.get("dest"), size=a.get("size"))
    return out


class Arena:
    def __init__(
        self,
        tasks: list[Task],
        data_dir: Path,
        *,
        cache_dir: Path | None = None,
        agents: dict[str, Callable[[], Agent]] | None = None,
        command_timeout: float = 120,
    ) -> None:
        self.tasks = {t.instance_id: t for t in tasks}
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.agents = agents if agents is not None else builtin_agents()
        self.command_timeout = command_timeout
        self.leaderboard = Leaderboard(data_dir / "arena.db")
        self.matches: dict[str, Match] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def contestants(self) -> list[dict[str, str]]:
        rows = [{"name": n, "description": f().description} for n, f in self.agents.items()]
        return rows + [{"name": HUMAN, "description": "You, in a browser terminal, racing the agents."}]

    def start_match(self, task_id: str, contestants: list[str]) -> Match:
        task = self.tasks.get(task_id)
        if task is None:
            raise ArenaError(f"no task {task_id}")
        if not contestants or len(set(contestants)) != len(contestants):
            raise ArenaError("pick at least one contestant, each one once")
        unknown = [c for c in contestants if c != HUMAN and c not in self.agents]
        if unknown:
            raise ArenaError(f"unknown contestant(s): {', '.join(unknown)}")

        match_id = uuid.uuid4().hex[:10]
        folder = self.data_dir / "matches" / match_id
        entries = {c: Entry(name=c, trajectory_path=folder / f"{c}.jsonl") for c in contestants}
        match = Match(id=match_id, task=task, entries=entries)
        with self._lock:
            self.matches[match_id] = match

        for name, entry in entries.items():
            if name == HUMAN:
                self._open_human(match, entry)
            else:
                threading.Thread(
                    target=self._run_agent, args=(match, entry), name=f"arena-{match_id}-{name}", daemon=True
                ).start()
        return match

    def get(self, match_id: str) -> Match:
        match = self.matches.get(match_id)
        if match is None:
            raise ArenaError(f"no match {match_id}")
        return match

    def wait(self, match_id: str, timeout: float = 3600, poll: float = 0.5) -> Match:
        deadline = time.time() + timeout
        match = self.get(match_id)
        while not match.done:
            if time.time() > deadline:
                raise ArenaError(f"match {match_id} still running after {timeout:g}s")
            time.sleep(poll)
        return match

    def match_state(self, match_id: str, since: dict[str, int] | None = None) -> dict[str, Any]:
        """Current state. `since` maps contestant -> steps already seen, so polls stay small."""
        match = self.get(match_id)
        since = since or {}
        entries = []
        for e in match.entries.values():
            steps = e.trajectory.steps if e.trajectory else []
            # once the agent is done, stop at its last step so our diff command stays hidden
            first, last = e.agent_steps
            end = last if e.state in ("grading", "done") and last >= first else len(steps)
            start = max(since.get(e.name, 0), first)
            entries.append(
                {
                    "name": e.name,
                    "state": e.state,
                    "seconds": e.seconds,
                    "steps": len(steps),
                    "new_steps": [summarize_step(s) for s in steps[start:end]],
                    "notes": e.notes[-50:],
                    "result": e.result,
                    "error": e.error,
                }
            )
        return {
            "id": match.id,
            "task": {
                "id": match.task.instance_id,
                "repo": match.task.repo,
                "problem": match.task.problem_statement,
                "workdir": match.task.workdir,
            },
            "created_at": match.created_at,
            "done": match.done,
            "winner": match.winner,
            "entries": entries,
        }

    def list_matches(self) -> list[dict[str, Any]]:
        rows = []
        for m in sorted(self.matches.values(), key=lambda m: -m.created_at):
            rows.append(
                {
                    "id": m.id,
                    "task": m.task.instance_id,
                    "contestants": list(m.entries),
                    "done": m.done,
                    "winner": m.winner,
                    "created_at": m.created_at,
                }
            )
        return rows

    def replay_html(self, match_id: str, name: str) -> str:
        from gauntlet_timemachine import render_html

        entry = self._entry(match_id, name)
        if entry.trajectory is None:
            raise ArenaError("nothing recorded yet")
        annotations = (entry.result or {}).get("annotations") or {}
        return render_html(
            entry.trajectory,
            title=f"{name} on {self.get(match_id).task.instance_id}",
            annotations={int(k): v for k, v in annotations.items()},
        )

    # ------------------------------------------------------------------ humans

    def human_exec(self, match_id: str, command: str) -> ExecResult:
        sb, task = self._human_sandbox(match_id)
        return sb.exec(command, workdir=task.workdir, timeout=self.command_timeout)

    def human_read(self, match_id: str, path: str) -> str:
        sb, _ = self._human_sandbox(match_id)
        return sb.read_text(path)

    def human_write(self, match_id: str, path: str, content: str) -> str:
        sb, _ = self._human_sandbox(match_id)
        return sb.write_file(path, content)

    def human_submit(self, match_id: str) -> None:
        match = self.get(match_id)
        entry = self._entry(match_id, HUMAN)
        if entry.state != "your_turn":
            raise ArenaError("already submitted")
        entry.state = "grading"
        threading.Thread(target=self._finish_human, args=(match, entry), daemon=True).start()

    def reap_idle_humans(self) -> None:
        cutoff = time.time() - HUMAN_IDLE_LIMIT
        for match in list(self.matches.values()):
            entry = match.entries.get(HUMAN)
            if entry and entry.state == "your_turn" and entry.last_active < cutoff:
                entry.note("idle too long, submitted automatically")
                self.human_submit(match.id)

    def shutdown(self) -> None:
        for match in self.matches.values():
            for entry in match.entries.values():
                if entry.sandbox is not None:
                    entry.sandbox.close()
                    entry.sandbox = None

    # ------------------------------------------------------------------ internals

    def _entry(self, match_id: str, name: str) -> Entry:
        entry = self.get(match_id).entries.get(name)
        if entry is None:
            raise ArenaError(f"{name} isn't in match {match_id}")
        return entry

    def _human_sandbox(self, match_id: str) -> tuple[Sandbox, Task]:
        match = self.get(match_id)
        entry = self._entry(match_id, HUMAN)
        if entry.state != "your_turn" or entry.sandbox is None:
            raise ArenaError("your sandbox is closed, you already submitted")
        entry.last_active = time.time()
        return entry.sandbox, match.task

    def _open(self, match: Match, entry: Entry) -> Sandbox:
        sb = open_task(
            match.task,
            timeout=self.command_timeout,
            trajectory_path=entry.trajectory_path,
            cache_dir=self.cache_dir,
        )
        entry.trajectory = sb.trajectory
        n = len(sb.trajectory) if sb.trajectory else 0
        entry.agent_steps = (n, n)
        return sb

    def _open_human(self, match: Match, entry: Entry) -> None:
        entry.started_at = time.time()
        try:
            entry.sandbox = self._open(match, entry)
            entry.state = "your_turn"
        except Exception as e:
            entry.state, entry.error = "error", str(e)

    def _finish_human(self, match: Match, entry: Entry) -> None:
        sb = entry.sandbox
        try:
            assert sb is not None
            self._collect(match, entry, sb)
        except Exception as e:
            log.exception("human submit failed")
            entry.state, entry.error = "error", str(e)
            entry.finished_at = time.time()
        finally:
            if sb is not None:
                sb.close()
            entry.sandbox = None

    def _run_agent(self, match: Match, entry: Entry) -> None:
        entry.state, entry.started_at = "running", time.time()
        try:
            sb = self._open(match, entry)
            try:
                agent = self.agents[entry.name]()
                try:
                    agent.solve(match.task, sb, entry.note)
                except Exception as e:  # a crashing agent still gets graded on what it left
                    log.exception("agent %s crashed", entry.name)
                    entry.note(f"agent crashed: {type(e).__name__}: {e}")
                self._collect(match, entry, sb)
            finally:
                sb.close()
        except Exception as e:
            log.exception("match %s / %s failed", match.id, entry.name)
            entry.state, entry.error = "error", str(e)
            entry.finished_at = entry.finished_at or time.time()

    def _collect(self, match: Match, entry: Entry, sb: Sandbox) -> None:
        traj = sb.trajectory
        end = len(traj) if traj else 0
        entry.agent_steps = (entry.agent_steps[0], end)
        entry.finished_at = time.time()  # the clock stops when the work does, not after grading
        entry.patch = extract_patch(sb, match.task)
        entry.state = "grading"
        self._grade(match, entry)

    def _grade(self, match: Match, entry: Entry) -> None:
        task = match.task
        evaluation = evaluate(task, entry.patch, cache_dir=self.cache_dir)

        # only the agent's own steps go to the cheat check, not our setup and diff commands
        agent_traj = None
        if entry.trajectory is not None:
            agent_traj = Trajectory(entry.trajectory.header)
            start, end = entry.agent_steps
            agent_traj.steps = entry.trajectory.steps[start:end]
        report = scan(agent_traj, entry.patch, protected_paths=task.test_files)

        counted = evaluation.resolved and report.verdict != "cheating"
        entry.result = {
            "resolved": evaluation.resolved,
            "counted": counted,
            "verdict": report.verdict,
            "findings": [f.model_dump() for f in report.findings],
            "annotations": {str(k): v for k, v in report.by_step().items()},
            "fail_to_pass": evaluation.fail_to_pass,
            "pass_to_pass_broken": [t for t, ok in evaluation.pass_to_pass.items() if not ok],
            "ignored_files": getattr(evaluation, "ignored_files", []),
            "patch_applied": evaluation.patch_applied,
            "grading_error": evaluation.error,
            "patch_lines": entry.patch.count("\n"),
            "patch": entry.patch[-20_000:],
        }
        entry.state = "done"
        self.leaderboard.record(
            {
                "match_id": match.id,
                "task_id": task.instance_id,
                "contestant": entry.name,
                "resolved": int(evaluation.resolved),
                "verdict": report.verdict,
                "counted": int(counted),
                "seconds": entry.seconds,
                "steps": entry.agent_steps[1] - entry.agent_steps[0],
                "created_at": time.time(),
            }
        )
