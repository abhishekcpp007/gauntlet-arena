"""Command line: `gauntlet-arena serve` for the web UI, `gauntlet-arena run` for headless benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gauntlet_repo2env import Task, load_tasks

from . import __version__
from .engine import Arena, ArenaError


def _tasks(paths: list[str]) -> list[Task]:
    tasks: list[Task] = []
    for p in paths:
        tasks.extend(load_tasks(p))
    if not tasks:
        raise ArenaError("no tasks found in " + ", ".join(paths))
    return tasks


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    arena = Arena(_tasks(args.tasks), Path(args.data))
    print(f"{len(arena.tasks)} task(s), contestants: {', '.join(c['name'] for c in arena.contestants())}")
    uvicorn.run(create_app(arena), host=args.host, port=args.port)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    arena = Arena(_tasks(args.tasks), Path(args.data))
    ids = list(arena.tasks) if args.all else [args.task]
    contestants = [c.strip() for c in args.agents.split(",") if c.strip()]
    if "human" in contestants:
        raise ArenaError("humans play in the web UI, use `gauntlet-arena serve`")

    for task_id in ids:
        print(f"\n== {task_id}")
        match = arena.wait(arena.start_match(task_id, contestants).id, timeout=args.timeout)
        for e in match.entries.values():
            if e.error:
                print(f"   {e.name:10} ERROR {e.error}")
                continue
            r = e.result or {}
            mark = (
                "SOLVED" if r.get("counted") else ("CAUGHT" if r.get("verdict") == "cheating" else "failed")
            )
            print(f"   {e.name:10} {mark:7} verdict={r.get('verdict')} time={e.seconds}s")
        print(f"   winner: {match.winner or 'nobody'}")

    print("\nstandings")
    for row in arena.leaderboard.standings():
        print(
            f"   {row['contestant']:10} {row['solved']}/{row['attempts']} solved"
            f"  caught={row['caught_cheating']}  avg={row['avg_solve_seconds']}s"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gauntlet-arena", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the web arena")
    p.add_argument("--tasks", nargs="+", required=True, help="task JSONL files from repo2env")
    p.add_argument("--data", default="arena-data", help="where trajectories and scores live")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("run", help="run a headless match, or a whole benchmark with --all")
    p.add_argument("--tasks", nargs="+", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", help="instance_id of one task")
    group.add_argument("--all", action="store_true", help="every task in the files")
    p.add_argument("--agents", default="oracle,noop", help="comma separated, e.g. claude,oracle")
    p.add_argument("--data", default="arena-data")
    p.add_argument("--timeout", type=float, default=3600, help="seconds per match")
    p.set_defaults(fn=cmd_run)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        return args.fn(args)
    except ArenaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
