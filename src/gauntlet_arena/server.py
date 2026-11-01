"""HTTP API plus the single-page UI."""

from __future__ import annotations

import os
import secrets
import threading
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from gauntlet_sandbox import SandboxError
from pydantic import BaseModel, Field

from . import __version__
from .engine import Arena, ArenaError


class StartMatch(BaseModel):
    task_id: str
    contestants: list[str] = Field(min_length=1)


class Command(BaseModel):
    command: str = Field(min_length=1)


class FileBody(BaseModel):
    path: str = Field(min_length=1)
    content: str


def _parse_since(raw: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (raw or "").split(","):
        name, _, n = part.partition(":")
        if name and n.isdigit():
            out[name] = int(n)
    return out


def create_app(arena: Arena, *, api_key: str | None = None) -> FastAPI:
    api_key = (api_key if api_key is not None else os.environ.get("GAUNTLET_API_KEY")) or None
    stop = threading.Event()

    def reaper() -> None:
        while not stop.wait(60):
            arena.reap_idle_humans()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=reaper, name="arena-reaper", daemon=True).start()
        yield
        stop.set()
        arena.shutdown()

    app = FastAPI(title="Gauntlet Arena", version=__version__, lifespan=lifespan)
    app.state.arena = arena

    def check_auth(authorization: str | None = Header(None), key: str | None = Query(None)) -> None:
        if api_key is None:
            return
        given = (authorization or "").removeprefix("Bearer ") or key or ""
        if not secrets.compare_digest(given, api_key):
            raise HTTPException(401, "missing or wrong API key")

    @app.exception_handler(ArenaError)
    async def arena_error(_: Request, exc: ArenaError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(SandboxError)
    async def sandbox_error(_: Request, exc: SandboxError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (files("gauntlet_arena") / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "tasks": len(arena.tasks)}

    api = APIRouter(prefix="/api", dependencies=[Depends(check_auth)])

    @api.get("/tasks")
    def tasks() -> list[dict[str, Any]]:
        return [
            {
                "id": t.instance_id,
                "repo": t.repo,
                "title": t.problem_statement.splitlines()[0] if t.problem_statement else t.instance_id,
                "problem": t.problem_statement,
                "fail_to_pass": len(t.fail_to_pass),
                "pass_to_pass": len(t.pass_to_pass),
            }
            for t in arena.tasks.values()
        ]

    @api.get("/contestants")
    def contestants() -> list[dict[str, str]]:
        return arena.contestants()

    @api.post("/matches", status_code=201)
    def start(req: StartMatch) -> dict[str, str]:
        return {"id": arena.start_match(req.task_id, req.contestants).id}

    @api.get("/matches")
    def matches() -> list[dict[str, Any]]:
        return arena.list_matches()

    @api.get("/matches/{match_id}")
    def match(match_id: str, since: str | None = None) -> dict[str, Any]:
        return arena.match_state(match_id, _parse_since(since))

    @api.get("/matches/{match_id}/{name}/replay", response_class=HTMLResponse)
    def replay(match_id: str, name: str) -> str:
        return arena.replay_html(match_id, name)

    @api.post("/matches/{match_id}/human/exec")
    def human_exec(match_id: str, body: Command) -> dict[str, Any]:
        return arena.human_exec(match_id, body.command).model_dump()

    @api.get("/matches/{match_id}/human/file")
    def human_read(match_id: str, path: str = Query(min_length=1)) -> dict[str, str]:
        try:
            return {"path": path, "content": arena.human_read(match_id, path)}
        except FileNotFoundError as e:
            raise HTTPException(404, f"no such file: {path}") from e
        except (IsADirectoryError, UnicodeDecodeError) as e:
            raise HTTPException(400, f"can't open {path} as text") from e

    @api.put("/matches/{match_id}/human/file")
    def human_write(match_id: str, body: FileBody) -> dict[str, str]:
        return {"path": arena.human_write(match_id, body.path, body.content)}

    @api.post("/matches/{match_id}/human/submit", status_code=202)
    def human_submit(match_id: str) -> dict[str, bool]:
        arena.human_submit(match_id)
        return {"ok": True}

    @api.get("/leaderboard")
    def leaderboard() -> list[dict[str, Any]]:
        return arena.leaderboard.standings()

    app.include_router(api)
    return app
