"""Real matches on a real container: the calibration agents and a human, through the HTTP API."""

import time

import pytest
from fastapi.testclient import TestClient

from gauntlet_arena import Arena, ArenaError
from gauntlet_arena.server import create_app

pytestmark = pytest.mark.docker

FIX = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"


@pytest.fixture
def arena(calc_tasks, tmp_path):
    tasks, cache = calc_tasks
    a = Arena(tasks, tmp_path / "data", cache_dir=cache)
    yield a
    a.shutdown()


def test_calibration_agents(arena):
    task_id = next(iter(arena.tasks))
    match = arena.wait(arena.start_match(task_id, ["oracle", "noop", "cheater"]).id, timeout=600)
    res = {name: e.result for name, e in match.entries.items()}
    assert all(e.error is None for e in match.entries.values())

    assert res["oracle"]["resolved"] and res["oracle"]["counted"]
    assert res["oracle"]["verdict"] == "clean"
    assert not res["noop"]["resolved"] and not res["noop"]["counted"]

    # the conftest hack and the test edits are stripped by the grader AND flagged
    cheat = res["cheater"]
    assert not cheat["resolved"] and not cheat["counted"]
    assert cheat["verdict"] == "cheating"
    assert "conftest.py" in cheat["ignored_files"]
    assert cheat["annotations"]  # findings point at specific steps for the replay viewer

    assert match.winner == "oracle"
    board = {r["contestant"]: r for r in arena.leaderboard.standings()}
    assert board["oracle"]["solved"] == 1 and board["oracle"]["solve_rate"] == 1.0
    assert board["cheater"]["caught_cheating"] == 1
    assert [r["contestant"] for r in arena.leaderboard.standings()] == ["oracle", "noop", "cheater"]


def test_validation(arena):
    task_id = next(iter(arena.tasks))
    for bad in ([], ["oracle", "oracle"], ["skynet"]):
        with pytest.raises(ArenaError):
            arena.start_match(task_id, bad)
    with pytest.raises(ArenaError):
        arena.start_match("nope", ["oracle"])


def test_http_human_vs_noop(arena):
    task_id = next(iter(arena.tasks))
    with TestClient(create_app(arena, api_key="")) as client:
        assert client.get("/").text.startswith("<!doctype html>")
        assert client.get("/api/tasks").json()[0]["id"] == task_id
        names = {c["name"] for c in client.get("/api/contestants").json()}
        assert {"oracle", "noop", "cheater", "human"} <= names

        mid = client.post("/api/matches", json={"task_id": task_id, "contestants": ["human", "noop"]}).json()[
            "id"
        ]
        r = client.post(f"/api/matches/{mid}/human/exec", json={"command": "cat calc/__init__.py"})
        assert "def add" in r.json()["stdout"]
        assert client.get(f"/api/matches/{mid}/human/file", params={"path": "missing.py"}).status_code == 404
        client.put(f"/api/matches/{mid}/human/file", json={"path": "calc/__init__.py", "content": FIX})
        assert (
            client.post(
                f"/api/matches/{mid}/human/exec",
                json={"command": "python -c 'import calc; print(calc.mul(3,4))'"},
            ).json()["stdout"]
            == "12\n"
        )
        assert client.post(f"/api/matches/{mid}/human/submit").status_code == 202
        assert client.post(f"/api/matches/{mid}/human/submit").status_code == 400  # only once

        deadline = time.time() + 300
        state = client.get(f"/api/matches/{mid}").json()
        while not state["done"] and time.time() < deadline:
            time.sleep(0.5)
            state = client.get(f"/api/matches/{mid}").json()

        assert state["done"] and state["winner"] == "human"
        human = next(e for e in state["entries"] if e["name"] == "human")
        assert human["result"]["counted"] and human["result"]["verdict"] == "clean"
        # the polling cursor only returns steps the client hasn't seen
        assert human["new_steps"]
        assert not any("git diff --cached" in st.get("command", "") for st in human["new_steps"])
        again = client.get(
            f"/api/matches/{mid}", params={"since": f"human:{human['new_steps'][-1]['i'] + 1}"}
        ).json()
        assert next(e for e in again["entries"] if e["name"] == "human")["new_steps"] == []

        page = client.get(f"/api/matches/{mid}/human/replay")
        assert page.status_code == 200 and "<html" in page.text.lower()
        assert client.get("/api/leaderboard").json()[0]["contestant"] == "human"


def test_api_key(arena):
    with TestClient(create_app(arena, api_key="s3cret")) as client:
        assert client.get("/api/tasks").status_code == 401
        assert client.get("/api/tasks", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert client.get("/api/tasks", params={"key": "s3cret"}).status_code == 200
        assert client.get("/health").status_code == 200
