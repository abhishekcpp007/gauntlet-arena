import subprocess
from pathlib import Path

import docker
import pytest


def _docker_ok() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


DOCKER_OK = _docker_ok()


def pytest_collection_modifyitems(config, items):
    if DOCKER_OK:
        return
    skip = pytest.mark.skip(reason="Docker daemon not reachable")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


PYPROJECT = """\
[project]
name = "calc"
version = "0.1.0"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["calc"]
"""


def make_calc_repo(root: Path) -> Path:
    """Two commits: add() with a test, then mul() with a test. The second one is the task."""
    repo = root / "calc"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    files = {
        "pyproject.toml": PYPROJECT,
        "calc/__init__.py": "def add(a, b):\n    return a + b\n",
        "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    }
    for name, body in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(body)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Initial calc")
    (repo / "calc/__init__.py").write_text(
        files["calc/__init__.py"] + "\n\ndef mul(a, b):\n    return a * b\n"
    )
    (repo / "tests/test_calc.py").write_text(
        "from calc import add, mul\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_mul():\n    assert mul(3, 4) == 12\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Add mul")
    return repo


@pytest.fixture(scope="session")
def calc_tasks(tmp_path_factory):
    """Build the env image once per session and mine the single task."""
    from gauntlet_repo2env import build_env, mine_tasks

    root = tmp_path_factory.mktemp("arena")
    cache = root / "cache"
    env = build_env(str(make_calc_repo(root)), cache_dir=cache)
    tasks = list(mine_tasks(env, cache_dir=cache))
    assert len(tasks) == 1
    yield tasks, cache
    try:
        docker.from_env().images.remove(env.image, force=True)
    except docker.errors.APIError:
        pass
