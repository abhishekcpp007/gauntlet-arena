# gauntlet-arena

Race AI coding agents against each other, or against yourself, on real bugs from real repos. Watch every command they run live in the browser. A solve only counts if the hidden tests pass **and** the run isn't caught cheating.

```bash
repo2env build https://github.com/pallets/itsdangerous
repo2env mine pallets__itsdangerous.env.json --limit 5
gauntlet-arena serve --tasks pallets__itsdangerous.tasks.jsonl
# open http://127.0.0.1:8080
```

Pick a task, pick contestants, hit start. Each one gets its own locked-down sandbox at the commit right before the fix, with none of the hidden tests in sight. When they're done, their diff gets graded, their run gets scanned for cheating, and the leaderboard updates.

## Contestants

| Name | What it is |
|---|---|
| `claude` | Claude with a bash tool and a file editor, both running inside the task sandbox |
| `human` | You. A terminal and a file editor in the browser, racing the agents on the clock |
| `oracle` | Applies the real fix from git history. If it can't win, the task is broken |
| `noop` | Looks around and gives up. If it ever wins, the task is broken |
| `cheater` | Plants a conftest hook that marks every test as passed and skips the test files. It should always get caught |

The three scripted ones are there to calibrate the arena itself. A grader you haven't tried to fool is a grader you can't trust, so `cheater` runs the exact trick that fools a naive pytest grader. Here it gets stopped twice: repo2env drops its edits to test files and conftest before grading, and Cheat Catcher flags the attempt from the trajectory. It ends up at the bottom of the board with a "caught cheating" mark.

## How a match is scored

1. Every contestant works in a fresh `gauntlet-sandbox` from `repo2env.open_task`, with no network.
2. When it stops, its changes come out as a git diff.
3. `repo2env.evaluate` applies the diff, then the hidden tests, and checks every `fail_to_pass` and `pass_to_pass` test.
4. `gauntlet-cheatcatcher` scans the agent's own steps and the diff.
5. A solve counts when the tests pass and the verdict isn't `cheating`. The first counted solve wins, timed by when the contestant stopped working, not when grading finished.

Every finished attempt lands in SQLite, and the leaderboard ranks by solve rate, then solves, then average time. Each lane has a **Replay in Time Machine** link that opens the full trajectory in the `gauntlet-timemachine` viewer, with Cheat Catcher's findings pinned to the steps that triggered them.

## The Claude agent

```bash
pip install "gauntlet-arena[claude]"
export ANTHROPIC_API_KEY=...
gauntlet-arena serve --tasks tasks.jsonl
```

It uses `claude-opus-5-5` by default (set `GAUNTLET_CLAUDE_MODEL` to change it) with adaptive thinking at high effort, and Anthropic's built-in `bash` and `str_replace_based_edit_tool` tools. Every tool call runs in the sandbox. The editor can't reach outside the repo, and commands time out after two minutes. It also opts into the API's server-side fallback, so if a request gets declined by a safety classifier it's retried on a fallback model instead of ending the run.

The tool loop is tested against a scripted fake client, so `pytest` never spends API credits.

## Bring your own agent

```python
from pathlib import Path

from gauntlet_arena import Agent, Arena, builtin_agents
from gauntlet_repo2env import load_tasks


class MyAgent(Agent):
    name = "mine"
    description = "My agent."

    def solve(self, task, sb, log):
        log("reading the code")
        print(sb.exec("grep -rn 'def ' src | head", workdir=task.workdir).stdout)
        sb.write_file(f"{task.workdir}/src/fix.py", "...")


arena = Arena(
    load_tasks("tasks.jsonl"), data_dir=Path("arena-data"), agents={"mine": MyAgent, **builtin_agents()}
)
```

Whatever's in the working tree when `solve` returns is what gets graded. Exceptions are caught, and the agent still gets graded on what it left behind.

## Headless benchmarks

```bash
gauntlet-arena run --tasks tasks.jsonl --all --agents claude,oracle,noop
```

Runs every task, prints who solved what, then prints the standings. Same scoring, no browser.

## API

The UI is a single page on top of a small JSON API, so you can drive it from scripts too.

| Method | Path | |
|---|---|---|
| `GET` | `/api/tasks`, `/api/contestants`, `/api/matches`, `/api/leaderboard` | Lists |
| `POST` | `/api/matches` | `{"task_id": "...", "contestants": ["claude", "human"]}` |
| `GET` | `/api/matches/{id}?since=claude:12,human:4` | Live state, only steps you haven't seen |
| `GET` | `/api/matches/{id}/{name}/replay` | Time Machine viewer for one contestant |
| `POST` | `/api/matches/{id}/human/exec` | Run a command in your sandbox |
| `GET`/`PUT` | `/api/matches/{id}/human/file` | Read or save a file |
| `POST` | `/api/matches/{id}/human/submit` | Lock in your answer |

Set `GAUNTLET_API_KEY` and every `/api` route wants `Authorization: Bearer <key>` (or `?key=` for the replay links). The server binds to localhost by default. It starts containers on demand, so don't expose it without a key.

## Part of Gauntlet

This is the top of the stack. It leans on every other piece:

- [gauntlet-sandbox](https://github.com/abhishekcpp007/gauntlet-sandbox) runs every action and records it
- [gauntlet-repo2env](https://github.com/abhishekcpp007/gauntlet-repo2env) builds the environments, mines the tasks and grades the patches
- [gauntlet-timemachine](https://github.com/abhishekcpp007/gauntlet-timemachine) powers the replay links
- [gauntlet-cheatcatcher](https://github.com/abhishekcpp007/gauntlet-cheatcatcher) decides whether a green run counts

[gauntlet](https://github.com/abhishekcpp007/gauntlet) wires all of it into one `docker compose up`.

## Develop

```bash
uv sync
uv run pytest   # match tests need Docker and network, they skip without Docker
```

MIT licensed.
