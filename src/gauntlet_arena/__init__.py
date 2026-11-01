"""gauntlet-arena: race AI coding agents and humans on real repo tasks."""

__version__ = "0.1.0"

from .agents import Agent, CheaterAgent, NoopAgent, OracleAgent, builtin_agents  # noqa: E402
from .engine import Arena, ArenaError, Leaderboard  # noqa: E402

__all__ = [
    "Agent",
    "Arena",
    "ArenaError",
    "CheaterAgent",
    "Leaderboard",
    "NoopAgent",
    "OracleAgent",
    "builtin_agents",
]
