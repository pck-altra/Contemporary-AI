"""Q1: maze pathfinding with A*.

Input : start Coord, goal Coord, grid List[List[str]] ('S'/'.'/'X'/'E').
Output: List[Coord] shortest path (start..goal inclusive); [] if unsolvable.
"""

from __future__ import annotations

from . import common
from .astar import Metrics, astar
from .common import Coord

__all__ = ["solve_maze", "OFFICIAL_GRID", "OFFICIAL_START", "OFFICIAL_GOAL"]

OFFICIAL_START: Coord = (0, 0)
OFFICIAL_GOAL: Coord = (3, 4)
OFFICIAL_GRID: list[list[str]] = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
OFFICIAL_EXPECTED: list[Coord] = [
    (0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4),
]


def solve_maze(
    start: Coord,
    goal: Coord,
    grid: list[list[str]],
    heuristic: str = "manhattan",
    tie_break: str = "h",
) -> tuple[list[Coord], Metrics]:
    """Return (path, metrics); path == [] when no route exists."""
    path, _cost, metrics = astar(
        start, goal, grid, heuristic=heuristic, tie_break=tie_break
    )
    return (path if path is not None else []), metrics


if __name__ == "__main__":  # smoke demo on the official case
    path, m = solve_maze(OFFICIAL_START, OFFICIAL_GOAL, OFFICIAL_GRID)
    print("path:", path)
    print("steps:", len(path) - 1 if path else None)
    print("expanded:", m.nodes_expanded, "generated:", m.nodes_generated)
    print("matches official:", path == OFFICIAL_EXPECTED)
