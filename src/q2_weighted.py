"""Q2: weighted terrain -- minimum move cost with A*.

Input : start Coord, goal Coord, grid of terrain weights (digits/ints);
        'S' is the start, 'E' the goal.
Output: int minimum total cost; -1 if unsolvable.

Cost rule (assignment p.5/p.7): moving A -> neighbour B costs the weight
of *entering* B; the last step into E costs 0 ("treat E as 0").
"""

from __future__ import annotations

from typing import Any

from .astar import Metrics, astar
from .common import Coord

__all__ = [
    "solve_weighted",
    "solve_weighted_path",
    "OFFICIAL_GRID",
    "OFFICIAL_START",
    "OFFICIAL_GOAL",
    "OFFICIAL_EXPECTED_COST",
]

OFFICIAL_START: Coord = (0, 0)
OFFICIAL_GOAL: Coord = (4, 4)
OFFICIAL_GRID = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]
OFFICIAL_EXPECTED_COST = 7
OFFICIAL_EXPECTED_PATH: list[Coord] = [
    (0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (4, 3), (4, 4),
]


def solve_weighted(
    start: Coord,
    goal: Coord,
    grid: list[list[Any]],
    heuristic: str = "manhattan",
    tie_break: str = "h",
) -> tuple[int, Metrics]:
    """Return (min_cost, metrics); -1 when no route exists."""
    _path, cost, metrics = astar(
        start, goal, grid, heuristic=heuristic, tie_break=tie_break
    )
    return (int(cost) if cost is not None else -1), metrics


def solve_weighted_path(
    start: Coord,
    goal: Coord,
    grid: list[list[Any]],
    heuristic: str = "manhattan",
    tie_break: str = "h",
) -> tuple[list[Coord], int, Metrics]:
    """Return (path, min_cost, metrics) -- used to show cost accumulation."""
    path, cost, metrics = astar(
        start, goal, grid, heuristic=heuristic, tie_break=tie_break,
        record_trace=True,
    )
    if path is None or cost is None:
        return [], -1, metrics
    return path, int(cost), metrics


if __name__ == "__main__":  # smoke demo on the official case
    cost, m = solve_weighted(OFFICIAL_START, OFFICIAL_GOAL, OFFICIAL_GRID)
    path, _, _ = solve_weighted_path(
        OFFICIAL_START, OFFICIAL_GOAL, OFFICIAL_GRID
    )
    print("cost:", cost, "(expected 7)")
    print("path:", path)
    print("expanded:", m.nodes_expanded, "generated:", m.nodes_generated)
    print("path ok:", path == OFFICIAL_EXPECTED_PATH)
