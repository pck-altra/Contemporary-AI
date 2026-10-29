"""Unit tests for Q1 / Q2 (official cases + edge cases + control invariants)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import q1_maze, q2_weighted
from src.astar import astar
from src.common import manhattan


def _path_cost(grid, path):
    from src.common import cell_cost

    return sum(cell_cost(grid[r][c]) for r, c in path[1:])


def test_q1_official_path():
    path, m = q1_maze.solve_maze(
        q1_maze.OFFICIAL_START, q1_maze.OFFICIAL_GOAL, q1_maze.OFFICIAL_GRID
    )
    assert path == q1_maze.OFFICIAL_EXPECTED
    assert len(path) - 1 == 7  # 8 coords, 7 moves


def test_q1_path_is_valid_and_shortest():
    grid = q1_maze.OFFICIAL_GRID
    path, _ = q1_maze.solve_maze((0, 0), (3, 4), grid)
    assert path[0] == (0, 0) and path[-1] == (3, 4)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # 4-neighbour adjacency
        assert grid[r2][c2] != "X"
    assert len(path) - 1 == manhattan((0, 0), (3, 4))  # == 7 -> optimal


def test_q1_unsolvable_returns_empty():
    grid = [
        ["S", "X", "."],
        ["X", "X", "."],
        [".", ".", "E"],
    ]
    path, _ = q1_maze.solve_maze((0, 0), (2, 2), grid)
    assert path == []


def test_q1_start_is_goal():
    path, _ = q1_maze.solve_maze((1, 1), (1, 1), [["S"]])
    assert path == [(1, 1)]


def test_q2_official_cost():
    cost, _ = q2_weighted.solve_weighted(
        q2_weighted.OFFICIAL_START,
        q2_weighted.OFFICIAL_GOAL,
        q2_weighted.OFFICIAL_GRID,
    )
    assert cost == q2_weighted.OFFICIAL_EXPECTED_COST == 7


def test_q2_path_and_cost_accumulation():
    grid = q2_weighted.OFFICIAL_GRID
    path, cost, _ = q2_weighted.solve_weighted_path((0, 0), (4, 4), grid)
    assert path == q2_weighted.OFFICIAL_EXPECTED_PATH
    assert _path_cost(grid, path) == cost == 7
    # entering E is free: path has 8 steps but only 7 cost
    assert len(path) - 1 == 8 and grid[path[-1][0]][path[-1][1]] == "E"


def test_q2_e_cost_zero_regression():
    """If E were (wrongly) charged 1, the official cost would be 8, not 7."""
    from src.common import cell_cost

    assert cell_cost("E") == 0


def test_q2_unsolvable_returns_minus_one():
    grid = [
        ["S", "X", "X"],
        ["X", "X", "X"],
        ["X", "X", "E"],
    ]
    cost, _ = q2_weighted.solve_weighted((0, 0), (2, 2), grid)
    assert cost == -1


def test_q2_zero_heuristic_same_optimum():
    """Dijkstra (h=0) must return the same optimum on the official case."""
    cost0, _ = q2_weighted.solve_weighted(
        (0, 0), (4, 4), q2_weighted.OFFICIAL_GRID, heuristic="zero"
    )
    assert cost0 == 7
    path0, _ = q1_maze.solve_maze(
        (0, 0), (3, 4), q1_maze.OFFICIAL_GRID, heuristic="zero"
    )
    assert len(path0) - 1 == 7
