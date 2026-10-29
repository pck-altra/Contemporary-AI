"""Grid cost model shared by Q1 (maze) and Q2 (weighted terrain).

Cost semantics (``cost-to-enter``): the cost of the edge u -> v equals the
cost of *entering* cell v.  An ``'X'`` cell is impassable.

Cost table
----------
'X'                -> None (blocked)
'.'                -> 1        (free cell, Q1)
'S'                -> 1        (start; the start cell itself is never charged)
'E'                -> 0        (Q2 rule: "the last step is free / treat E as 0")
digits / int       -> int(ch)  (terrain weight, Q2)
anything else      -> 1        (e.g. 'A'/'B'/'T*' for Q4)
"""

from __future__ import annotations

from typing import Any, Optional

Coord = tuple[int, int]

# Four-neighbour moves: up, down, left, right.  No diagonal moves.
DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def cell_cost(ch: Any) -> Optional[int]:
    """Return the cost of entering cell `ch`, or None if impassable."""
    if isinstance(ch, int):  # already a number (mixed-type grids allowed)
        return int(ch)
    if isinstance(ch, str):
        if ch == "X":
            return None
        if ch == "E":
            return 0
        if ch == "S" or ch == ".":
            return 1
        if ch.isdigit() or (ch.startswith("-") and ch[1:].isdigit()):
            return int(ch)
    return 1


def in_bounds(grid: list[list[Any]], r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def neighbours(grid: list[list[Any]], node: Coord):
    """Yield ``(position, enter_cost)`` for every passable 4-neighbour."""
    r, c = node
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if not in_bounds(grid, nr, nc):
            continue
        cost = cell_cost(grid[nr][nc])
        if cost is None:
            continue
        yield (nr, nc), cost


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
