"""A* search core shared by Q1 (maze), Q2 (weighted terrain) and Q4 (distances).

Interface
---------
>>> path, cost, metrics = astar(start, goal, grid, heuristic="manhattan")
>>> # unsolvable -> (None, None, metrics)

The search state of a node is its grid coordinate; the evaluation is
``f(n) = g(n) + h(n)`` where ``g`` is the true accumulated cost from the
start and ``h`` a goal-distance estimate.  Nodes are expanded in ascending
``f`` order from a binary min-heap (``heapq``).
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .common import Coord, cell_cost, in_bounds, manhattan, neighbours


# ---------------------------------------------------------------- metrics
@dataclass
class Metrics:
    """Search-efficiency counters (used for all control experiments)."""

    nodes_expanded: int = 0      # nodes popped from the heap and expanded
    nodes_generated: int = 0     # nodes ever pushed into the heap
    max_open_size: int = 0       # peak heap size
    reopens: int = 0             # closed nodes re-opened via a better path
    elapsed_ms: float = 0.0      # wall time of the search
    # expansion order log (node, g, h, f); filled only when record_trace=True,
    # afterwards also reused by the h=0 counter-experiment on Q2.
    trace: list = field(default_factory=list)


# ------------------------------------------------------------ heuristics
Heuristic = Callable[[Coord, Coord], float]


def _zero(_node: Coord, _goal: Coord) -> float:
    return 0.0  # -> Dijkstra


def _manhattan(node: Coord, goal: Coord) -> float:
    return float(manhattan(node, goal))


def _euclid(node: Coord, goal: Coord) -> float:
    return math.hypot(node[0] - goal[0], node[1] - goal[1])


def _overestimate(node: Coord, goal: Coord, factor: float) -> Heuristic:
    def h(n: Coord, g: Coord) -> float:
        return factor * manhattan(n, g)

    h.__name__ = f"manhattan_x{factor:g}"
    return h


def _manhattan_minus1(node: Coord, goal: Coord) -> float:
    """Strictly admissible variant for Q2 (E is free to enter)."""
    return float(max(0, manhattan(node, goal) - 1))


def perfect_heuristic(grid: list[list[Any]], goal: Coord) -> Heuristic:
    """Precompute the exact remaining cost h* with a reverse Dijkstra.

    Running Dijkstra backwards from `goal` over the *reversed* graph gives
    h*(n) for every cell n.  Because our edge costs are symmetric
    (cost-to-enter of the target cell), the reversed graph has the same
    weights of the forward graph, so a single forward Dijkstra run from
    `goal` suffices -- every cell-to-enter of a neighbour is known.
    """
    dist: dict[Coord, float] = {goal: 0.0}
    pq: list[tuple[float, int, Coord]] = [(0.0, 0, goal)]
    counter = itertools.count()
    while pq:
        d, _, cur = heapq.heappop(pq)
        if d > dist[cur]:
            continue
        for nxt, c in neighbours(grid, cur):
            nd = d + c
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                heapq.heappush(pq, (nd, next(counter), nxt))

    def h(node: Coord, _goal: Coord) -> float:
        return dist.get(node, math.inf)

    h.__name__ = "perfect"
    return h


HEURISTICS: dict[str, Heuristic] = {
    "zero": _zero,
    "manhattan": _manhattan,
    "euclid": _euclid,
    "manhattan_minus1": _manhattan_minus1,
}


def get_heuristic(name: str, grid=None, goal=None) -> Heuristic:
    """Resolve a heuristic by name.

    Supported: 'zero', 'manhattan', 'euclid', 'manhattan_minus1',
    'manhattan_x<k>' (inadmissible scaling, for the counter-experiment),
    'perfect' (requires grid + goal).
    """
    if name in HEURISTICS:
        return HEURISTICS[name]
    if name.startswith("manhattan_x"):
        return _overestimate(None, None, float(name.split("x", 1)[1]))
    if name == "perfect":
        assert grid is not None and goal is not None
        return perfect_heuristic(grid, goal)
    raise ValueError(f"unknown heuristic: {name}")


# ------------------------------------------------------------------ A*
def astar(
    start: Coord,
    goal: Coord,
    grid: list[list[Any]],
    heuristic: str = "manhattan",
    tie_break: str = "h",
    record_trace: bool = False,
) -> tuple[Optional[list[Coord]], Optional[float], Metrics]:
    """Run A* from `start` to `goal`.

    Parameters
    ----------
    tie_break : how to order heap entries with equal f:
        'h'  -> smaller h first (default; target-oriented)
        'g'  -> larger g first (deeper node first)
        'fifo' -> plain insertion order
    record_trace : also log every expansion (node, g, h, f).

    Returns
    -------
    (path, cost, metrics); path is a list of coords from start to goal
    (inclusive) and cost its total enter-cost.  Unsolvable -> (None, None).
    """
    metrics = Metrics()
    if start == goal:
        return [start], 0.0, metrics
    if not in_bounds(grid, *start) or not in_bounds(grid, *goal):
        return None, None, metrics
    if cell_cost(grid[start[0]][start[1]]) is None:
        return None, None, metrics

    h = get_heuristic(heuristic, grid, goal)
    t0 = time.perf_counter()

    counter = itertools.count()  # monotonic id -> deterministic FIFO tie-break
    g: dict[Coord, float] = {start: 0.0}
    came: dict[Coord, Coord] = {}
    closed: set[Coord] = set()
    h0 = h(start, goal)
    open_heap: list[tuple] = [(h0, _tie(h0, 0.0, tie_break), next(counter), start)]
    metrics.nodes_generated = 1
    metrics.max_open_size = 1

    while open_heap:
        _, _, _, cur = heapq.heappop(open_heap)
        if cur in closed:  # stale heap entry
            continue
        closed.add(cur)
        gc = g[cur]
        hc = h(cur, goal)
        metrics.nodes_expanded += 1
        if record_trace:
            metrics.trace.append((cur, gc, hc, gc + hc))
        if cur == goal:
            metrics.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _reconstruct(came, start, goal), gc, metrics
        for nxt, c in neighbours(grid, cur):
            new_g = gc + c
            if nxt in closed and new_g >= g.get(nxt, math.inf):
                continue
            if new_g < g.get(nxt, math.inf):
                if nxt in closed:
                    closed.discard(nxt)  # reopen with the better path
                    metrics.reopens += 1
                g[nxt] = new_g
                came[nxt] = cur
                hn = h(nxt, goal)
                heapq.heappush(
                    open_heap,
                    (new_g + hn, _tie(hn, new_g, tie_break), next(counter), nxt),
                )
                metrics.nodes_generated += 1
                metrics.max_open_size = max(metrics.max_open_size, len(open_heap))

    metrics.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return None, None, metrics


def _tie(hn: float, gn: float, mode: str) -> float:
    if mode == "h":  # prefer node closer to the goal
        return hn
    if mode == "g":  # prefer deeper node (larger g first)
        return -gn
    if mode == "fifo":  # plain breadth-ish order on equal f
        return 0.0
    raise ValueError(f"unknown tie_break: {mode}")


def _reconstruct(came: dict[Coord, Coord], start: Coord, goal: Coord) -> list[Coord]:
    path = [goal]
    cur = goal
    while cur != start:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return path
