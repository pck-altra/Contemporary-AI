"""Q4: treasure allocation under competition (A* + Minimax/Alpha-Beta).

Map (7x7), players A (MAX) and B (MIN) alternately pick one remaining
treasure; the picker travels to it along the minimum-cost path.

Official rules:
- 'X' blocked; digits = cost of *entering* that cell; A/B/T cells cost 1.
- 4 rounds in order A -> B -> A -> B; each round picks exactly one treasure
  and the picker moves to it.
- Passing over other treasures does not collect them; taken-treasure cells
  stay passable; the other player never blocks pathfinding.
- Gain(p, Ti) = V[i] - d(p, Ti);  U = Score_A - Score_B.
  A maximises U, B minimises U.  At depth H (or when no treasure remains),
  the current U is the leaf evaluation.

Key design insight: because movement costs depend only on the two players'
current positions and the target treasure (never on the taken set), the
6x6 distance matrix is constant for the whole game.  Pathfinding (A*) and
game search are fully decoupled: task 1 precomputes the matrix once, task 3
only does table lookups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .astar import astar
from .common import Coord
from .minimax import SearchStats, alphabeta, minimax

__all__ = [
    "OFFICIAL_GRID",
    "TREASURES",
    "VALUES",
    "POS",
    "KEYS",
    "distance_matrix",
    "solve_treasure",
    "OFFICIAL_EXPECTED",
]

# ------------------------------------------------------------------ data
OFFICIAL_GRID: list[list] = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
TREASURES = ["T1", "T2", "T3", "T4"]
VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
POS: dict[str, Coord] = {
    "A": (0, 0), "B": (6, 6),
    "T1": (0, 6), "T2": (2, 2), "T3": (4, 4), "T4": (6, 0),
}
KEYS = ["A", "B", "T1", "T2", "T3", "T4"]

OFFICIAL_EXPECTED = {2: ("T2", -1), 4: ("T3", 3)}


# ---------------------------------------------------------- task 1: dist
def distance_matrix(grid=None) -> dict[tuple[str, str], float]:
    """6x6 minimum move costs between A/B/T1..T4 via A* (task 1)."""
    grid = OFFICIAL_GRID if grid is None else grid
    dist: dict[tuple[str, str], float] = {}
    for a in KEYS:
        for b in KEYS:
            _p, cost, _m = astar(POS[a], POS[b], grid, heuristic="manhattan")
            dist[(a, b)] = math.inf if cost is None else cost
    return dist


# ------------------------------------------------- task 2: game state
@dataclass
class Q4State:
    """Game state: both positions, remaining treasures, scores, player, ply."""

    pos_a: str
    pos_b: str
    remaining: frozenset
    score_a: float = 0.0
    score_b: float = 0.0
    turn: int = 0          # 0 -> A (MAX) to move; 1 -> B (MIN) to move
    depth: int = 0

    @property
    def utility(self) -> float:
        return self.score_a - self.score_b


# ------------------------------------------------- task 3: game search
def solve_treasure(
    H: int,
    use_alphabeta: bool = True,
    order: str = "lr",
    grid=None,
    dist: Optional[dict] = None,
) -> tuple[str, float, SearchStats, dict]:
    """Solve Q4 at depth H.  Returns (best_target, best_value, stats, dist).

    order: 'lr' (T1..T4), 'gain' (MAX: gain desc, MIN: gain asc),
           'value' (treasure value desc), 'reverse' (T4..T1).
    """
    grid = OFFICIAL_GRID if grid is None else grid
    dist = distance_matrix(grid) if dist is None else dist
    stats = SearchStats()
    root = Q4State(pos_a="A", pos_b="B",
                   remaining=frozenset(TREASURES), depth=0)

    def _children(s: Q4State):
        cands = sorted(s.remaining)  # deterministic base order
        player = s.pos_a if s.turn == 0 else s.pos_b
        scored = [(t, VALUES[t] - dist[(player, t)]) for t in cands]
        if order == "lr":
            seq = cands
        elif order == "reverse":
            seq = list(reversed(cands))
        elif order == "gain":
            seq = [t for t, _ in sorted(
                scored, key=lambda kv: (-kv[1] if s.turn == 0 else kv[1]))]
        elif order == "value":
            seq = [t for t, _ in sorted(scored, key=lambda kv: -VALUES[kv[0]])]
        else:
            raise ValueError(f"unknown order: {order}")
        out = []
        for t in seq:
            gain = VALUES[t] - dist[(player, t)]
            if s.turn == 0:  # A's move
                out.append(Q4State(t, s.pos_b, s.remaining - {t},
                                   s.score_a + gain, s.score_b, 1, s.depth + 1))
            else:            # B's move
                out.append(Q4State(s.pos_a, t, s.remaining - {t},
                                   s.score_a, s.score_b + gain, 0, s.depth + 1))
        return out

    def _is_max(s: Q4State):
        return s.turn == 0

    def _eval(s: Q4State):
        # leaf = depth H reached or no treasure left -> current U
        return s.utility

    def _is_leaf_state(s: Q4State):
        return s.depth >= H or not s.remaining

    # The generic kernels treat "no children" as leaf; wrap so the
    # H-truncation also terminates expansion.
    def _children_h(s: Q4State):
        return [] if _is_leaf_state(s) else _children(s)

    if use_alphabeta:
        best_value = alphabeta(root, _children_h, _is_max, _eval, None, stats)
    else:
        best_value = minimax(root, _children_h, _is_max, _eval, None, stats)

    # best first move = argmax over root children (A is MAX at root)
    best_target, best_v = None, -math.inf
    branch_values: dict[str, float] = {}
    for c in _children(root):
        v = (alphabeta if use_alphabeta else minimax)(
            c, _children_h, _is_max, _eval, None, SearchStats())
        branch_values[c.pos_a] = v
        if v > best_v:
            best_v, best_target = v, c.pos_a
    assert best_v == best_value
    return best_target, best_value, stats, dist


def root_branch_values(
    H: int,
    grid=None,
    dist: Optional[dict] = None,
) -> dict[str, float]:
    """Return {first_move: minimax value} at the root (for the report)."""
    from .minimax import SearchStats as _SS  # local import: avoid cycle

    grid = OFFICIAL_GRID if grid is None else grid
    dist = distance_matrix(grid) if dist is None else dist
    root = Q4State(pos_a="A", pos_b="B",
                   remaining=frozenset(TREASURES), depth=0)

    def _children(s: Q4State):
        out = []
        player = s.pos_a if s.turn == 0 else s.pos_b
        for t in sorted(s.remaining):
            gain = VALUES[t] - dist[(player, t)]
            if s.turn == 0:
                out.append(Q4State(t, s.pos_b, s.remaining - {t},
                                   s.score_a + gain, s.score_b, 1, s.depth + 1))
            else:
                out.append(Q4State(s.pos_a, t, s.remaining - {t},
                                   s.score_a, s.score_b + gain, 0, s.depth + 1))
        return out

    def _is_max(s: Q4State):
        return s.turn == 0

    def _eval(s: Q4State):
        return s.utility

    def _children_h(s: Q4State):
        return [] if (s.depth >= H or not s.remaining) else _children(s)

    return {c.pos_a: minimax(c, _children_h, _is_max, _eval, None, _SS())
            for c in _children(root)}


if __name__ == "__main__":  # smoke demo on the official case
    dist = distance_matrix()
    print("distance matrix (A,B,T1..T4):")
    print("      " + "".join(f"{k:>5}" for k in KEYS))
    for a in KEYS:
        print(f"{a:>4}  " + "".join(f"{dist[(a, b)]:>5.0f}" for b in KEYS))
    for H in (2, 4):
        for algo in (False, True):
            t, v, s, _ = solve_treasure(H, use_alphabeta=algo, dist=dist)
            print(f"H={H} {'alphabeta' if algo else 'minimax  '}: "
                  f"best={t} value={v} nodes={s.nodes_visited} "
                  f"leaves={s.leaves} cutoffs={s.cutoffs} "
                  f"(expected {OFFICIAL_EXPECTED[H]})")
