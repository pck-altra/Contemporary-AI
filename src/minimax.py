"""Minimax + Alpha-Beta core shared by Q3 (game tree) and Q4 (treasure game).

A game search works on two callbacks:

    children(state) -> ordered list of successor states (or moves)
    is_max(state)   -> True if the player to move at `state` is MAX

Both plain Minimax and Alpha-Beta return Open/closed-node statistics so
that the control experiments (visit-order effect, pruning efficiency) can
be quantified.  The golden invariant is:

    minimax(...) == alphabeta(...)   for every visit order
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Optional, Sequence


@dataclass
class SearchStats:
    """Node statistics of one game-tree search."""

    nodes_visited: int = 0   # internal + leaf nodes actually evaluated
    leaves: int = 0          # leaf nodes evaluated
    cutoffs: int = 0         # alpha/beta cutoffs triggered
    max_depth: int = 0       # deepest ply reached (root = 0)
    order: list = field(default_factory=list)  # visit log for the report


ChildFn = Callable[[Any], Sequence[Any]]
MaxFn = Callable[[Any], bool]
EvalFn = Callable[[Any], float]
KeyFn = Callable[[Any], Any]


def minimax(
    state: Any,
    children: ChildFn,
    is_max: MaxFn,
    evaluate: EvalFn,
    order: Optional[KeyFn] = None,
    stats: Optional[SearchStats] = None,
    _depth: int = 0,
) -> float:
    """Plain Minimax (no pruning).  Child order does not affect the value."""
    st = stats or SearchStats()
    st.nodes_visited += 1
    st.max_depth = max(st.max_depth, _depth)
    succ = list(children(state))
    if not succ:  # leaf
        st.leaves += 1
        st.order.append(("leaf", state))
        return evaluate(state)
    if order is not None:
        succ = sorted(succ, key=order)
    vals = [minimax(c, children, is_max, evaluate, order, st, _depth + 1) for c in succ]
    return (max if is_max(state) else min)(vals)


def alphabeta(
    state: Any,
    children: ChildFn,
    is_max: MaxFn,
    evaluate: EvalFn,
    order: Optional[KeyFn] = None,
    stats: Optional[SearchStats] = None,
    alpha: float = -math.inf,
    beta: float = math.inf,
    _depth: int = 0,
) -> float:
    """Minimax with alpha-beta pruning.  Same value as minimax, fewer nodes."""
    st = stats or SearchStats()
    st.nodes_visited += 1
    st.max_depth = max(st.max_depth, _depth)
    succ = list(children(state))
    if not succ:  # leaf
        st.leaves += 1
        st.order.append(("leaf", state))
        return evaluate(state)
    if order is not None:
        succ = sorted(succ, key=order)
    if is_max(state):
        value = -math.inf
        for c in succ:
            value = max(value, alphabeta(c, children, is_max, evaluate, order, st, alpha, beta, _depth + 1))
            alpha = max(alpha, value)
            if alpha >= beta:  # beta-cutoff
                st.cutoffs += 1
                break
        return value
    value = math.inf
    for c in succ:
        value = min(value, alphabeta(c, children, is_max, evaluate, order, st, alpha, beta, _depth + 1))
        beta = min(beta, value)
        if alpha >= beta:  # alpha-cutoff
            st.cutoffs += 1
            break
    return value


# ------------------------------------------------------- explicit trees
class TreeNode:
    """Q3 node type, exactly as defined in the assignment (p.8)."""

    def __init__(self, value=None, children=None):
        self.value = value        # leaf score
        self.children = children if children is not None else []


def tree_children(node: TreeNode) -> Sequence[TreeNode]:
    return node.children


def tree_evaluate(node: TreeNode) -> float:
    return node.value


def solve_tree(
    root: TreeNode,
    maximizing: bool = True,
    use_alphabeta: bool = True,
    order: Optional[KeyFn] = None,
) -> tuple[float, SearchStats]:
    """Evaluate a MAX/MIN alternating tree from `root`.

    The player to move alternates every ply, so `is_max` is derived from
    the depth parity instead of stored flags: even ply -> MAX.
    Returns (value, stats).
    """
    stats = SearchStats()
    wrapped = _Wrap(root, 0)

    def _wchildren(w):
        return [_Wrap(c, w.depth + 1) for c in w.node.children]

    def _wis_max(w):
        return (w.depth % 2 == 0) if maximizing else (w.depth % 2 == 1)

    def _weval(w):
        return w.node.value

    if use_alphabeta:
        v = alphabeta(wrapped, _wchildren, _wis_max, _weval, order, stats)
    else:
        v = minimax(wrapped, _wchildren, _wis_max, _weval, order, stats)
    return v, stats


class _Wrap:
    __slots__ = ("node", "depth")

    def __init__(self, node, depth):
        self.node = node
        self.depth = depth

    def __repr__(self):
        return f"Wrap(depth={self.depth}, value={self.node.value})"
