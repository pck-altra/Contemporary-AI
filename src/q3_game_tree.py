"""Q3: best score of a minimax game tree via alpha-beta pruning.

Input : root TreeNode (root is the MAX player; players alternate by ply).
Output: int/float best achievable score for MAX.
"""

from __future__ import annotations

from itertools import permutations
from typing import Optional

from .minimax import SearchStats, TreeNode, solve_tree

__all__ = [
    "solve_best_score",
    "build_official_tree",
    "OFFICIAL_EXPECTED",
    "order_experiment",
]

OFFICIAL_EXPECTED = 5


def build_official_tree() -> TreeNode:
    """Build the assignment tree (p.12):

              (MAX)
              /   \\
           (MIN) (MIN)
           / \\   / \\
        (MAX)(MAX)(MAX)
        / \\  / \\  / \\
        3  5 6  9 1  2
    """
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


def solve_best_score(
    root: TreeNode,
    use_alphabeta: bool = True,
    order: Optional[str] = None,
) -> tuple[float, SearchStats]:
    """Return (best_score, stats).

    order: None/"lr" (left-to-right, the assignment default), "rl"
    (right-to-left), or "best"/"worst" for the control experiment.
    """
    if order in (None, "lr"):
        key = None  # natural left-to-right insertion order
        tree = root
    elif order == "rl":
        key = None
        tree = _reversed_copy(root)
    elif order == "best":
        key = _best_key
        tree = root
    elif order == "worst":
        key = _worst_key
        tree = root
    else:
        raise ValueError(f"unknown order: {order}")
    return solve_tree(tree, maximizing=True, use_alphabeta=use_alphabeta, order=key)


def _reversed_copy(node: TreeNode) -> TreeNode:
    """Deep copy with every child list reversed (for the rl experiment)."""
    return TreeNode(
        value=node.value,
        children=[_reversed_copy(c) for c in reversed(node.children)],
    )


def _best_key(w):
    # "oracle" order: children of a MAX parent (odd depth) try larger
    # static values first; children of a MIN parent (even depth) try
    # smaller ones first.  sorted() is ascending, hence the signs.
    return -_static_value(w) if (w.depth % 2 == 1) else _static_value(w)


def _worst_key(w):
    return -_best_key(w)


def _wis_max_static(w) -> bool:
    return w.depth % 2 == 0


def _static_value(w) -> float:
    """Exact minimax value of the wrapped subtree (computed once, cached)."""
    node = w.node
    if not node.children:
        return node.value
    vals = [_static_value(type(w)(c, w.depth + 1)) for c in node.children]
    return (max if w.depth % 2 == 0 else min)(vals)


def index_tree(root: TreeNode) -> TreeNode:
    """Return an independent deep copy whose nodes carry left-to-right ids.

    The ids let `order_experiment` permute child visit orders without
    mutating the caller's tree.
    """
    counter = iter(range(10**9))

    def clone(node: TreeNode) -> TreeNode:
        return TreeNode(
            value=node.value, children=[clone(c) for c in node.children]
        )

    return clone(root)


def order_experiment(root: TreeNode) -> dict:
    """Exhaustively permute child orders of the official tree (Exp-C).

    For each internal node we try every child permutation; for every global
    combination we run both Minimax and Alpha-Beta and record node counts.
    Returns a summary dict used by experiments.py and the report.
    """
    from .minimax import _Wrap, alphabeta, minimax

    # collect internal nodes in BFS order with their child counts
    # (deduplicated by identity: the official tree shares max2, see p.12)
    internals: list[TreeNode] = []
    seen: set[int] = set()
    queue = [root]
    while queue:
        n = queue.pop(0)
        if n.children:
            if id(n) not in seen:
                seen.add(id(n))
                internals.append(n)
            queue.extend(n.children)
    perms_per_node = [list(permutations(range(len(n.children)))) for n in internals]

    import itertools as _it

    rows = []
    for combo in _it.product(*perms_per_node):
        order_map = {id(n): p for n, p in zip(internals, combo)}

        def _wchildren(w, _om=order_map):
            kids = [_Wrap(c, w.depth + 1) for c in w.node.children]
            perm = _om.get(id(w.node))
            if perm is not None:
                kids = [kids[i] for i in perm]
            return kids

        def _wis_max(w):
            return w.depth % 2 == 0

        def _weval(w):
            return w.node.value

        s_mm, s_ab = SearchStats(), SearchStats()
        v_mm = minimax(_Wrap(root, 0), _wchildren, _wis_max, _weval, None, s_mm)
        v_ab = alphabeta(_Wrap(root, 0), _wchildren, _wis_max, _weval, None, s_ab)
        assert v_mm == v_ab == OFFICIAL_EXPECTED
        rows.append(
            {
                "perm": combo,
                "mm_nodes": s_mm.nodes_visited,
                "mm_leaves": s_mm.leaves,
                "ab_nodes": s_ab.nodes_visited,
                "ab_leaves": s_ab.leaves,
                "ab_cutoffs": s_ab.cutoffs,
                "value": v_ab,
            }
        )
    ab_nodes = [r["ab_nodes"] for r in rows]
    return {
        "n_combos": len(rows),
        "rows": rows,
        "mm_nodes": rows[0]["mm_nodes"],
        "ab_min": min(ab_nodes),
        "ab_max": max(ab_nodes),
        "ab_mean": sum(ab_nodes) / len(ab_nodes),
    }


if __name__ == "__main__":  # smoke demo on the official tree
    root = build_official_tree()
    for algo in (True, False):
        v, s = solve_best_score(root, use_alphabeta=algo)
        print(f"{'alphabeta' if algo else 'minimax ':9s} value={v} "
              f"nodes={s.nodes_visited} leaves={s.leaves} cutoffs={s.cutoffs}")
    print("expected:", OFFICIAL_EXPECTED)
