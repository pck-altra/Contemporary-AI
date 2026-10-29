"""Unit tests for Q3 (game tree) and Q4 (treasure game)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import q3_game_tree, q4_treasure
from src.minimax import SearchStats, alphabeta, minimax


def test_q3_official_value_is_5():
    root = q3_game_tree.build_official_tree()
    for ab in (False, True):
        v, _s = q3_game_tree.solve_best_score(root, use_alphabeta=ab)
        assert v == q3_game_tree.OFFICIAL_EXPECTED == 5


def test_q3_golden_invariant_all_orders():
    """Alpha-Beta must equal Minimax under every visit order."""
    root = q3_game_tree.build_official_tree()
    for order in ("lr", "rl", "best", "worst"):
        v_mm, _ = q3_game_tree.solve_best_score(root, use_alphabeta=False,
                                                order=order)
        v_ab, _ = q3_game_tree.solve_best_score(root, use_alphabeta=True,
                                                order=order)
        assert v_mm == v_ab == 5


def test_q3_single_leaf():
    from src.minimax import TreeNode

    v, s = q3_game_tree.solve_best_score(TreeNode(value=42))
    assert v == 42 and s.leaves == 1


def test_q3_pruning_actually_skips_leaves():
    root = q3_game_tree.build_official_tree()
    _v_mm, s_mm = q3_game_tree.solve_best_score(root, use_alphabeta=False)
    _v_ab, s_ab = q3_game_tree.solve_best_score(root, use_alphabeta=True)
    assert s_ab.leaves < s_mm.leaves
    assert s_ab.cutoffs >= 1


def test_q3_exhaustive_range():
    summary = q3_game_tree.order_experiment(q3_game_tree.build_official_tree())
    assert summary["n_combos"] == 64
    assert summary["mm_nodes"] == 15
    assert summary["ab_min"] == 11 and summary["ab_max"] == 15


def test_q4_distance_matrix_spot_checks():
    dist = q4_treasure.distance_matrix()
    assert dist[("A", "A")] == 0
    assert int(dist[("A", "T2")]) == 4
    assert int(dist[("B", "T3")]) == 4
    assert int(dist[("A", "B")]) == 12
    assert int(dist[("T1", "T4")]) == 12
    # symmetry on this map
    for a in q4_treasure.KEYS:
        for b in q4_treasure.KEYS:
            assert dist[(a, b)] == dist[(b, a)]


def test_q4_official_h2_h4():
    dist = q4_treasure.distance_matrix()
    for H, (exp_t, exp_v) in q4_treasure.OFFICIAL_EXPECTED.items():
        for ab in (False, True):
            t, v, _s, _d = q4_treasure.solve_treasure(
                H, use_alphabeta=ab, dist=dist)
            assert t == exp_t, f"H={H} ab={ab}: {t} != {exp_t}"
            assert v == exp_v, f"H={H} ab={ab}: {v} != {exp_v}"


def test_q4_golden_invariant_all_orders():
    dist = q4_treasure.distance_matrix()
    for H in (2, 4):
        for order in ("lr", "reverse", "gain", "value"):
            t_mm, v_mm, _, _ = q4_treasure.solve_treasure(
                H, use_alphabeta=False, order=order, dist=dist)
            t_ab, v_ab, _, _ = q4_treasure.solve_treasure(
                H, use_alphabeta=True, order=order, dist=dist)
            assert (t_mm, v_mm) == (t_ab, v_ab)


def test_q4_branch_values_explain_decision_shift():
    """H=2 greedy (T2) vs H=4 deny+punish (T3): the report's key claim."""
    dist = q4_treasure.distance_matrix()
    bv2 = q4_treasure.root_branch_values(2, dist=dist)
    bv4 = q4_treasure.root_branch_values(4, dist=dist)
    assert max(bv2, key=bv2.get) == "T2" and bv2["T2"] == -1
    assert max(bv4, key=bv4.get) == "T3" and bv4["T3"] == 3


def test_q4_h6_equals_h4():
    """Only 4 treasures exist: H=6 degenerates to the H=4 game."""
    dist = q4_treasure.distance_matrix()
    t6, v6, _, _ = q4_treasure.solve_treasure(6, use_alphabeta=True, dist=dist)
    assert (t6, v6) == ("T3", 3.0)


def test_q4_minimax_alphabeta_kernel_equivalence():
    """Generic kernels agree on a small synthetic tree."""
    from src.minimax import TreeNode, tree_children, tree_evaluate

    def is_max_depth(state):
        node, d = state
        return d % 2 == 0

    root = (TreeNode(children=[TreeNode(value=1), TreeNode(value=9)]), 0)

    def children(state):
        node, d = state
        return [(c, d + 1) for c in node.children]

    def evaluate(state):
        node, _d = state
        return node.value

    assert minimax(root, children, is_max_depth, evaluate, None,
                   SearchStats()) == 9
    assert alphabeta(root, children, is_max_depth, evaluate, None,
                     SearchStats()) == 9
