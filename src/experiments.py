"""Batch control experiments for Experiment 2 (Exp-A ~ Exp-H).

Runs every comparison required by the report and saves tidy CSVs under
``results/``.  All randomness is seeded; timings are medians over repeats.

Usage:
    python -m src.experiments
    python src/experiments.py
"""
from __future__ import annotations

import csv
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import q1_maze, q2_weighted, q3_game_tree, q4_treasure  # noqa: E402
from src.astar import astar  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

REPEATS = 5  # medians for elapsed_ms


def timed(fn, repeats=REPEATS):
    vals = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return out, statistics.median(vals)


def write_csv(name, rows):
    path = RESULTS / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def random_grid(n, obstacle_p=0.22, weighted=False, seed=0):
    rng = random.Random(seed)
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            if (r, c) in ((0, 0), (n - 1, n - 1)):
                row.append("S" if (r, c) == (0, 0) else "E")
                continue
            if rng.random() < obstacle_p:
                row.append("X")
            else:
                row.append(rng.randint(1, 8) if weighted else ".")
        grid.append(row)
    return grid


def open_grid(n, weighted=False, seed=0, obstacle_p=0.08):
    """Guaranteed-solvable random map: carve a random monotone S->E path."""
    rng = random.Random(seed)
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            if (r, c) in ((0, 0), (n - 1, n - 1)):
                row.append("S" if (r, c) == (0, 0) else "E")
                continue
            if rng.random() < obstacle_p:
                row.append("X")
            else:
                row.append(rng.randint(1, 8) if weighted else ".")
        grid.append(row)
    # carve an open corridor: right along row 0, then down last column
    for c in range(n):
        if grid[0][c] == "X":
            grid[0][c] = 1 if weighted else "."
    for r in range(n):
        if grid[r][n - 1] == "X":
            grid[r][n - 1] = 1 if weighted else "."
    return grid


def exp_a_heuristics():
    """Exp-A: heuristic sweep on Q1/Q2 official + a large random map."""
    rows = []
    cases = [
        ("Q1-official", q1_maze.OFFICIAL_START, q1_maze.OFFICIAL_GOAL,
         q1_maze.OFFICIAL_GRID),
        ("Q2-official", q2_weighted.OFFICIAL_START, q2_weighted.OFFICIAL_GOAL,
         q2_weighted.OFFICIAL_GRID),
    ]
    big = open_grid(25, obstacle_p=0.22, weighted=False, seed=42)
    cases.append(("Q1-big25", (0, 0), (24, 24), big))
    bigw = open_grid(25, obstacle_p=0.15, weighted=True, seed=7)
    cases.append(("Q2-big25w", (0, 0), (24, 24), bigw))
    for case, s, g, grid in cases:
        for h in ("zero", "manhattan", "manhattan_minus1", "euclid", "perfect"):
            (res, ms) = timed(lambda s=s, g=g, grid=grid, h=h:
                              astar(s, g, grid, heuristic=h))
            path, cost, m = res
            rows.append({
                "case": case, "heuristic": h,
                "expanded": m.nodes_expanded, "generated": m.nodes_generated,
                "max_open": m.max_open_size, "reopens": m.reopens,
                "cost": cost if cost is not None else -1,
                "path_len": len(path) - 1 if path else -1,
                "elapsed_ms": round(ms, 3),
            })
    return write_csv("expA_heuristics.csv", rows)


def exp_b_tiebreak():
    rows = []
    cases = [
        ("Q1-official", q1_maze.OFFICIAL_START, q1_maze.OFFICIAL_GOAL,
         q1_maze.OFFICIAL_GRID),
        ("Q2-official", q2_weighted.OFFICIAL_START, q2_weighted.OFFICIAL_GOAL,
         q2_weighted.OFFICIAL_GRID),
    ]
    big = random_grid(25, obstacle_p=0.22, weighted=False, seed=42)
    cases.append(("Q1-big25", (0, 0), (24, 24), big))
    for case, s, g, grid in cases:
        for tb in ("h", "g", "fifo"):
            (res, ms) = timed(lambda s=s, g=g, grid=grid, tb=tb: astar(
                s, g, grid, heuristic="manhattan", tie_break=tb))
            path, cost, m = res
            rows.append({
                "case": case, "tie_break": tb,
                "expanded": m.nodes_expanded, "generated": m.nodes_generated,
                "cost": cost if cost is not None else -1,
                "elapsed_ms": round(ms, 3),
            })
    return write_csv("expB_tiebreak.csv", rows)


def exp_weighted_astar():
    """Innovation: weighted A* (w*h) on Q2 -- optimality boundary.

    The official map is too small to separate w (all give 7), so we also
    sweep w on a larger weighted map where overestimation actually bites.
    """
    rows = []
    s, g, grid = (q2_weighted.OFFICIAL_START, q2_weighted.OFFICIAL_GOAL,
                  q2_weighted.OFFICIAL_GRID)
    bigw = open_grid(25, obstacle_p=0.15, weighted=True, seed=7)
    opt_bigw = astar((0, 0), (24, 24), bigw, heuristic="zero")[1]
    cases = [("Q2-official", s, g, grid, 7.0),
             ("Q2-big25w", (0, 0), (24, 24), bigw, opt_bigw)]
    for case, s2, g2, grid2, opt in cases:
        for w in (1.0, 1.5, 2.0, 3.0, 5.0):
            (res, ms) = timed(lambda s2=s2, g2=g2, grid2=grid2, w=w: astar(
                s2, g2, grid2, heuristic=f"manhattan_x{w:g}"))
            path, cost, m = res
            rows.append({
                "case": case, "w": w,
                "cost": cost if cost is not None else -1,
                "optimal_cost": opt,
                "optimal": (cost == opt),
                "expanded": m.nodes_expanded, "generated": m.nodes_generated,
                "elapsed_ms": round(ms, 3),
            })
    # inadmissible demo on Q1-big map (manhattan x3 stays optimal here by luck)
    return write_csv("exp_weighted_astar.csv", rows)


def exp_c_q3_orders():
    root = q3_game_tree.build_official_tree()
    rows = []
    for order in ("lr", "rl", "best", "worst"):
        for ab in (False, True):
            (res, ms) = timed(lambda order=order, ab=ab:
                              q3_game_tree.solve_best_score(root, use_alphabeta=ab,
                                                            order=order))
            v, s = res
            rows.append({
                "order": order, "algo": "alphabeta" if ab else "minimax",
                "value": v, "nodes": s.nodes_visited, "leaves": s.leaves,
                "cutoffs": s.cutoffs, "elapsed_ms": round(ms, 3),
            })
    write_csv("expC_q3_orders.csv", rows)
    # exhaustive permutation summary (dedupe shared node first)
    summary = q3_game_tree.order_experiment(q3_game_tree.build_official_tree())
    with open(RESULTS / "expC_q3_exhaustive.txt", "w", encoding="utf-8") as f:
        f.write(f"combos={summary['n_combos']} "
                f"mm_nodes={summary['mm_nodes']} "
                f"ab_min={summary['ab_min']} ab_max={summary['ab_max']} "
                f"ab_mean={summary['ab_mean']:.2f}\n")
    print(f"exhaustive Q3 combos: {summary['n_combos']}, "
          f"ab range [{summary['ab_min']},{summary['ab_max']}]")
    return rows


def exp_q4_sweep():
    dist = q4_treasure.distance_matrix()
    rows = []
    for H in (2, 4, 6):
        for order in ("lr", "reverse", "gain", "value"):
            for ab in (False, True):
                (res, ms) = timed(
                    lambda H=H, order=order, ab=ab:
                    q4_treasure.solve_treasure(H, use_alphabeta=ab,
                                               order=order, dist=dist))
                t, v, s, _ = res
                rows.append({
                    "H": H, "order": order,
                    "algo": "alphabeta" if ab else "minimax",
                    "best_target": t, "best_value": v,
                    "nodes": s.nodes_visited, "leaves": s.leaves,
                    "cutoffs": s.cutoffs, "elapsed_ms": round(ms, 3),
                })
    # distance matrix dump
    with open(RESULTS / "q4_distance_matrix.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + q4_treasure.KEYS)
        for a in q4_treasure.KEYS:
            w.writerow([a] + [int(dist[(a, b)]) for b in q4_treasure.KEYS])
    return write_csv("expDE_q4_orders.csv", rows)


def exp_g_scale():
    """Exp-G: A* expansion vs map size (seeded, guaranteed solvable)."""
    rows = []
    for n in (7, 15, 25, 31):
        grid = open_grid(n, obstacle_p=0.20, weighted=False, seed=100 + n)
        for h in ("zero", "manhattan", "perfect"):
            (res, ms) = timed(lambda grid=grid, n=n, h=h:
                              astar((0, 0), (n - 1, n - 1), grid, heuristic=h))
            path, cost, m = res
            rows.append({
                "n": n, "heuristic": h,
                "expanded": m.nodes_expanded, "generated": m.nodes_generated,
                "solvable": path is not None,
                "elapsed_ms": round(ms, 3),
            })
    return write_csv("expG_scale.csv", rows)


def main():
    print("== Exp-A heuristics ==")
    exp_a_heuristics()
    print("== Exp-B tie-break ==")
    exp_b_tiebreak()
    print("== Weighted A* ==")
    exp_weighted_astar()
    print("== Exp-C Q3 orders ==")
    exp_c_q3_orders()
    print("== Exp-D/E Q4 sweep ==")
    exp_q4_sweep()
    print("== Exp-G scale ==")
    exp_g_scale()
    print("done.")


if __name__ == "__main__":
    main()
