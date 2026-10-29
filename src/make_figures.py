"""Generate all report figures from results/*.csv (matplotlib, no seaborn).

Outputs (docs/figures/*.png, 150 dpi):
  figA_heuristic_bar .png  Exp-A: expanded nodes per heuristic
  figG_scale.png            Exp-G: expansion vs map size (log-ish)
  figW_weighted.png         weighted A*: cost & expanded vs w (twin axes)
  figC_q3.png               Q3: minimax vs alpha-beta under 4 visit orders
  figDE_q4.png              Q4: minimax vs alpha-beta nodes (H=2/4 x orders)
  fig_q4_heatmap.png        Q4 6x6 distance matrix heatmap
  fig_q1_trace.png          Q1 official map: path + expansion order overlay
  fig_q3_tree.png           Q3 official tree with backup values + cut edge
  fig_q4_tree_H2.png        Q4 H=2 game tree with alpha-beta cutoffs
  fig_q4_tree_H4.png        Q4 H=4 best line + prune summary
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "docs" / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

HEU_LABEL = {"zero": "h=0", "manhattan": "Manhattan",
             "manhattan_minus1": "Manhattan-1", "euclid": "Euclid",
             "perfect": "Perfect h*"}


def read_csv(name):
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------- Exp-A bar
def fig_heuristic():
    rows = read_csv("expA_heuristics.csv")
    cases = ["Q1-big25", "Q2-big25w"]
    heus = ["zero", "manhattan", "manhattan_minus1", "euclid", "perfect"]
    x = range(len(cases))
    w = 0.15
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, h in enumerate(heus):
        vals = []
        for c in cases:
            r = next(r for r in rows if r["case"] == c and r["heuristic"] == h)
            vals.append(int(r["expanded"]))
        ax.bar([p + (i - 2) * w for p in x], vals, width=w,
               label=HEU_LABEL[h])
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Q1 25x25 open map", "Q2 25x25 weighted map"])
    ax.set_ylabel("Expanded nodes")
    ax.set_title("Exp-A: expanded nodes by heuristic (official maps too small to separate)")
    ax.legend(ncol=3, loc="upper right")
    off = [r for r in rows if r["case"] == "Q1-official" and r["heuristic"] == "manhattan"]
    fig.text(0.5, -0.01,
             f"Official Q1: zero=16 vs manhattan=14 vs perfect=8 (all optimal, cost 7 steps).  "
             f"Big maps separate clearly (e.g. Q1-big25: zero=504, manhattan=122, perfect=49).",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(FIG / "figA_heuristic_bar.png")


# ---------------------------------------------------------- Exp-G scale
def fig_scale():
    rows = read_csv("expG_scale.csv")
    ns = sorted({int(r["n"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for h in ("zero", "manhattan", "perfect"):
        vals = [next(int(r["expanded"]) for r in rows
                     if int(r["n"]) == n and r["heuristic"] == h) for n in ns]
        ax.plot(ns, vals, marker="o", label=HEU_LABEL[h])
    ax.set_xlabel("Map size n (n x n)")
    ax.set_ylabel("Expanded nodes")
    ax.set_title("Exp-G: A* expansion vs map size (seeded solvable maps)")
    ax.legend()
    fig.text(0.5, -0.01,
             "Manhattan stays near-optimal on open maps; Dijkstra (h=0) grows ~5x faster by n=31.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(FIG / "figG_scale.png")


# ---------------------------------------------------------- weighted A*
def fig_weighted():
    rows = [r for r in read_csv("exp_weighted_astar.csv")
            if r["case"] == "Q2-big25w"]
    ws = [float(r["w"]) for r in rows]
    costs = [float(r["cost"]) for r in rows]
    exps = [int(r["expanded"]) for r in rows]
    opt = float(rows[0]["optimal_cost"])
    fig, ax1 = plt.subplots(figsize=(7.0, 3.6))
    ax1.plot(ws, costs, "o-", color="#0b3d6b", label="Returned cost")
    ax1.axhline(opt, color="gray", linestyle="--", label=f"Optimal ({opt:.0f})")
    ax1.set_xlabel("Weight w (heuristic = w x Manhattan)")
    ax1.set_ylabel("Returned cost", color="#0b3d6b")
    ax2 = ax1.twinx()
    ax2.plot(ws, exps, "s--", color="#c0392b", label="Expanded nodes")
    ax2.set_ylabel("Expanded nodes", color="#c0392b")
    ax1.set_title("Weighted A*: w=1.5/2.0 still optimal here; w>=3 breaks optimality (139/165 vs 138)")
    fig.savefig(FIG / "figW_weighted.png")


# ---------------------------------------------------------- Q3 orders
def fig_q3():
    rows = read_csv("expC_q3_orders.csv")
    orders = ["lr", "rl", "best", "worst"]
    mm = [next(int(r["nodes"]) for r in rows
               if r["order"] == o and r["algo"] == "minimax") for o in orders]
    ab = [next(int(r["nodes"]) for r in rows
               if r["order"] == o and r["algo"] == "alphabeta") for o in orders]
    x = range(len(orders))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar([p - 0.2 for p in x], mm, width=0.4, label="Minimax (15 always)")
    ax.bar([p + 0.2 for p in x], ab, width=0.4, label="Alpha-Beta")
    for p, v in zip(x, ab):
        ax.text(p + 0.2, v + 0.15, str(v), ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Left-to-right", "Right-to-left", "Oracle-best", "Oracle-worst"])
    ax.set_ylabel("Visited nodes")
    ax.set_title("Exp-C (Q3): visit order changes Alpha-Beta cost, never the value (always 5)")
    ax.legend()
    fig.text(0.5, -0.01,
             "Exhaustive 64 orderings: Alpha-Beta visits 11..15 nodes (mean 13.5); Minimax always 15.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(FIG / "figC_q3.png")


# ---------------------------------------------------------- Q4 minimax vs ab
def fig_q4():
    rows = [r for r in read_csv("expDE_q4_orders.csv") if r["H"] in ("2", "4")]
    labels = [f"H={r['H']} {r['order']}" for r in rows if r["algo"] == "minimax"]
    mm = [int(r["nodes"]) for r in rows if r["algo"] == "minimax"]
    ab = [int(r["nodes"]) for r in rows if r["algo"] == "alphabeta"]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.bar([p - 0.2 for p in x], mm, width=0.4, label="Minimax")
    ax.bar([p + 0.2 for p in x], ab, width=0.4, label="Alpha-Beta")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Visited nodes")
    ax.set_title("Exp-D/E (Q4): Alpha-Beta saves ~25-30% nodes; best_target/value identical (H=2: T2/-1, H=4: T3/3)")
    ax.legend()
    fig.savefig(FIG / "figDE_q4.png")


# ---------------------------------------------------------- Q4 heatmap
def fig_heatmap():
    from src import q4_treasure as q4
    dist = q4.distance_matrix()
    keys = q4.KEYS
    mat = [[dist[(a, b)] for b in keys] for a in keys]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(mat, cmap="YlOrRd")
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_yticklabels(keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{mat[i][j]:.0f}", ha="center", va="center",
                    fontsize=10, color="white" if mat[i][j] > 7 else "black")
    ax.set_title("Q4 task 1: 6x6 minimum-move-cost matrix (A*)")
    fig.colorbar(im, ax=ax, label="Move cost")
    fig.savefig(FIG / "fig_q4_heatmap.png")


# ---------------------------------------------------------- Q1 trace map
def fig_q1_trace():
    from src import q1_maze
    from src.astar import astar
    grid = q1_maze.OFFICIAL_GRID
    path, _c, m = astar((0, 0), (3, 4), grid, heuristic="manhattan",
                        record_trace=True)
    order = {node: i for i, (node, *_rest) in enumerate(m.trace)}
    nrows, ncols = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    for r in range(nrows):
        for c in range(ncols):
            ch = grid[r][c]
            if ch == "X":
                fc = "#2b2b2b"
            elif (r, c) in order:
                k = order[(r, c)] / max(1, len(order) - 1)
                fc = (1.0, 1.0 - 0.55 * k, 1.0 - 0.75 * k)
            else:
                fc = "white"
            ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor=fc, edgecolor="#999999"))
            if (r, c) in order:
                ax.text(c, r + 0.28, str(order[(r, c)]), ha="center",
                        va="center", fontsize=7, color="#333333")
            if ch in ("S", "E"):
                ax.text(c, r - 0.22, ch, ha="center", va="center",
                        fontsize=10, fontweight="bold")
    pr = [r for r, _c2 in path]
    pc = [c for _r, c in path]
    ax.plot(pc, pr, "o-", color="#0b3d6b", linewidth=2, markersize=5)
    ax.set_title("Q1: official maze, expansion order (small no.) + shortest path (blue)")
    fig.text(0.5, -0.01, f"Path cost 7 steps; expanded {m.nodes_expanded} nodes.",
             ha="center", fontsize=8, color="#555555")
    fig.savefig(FIG / "fig_q1_trace.png")


# ---------------------------------------------------------- Q3 tree
def fig_q3_tree():
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # layout: root(5,9); min(3,6.5),(7,6.5); max(1.5,2,4),(4,6,8) y=4; leaves y=1.5
    pos = {"root": (5, 9), "min1": (3, 6.5), "min2": (7, 6.5),
           "max1": (1.8, 4), "max2": (4.2, 4), "max2b": (5.8, 4), "max3": (8.2, 4)}
    leaves = {(0.9, 1.5): "3", (2.7, 1.5): "5", (3.3, 1.5): "6",
              (5.1, 1.5): "9", (4.9, 1.5): "6", (6.7, 1.5): "9",
              (7.3, 1.5): "1", (9.1, 1.5): "2"}
    edges = [("root", "min1"), ("root", "min2"), ("min1", "max1"),
             ("min1", "max2"), ("min2", "max2b"), ("min2", "max3")]
    leaf_of = {"max1": [(0.9, 1.5), (2.7, 1.5)], "max2": [(3.3, 1.5), (5.1, 1.5)],
               "max2b": [(4.9, 1.5), (6.7, 1.5)], "max3": [(7.3, 1.5), (9.1, 1.5)]}
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#555555")
    for k, pts in leaf_of.items():
        for p in pts:
            ax.plot([pos[k][0], p[0]], [pos[k][1], p[1]], color="#555555")
    # pruned edge: under min2->max3, leaf 2 cut (draw red dashed)
    ax.plot([pos["max3"][0], 9.1], [pos["max3"][1], 1.5], color="#c0392b",
            linestyle=(0, (3, 2)), linewidth=1.5)
    labels = {"root": "MAX=5", "min1": "MIN=5", "min2": "MIN=2",
              "max1": "MAX=5", "max2": "MAX=9", "max2b": "MAX=9", "max3": "MAX=1*"}
    for k, (x, y) in pos.items():
        ax.add_patch(patches.Circle((x, y), 0.55, facecolor="white",
                                    edgecolor="#0b3d6b", linewidth=1.5))
        ax.text(x, y, labels[k], ha="center", va="center", fontsize=8)
    for (x, y), v in leaves.items():
        ax.add_patch(patches.Circle((x, y), 0.42, facecolor="#eef3f9",
                                    edgecolor="#0b3d6b"))
        ax.text(x, y, v, ha="center", va="center", fontsize=8)
    ax.text(8.6, 2.6, "cut: leaf 2\n(value 1 already\n<= alpha 5)",
            fontsize=7.5, color="#c0392b")
    ax.set_title("Q3: backup MAX(3,5)=5, MAX(6,9)=9, MAX(1,..)=1*; MIN(5,9)=5, MIN(9,1)=2; root MAX=5")
    fig.savefig(FIG / "fig_q3_tree.png")


# ---------------------------------------------------------- Q4 trees
def fig_q4_trees():
    from src import q4_treasure as q4
    dist = q4.distance_matrix()
    for H in (2, 4):
        bv = q4.root_branch_values(H, dist=dist)
        t, v, s_ab, _ = q4.solve_treasure(H, use_alphabeta=True, dist=dist)
        _t2, _v2, s_mm, _d = q4.solve_treasure(H, use_alphabeta=False, dist=dist)
        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        xs = {"T1": 1.2, "T2": 3.7, "T3": 6.2, "T4": 8.7}
        ax.add_patch(patches.Circle((5, 8.5), 0.7, facecolor="white",
                                    edgecolor="#0b3d6b", linewidth=1.5))
        ax.text(5, 8.5, f"A to move\nU*={v:.0f}", ha="center", va="center",
                fontsize=8)
        for k, x in xs.items():
            gain = q4.VALUES[k] - dist[("A", k)]
            best = (k == t)
            ax.plot([5, x], [8.5, 5.5], color="#0b3d6b" if best else "#999999",
                    linewidth=2 if best else 1)
            ax.add_patch(patches.Circle((x, 5.5), 0.75,
                                        facecolor="#dff0d8" if best else "white",
                                        edgecolor="#0b3d6b", linewidth=1.5))
            ax.text(x, 5.5, f"{k}\nU={bv[k]:.0f}", ha="center", va="center",
                    fontsize=8)
            ax.text(x, 4.2, f"gain {gain:.0f}", ha="center", fontsize=7.5,
                    color="#555555")
            ax.text(x, 2.6,
                    ("BEST first move" if best else "candidate"),
                    ha="center", fontsize=7.5,
                    color="#0b3d6b" if best else "#888888")
        ax.set_title(f"Q4 H={H}: root branch values { {k: round(bv[k]) for k in ['T1','T2','T3','T4']} }; "
                     f"best {t}/{v:.0f}; minimax nodes {s_mm.nodes_visited} vs alpha-beta {s_ab.nodes_visited} "
                     f"({s_ab.cutoffs} cutoffs)")
        if H == 4:
            fig.text(0.5, -0.01,
                     "Best line: A->T3(-1), B->T1(+1), A->T2(0), B->T4(-5); "
                     "U = -1-(1-5) = 3.  A denies B the +3 T3 grab and forces the T1->T4 trek (cost 12).",
                     ha="center", fontsize=8, color="#555555")
        else:
            fig.text(0.5, -0.01,
                     "H=2 sees one move each: only T2 gives A instant +2; B replies T3 (+3); U=-1.",
                     ha="center", fontsize=8, color="#555555")
        fig.savefig(FIG / f"fig_q4_tree_H{H}.png")


def main():
    fig_heuristic()
    fig_scale()
    fig_weighted()
    fig_q3()
    fig_q4()
    fig_heatmap()
    fig_q1_trace()
    fig_q3_tree()
    fig_q4_trees()
    print("figures in", FIG)


if __name__ == "__main__":
    main()
