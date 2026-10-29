# 当代人工智能 · 实验二 · 搜索算法（A* / Minimax / Alpha-Beta）

> 学号：10245501437 ｜ 截止：2026-11-12 23:59:59 ｜ 四题官方用例全部复现通过。

## 一、目录结构

```
10245501437/
├── README.md                 ← 你正在看这个
├── run_all.sh                ← 一键复现：测试 → 实验 → 图表
├── src/
│   ├── common.py             ← 网格代价模型（cost-to-enter）、四邻接、曼哈顿距离
│   ├── astar.py              ← A* 核心（题一、题二、题四共用）+ 5 种启发函数
│   ├── minimax.py            ← Minimax / Alpha-Beta 通用内核 + SearchStats
│   ├── q1_maze.py            ← 题一：迷宫寻路（输出路径，无解 []）
│   ├── q2_weighted.py        ← 题二：加权地形（输出最小代价，无解 -1）
│   ├── q3_game_tree.py       ← 题三：博弈树最佳得分 + 访问顺序对照
│   ├── q4_treasure.py        ← 题四：距离矩阵 + 博弈状态 + Minimax/Alpha-Beta
│   ├── experiments.py        ← 对照实验跑批（Exp-A/B/C/D/E/G + 加权 A*），落盘 CSV
│   └── make_figures.py       ← 由 CSV 生成 docs/figures/*.png（共 10 张）
├── tests/
│   ├── test_q1_q2.py         ← 题一、题二官方用例 + 边界 + 不可采纳对照
│   └── test_q3_q4.py         ← 题三、题四官方用例 + 黄金不变式 + 决策迁移断言
├── results/                  ← 实验数据（CSV，可由 experiments.py 完整再生）
└── docs/figures/             ← 报告用图（PNG，可由 make_figures.py 完整再生）
```

## 二、环境与依赖

- Python ≥ 3.10（实测 3.14），仅依赖标准库 + `matplotlib`（画图）+ `pytest`（测试）。
- 无随机性：全部对照实验使用固定种子；计时取 5 次中位数。

```bash
pip install matplotlib pytest
```

## 三、一键复现

```bash
cd 10245501437
bash run_all.sh
```

`run_all.sh` 依次执行：① `pytest`（20 个测试）；② `experiments.py`（跑批落盘 `results/*.csv`）；
③ `make_figures.py`（生成 `docs/figures/*.png`）。输出应与仓库内已有 CSV/PNG 一致。

## 四、官方用例复现结果

| 题 | 输入 | 输出 | 状态 |
|---|---|---|---|
| 一 迷宫寻路 | 起点 (0,0)，终点 (3,4)，5×5 迷宫 | `[(0,0),(0,1),(0,2),(1,2),(1,3),(1,4),(2,4),(3,4)]`（7 步） | ✅ |
| 二 加权地形 | 起点 (0,0)，终点 (4,4)，5×5 权值表 | `7`（8 步、E 免费；路径见报告） | ✅ |
| 三 博弈树 | 题目树（根 MAX，叶 3,5,6,9,1,2 结构见报告） | `5`（Minimax ≡ Alpha-Beta） | ✅ |
| 四 宝藏分配 | 7×7 地图，H=2 / H=4 | H=2 → `T2/-1`；H=4 → `T3/3` | ✅ |

## 五、关键实现约定（最易错的三处）

1. **代价语义 = 进入目标格的代价**（cost-to-enter）。题二终点 `E` 进入代价为 **0**（否则官方代价会变成 8）。
2. **题四 A/B/T 格代价为 1**（否则 H=4 结果会变成 2 而非官方的 3）。
3. **黄金不变式**：任何访问顺序下 `Alpha-Beta == Minimax`（`tests/test_q3_q4.py` 全覆盖，静默错误的哨兵）。

## 六、对照实验速览

- Exp-A（启发函数）：官方小图分不开（16 vs 14 vs 8），25×25 大图拉开（504 vs 122 vs 49）。
- Exp-C（题三顺序）：64 种全排列下 Alpha-Beta 访问 11~15 结点（均值 13.5），Minimax 恒为 15。
- Exp-D/E（题四）：H=4 时 Alpha-Beta 省约 28% 结点（65→47）；`value` 排序最优（42 结点），`gain` 排序反而最差（50 结点）。
- 加权 A*（创新）：自设大图上 w=1.5/2.0 仍最优（138），w=3 起失守（139/165），官方小图上 w 全最优（地图太小）。
- Exp-G（规模）：n=31 时 Dijkstra 展开 768 结点，曼哈顿仅 149，完美启发 61。

详见实验报告正文与 `results/*.csv`。
