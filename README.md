# 当代人工智能 · 实验一  ·  新闻 10 分类

> 20 Newsgroups 文本分类 —— 朴素贝叶斯 / 线性 SVM / 逻辑回归 / MLP / BERT 五种模型的对比与融合实验。

**作者**：潘淳恺  ·  **学号**：10245501437  ·  **时间**：2026.9.17  ·  **分支**：`homework01`

---

## 一、项目简介

在 20 Newsgroups 的英文新闻语料（10 类子集）上完成文本分类，对比 5 种算法的精度、收敛速度、过拟合程度，并尝试**投票融合**进一步提升精度。

**核心结论**：

| 模型 | 验证集 Accuracy | 训练耗时 | 备注 |
|---|---|---|---|
| MultinomialNB    | 0.9009 | 秒级   | 传统 baseline |
| LinearSVC (SGD)  | 0.9315 | 秒级   | 全收敛后精度最高 / 性价比最高 |
| LogisticRegression | 0.9138 | 数十秒 | 收敛慢，20 步远不够 |
| MLP              | ~0.91  | 51×SVC | 性价比最低 |
| **BERT (base)**  | **0.9342** | 28.7min | 异构最强 |

---

## 二、目录结构

```
.
├── README.md                       ← 你正在看这个
├── seeds.py                        ← 集中管理随机种子（关键）
├── .gitignore
├── data/                           ← 输入数据
│   ├── train_data.csv              7368 条已标注训练样本（10 类）
│   └── test_data_unlabeled.csv     2457 条无标签测试样本
├── src/                            ← 全部源代码
│   ├── nb_baseline.py              (1) MultinomialNB 基线 + 增量学习曲线
│   ├── compare_models.py           (2) 4 模型 20 epoch 对比图
│   ├── compare_models_v2.py        (3) 邮件头清洗版对比
│   ├── save_best_predictions.py    (4) MLP / LinearSVC 全收敛版预测
│   ├── draft_main.py               (5) 实验模板
│   └── ensemble_fusion.py          (6) 多模型投票融合
├── predictions/                    ← 测试集预测 CSV
│   ├── predictions_nb_baseline.csv
│   ├── predictions_svc_raw.csv     ★ 性价比最高的传统模型
│   ├── predictions_bert_raw.csv    ★ BERT 单模型最强
│   ├── predictions_fusion_bert_mlp_svc.csv   ★★★ 推荐提交
│   └── ...
└── docs/                           ← 报告与图表
    ├── 实验报告.md                 实验主报告
    ├── NB_BASELINE_README.md       NB baseline 详细说明
    ├── TUNING.md                   超参数调参指南
    ├── training_log_v2.txt         BERT 训练日志
    └── figures/
        ├── nb_loss_curve.png
        ├── compare_loss_curve.png
        ├── compare_loss_curve_cleaned.png
        └── compare_raw_vs_cleaned.png
```

---

## 三、可复现性

**所有随机操作都从 `seeds.py` 统一取种子**，确保任何人跑都能得到一致结果：

```python
SEED              = 42       # 数据划分 + 分类器初始化
NB_BATCH_BASE     = 0        # NB partial_fit 初始 batch 种子
TEST_SIZE         = 0.2      # 验证集比例
TFIDF_PARAMS      = {...}    # 统一 TF-IDF 超参
```

每个脚本开头都调用 `set_global_seeds()`，锁定 Python / NumPy / PyTorch 的随机状态。

---

## 四、复现命令

```bash
cd homework01

# (1) NB baseline
python src/nb_baseline.py

# (2) 4 模型 20 epoch 对比
python src/compare_models.py

# (3) 邮件头清洗版对比
python src/compare_models_v2.py

# (4) MLP / LinearSVC 全收敛版预测
python src/save_best_predictions.py

# (5) 投票融合（直接读 CSV，无需训练）
python src/ensemble_fusion.py
```

依赖：`numpy`, `pandas`, `scikit-learn>=1.5`, `matplotlib`。

---

## 五、关键实验结果

详见 [docs/实验报告.md](docs/实验报告.md)。

| 模型 | @Epoch 20 | 全收敛 |
|---|---|---|
| MultinomialNB     | 0.9009 | 0.9009 |
| LinearSVC (SGD)   | 0.9261 | **0.9315** |
| LogisticRegression| 0.8500 | 0.9138 |
| MLP               | 0.9100 | 0.9100 |
| BERT (5 epoch)    | —      | **0.9342** |
| 融合 (BERT+MLP+SVC 硬投票) | — | (估算 ≈0.94+) |

**推荐提交**：`predictions/predictions_fusion_bert_mlp_svc.csv`

---

## 六、License

仅用于《当代人工智能》课程作业。
