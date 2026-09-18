# MultinomialNB 快速 Baseline 实验说明

## 一、实验目的

本次实验是整个新闻 10 分类项目的**第一步**，目标有两个：

1. **验证数据流程通畅**：确认 CSV 加载、文本预处理、TF-IDF 特征提取、模型训练、预测、保存 csv 这条流水线完整无 bug，为后续上更复杂的模型铺路。
2. **建立精度基线**：得到一个数字，后续的 LR / SVM / MLP 等都要超过它才算有意义。

选 MultinomialNB（多项式朴素贝叶斯）做 baseline 的原因：
- 训练只需秒级，几分钟就能跑完
- 它是文本分类的传统经典模型（与 TF-IDF 几乎是黄金搭档）
- 不需要调很多超参数就能出结果（这里只用默认的 α=1 Laplace 平滑）

---

## 二、数据与划分

- 数据文件：[train_data.csv](train_data.csv)、[test_data_unlabeled.csv](test_data_unlabeled.csv)
- 训练集 7368 条，测试集 2457 条，10 个类别（编号 0–9）
- 每个类在训练集中分布基本均衡（705–747 条），不需要做 class_weight 或重采样

**划分策略**（`sklearn.model_selection.train_test_split`）：

| 子集 | 样本数 | 比例 |
|---|---|---|
| 训练集（train） | 5894 | 80% |
| 验证集（val）  | 1474 | 20% |

参数：`test_size=0.2, stratify=y, random_state=42`

- `stratify=y` 保证每个类在 train/val 中的比例与原始一致
- `random_state=42` 固定划分，便于复现与对比

---

## 三、特征工程：TF-IDF

使用 `sklearn.feature_extraction.text.TfidfVectorizer`：

| 参数 | 取值 | 含义 |
|---|---|---|
| `sublinear_tf` | True | TF 用 `log(1+tf)` 替换原始计数，抑制长文本 |
| `min_df` | 5 | 出现文档数 < 5 的稀有词丢弃 |
| `max_df` | 0.95 | 出现在 >95% 文档的常见词丢弃（去停用词作用） |
| `ngram_range` | (1, 2) | 同时取 unigram 和 bigram |
| `max_features` | 50000 | 特征维度上限 50000 |
| `token_pattern` | `r"(?u)\b[A-Za-z][A-Za-z]+\b"` | 仅保留纯字母词 |

**结果**：5894 × 50000 的稀疏矩阵，**非零占比仅 0.46%**（极度稀疏，符合文本特征的特点）。

---

## 四、模型与损失追踪

### 4.1 模型

`sklearn.naive_bayes.MultinomialNB(alpha=1.0)`

### 4.2 损失曲线记录方法

MultinomialNB 本身是 batch 训练，没有 epoch 概念。为了画出"损失随训练量变化"的曲线，采用 **增量采样法**：

```
对 frac in [0.08, 0.12, ..., 1.00]:     # 25 个采样点
    用前 frac × 100% 条数据重新训练一个 NB
    在训练子集上算 train_loss
    在全量验证集上算 val_loss 和 val_acc
```

`log_loss = -Σ P(y_true) · log P(y_pred)`，等价于交叉熵，越小代表概率越确定正确。

### 4.3 关键观察点
- **训练损失曲线（蓝）单调下降**：训练数据越多，模型对数据的拟合越好
- **验证损失曲线（红）单调下降**：没有明显的过拟合拐点，说明 NB 对这个量级的数据"还没吃饱"，还有扩容空间
- **验证 Accuracy 曲线（绿）**：从 8% 数据的 0.32 飙升到 100% 数据的 0.90

---

## 五、实验结果

### 5.1 最终精度（全量 5894 条训练）

| 指标 | 值 |
|---|---|
| **验证集 Accuracy** | **0.9009** |
| Macro Precision | 0.9044 |
| Macro Recall    | 0.9011 |
| Macro F1        | 0.9015 |

### 5.2 各类别 F1（从分类报告截取）

| Class | F1 | 备注 |
|---|---|---|
| 0 | 0.8396 | 中等 |
| 1 | 0.8495 | 中等 |
| 2 | 0.8213 | 较弱（与 1, 7 类易混）|
| 3 | 0.8874 | 中等偏上 |
| 4 | 0.9388 | 较强 |
| 5 | 0.9667 | 强 |
| 6 | 0.9298 | 强 |
| 7 | 0.8235 | 较弱（与 2 类混淆较多）|
| 8 | 0.9691 | 强 |
| 9 | 0.9892 | 几乎完美 |

混淆矩阵显示 **class 2 和 class 7 之间互相混淆严重**（15 + 15），这两个主题在词义上有重叠，NB 用词袋假设解决不了，需要 bigram 增强或换非线性模型。

### 5.3 最佳训练量

- 100% 训练数据时 val_acc = 0.9009
- 92% 训练数据时 val_acc = 0.9016（极小差异，**无统计显著性**）

结论：**当前 NB 模型已经接近"数据饱和"**，单纯加数据收益有限，要提升需要换模型或加特征。

---

## 六、复现命令

```bash
# 1. 进入 ai1 环境（已装 numpy, pandas, scikit-learn, matplotlib）
conda activate ai1

# 2. 在 data-Project1 目录下运行
cd "/Users/pck/Documents/ecnu/课程/当代人工智能/作业1/data-Project1"
python nb_baseline.py
```

**输出文件**：
- `nb_loss_curve.png` — 损失曲线 + 验证准确率曲线 + 混淆矩阵
- `predictions_nb_baseline.csv` — 测试集 2457 条预测结果（无表头，每行一个整数类别）

---

## 七、结论与下一步

MultinomialNB 已达到 **90.09%** 验证精度，作为 baseline 表现合格。下一步建议：

1. 用 **Logistic Regression** 替换 NB，预计精度能提升 2–4 个百分点（更鲁棒的判别模型）
2. 用 **LinearSVC** 做对照实验
3. 调 TF-IDF 超参数：`ngram_range=(1,3)`、调整 `min_df/max_df`
4. 对 class 2 与 class 7 的混淆，尝试加 bigram 权重或剥离邮件头以暴露正文特征