"""
新闻10分类 — MultinomialNB 快速 Baseline 实验
=============================================
实验目标：
  1. 使用 MultinomialNB 建立第一个 baseline，验证数据流程通畅
  2. 记录训练过程中的对数似然损失（log-likelihood loss）
  3. 绘制损失变化曲线，并输出分类评估报告

作者：AI 助手
日期：2026-09-17

复现命令（在项目根目录 data-Project1/ 下）：
    python src/nb_baseline.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    confusion_matrix,
)
import warnings
warnings.filterwarnings('ignore')

# 统一管理：路径 + 随机种子
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeds import (
    SEED, DATA_DIR, FIG_DIR, PRED_DIR,
    TFIDF_PARAMS, TEST_SIZE,
    set_global_seeds, get_nb_batch_seed, split_data, predict_path, figure_path,
)
set_global_seeds()

# 使用非交互式后端，避免 plt.show() 阻塞
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 一、数据加载
# ============================================================
print("=" * 60)
print("一、数据加载")
print("=" * 60)

train_df = pd.read_csv(str(DATA_DIR / "train_data.csv"))
test_df  = pd.read_csv(str(DATA_DIR / "test_data_unlabeled.csv"))

X_text_all = train_df['text'].astype(str).tolist()
y_all      = train_df['target'].values
X_test_text = test_df['text'].astype(str).tolist()

print(f"  训练集总样本数 : {len(X_text_all)}")
print(f"  测试集样本数  : {len(X_test_text)}")
print(f"  类别数量      : {len(np.unique(y_all))}")
print(f"  类别分布      :")
for c in sorted(np.unique(y_all)):
    print(f"    class {c}: {(y_all == c).sum()} 条")

# ============================================================
# 二、数据划分
# ============================================================
print("\n" + "=" * 60)
print("二、数据划分")
print("=" * 60)

X_train_text, X_val_text, y_train, y_val = split_data(
    X_text_all, y_all, test_size=TEST_SIZE, stratify=y_all,
)

print(f"  训练集 : {len(X_train_text)} 条")
print(f"  验证集 : {len(X_val_text)} 条")

# ============================================================
# 三、TF-IDF 特征工程
# ============================================================
print("\n" + "=" * 60)
print("三、TF-IDF 特征工程")
print("=" * 60)

vectorizer = TfidfVectorizer(**TFIDF_PARAMS)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf   = vectorizer.transform(X_val_text)
X_test_tfidf  = vectorizer.transform(X_test_text)

print(f"  TF-IDF 特征维度 : {X_train_tfidf.shape[1]}")
print(f"  非零元素占比    : {X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]) * 100:.4f}%")
print(f"  训练集稀疏矩阵  : {X_train_tfidf.shape}")
print(f"  验证集稀疏矩阵  : {X_val_tfidf.shape}")

# ============================================================
# 四、损失曲线绘制（增量训练版本）
# ============================================================
print("\n" + "=" * 60)
print("四、MultinomialNB 增量训练 — 损失曲线记录")
print("=" * 60)

# MultinomialNB 增量训练：逐步增加训练数据量，
# 记录每个阶段的训练损失和验证损失，绘制学习曲线
#
# 原理说明：
#   MultinomialNB 的损失 = -Σ log P(y|x) （对数似然损失，越小越好）
#   随着训练数据增多，模型对数据分布的估计越来越准：
#     - 训练损失：会持续下降（模型见过的样本越多，预测越准）
#     - 验证损失：初期下降，达到最优后可能略微上升（过拟合开始）
#   两条曲线的 gap 反映过拟合程度

n_points = 25  # 取 25 个采样点
train_sizes = np.linspace(0.08, 1.0, n_points).round(2)  # 从 8% 到 100%

classes = np.unique(y_train)  # 10 个固定类别，用于 log_loss 标签

train_losses_curve = []
val_losses_curve   = []
val_accs_curve     = []

print(f"  训练 {n_points} 个采样点，记录每个阶段的损失...")

for frac in train_sizes:
    n_samples = max(1, int(len(X_train_text) * frac))
    X_sub = X_train_tfidf[:n_samples]
    y_sub = y_train[:n_samples]

    nb_sub = MultinomialNB(alpha=1.0, random_state=SEED)
    nb_sub.fit(X_sub, y_sub)

    # 训练集 log_loss（在当前子集上）
    train_prob_sub = nb_sub.predict_proba(X_sub)
    tl = log_loss(y_sub, train_prob_sub, labels=classes)

    # 验证集 log_loss 和 accuracy（全量验证集）
    val_prob = nb_sub.predict_proba(X_val_tfidf)
    vl = log_loss(y_val, val_prob, labels=classes)
    va = accuracy_score(y_val, nb_sub.predict(X_val_tfidf))

    train_losses_curve.append(tl)
    val_losses_curve.append(vl)
    val_accs_curve.append(va)

    if int(frac * 100) % 20 == 0 or frac == 1.0:
        print(f"  {frac*100:5.1f}% ({n_samples:5d} 条) | "
              f"Train Loss: {tl:.6f} | Val Loss: {vl:.6f} | Val Acc: {va:.4f}")

print(f"\n  最终验证集 Accuracy : {val_accs_curve[-1]:.4f} ({val_accs_curve[-1]*100:.2f}%)")
print(f"  最佳验证 Accuracy   : {max(val_accs_curve):.4f} (出现在 {train_sizes[np.argmax(val_accs_curve)]*100:.0f}% 训练数据时)")

# ============================================================
# 五、最终模型评估
# ============================================================
print("\n" + "=" * 60)
print("五、最终模型评估（全量训练数据）")
print("=" * 60)

# 用全量训练数据训练的最终模型
nb_final_eval = MultinomialNB(alpha=1.0, random_state=SEED)
nb_final_eval.fit(X_train_tfidf, y_train)

y_val_pred = nb_final_eval.predict(X_val_tfidf)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"  验证集 Accuracy : {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"\n  分类报告:")
print(classification_report(y_val, y_val_pred, digits=4))

# 混淆矩阵
cm = confusion_matrix(y_val, y_val_pred)
print(f"  混淆矩阵 (行=真实, 列=预测):")
print(cm)

# ============================================================
# 六、损失曲线绘制
# ============================================================
print("\n" + "=" * 60)
print("六、绘制损失曲线")
print("=" * 60)

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---- 左图：Log-Loss 学习曲线 ----
ax1 = axes[0]
ax1.plot(train_sizes * len(X_train_text), train_losses_curve,
         'b-o', label='Train Log-Loss', linewidth=2, markersize=4)
ax1.plot(train_sizes * len(X_train_text), val_losses_curve,
         'r-s', label='Val Log-Loss', linewidth=2, markersize=4)

# 标注最低验证损失点
best_idx = np.argmin(val_losses_curve)
best_n = int(train_sizes[best_idx] * len(X_train_text))
ax1.axvline(best_n, color='red', linestyle='--', alpha=0.5)
ax1.scatter([best_n], [val_losses_curve[best_idx]],
             color='red', s=80, zorder=5, marker='*')
ax1.annotate(f'Best Val\n{best_n} samples\nLoss={val_losses_curve[best_idx]:.4f}',
             xy=(best_n, val_losses_curve[best_idx]),
             xytext=(best_n + 300, val_losses_curve[best_idx] + 0.15),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, color='red')

ax1.set_xlabel('Training Samples Seen', fontsize=12)
ax1.set_ylabel('Log-Loss', fontsize=12)
ax1.set_title('MultinomialNB: Log-Loss Learning Curve', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.4)

# ---- 中图：验证集 Accuracy 学习曲线 ----
ax2 = axes[1]
ax2.plot(train_sizes * len(X_train_text), val_accs_curve,
         'g-^', linewidth=2, markersize=4, label='Val Accuracy')

best_acc_idx = np.argmax(val_accs_curve)
best_acc_n = int(train_sizes[best_acc_idx] * len(X_train_text))
ax2.axvline(best_acc_n, color='green', linestyle='--', alpha=0.5)
ax2.scatter([best_acc_n], [val_accs_curve[best_acc_idx]],
            color='green', s=80, zorder=5, marker='*')
ax2.annotate(f'Best Acc\n{best_acc_n} samples\nAcc={val_accs_curve[best_acc_idx]:.4f}',
             xy=(best_acc_n, val_accs_curve[best_acc_idx]),
             xytext=(best_acc_n + 300, val_accs_curve[best_acc_idx] - 0.01),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=9, color='green')

ax2.set_xlabel('Training Samples Seen', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('MultinomialNB: Validation Accuracy vs Training Size', fontsize=13)
ax2.grid(True, alpha=0.4)

# ---- 右图：混淆矩阵热力图 ----
ax3 = axes[2]
im = ax3.imshow(cm, cmap='Blues', aspect='auto')
ax3.set_xticks(np.arange(10))
ax3.set_yticks(np.arange(10))
ax3.set_xticklabels(np.arange(10))
ax3.set_yticklabels(np.arange(10))
ax3.set_xlabel('Predicted Label', fontsize=12)
ax3.set_ylabel('True Label', fontsize=12)
ax3.set_title(f'Confusion Matrix (Val Acc: {val_acc:.4f})', fontsize=13)
plt.colorbar(im, ax=ax3, shrink=0.8)

for i in range(10):
    for j in range(10):
        val_c = cm[i, j]
        color = 'white' if val_c > cm.max() / 2 else 'black'
        ax3.text(j, i, str(val_c), ha='center', va='center',
                color=color, fontsize=8)

plt.suptitle(
    'MultinomialNB Baseline — News 10-Class Classification\n'
    'Alpha=1.0 (Laplace), TF-IDF(unigram+bigram, sublinear_tf, max_features=50000)',
    fontsize=13, fontweight='bold', y=1.05,
)
plt.tight_layout()

fig_path = str(FIG_DIR / "nb_loss_curve.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  图像已保存: {fig_path}")
plt.close()

# ============================================================
# 七、对比：使用全部训练数据重新训练 + 最终预测
# ============================================================
print("\n" + "=" * 60)
print("七、全量重训 + 测试集预测")
print("=" * 60)

# 用全部训练数据 fit 一次
nb_final = MultinomialNB(alpha=1.0, random_state=SEED)
nb_final.fit(X_train_tfidf, y_train)

y_test_pred = nb_final.predict(X_test_tfidf)
print(f"  测试集预测完成，共 {len(y_test_pred)} 条")
print(f"  预测类别分布:")
for c in sorted(np.unique(y_test_pred)):
    print(f"    class {c}: {(y_test_pred == c).sum()} 条")

# 保存预测结果
pred_path = str(PRED_DIR / "predictions_nb_baseline.csv")
pd.DataFrame(y_test_pred).to_csv(pred_path, index=False, header=False)
print(f"  预测文件已保存: {pred_path}")

print("\n" + "=" * 60)
print("实验完成")
print("=" * 60)
