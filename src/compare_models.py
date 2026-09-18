"""
四种分类器对比实验
=================
朴素贝叶斯 (MultinomialNB) / 线性支持向量机 (LinearSVC) /
逻辑回归 (LogisticRegression) / 多层感知机 (MLPClassifier)

对每个模型：
  1. 列出关键超参数
  2. 跟踪迭代过程的训练损失与验证损失
  3. 绘制迭代损失曲线（4 模型同图对比）
  4. 最终验证精度对比

统一训练步数：N_EPOCHS = 20
  - NB: partial_fit，每 epoch 内打乱 batch 顺序
  - LR: warm_start=True, max_iter=1（lbfgs 迭代）
  - SVC: warm_start=True, max_iter=1（liblinear 迭代）
  - MLP: warm_start=True, max_iter=1（adam 每 epoch 跑一遍数据）

损失函数说明：
  - NB / LR / MLP: log_loss (交叉熵)，概率输出可直接用
  - SVC: hinge_loss (合页损失)，用 decision_function 算

作者：AI 助手
日期：2026-09-17
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, hinge_loss
import warnings, time
warnings.filterwarnings('ignore')

# 集中管理：路径 + 随机种子
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeds import (
    SEED, DATA_DIR, FIG_DIR, PRED_DIR,
    TFIDF_PARAMS, TEST_SIZE,
    set_global_seeds, get_nb_batch_seed, split_data,
)
set_global_seeds()

# ============================================================
# 一、数据加载与划分（与 NB baseline 一致，确保可复现对比）
# ============================================================
print("="*60)
print("一、数据加载与划分")
print("="*60)
train_df = pd.read_csv(str(DATA_DIR / "train_data.csv"))
test_df  = pd.read_csv(str(DATA_DIR / "test_data_unlabeled.csv"))

X_text_all  = train_df['text'].astype(str).tolist()
y_all       = train_df['target'].values
X_test_text = test_df['text'].astype(str).tolist()

X_train_text, X_val_text, y_train, y_val = split_data(
    X_text_all, y_all, test_size=TEST_SIZE, stratify=y_all,
)
print(f"  训练集: {len(X_train_text)} 条 | 验证集: {len(X_val_text)} 条 | 测试集: {len(X_test_text)} 条")

# ============================================================
# 二、TF-IDF 特征工程
# ============================================================
print("\n" + "="*60)
print("二、TF-IDF 特征工程")
print("="*60)
vec = TfidfVectorizer(**TFIDF_PARAMS)
X_train_tfidf = vec.fit_transform(X_train_text)
X_val_tfidf   = vec.transform(X_val_text)
X_test_tfidf  = vec.transform(X_test_text)
classes = np.unique(y_train)
print(f"  特征维度: {X_train_tfidf.shape[1]}")

N_EPOCHS = 20
BATCH    = 64  # NB 的 mini-batch 大小

# ============================================================
# 模型 1: MultinomialNB
# ============================================================
print("\n" + "="*60)
print("模型 1: MultinomialNB (多项式朴素贝叶斯)")
print("="*60)
print("【超参数】")
print("  alpha           = 1.0    (Laplace 平滑)")
print("  class_prior     = None   (从训练数据估计)")
print("  fit_prior       = True   (学习 P(c))")
print("【训练策略】")
print(f"  partial_fit 多 epoch, 每 epoch 打乱 batch={BATCH}")

nb_losses = {'train': [], 'val': []}
t0 = time.time()
nb = MultinomialNB(alpha=1.0)

# 第一批 partial_fit 必须传 classes
init_idx = np.random.RandomState(0).permutation(len(y_train))[:BATCH]  # 固定种子，可复现
nb.partial_fit(X_train_tfidf[init_idx], y_train[init_idx], classes=classes)

for epoch in range(1, N_EPOCHS + 1):
    indices = np.random.RandomState(get_nb_batch_seed(epoch)).permutation(len(y_train))
    for bi in np.array_split(indices, len(y_train) // BATCH):
        nb.partial_fit(X_train_tfidf[bi], y_train[bi])
    nb_losses['train'].append(log_loss(y_train, nb.predict_proba(X_train_tfidf), labels=classes))
    nb_losses['val'].append(  log_loss(y_val,   nb.predict_proba(X_val_tfidf),   labels=classes))
    print(f"  Epoch {epoch:2d} | Train {nb_losses['train'][-1]:.4f} | Val {nb_losses['val'][-1]:.4f}")

nb_acc = accuracy_score(y_val, nb.predict(X_val_tfidf))
nb_time = time.time() - t0
print(f"  最终 Val Acc = {nb_acc:.4f} | 耗时 {nb_time:.1f}s")

# ============================================================
# 模型 2: LinearSVC (用 SGDClassifier(loss='hinge') 等价实现)
# ============================================================
# 注意：sklearn 的 LinearSVC 没有 warm_start 参数，
#       所以这里用 SGDClassifier(loss='hinge', penalty='l2') 来做线性 SVM，
#       它支持 warm_start 和 partial_fit，可以逐迭代训练。
print("\n" + "="*60)
print("模型 2: Linear SVM (SGDClassifier, loss='hinge')")
print("="*60)
print("【超参数】")
print("  loss            = 'hinge'      (合页损失 = LinearSVC 的目标)")
print("  penalty         = 'l2'         (L2 正则化)")
print("  alpha           = 1e-4         (正则化强度, 等价于 LinearSVC 的 C≈1/C·n)")
print("  learning_rate   = 'optimal'    (学习率由 sklearn 自动设定)")
print("  eta0            = 0.01         (初始学习率)")
print("【训练策略】")
print("  warm_start=True, max_iter=1，每 fit() 1 个 epoch")

svc_losses = {'train': [], 'val': []}
t0 = time.time()
svc = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4,
                    max_iter=1, tol=1e-12, warm_start=True,
                    learning_rate='optimal', eta0=0.01,
                    random_state=SEED)

for epoch in range(1, N_EPOCHS + 1):
    svc.fit(X_train_tfidf, y_train)
    svc_losses['train'].append(hinge_loss(y_train, svc.decision_function(X_train_tfidf), labels=classes))
    svc_losses['val'].append(  hinge_loss(y_val,   svc.decision_function(X_val_tfidf),   labels=classes))
    print(f"  Epoch {epoch:2d} | Train {svc_losses['train'][-1]:.4f} | Val {svc_losses['val'][-1]:.4f}")

svc_acc = accuracy_score(y_val, svc.predict(X_val_tfidf))
svc_time = time.time() - t0
print(f"  最终 Val Acc = {svc_acc:.4f} | 耗时 {svc_time:.1f}s")

# ============================================================
# 模型 3: LogisticRegression
# ============================================================
print("\n" + "="*60)
print("模型 3: LogisticRegression (逻辑回归 / Softmax)")
print("="*60)
print("【超参数】")
print("  C               = 1.0      (正则化强度倒数)")
print("  penalty         = 'l2'     (L2 正则化)")
print("  solver          = 'lbfgs'  (拟牛顿法，多分类用 softmax)")
print("  multi_class     = 'auto'   (10 类用 multinomial 一次算)")
print("  max_iter        = 1000     (默认上限，这里用 warm_start 累加)")
print("【训练策略】")
print("  warm_start=True, max_iter=1，每 fit() 1 个 lbfgs 迭代")

lr_losses = {'train': [], 'val': []}
t0 = time.time()
lr = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                        max_iter=1, tol=1e-12, warm_start=True,
                        n_jobs=-1, random_state=SEED)

for epoch in range(1, N_EPOCHS + 1):
    lr.fit(X_train_tfidf, y_train)
    lr_losses['train'].append(log_loss(y_train, lr.predict_proba(X_train_tfidf), labels=classes))
    lr_losses['val'].append(  log_loss(y_val,   lr.predict_proba(X_val_tfidf),   labels=classes))
    print(f"  Epoch {epoch:2d} | Train {lr_losses['train'][-1]:.4f} | Val {lr_losses['val'][-1]:.4f}")

lr_acc = accuracy_score(y_val, lr.predict(X_val_tfidf))
lr_time = time.time() - t0
print(f"  最终 Val Acc = {lr_acc:.4f} | 耗时 {lr_time:.1f}s")

# ============================================================
# 模型 4: MLPClassifier
# ============================================================
print("\n" + "="*60)
print("模型 4: MLPClassifier (多层感知机)")
print("="*60)
print("【超参数】")
print("  hidden_layer_sizes = (100,)      (单隐藏层 100 神经元)")
print("  activation         = 'relu'      (ReLU 激活)")
print("  solver             = 'adam'      (Adam 优化器)")
print("  alpha              = 1e-4        (L2 正则化强度)")
print("  learning_rate_init = 1e-3        (初始学习率)")
print("  batch_size         = 200         (mini-batch 大小)")
print("【训练策略】")
print("  warm_start=True, max_iter=1，每 fit() 1 个 epoch")

mlp_losses = {'train': [], 'val': []}
mlp_batch_losses = []  # 额外的 MLP 内部每 batch 损失（loss_curve_）
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                    alpha=1e-4, learning_rate_init=1e-3, batch_size=200,
                    max_iter=1, tol=1e-12, warm_start=True,
                    random_state=SEED, verbose=False)

for epoch in range(1, N_EPOCHS + 1):
    mlp.fit(X_train_tfidf, y_train)
    mlp_batch_losses.extend(mlp.loss_curve_)
    mlp_losses['train'].append(log_loss(y_train, mlp.predict_proba(X_train_tfidf), labels=classes))
    mlp_losses['val'].append(  log_loss(y_val,   mlp.predict_proba(X_val_tfidf),   labels=classes))
    print(f"  Epoch {epoch:2d} | Train {mlp_losses['train'][-1]:.4f} | Val {mlp_losses['val'][-1]:.4f}")

mlp_acc = accuracy_score(y_val, mlp.predict(X_val_tfidf))
mlp_time = time.time() - t0
print(f"\n  最终 Val Acc(20 步) = {mlp_acc:.4f} | 耗时 {mlp_time:.1f}s")

# ============================================================
# 全收敛版（用同样的超参数但跑到收敛，得到公平最终精度对比）
# ============================================================
print("\n" + "="*60)
print("附录：各模型全收敛后的最终精度（公平对比）")
print("="*60)

from sklearn.svm import LinearSVC as _LinearSVC

# NB: 全量重训
nb_final = MultinomialNB(alpha=1.0)
nb_final.fit(X_train_tfidf, y_train)
nb_acc_conv = accuracy_score(y_val, nb_final.predict(X_val_tfidf))
print(f"  NB   (alpha=1.0, 全量): {nb_acc_conv:.4f}")

# LinearSVC (真正的 liblinear 实现)
svc_final = _LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
svc_final.fit(X_train_tfidf, y_train)
svc_acc_conv = accuracy_score(y_val, svc_final.predict(X_val_tfidf))
print(f"  SVC  (C=1.0, liblinear): {svc_acc_conv:.4f}")

# LR (lbfgs 跑到收敛)
lr_final = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=SEED)
lr_final.fit(X_train_tfidf, y_train)
lr_acc_conv = accuracy_score(y_val, lr_final.predict(X_val_tfidf))
print(f"  LR   (C=1.0, lbfgs 收敛): {lr_acc_conv:.4f}")

# MLP (adam 跑到收敛)
mlp_final = MLPClassifier(hidden_layer_sizes=(100,), alpha=1e-4,
                          learning_rate_init=1e-3, batch_size=200,
                          max_iter=100, random_state=SEED, early_stopping=False)
mlp_final.fit(X_train_tfidf, y_train)
mlp_acc_conv = accuracy_score(y_val, mlp_final.predict(X_val_tfidf))
print(f"  MLP  ((100,), 100 epoch): {mlp_acc_conv:.4f}")

# ============================================================
# 三、绘制对比图
# ============================================================
print("\n" + "="*60)
print("三、绘制对比图")
print("="*60)

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

colors = {'NB':  '#1f77b4',  # 蓝
          'SVC': '#ff7f0e',  # 橙
          'LR':  '#2ca02c',  # 绿
          'MLP': '#d62728'}  # 红

epochs = np.arange(1, N_EPOCHS + 1)

# ----- 子图 1: 训练集 loss（NB / LR / MLP 用 log_loss，SVC 用 hinge_loss）
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, nb_losses['train'],  '-o', color=colors['NB'],  label='NB',  markersize=4, linewidth=1.5)
ax1.plot(epochs, lr_losses['train'],  '-^', color=colors['LR'],  label='LR',  markersize=4, linewidth=1.5)
ax1.plot(epochs, mlp_losses['train'], '-v', color=colors['MLP'], label='MLP', markersize=4, linewidth=1.5)
ax1_twin = ax1.twinx()
ax1_twin.plot(epochs, svc_losses['train'], '-s', color=colors['SVC'], label='SVC (hinge)', markersize=4, linewidth=1.5)
ax1.set_xlabel('Epoch / Iteration');  ax1.set_ylabel('log_loss (NB / LR / MLP)')
ax1_twin.set_ylabel('hinge_loss (SVC)', color=colors['SVC'])
ax1_twin.tick_params(axis='y', labelcolor=colors['SVC'])
ax1.set_title('Train Loss vs Epoch')
ax1.grid(True, alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# ----- 子图 2: 验证集 loss
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, nb_losses['val'],  '-o', color=colors['NB'],  label='NB',  markersize=4, linewidth=1.5)
ax2.plot(epochs, lr_losses['val'],  '-^', color=colors['LR'],  label='LR',  markersize=4, linewidth=1.5)
ax2.plot(epochs, mlp_losses['val'], '-v', color=colors['MLP'], label='MLP', markersize=4, linewidth=1.5)
ax2_twin = ax2.twinx()
ax2_twin.plot(epochs, svc_losses['val'], '-s', color=colors['SVC'], label='SVC (hinge)', markersize=4, linewidth=1.5)
ax2.set_xlabel('Epoch / Iteration');  ax2.set_ylabel('log_loss (NB / LR / MLP)')
ax2_twin.set_ylabel('hinge_loss (SVC)', color=colors['SVC'])
ax2_twin.tick_params(axis='y', labelcolor=colors['SVC'])
ax2.set_title('Validation Loss vs Epoch')
ax2.grid(True, alpha=0.3)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# ----- 子图 3: 泛化差距 train - val（仅 NB / LR / MLP 同尺度）
ax3 = fig.add_subplot(gs[1, 0])
for name in ['NB', 'LR', 'MLP']:
    losses = {'NB': nb_losses, 'LR': lr_losses, 'MLP': mlp_losses}[name]
    gap = [t - v for t, v in zip(losses['train'], losses['val'])]
    ax3.plot(epochs, gap, '-o', color=colors[name], label=name, markersize=4)
ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Train Loss − Val Loss')
ax3.set_title('Generalization Gap (正值=欠拟合, 负值=过拟合)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ----- 子图 4: MLP 内部每 batch 损失 (loss_curve_)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(np.arange(1, len(mlp_batch_losses) + 1), mlp_batch_losses,
         color=colors['MLP'], linewidth=0.8)
ax4.set_xlabel('Mini-batch Step')
ax4.set_ylabel('Training Loss (per batch)')
ax4.set_title(f'MLP Internal Loss Curve ({len(mlp_batch_losses)} batches, 20 epochs)')
ax4.grid(True, alpha=0.3)

# ----- 子图 5: 最终 Val Accuracy 柱状图（含"20 步"和"全收敛"两组）
ax5 = fig.add_subplot(gs[2, 0])
names = ['NB', 'LinearSVC', 'LR', 'MLP']
accs_20   = [nb_acc, svc_acc, lr_acc, mlp_acc]
accs_conv = [nb_acc_conv, svc_acc_conv, lr_acc_conv, mlp_acc_conv]

x = np.arange(len(names))
w = 0.38
b1 = ax5.bar(x - w/2, accs_20,   w, label='@Epoch 20 (迭代曲线末点)',
             color=[colors[n] if n in colors else '#9467bd' for n in names],
             edgecolor='black', linewidth=0.5)
b2 = ax5.bar(x + w/2, accs_conv, w, label='Converged (全收敛)',
             color=[colors[n] if n in colors else '#9467bd' for n in names],
             alpha=0.45, edgecolor='black', linewidth=0.5,
             hatch='//')
for bar, a in zip(b1, accs_20):
    ax5.text(bar.get_x() + bar.get_width()/2, a + 0.002, f'{a:.4f}',
             ha='center', fontsize=9, fontweight='bold')
for bar, a in zip(b2, accs_conv):
    ax5.text(bar.get_x() + bar.get_width()/2, a + 0.002, f'{a:.4f}',
             ha='center', fontsize=9, fontweight='bold')
ax5.set_xticks(x); ax5.set_xticklabels(names)
ax5.set_ylabel('Validation Accuracy')
ax5.set_title('Final Validation Accuracy: 20-iter vs Fully Converged')
ax5.set_ylim(min(accs_20 + accs_conv) - 0.02, 1.0)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# ----- 子图 6: 训练耗时柱状图（20 步）
ax6 = fig.add_subplot(gs[2, 1])
times = [nb_time, svc_time, lr_time, mlp_time]
bars = ax6.bar(names, times, color=[colors[n] if n in colors else '#9467bd' for n in names],
               edgecolor='black', linewidth=0.5)
for bar, t in zip(bars, times):
    ax6.text(bar.get_x() + bar.get_width()/2, t + max(times)*0.02, f'{t:.1f}s',
             ha='center', fontsize=11, fontweight='bold')
ax6.set_ylabel('Training Time (seconds, 20 epochs)')
ax6.set_title('Training Time Comparison (20 epochs)')
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    'Comparison Experiment: NB / LinearSVC / LogisticRegression / MLP\n'
    'News 10-Class Classification  ·  TF-IDF(unigram+bigram, 50k features)  ·  20 training steps',
    fontsize=14, fontweight='bold', y=1.00)
plt.savefig(str(FIG_DIR / "compare_loss_curve.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: {DATA_DIR}/compare_loss_curve.png")

# ============================================================
# 四、测试集预测（每个模型用最终版本预测一次）
# ============================================================
print("\n" + "="*60)
print("四、测试集预测")
print("="*60)

# NB 重新用全量训练数据 fit 一次
nb_full = MultinomialNB(alpha=1.0)
nb_full.fit(X_train_tfidf, y_train)

# LR/SVC/MLP 直接用迭代训练后的版本
preds = {
    'nb':  nb_full.predict(X_test_tfidf),
    'svc': svc.predict(X_test_tfidf),
    'lr':  lr.predict(X_test_tfidf),
    'mlp': mlp.predict(X_test_tfidf),
}
for name, pred in preds.items():
    path = str(PRED_DIR / f"predictions_{name}_compare.csv")
    pd.DataFrame(pred).to_csv(path, index=False, header=False)
    print(f"  {name:4s}: {len(pred)} 条 → {path}")

# ============================================================
# 五、汇总
# ============================================================
print("\n" + "="*60)
print("五、最终汇总")
print("="*60)
print(f"{'模型':<14}{'核心超参数':<32}{'@20步':>10}{'全收敛':>10}{'耗时(s)':>10}")
print("-" * 76)
print(f"{'MultinomialNB':<14}{'alpha=1.0':<32}{nb_acc:>10.4f}{nb_acc_conv:>10.4f}{nb_time:>10.1f}")
print(f"{'LinearSVC':<14}{'C=1.0, hinge, L2':<32}{svc_acc:>10.4f}{svc_acc_conv:>10.4f}{svc_time:>10.1f}")
print(f"{'LogisticReg':<14}{'C=1.0, lbfgs, L2':<32}{lr_acc:>10.4f}{lr_acc_conv:>10.4f}{lr_time:>10.1f}")
print(f"{'MLP':<14}{'(100,), relu, adam':<32}{mlp_acc:>10.4f}{mlp_acc_conv:>10.4f}{mlp_time:>10.1f}")
print("\n实验完成。")
