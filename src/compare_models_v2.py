"""
四分类器对比实验 v2
===================
改进点：
  1. 支持邮件头清洗（clean_email_headers 参数）
  2. 中文/全角符号字体修复（Arial Unicode MS）
  3. 输出图表改为 4 联布局（去掉中间的泛化差距与 MLP 内部曲线）
  4. 提供 side-by-side 对比图（原始 vs 清洗）

作者：AI 助手
日期：2026-09-17
"""

import sys
from pathlib import Path

import re
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ------------------------------------------------------------
# 0. 中文/全角字体修复（macOS 通用 Arial Unicode 兜底）
# ------------------------------------------------------------
def setup_cjk_font():
    """加载一个含 CJK 字形的字体作为默认 sans-serif。"""
    candidates = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        try:
            fm.fontManager.addfont(path)
            # 取 font family name
            fp = fm.FontProperties(fname=path)
            family = fp.get_name()
            matplotlib.rcParams['font.sans-serif'] = [family, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return family
        except Exception:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False
    return None

CJK_FAMILY = setup_cjk_font()
print(f"[字体] 使用 CJK 字体: {CJK_FAMILY}")

# ------------------------------------------------------------
# 1. 邮件头清洗
# ------------------------------------------------------------
HEADER_LINE_RE = re.compile(
    r'^[A-Z][A-Za-z0-9-]*:\s'        # 匹配 "From:" / "Subject:" 等
)
QUOTE_RE = re.compile(r'^\s*>+\s?')   # 匹配 "> " / ">> " 等引文行
SIG_RE   = re.compile(r'^--\s?$')     # 签名分隔行 "-- " 单独成行
EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
URL_RE   = re.compile(r'\b\S+://\S+|www\.\S+')
NNTP_ID_RE = re.compile(r'^\s*<\s*\S+@\S+\s*>\s*$')  # 纯消息 ID 行

def clean_email(text: str) -> str:
    """
    去除 20 Newsgroups 风格邮件的"噪声"：
      - From/Subject/Organization/Reply-To... 等邮件头（第一空行之前）
      - 签名分隔行 '-- ' 之后
      - 行首 '>' 引文
      - 邮件地址与 URL
      - 单独成行的 NNTP 消息 ID（<...@...>）
    保留所有正文字符。
    """
    lines = text.splitlines()
    # (a) 跳过开头的邮件头直到第一个空行
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '':
            body_start = i + 1
            break
        if HEADER_LINE_RE.match(line):
            continue
        body_start = i
        break

    body = lines[body_start:]

    # (b) 移除签名块 ('-- ' 单独成行后全部丢弃)
    cleaned = []
    for line in body:
        if SIG_RE.match(line):
            break
        cleaned.append(line)

    body = '\n'.join(cleaned)

    # (c) 邮件地址 / URL
    body = EMAIL_RE.sub(' ', body)
    body = URL_RE.sub(' ', body)

    # (d) 行首引文 '> ' 与 单独行消息 ID
    new_lines = []
    for line in body.splitlines():
        if QUOTE_RE.match(line):
            continue
        if NNTP_ID_RE.match(line):
            continue
        new_lines.append(line)

    return '\n'.join(new_lines).strip()

# ------------------------------------------------------------
# 2. 训练 + 监控
# ------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, hinge_loss
import warnings
warnings.filterwarnings('ignore')

# 集中管理：路径 + 随机种子
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeds import (
    SEED, DATA_DIR, FIG_DIR, PRED_DIR,
    TFIDF_PARAMS, TEST_SIZE,
    set_global_seeds, get_nb_batch_seed, split_data,
)
set_global_seeds()


def run_experiment(clean_email_headers: bool, n_epochs: int = 20):
    """
    跑完 4 个模型，返回 (results, X_train_tfidf, X_val_tfidf, ...) 的字典。
    results: { model_name: { 'epochs_train': [...], 'epochs_val': [...],
                              'epoch_acc_20': float, 'final_acc_conv': float,
                              'time': float, 'mlp_batch_losses': [...](可选) } }
    """
    tag = "清洗后" if clean_email_headers else "原始"
    print("\n" + "=" * 70)
    print(f"开始实验 [{tag}]  n_epochs={n_epochs}")
    print("=" * 70)

    # ---- 数据加载
    train_df = pd.read_csv(str(DATA_DIR / "train_data.csv"))
    test_df  = pd.read_csv(str(DATA_DIR / "test_data_unlabeled.csv"))

    if clean_email_headers:
        X_text_all  = [clean_email(t) for t in train_df['text'].astype(str)]
        X_test_text = [clean_email(t) for t in test_df['text'].astype(str)]
        print(f"[清洗] 训练样本平均字符数 {np.mean([len(t) for t in X_text_all]):.1f}")
    else:
        X_text_all  = train_df['text'].astype(str).tolist()
        X_test_text = test_df['text'].astype(str).tolist()
        print(f"[原始] 训练样本平均字符数 {np.mean([len(t) for t in X_text_all]):.1f}")
        # 如果清洗关闭，也用 mean 不计算原文长度，避免数据量不一致

    y_all = train_df['target'].values

    X_train_text, X_val_text, y_train, y_val = split_data(
        X_text_all, y_all, test_size=TEST_SIZE, stratify=y_all,
    )
    print(f"训练 {len(X_train_text)} | 验证 {len(X_val_text)}")

    # ---- TF-IDF
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = vec.fit_transform(X_train_text)
    X_val_tfidf   = vec.transform(X_val_text)
    X_test_tfidf  = vec.transform(X_test_text)
    classes = np.unique(y_train)
    print(f"TF-IDF 维度: {X_train_tfidf.shape[1]}")

    batch = 64
    results = {}
    extras = {"X_train_tfidf": X_train_tfidf, "X_val_tfidf": X_val_tfidf,
              "X_test_tfidf": X_test_tfidf, "y_train": y_train, "y_val": y_val}

    # ============================================================
    # (1) MultinomialNB  - partial_fit 多 epoch
    # ============================================================
    print("\n--- (1) MultinomialNB (alpha=1.0) ---")
    train_loss, val_loss = [], []
    nb = MultinomialNB(alpha=1.0)
    init = np.random.RandomState(0).permutation(len(y_train))[:batch]  # 固定种子可复现
    nb.partial_fit(X_train_tfidf[init], y_train[init], classes=classes)
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        order = np.random.RandomState(get_nb_batch_seed(epoch)).permutation(len(y_train))
        for bi in np.array_split(order, len(y_train) // batch):
            nb.partial_fit(X_train_tfidf[bi], y_train[bi])
        train_loss.append(log_loss(y_train, nb.predict_proba(X_train_tfidf), labels=classes))
        val_loss.append(  log_loss(y_val,   nb.predict_proba(X_val_tfidf),   labels=classes))
    nb_t = time.time() - t0

    nb_full = MultinomialNB(alpha=1.0, random_state=SEED)
    nb_full.fit(X_train_tfidf, y_train)
    nb_acc_conv = accuracy_score(y_val, nb_full.predict(X_val_tfidf))
    nb_acc_e20  = accuracy_score(y_val, nb.predict(X_val_tfidf))
    print(f"  Epoch 20: train={train_loss[-1]:.4f} val={val_loss[-1]:.4f} "
          f"acc20={nb_acc_e20:.4f} acc_conv={nb_acc_conv:.4f} time={nb_t:.1f}s")
    results['NB'] = {
        'train': train_loss, 'val': val_loss,
        'acc_20': nb_acc_e20, 'acc_conv': nb_acc_conv, 'time': nb_t,
        'mlp_batches': None,
    }
    extras['NB_pred'] = nb_full.predict(X_test_tfidf)

    # ============================================================
    # (2) Linear SVM via SGDClassifier(loss='hinge')
    # ============================================================
    print("\n--- (2) Linear SVM (SGDClassifier, loss='hinge') ---")
    train_loss, val_loss = [], []
    sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4,
                        max_iter=1, tol=1e-12, warm_start=True,
                        learning_rate='optimal', eta0=0.01,
                        random_state=SEED)
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        sgd.fit(X_train_tfidf, y_train)
        train_loss.append(hinge_loss(y_train, sgd.decision_function(X_train_tfidf), labels=classes))
        val_loss.append(  hinge_loss(y_val,   sgd.decision_function(X_val_tfidf),   labels=classes))
    sgd_t = time.time() - t0

    sgd_full = LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
    sgd_full.fit(X_train_tfidf, y_train)
    sgd_acc_conv = accuracy_score(y_val, sgd_full.predict(X_val_tfidf))
    sgd_acc_e20  = accuracy_score(y_val, sgd.predict(X_val_tfidf))
    print(f"  Epoch 20: train={train_loss[-1]:.4f} val={val_loss[-1]:.4f} "
          f"acc20={sgd_acc_e20:.4f} acc_conv={sgd_acc_conv:.4f} time={sgd_t:.1f}s")
    results['SVC'] = {
        'train': train_loss, 'val': val_loss,
        'acc_20': sgd_acc_e20, 'acc_conv': sgd_acc_conv, 'time': sgd_t,
        'mlp_batches': None,
    }
    extras['SVC_pred'] = sgd_full.predict(X_test_tfidf)

    # ============================================================
    # (3) Logistic Regression (lbfgs)
    # ============================================================
    print("\n--- (3) LogisticRegression (C=1.0, lbfgs) ---")
    train_loss, val_loss = [], []
    lr = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                            max_iter=1, tol=1e-12, warm_start=True,
                            n_jobs=-1, random_state=SEED)
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        lr.fit(X_train_tfidf, y_train)
        train_loss.append(log_loss(y_train, lr.predict_proba(X_train_tfidf), labels=classes))
        val_loss.append(  log_loss(y_val,   lr.predict_proba(X_val_tfidf),   labels=classes))
    lr_t = time.time() - t0

    lr_full = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=SEED)
    lr_full.fit(X_train_tfidf, y_train)
    lr_acc_conv = accuracy_score(y_val, lr_full.predict(X_val_tfidf))
    lr_acc_e20  = accuracy_score(y_val, lr.predict(X_val_tfidf))
    print(f"  Epoch 20: train={train_loss[-1]:.4f} val={val_loss[-1]:.4f} "
          f"acc20={lr_acc_e20:.4f} acc_conv={lr_acc_conv:.4f} time={lr_t:.1f}s")
    results['LR'] = {
        'train': train_loss, 'val': val_loss,
        'acc_20': lr_acc_e20, 'acc_conv': lr_acc_conv, 'time': lr_t,
        'mlp_batches': None,
    }
    extras['LR_pred'] = lr_full.predict(X_test_tfidf)

    # ============================================================
    # (4) MLP (adam)
    # ============================================================
    print("\n--- (4) MLP ((100,), relu, adam) ---")
    train_loss, val_loss = [], []
    mlp_batches = []
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                        alpha=1e-4, learning_rate_init=1e-3, batch_size=200,
                        max_iter=1, tol=1e-12, warm_start=True,
                        random_state=SEED, verbose=False)
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        mlp.fit(X_train_tfidf, y_train)
        mlp_batches.extend(mlp.loss_curve_)
        train_loss.append(log_loss(y_train, mlp.predict_proba(X_train_tfidf), labels=classes))
        val_loss.append(  log_loss(y_val,   mlp.predict_proba(X_val_tfidf),   labels=classes))
    mlp_t = time.time() - t0

    mlp_full = MLPClassifier(hidden_layer_sizes=(100,), alpha=1e-4,
                             learning_rate_init=1e-3, batch_size=200,
                             max_iter=100, random_state=SEED, early_stopping=False)
    mlp_full.fit(X_train_tfidf, y_train)
    mlp_acc_conv = accuracy_score(y_val, mlp_full.predict(X_val_tfidf))
    mlp_acc_e20  = accuracy_score(y_val, mlp.predict(X_val_tfidf))
    print(f"  Epoch 20: train={train_loss[-1]:.4f} val={val_loss[-1]:.4f} "
          f"acc20={mlp_acc_e20:.4f} acc_conv={mlp_acc_conv:.4f} time={mlp_t:.1f}s")
    results['MLP'] = {
        'train': train_loss, 'val': val_loss,
        'acc_20': mlp_acc_e20, 'acc_conv': mlp_acc_conv, 'time': mlp_t,
        'mlp_batches': mlp_batches,
    }
    extras['MLP_pred'] = mlp_full.predict(X_test_tfidf)

    return results, extras


# ------------------------------------------------------------
# 3. 4 联对比图
# ------------------------------------------------------------
def plot_4panel(results, title_suffix, save_path):
    colors = {'NB': '#1f77b4', 'SVC': '#ff7f0e', 'LR': '#2ca02c', 'MLP': '#d62728'}
    epochs = np.arange(1, len(results['NB']['train']) + 1)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # ---- (1) Train Loss
    ax1 = fig.add_subplot(gs[0, 0])
    for name in ['NB', 'LR', 'MLP']:
        ax1.plot(epochs, results[name]['train'], '-o',
                 color=colors[name], label=name, markersize=3.5, linewidth=1.4)
    ax1b = ax1.twinx()
    ax1b.plot(epochs, results['SVC']['train'], '-s', color=colors['SVC'],
              label='SVC (hinge)', markersize=3.5, linewidth=1.4)
    ax1.set_xlabel('Epoch / Iteration'); ax1.set_ylabel('log_loss')
    ax1b.set_ylabel('hinge_loss', color=colors['SVC'])
    ax1b.tick_params(axis='y', labelcolor=colors['SVC'])
    ax1.set_title('训练损失 vs Epoch  (Train Loss vs Epoch)')
    ax1.grid(True, alpha=0.3)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc='upper right', fontsize=9)

    # ---- (2) Validation Loss
    ax2 = fig.add_subplot(gs[0, 1])
    for name in ['NB', 'LR', 'MLP']:
        ax2.plot(epochs, results[name]['val'], '-o',
                 color=colors[name], label=name, markersize=3.5, linewidth=1.4)
    ax2b = ax2.twinx()
    ax2b.plot(epochs, results['SVC']['val'], '-s', color=colors['SVC'],
              label='SVC (hinge)', markersize=3.5, linewidth=1.4)
    ax2.set_xlabel('Epoch / Iteration'); ax2.set_ylabel('log_loss')
    ax2b.set_ylabel('hinge_loss', color=colors['SVC'])
    ax2b.tick_params(axis='y', labelcolor=colors['SVC'])
    ax2.set_title('验证损失 vs Epoch  (Validation Loss vs Epoch)')
    ax2.grid(True, alpha=0.3)
    l1, lab1 = ax2.get_legend_handles_labels()
    l2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, lab1 + lab2, loc='upper right', fontsize=9)

    # ---- (3) Final Val Accuracy  (20 epoch vs converged)
    ax3 = fig.add_subplot(gs[1, 0])
    names = ['NB', 'SVC', 'LR', 'MLP']
    acc_20   = [results[n]['acc_20']   for n in names]
    acc_conv = [results[n]['acc_conv'] for n in names]
    x = np.arange(len(names)); w = 0.38
    b1 = ax3.bar(x - w/2, acc_20,   w,
                 color=[colors[n] for n in names],
                 edgecolor='black', linewidth=0.5, label='Epoch 20 处')
    b2 = ax3.bar(x + w/2, acc_conv, w,
                 color=[colors[n] for n in names], alpha=0.45,
                 edgecolor='black', linewidth=0.5, hatch='//', label='全收敛后')
    for bar, a in zip(b1, acc_20):
        ax3.text(bar.get_x() + bar.get_width()/2, a + 0.002, f'{a:.4f}',
                 ha='center', fontsize=9, fontweight='bold')
    for bar, a in zip(b2, acc_conv):
        ax3.text(bar.get_x() + bar.get_width()/2, a + 0.002, f'{a:.4f}',
                 ha='center', fontsize=9, fontweight='bold')
    ax3.set_xticks(x); ax3.set_xticklabels(names)
    ax3.set_ylabel('验证集精度')
    ax3.set_title('最终验证精度对比  (Final Val Accuracy)')
    ax3.set_ylim(min(acc_20 + acc_conv) - 0.02, 1.0)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # ---- (4) Training Time
    ax4 = fig.add_subplot(gs[1, 1])
    times = [results[n]['time'] for n in names]
    bars = ax4.bar(names, times,
                   color=[colors[n] for n in names],
                   edgecolor='black', linewidth=0.5)
    for bar, t in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, t + max(times)*0.02, f'{t:.1f}s',
                 ha='center', fontsize=10, fontweight='bold')
    ax4.set_ylabel('训练耗时 (20 epoch, 秒)')
    ax4.set_title('训练耗时对比  (Training Time Cost)')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'四模型对比  /  {title_suffix}\n'
                 f'(Comparison: NB / LinearSVC / LogisticRegression / MLP)  ·  '
                 f'TF-IDF(unigram+bigram, 50k features)  ·  20 training steps',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ------------------------------------------------------------
# 4. 两版对比图：同一张图上对比原始 vs 清洗
# ------------------------------------------------------------
def plot_comparison(res_raw, res_clean, save_path):
    """在两个版本上各画 'train+val loss+acc' 的 4 联图"""
    colors = {'NB': '#1f77b4', 'SVC': '#ff7f0e', 'LR': '#2ca02c', 'MLP': '#d62728'}
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.40)
    epochs = np.arange(1, len(res_raw['NB']['train']) + 1)

    def _plot_train_val(ax_idx, res, title):
        ax = fig.add_subplot(gs[0, ax_idx])
        for name in ['NB', 'LR', 'MLP']:
            ax.plot(epochs, res[name]['train'], '-', color=colors[name],
                    label=f'{name}-train', linewidth=1.0, alpha=0.7)
            ax.plot(epochs, res[name]['val'],   '--', color=colors[name],
                    label=f'{name}-val', linewidth=1.4)
        ax.set_xlabel('Epoch'); ax.set_ylabel('loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    def _plot_acc_time(ax_idx, res, title):
        ax = fig.add_subplot(gs[1, ax_idx])
        names = ['NB', 'SVC', 'LR', 'MLP']
        acc_conv = [res[n]['acc_conv'] for n in names]
        ax.bar(names, acc_conv,
               color=[colors[n] for n in names],
               edgecolor='black', linewidth=0.5)
        for i, (n, a) in enumerate(zip(names, acc_conv)):
            ax.text(i, a + 0.003, f'{a:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylim(min(acc_conv) - 0.02, 1.0)
        ax.set_ylabel('最终验证精度')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    # 上排：4 列分别 (原始 train/val / 清洗 train/val)
    _plot_train_val(0, res_raw,   '[原始] train/val loss vs epoch')
    _plot_train_val(1, res_clean, '[清洗] train/val loss vs epoch')
    # 上排右 2：直接放精度比较 / 单模型对比
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(4); w = 0.38
    raw_acc   = [res_raw[n]['acc_conv']   for n in ['NB', 'SVC', 'LR', 'MLP']]
    clean_acc = [res_clean[n]['acc_conv'] for n in ['NB', 'SVC', 'LR', 'MLP']]
    ax.bar(x - w/2, raw_acc,   w, label='原始',    color='#888', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, clean_acc, w, label='清洗后',  color='#4daf4a', edgecolor='black', linewidth=0.5)
    for i, (r, c) in enumerate(zip(raw_acc, clean_acc)):
        diff = (c - r) * 100
        sign = '+' if diff >= 0 else ''
        ax.text(i, max(r, c) + 0.003, f'{r:.4f}\n{c:.4f}\n{sign}{diff:.2f}pp',
                ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(['NB', 'SVC', 'LR', 'MLP'])
    ax.set_ylabel('Val Accuracy')
    ax.set_title('最终精度对比：原始 vs 清洗')
    ax.set_ylim(min(raw_acc + clean_acc) - 0.02, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 上排第 4 列：泛化差距（train - val）在 Epoch 20 处的对比
    ax = fig.add_subplot(gs[0, 3])
    raw_gap = [res_raw[n]['train'][-1] - res_raw[n]['val'][-1]     for n in ['NB', 'SVC', 'LR', 'MLP']]
    cln_gap = [res_clean[n]['train'][-1] - res_clean[n]['val'][-1] for n in ['NB', 'SVC', 'LR', 'MLP']]
    x = np.arange(4); w = 0.38
    ax.bar(x - w/2, raw_gap, w, label='原始', color='#888', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, cln_gap, w, label='清洗后', color='#4daf4a', edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(['NB', 'SVC', 'LR', 'MLP'])
    ax.set_ylabel('Train − Val  (@ Epoch 20)')
    ax.set_title('过拟合差距对比（train-loss − val-loss）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 下排：耗时对比
    ax = fig.add_subplot(gs[1, 0:2].subgridspec(1, 1)[0, 0])
    names = ['NB', 'SVC', 'LR', 'MLP']
    raw_t   = [res_raw[n]['time']   for n in names]
    clean_t = [res_clean[n]['time'] for n in names]
    x = np.arange(4); w = 0.38
    ax.bar(x - w/2, raw_t,   w, label='原始',   color='#888', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, clean_t, w, label='清洗后', color='#4daf4a', edgecolor='black', linewidth=0.5)
    for i, (r, c) in enumerate(zip(raw_t, clean_t)):
        ax.text(i - w/2, r + max(raw_t + clean_t)*0.02, f'{r:.1f}s', ha='center', fontsize=9)
        ax.text(i + w/2, c + max(raw_t + clean_t)*0.02, f'{c:.1f}s', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('训练耗时 (秒)')
    ax.set_title('训练耗时对比 (20 epoch)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 下排后两列：清洗后 vs 原始 每类的相对提升百分比与 PP 提升
    ax = fig.add_subplot(gs[1, 2:4].subgridspec(1, 1)[0, 0])
    deltas_pp = [(c - r) * 100 for r, c in zip(raw_acc, clean_acc)]
    colors_pp = ['#d62728' if d < 0 else '#2ca02c' for d in deltas_pp]
    bars = ax.bar(names, deltas_pp, color=colors_pp, edgecolor='black', linewidth=0.5)
    for bar, d in zip(bars, deltas_pp):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + (0.05 if h >= 0 else -0.15),
                f'{d:+.2f}pp',
                ha='center', fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax.set_ylabel('清洗后精度差 (pp, 百分点)')
    ax.set_title('邮件头清洗带来的精度提升（百分点）')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('原始 vs 邮件头清洗 —— 四分类器对比  /  Raw vs Email-Header Cleaned',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ------------------------------------------------------------
# 5. 主流程
# ------------------------------------------------------------
if __name__ == "__main__":
    # ---- 第一次：原始数据
    res_raw,   extras_raw   = run_experiment(clean_email_headers=False, n_epochs=20)
    plot_4panel(res_raw, "原始数据  (Raw data, with email headers)",
                str(FIG_DIR / "compare_loss_curve.png"))

    # ---- 第二次：清洗邮件头
    res_clean, extras_clean = run_experiment(clean_email_headers=True,  n_epochs=20)
    plot_4panel(res_clean, "清洗邮件头后  (Cleaned, headers stripped)",
                str(FIG_DIR / "compare_loss_curve_cleaned.png"))

    # ---- 两版对比图
    plot_comparison(res_raw, res_clean,
                    str(FIG_DIR / "compare_raw_vs_cleaned.png"))

    # ---- 汇总
    print("\n" + "=" * 70)
    print("最终汇总：原始 vs 清洗")
    print("=" * 70)
    print(f"{'Model':<8}{'Raw Acc':>12}{'Clean Acc':>14}{'Diff(pp)':>12}{'Raw Time':>12}{'Cln Time':>12}")
    print("-" * 70)
    for n in ['NB', 'SVC', 'LR', 'MLP']:
        r = res_raw[n]['acc_conv'];  c = res_clean[n]['acc_conv']
        rt = res_raw[n]['time'];      ct = res_clean[n]['time']
        print(f"{n:<8}{r:>12.4f}{c:>14.4f}{(c-r)*100:>+12.2f}{rt:>12.1f}{ct:>12.1f}")

    # 保存清洗后的 4 个测试集预测
    for name, key in [('nb','NB_pred'), ('svc','SVC_pred'),
                      ('lr','LR_pred'),  ('mlp','MLP_pred')]:
        path = str(PRED_DIR / f"predictions_{name}_cleaned.csv")
        pd.DataFrame(extras_clean[key]).to_csv(path, index=False, header=False)
        print(f"已保存清洗后预测: {path}")
