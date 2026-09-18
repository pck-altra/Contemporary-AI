"""
单独跑一次原始数据上的 MLP / LinearSVC（全收敛版本），
用于生成最优模型的 prediction 文件。
"""
import sys
from pathlib import Path

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 集中管理：路径 + 随机种子
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeds import (
    SEED, DATA_DIR, PRED_DIR,
    TFIDF_PARAMS, TEST_SIZE,
    set_global_seeds, split_data,
)
set_global_seeds()

print("=" * 60)
print("重新生成最优模型（原始数据）的预测文件")
print("=" * 60)

# ---- 加载数据（用与 v2 相同的种子和划分）
train_df = pd.read_csv(str(DATA_DIR / "train_data.csv"))
test_df  = pd.read_csv(str(DATA_DIR / "test_data_unlabeled.csv"))
X_text_all  = train_df['text'].astype(str).tolist()
X_test_text = test_df['text'].astype(str).tolist()
y_all = train_df['target'].values

X_train_text, X_val_text, y_train, y_val = split_data(
    X_text_all, y_all, test_size=TEST_SIZE, stratify=y_all,
)

# ---- TF-IDF（与 v2 完全一致）
vec = TfidfVectorizer(**TFIDF_PARAMS)
X_train_tfidf = vec.fit_transform(X_train_text)
X_val_tfidf   = vec.transform(X_val_text)
X_test_tfidf  = vec.transform(X_test_text)
print(f"TF-IDF 维度: {X_train_tfidf.shape[1]}")

# ---- MLP (原始数据) - 全收敛版本
print("\n训练 MLP（原始，全收敛）...")
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                    alpha=1e-4, learning_rate_init=1e-3, batch_size=200,
                    max_iter=100, random_state=SEED, early_stopping=False)
mlp.fit(X_train_tfidf, y_train)
t_mlp = time.time() - t0
y_val_pred = mlp.predict(X_val_tfidf)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"  MLP val_acc = {val_acc:.4f}  (time={t_mlp:.1f}s)")

mlp_test_pred = mlp.predict(X_test_tfidf)
pd.DataFrame(mlp_test_pred).to_csv(str(PRED_DIR / "predictions_mlp_raw.csv"),
                                   index=False, header=False)
print(f"  保存: predictions_mlp_raw.csv  shape={mlp_test_pred.shape}")

# ---- LinearSVC (原始数据) - 全收敛版本
print("\n训练 LinearSVC（原始，全收敛）...")
t0 = time.time()
svc = LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
svc.fit(X_train_tfidf, y_train)
t_svc = time.time() - t0
y_val_pred = svc.predict(X_val_tfidf)
val_acc_svc = accuracy_score(y_val, y_val_pred)
print(f"  SVC val_acc = {val_acc_svc:.4f}  (time={t_svc:.1f}s)")

svc_test_pred = svc.predict(X_test_tfidf)
pd.DataFrame(svc_test_pred).to_csv(str(PRED_DIR / "predictions_svc_raw.csv"),
                                   index=False, header=False)
print(f"  保存: predictions_svc_raw.csv  shape={svc_test_pred.shape}")

# ---- MLP 详细分类报告 (validation set)
print("\n" + "=" * 60)
print("MLP 验证集分类报告 (原始数据)")
print("=" * 60)
print(classification_report(y_val, mlp.predict(X_val_tfidf),
                            digits=4, zero_division=0))

print("=" * 60)
print("LinearSVC 验证集分类报告 (原始数据)")
print("=" * 60)
print(classification_report(y_val, svc.predict(X_val_tfidf),
                            digits=4, zero_division=0))

# ---- 两者一致率：测试集上一致预测比例
agree = (mlp_test_pred == svc_test_pred).mean()
print(f"\n测试集上 MLP 与 LinearSVC 预测一致率: {agree:.4f}")
