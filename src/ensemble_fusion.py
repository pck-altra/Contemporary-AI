"""
ensemble_fusion.py
==================
多模型预测融合 —— 基于已经保存的 predictions_*.csv 做后处理投票融合。

输入：
  predictions/
    ├── predictions_bert_raw.csv
    ├── predictions_svc_raw.csv
    ├── predictions_mlp_raw.csv
    └── predictions_nb_baseline.csv

输出：
  predictions/
    ├── predictions_final_bert_only.csv       (BERT 单独，作为对照)
    ├── predictions_fusion_bert_mlp_svc.csv   (BERT+MLP+SVC 硬投票，推荐)
    └── predictions_fusion_weighted.csv       (BERT×2 + SVC×1 + MLP×1 加权)

用法：
  cd data-Project1
  python src/ensemble_fusion.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# 让脚本无论从哪个目录运行都能找到 seeds.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeds import (
    PRED_DIR, set_global_seeds, SEED,
    predict_path, figure_path,
)

set_global_seeds(SEED)


# ============================================================
# 加载预测
# ============================================================
def load_predictions():
    """读入所有候选模型的测试集预测，返回 dict[name -> np.ndarray]"""
    sources = {
        "BERT": "predictions_bert_raw.csv",
        "MLP":  "predictions_mlp_raw.csv",
        "SVC":  "predictions_svc_raw.csv",
        "NB":   "predictions_nb_baseline.csv",
    }
    preds = {}
    for name, fname in sources.items():
        path = PRED_DIR / fname
        if not path.exists():
            print(f"[warn] 缺少 {path}，跳过 {name}")
            continue
        arr = pd.read_csv(path, header=None).iloc[:, 0].values
        preds[name] = arr
        print(f"  载入 {name:>4s}: {len(arr)} 条, 类别分布 {dict(Counter(arr))}")
    if not preds:
        raise FileNotFoundError("predictions/ 下没有任何预测文件，先跑其他脚本。")
    return preds


# ============================================================
# 融合函数
# ============================================================
def majority_vote(*predictions) -> np.ndarray:
    """硬投票：取众数；并列时取第一个模型的值"""
    arr = np.array(predictions)
    out = np.zeros(arr.shape[1], dtype=int)
    for i in range(arr.shape[1]):
        out[i] = Counter(arr[:, i]).most_common(1)[0][0]
    return out


def weighted_vote(predictions, weights) -> np.ndarray:
    """加权投票：按权重累加各模型的票"""
    arr = np.array(predictions)
    n_classes = int(arr.max()) + 1
    out = np.zeros(arr.shape[1], dtype=int)
    for i in range(arr.shape[1]):
        scores = np.zeros(n_classes)
        for j, w in enumerate(weights):
            scores[arr[j, i]] += w
        out[i] = int(scores.argmax())
    return out


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("多模型预测融合（Ensemble Fusion）")
    print("=" * 70)

    preds = load_predictions()
    n = len(next(iter(preds.values())))
    print(f"\n样本数: {n}")

    # ---------- 一、两两一致率矩阵 ----------
    names = list(preds.keys())
    print("\n[一] 两两一致率（Pairwise Agreement）")
    print(f"{'':>8}" + "".join([f"{n:>10}" for n in names]))
    for n1 in names:
        row = f"{n1:>8}"
        for n2 in names:
            row += f"{'  --    ':>10}" if n1 == n2 \
                   else f"{(preds[n1] == preds[n2]).mean():>10.4f}"
        print(row)

    # ---------- 二、几个候选融合方案 ----------
    schemes = {}

    # 1. BERT 单独
    if "BERT" in preds:
        schemes["final_bert_only"] = ("BERT 单独", preds["BERT"])

    # 2. BERT+MLP+SVC 硬投票（推荐）
    if all(k in preds for k in ("BERT", "MLP", "SVC")):
        schemes["fusion_bert_mlp_svc"] = (
            "BERT+MLP+SVC 硬投票",
            majority_vote(preds["BERT"], preds["MLP"], preds["SVC"]),
        )

    # 3. 全 4 模型硬投票
    if len(names) == 4:
        schemes["fusion_all4"] = (
            "全 4 模型硬投票",
            majority_vote(*[preds[k] for k in names]),
        )

    # 4. 加权 BERT×2 + SVC×1 + MLP×1
    if all(k in preds for k in ("BERT", "SVC", "MLP")):
        schemes["fusion_weighted"] = (
            "BERT×2 + SVC×1 + MLP×1 加权",
            weighted_vote([preds["BERT"], preds["SVC"], preds["MLP"]], [2.0, 1.0, 1.0]),
        )

    # ---------- 三、保存结果 ----------
    print("\n[二] 融合方案输出")
    ref = preds.get("BERT")
    for key, (label, fused) in schemes.items():
        out_path = PRED_DIR / f"predictions_{key}.csv"
        pd.DataFrame(fused).to_csv(out_path, index=False, header=False)
        diff_bert = (fused != ref).sum() if ref is not None else None
        msg = f"  {label:<35s} → {out_path.name}"
        if diff_bert is not None:
            msg += f"  (与 BERT 差异 {diff_bert:>4d} 条 / {len(fused)})"
        print(msg)

    # ---------- 四、方案差异矩阵 ----------
    print("\n[三] 方案差异矩阵（行 vs 列 的不一致条数）")
    keys = list(schemes.keys())
    print(f"{'':35s}" + "".join([f"{k[:8]:>10s}" for k in keys]))
    for k1 in keys:
        row = f"{schemes[k1][0]:35s}"
        for k2 in keys:
            row += f"{(schemes[k1][1] != schemes[k2][1]).sum():>10d}"
        print(row)

    print("\n[完成] 所有融合预测已写入 predictions/")


if __name__ == "__main__":
    main()
