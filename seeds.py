"""
seeds.py
========
集中管理项目中的所有随机种子与公共配置，确保所有实验**完全可复现**。

用法
----
在任意脚本顶部加入：

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from seeds import SEED, DATA_DIR, PRED_DIR, DOCS_DIR, FIG_DIR, \
                      set_global_seeds, get_nb_batch_seed, TFIDF_PARAMS

    set_global_seeds()                       # 设置全局种子
    train_test_split(..., random_state=SEED) # 使用统一种子
    SGDClassifier(..., random_state=SEED)    # 分类器种子

不要在脚本里硬编码 random_state=42，统一从本文件导入。
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# ============================================================
# 1. 路径常量 —— 所有脚本统一从这里取
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"           # 训练/测试 CSV
PRED_DIR     = PROJECT_ROOT / "predictions"    # 输出预测
DOCS_DIR     = PROJECT_ROOT / "docs"           # 文档/日志
FIG_DIR      = DOCS_DIR / "figures"            # 图表

# 兼容旧脚本：把它们也导出为字符串（部分 sklearn/pathlib API 不接受 Path）
DATA_DIR_STR  = str(DATA_DIR)
PRED_DIR_STR  = str(PRED_DIR)
DOCS_DIR_STR  = str(DOCS_DIR)
FIG_DIR_STR   = str(FIG_DIR)

# 确保目录存在（首次运行时自动创建）
for _p in (DATA_DIR, PRED_DIR, DOCS_DIR, FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. 种子常量 —— 整个项目唯一来源
# ============================================================
SEED              = 42       # 数据划分 + 分类器初始化的统一种子
NB_BATCH_BASE     = 0        # NB partial_fit 初始 batch 的种子
NB_BATCH_EPOCH_SEED = lambda epoch: epoch   # 每个 epoch 用自身序号作种子（可复现）

# ============================================================
# 3. 数据划分 & TF-IDF 公共参数
# ============================================================
TEST_SIZE         = 0.2      # 验证集比例（80/20）
STRATIFY          = True     # 分层采样保持类别比例

TFIDF_PARAMS = dict(
    sublinear_tf   = True,
    min_df         = 5,
    max_df         = 0.95,
    ngram_range    = (1, 2),
    max_features   = 50000,
    strip_accents  = "unicode",
    lowercase      = True,
    token_pattern  = r"(?u)\b[A-Za-z][A-Za-z]+\b",
)

# ============================================================
# 4. 工具函数
# ============================================================
def set_global_seeds(seed: int = SEED) -> None:
    """
    给 Python / NumPy / PyTorch（如果装了）统一设置种子。
    在脚本最开头调用一次即可。
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass


def get_nb_batch_seed(epoch: int) -> int:
    """
    MultinomialNB 的 partial_fit 在每个 epoch 用不同种子打乱数据，
    为了保证可复现，epoch k 用 seed=k。
    """
    return NB_BATCH_EPOCH_SEED(epoch)


def split_data(X, y, test_size: float = TEST_SIZE, seed: int = SEED, stratify=None):
    """
    统一的数据划分入口，避免每个脚本重复写 train_test_split(...) 参数。
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X, y,
        test_size  = test_size,
        stratify   = stratify,
        random_state = seed,
    )


def predict_path(name: str) -> str:
    """生成一个预测文件路径，例如 predict_path('nb_baseline') → .../predictions/predictions_nb_baseline.csv"""
    return str(PRED_DIR / f"predictions_{name}.csv")


def figure_path(name: str) -> str:
    """生成一个图表路径，例如 figure_path('compare_curve') → .../docs/figures/compare_curve.png"""
    return str(FIG_DIR / f"{name}.png")


# ============================================================
# 5. 启动时打印（方便确认）
# ============================================================
if __name__ == "__main__":
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}")
    print(f"PRED_DIR     : {PRED_DIR}")
    print(f"DOCS_DIR     : {DOCS_DIR}")
    print(f"FIG_DIR      : {FIG_DIR}")
    print(f"SEED         : {SEED}")
    print(f"TEST_SIZE    : {TEST_SIZE}")
    print(f"TFIDF max_features : {TFIDF_PARAMS['max_features']}")
    set_global_seeds()
    print("\n[OK] 全局种子已设置完成。")
