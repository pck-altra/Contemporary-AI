# Contemporary-AI

> 当代人工智能课程作业仓库

## 分支说明

- **`main`**：占位分支，仅保留此 README
- **`homework01`**：[实验一 · 新闻 10 分类](./tree/homework01) —— 5 种分类器对比与融合
  - 朴素贝叶斯 / 线性 SVM / 逻辑回归 / MLP / BERT
  - 投票融合：BERT + MLP + LinearSVC 硬投票
- **`homework02`**：[实验二 · 搜索算法](./tree/homework02) —— A* / Minimax / Alpha-Beta
  - 题一迷宫寻路 A*（7 步最优路径）/ 题二加权地形（代价 7）
  - 题三博弈树（Minimax ≡ Alpha-Beta = 5）/ 题四宝藏分配（H=2 T2/-1，H=4 T3/3）
  - `bash run_all.sh` 一键复现：20 个 pytest → 实验 CSV → 报告图表

- **`homework03`**：[实验三 · CIFAR-10 图像分类](https://github.com/pck-altra/Contemporary-AI/tree/homework03) —— CNN 设计、消融实验与误差分析
  - 普通 CNN / 残差 CNN / SE 通道注意力，比较数据增强、标签平滑与 Mixup
  - 六组 24 轮消融实验与一组 60 轮训练；最终测试准确率 **89.22%**，Macro-F1 **0.8921**
  - 训练曲线、每类指标、混淆矩阵、误分类分析、Grad-CAM 与遮挡敏感性检验
  - [五页实验报告](https://github.com/pck-altra/Contemporary-AI/blob/homework03/%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.pdf)，代码、训练权重与结果见该分支

## 课程信息

- 课程：当代人工智能
- 学期：2026 秋
- 作业提交人：[潘淳恺](https://github.com/pck-altra)

更多内容见各作业分支。
