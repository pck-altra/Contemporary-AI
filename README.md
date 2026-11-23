# CIFAR-10 图像分类实验

仓库：[pck-altra/Contemporary-AI](https://github.com/pck-altra/Contemporary-AI) · 分支：[homework03](https://github.com/pck-altra/Contemporary-AI/tree/homework03)

## 环境

Python 3.10–3.12，PyTorch 2.5.1。训练设备可选 CUDA、Apple MPS 或 CPU。

```bash
python -m pip install -r requirements.txt
```

## 数据与训练

```bash
python src/download_data.py
python -m unittest discover -s tests -v
python src/train.py --epochs 24 --batch-size 256 --seed 2026 --device auto
python src/train.py --runs E4_smooth --epochs 60 --batch-size 256 --seed 2026 --suffix _long60 --patience 12 --device auto
```

数据使用 CIFAR-10 官方 Python 版本，脚本核验 MD5。每类固定划分 4,500 张训练、500 张验证，官方测试集保持不变。标准化统计量仅由训练集估计。原始数据不随代码保存，可用下载脚本获取。

| 实验 | 结构 | 增强 | SE | 平滑 | Mixup |
|---|---|---|---|---|---|
| E0 | 普通 CNN | 无 | 无 | 0 | 0 |
| E1 | 普通 CNN | crop/flip | 无 | 0 | 0 |
| E2 | 残差 CNN | crop/flip | 无 | 0 | 0 |
| E3 | 残差 CNN | crop/flip | 有 | 0 | 0 |
| E4 | 残差 CNN | crop/flip | 有 | 0.1 | 0 |
| E5 | 残差 CNN | crop/flip | 有 | 0.1 | 0.2 |

E0–E5 各训练 24 轮；E4-60 使用 E4 配方从零训练 60 轮。最终检查点为第 54 轮，验证准确率 89.58%，测试准确率 89.22%。已完成的实验默认跳过；`--force` 可覆盖重训，`--runs` 可指定单组。单组中断后从头重训。

## 评估与作图

```bash
python src/audit_data.py
python src/evaluate.py --device auto
python src/occlusion.py --device auto
python src/make_figures.py
python src/report_figures.py
python src/verify_results.py
python src/build_report.py
```

模型按验证准确率选择，评估前固定候选与权重哈希。`selection.json` 已存在且候选变化时，评估脚本拒绝覆盖；新一轮独立实验应使用新的结果目录，避免与已有测试记录混合。固定种子仍可能因加速器内核产生小幅差异。

报告生成需要中文 TTF 字体；`src/build_report.py` 的 `FONT` 默认为 macOS 的 `/Library/Fonts/Arial Unicode.ttf`。

## 文件结构

- `实验报告.pdf`：五页双栏报告。
- `src/`：模型、训练、评估与分析代码。
- `tests/`：数据划分、模型梯度、指标及 Grad-CAM 检查。
- `checkpoints/`：七组最佳验证检查点。
- `results/`：逐轮日志、测试概率、指标、划分索引与遮挡结果。
- `figures/`：曲线、混淆矩阵、误分类和热图；`report/` 为双栏图。

## 图像预测

```bash
python src/predict.py path/to/image.png --device auto
```

输入转为 RGB 并缩放至 32×32，输出十类中的前三项。模型不处理十类以外的开放集识别。
