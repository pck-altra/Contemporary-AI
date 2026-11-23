"""Generate the five-page, two-column experiment report."""

from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
FONT = "/Library/Fonts/Arial Unicode.ttf"
PAGE_W, PAGE_H = 595.28, 841.89
MARGIN, GAP, BOTTOM = 33, 18, 34
WIDTH = (PAGE_W - 2 * MARGIN - GAP) / 2
pdfmetrics.registerFont(TTFont("Chinese", FONT))
STYLES = {
    "body": ParagraphStyle(
        "Body", fontName="Chinese", fontSize=9, leading=13.5, wordWrap="CJK", spaceAfter=5.5
    ),
    "heading": ParagraphStyle(
        "Heading",
        fontName="Chinese",
        fontSize=11.2,
        leading=16.5,
        wordWrap="CJK",
        spaceBefore=2,
        spaceAfter=5.5,
    ),
    "sub": ParagraphStyle(
        "Sub",
        fontName="Chinese",
        fontSize=9.5,
        leading=14,
        wordWrap="CJK",
        spaceBefore=2,
        spaceAfter=4,
    ),
    "small": ParagraphStyle(
        "Small", fontName="Chinese", fontSize=7.4, leading=10.8, wordWrap="CJK", spaceAfter=4.5
    ),
    "cell": ParagraphStyle("Cell", fontName="Chinese", fontSize=7.8, leading=11.3, wordWrap="CJK"),
}


class Column:
    def __init__(self, c, x, top):
        self.c, self.x, self.y = c, x, top

    def paragraph(self, text, style="body"):
        st = STYLES[style]
        self.y -= st.spaceBefore
        para = Paragraph(text, st)
        _, height = para.wrap(WIDTH - 7, 1000)
        self._check(height, text[:35])
        para.drawOn(self.c, self.x, self.y - height)
        self.y -= height + st.spaceAfter

    def _check(self, height, label):
        if self.y - height < BOTTOM:
            raise ValueError(
                f"Page {self.c.getPageNumber()} column {self.x}: overflow at {label}: {self.y-height:.1f}"
            )

    def table(self, rows, widths):
        cells = [[Paragraph(str(v), STYLES["cell"]) for v in row] for row in rows]
        tab = Table(cells, colWidths=[WIDTH * x for x in widths])
        tab.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                    ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        _, height = tab.wrap(WIDTH, 1000)
        self._check(height, "table")
        tab.drawOn(self.c, self.x, self.y - height)
        self.y -= height + 7

    def figure(self, name, caption, maxheight=400):
        image = ImageReader(str(ROOT / "figures/report" / name))
        width, height = image.getSize()
        scale = min(WIDTH / width, maxheight / height)
        width *= scale
        height *= scale
        self._check(height, name)
        self.c.drawImage(
            image,
            self.x + (WIDTH - width) / 2,
            self.y - height,
            width=width,
            height=height,
            mask="auto",
        )
        self.y -= height + 4
        self.paragraph(caption, "small")


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    selected = load("results/selection.json")["selected"]
    metrics = load(f"results/{selected}/test_metrics.json")
    summary = load(f"results/{selected}/summary.json")
    allruns = ["E0_plain", "E1_aug", "E2_residual", "E3_se", "E4_smooth", "E5_mixup", selected]
    summaries = {r: load(f"results/{r}/summary.json") for r in allruns}
    tests = {r: load(f"results/{r}/test_metrics.json") for r in allruns}
    occlusion = load("results/occlusion.json")
    out = ROOT / "实验报告.pdf"
    c = canvas.Canvas(str(out), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("当代人工智能实验三：CIFAR-10 图像分类")
    c.setAuthor("")
    c.setSubject("卷积网络设计、消融实验与误差分析")
    c.setCreator("Python / ReportLab")

    def page(number, first=False):
        c.setFillColor(colors.black)
        c.setFont("Chinese", 7.2)
        c.drawString(MARGIN, PAGE_H - 22, "当代人工智能 · 实验三")
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 22, "CIFAR-10 图像分类")
        c.setStrokeColor(colors.HexColor("#999999"))
        c.setLineWidth(0.35)
        c.line(MARGIN, PAGE_H - 28, PAGE_W - MARGIN, PAGE_H - 28)
        c.setFont("Chinese", 7)
        c.drawCentredString(PAGE_W / 2, 19, f"{number} / 5")
        top = PAGE_H - (160 if first else 44)
        c.setStrokeColor(colors.HexColor("#b5b5b5"))
        c.line(PAGE_W / 2, BOTTOM, PAGE_W / 2, top)
        return Column(c, MARGIN, top), Column(c, MARGIN + WIDTH + GAP, top)

    # Page 1: task, split and architecture.
    left, right = page(1, True)
    c.setFont("Chinese", 16.5)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 54, "图像分类模型的设计与实现")
    c.setFont("Chinese", 9)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 72, "基于紧凑 CNN 的消融实验与泛化分析")
    abstract = Paragraph(
        "摘要：以 CIFAR-10 为对象，在固定训练/验证划分下，依次比较数据增强、残差连接、SE 注意力、标签平滑和 Mixup，并单独检验延长训练预算的效果。最终模型测试准确率为 89.22%，比无增强基线提高 8.40 个百分点。结合训练曲线、混淆矩阵、误分类与遮挡实验，分析性能收益及解释方法的局限。",
        STYLES["body"],
    )
    _, ah = abstract.wrap(PAGE_W - 2 * MARGIN, 1000)
    abstract.drawOn(c, MARGIN, PAGE_H - 85 - ah)
    repository = Paragraph(
        '代码仓库：<link href="https://github.com/pck-altra/Contemporary-AI">https://github.com/pck-altra/Contemporary-AI</link><br/>'
        '分支位置：<link href="https://github.com/pck-altra/Contemporary-AI/tree/homework03">https://github.com/pck-altra/Contemporary-AI/tree/homework03</link>',
        STYLES["small"],
    )
    _, rh = repository.wrap(PAGE_W - 2 * MARGIN, 1000)
    repository.drawOn(c, MARGIN, PAGE_H - 128 - rh)


    left.paragraph("1 数据划分与问题分析", "heading")
    left.paragraph(
        "CIFAR-10 包含 60,000 张 32×32 RGB 图像，分为 airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck 十类[1]。官方训练集共 50,000 张，测试集共 10,000 张。任务是学习图像到类别概率的映射，并以独立测试准确率衡量泛化能力。"
    )
    left.table(
        [
            ["集合", "总量", "每类", "用途"],
            ["训练", "45,000", "4,500", "梯度更新"],
            ["验证", "5,000", "500", "调参与选轮次"],
            ["测试", "10,000", "1,000", "最终评估"],
        ],
        [0.19, 0.22, 0.18, 0.41],
    )
    left.paragraph(
        "划分使用 NumPy Generator(seed=2026)：逐类打乱官方训练索引，每类前 500 张用于验证，其余训练。各组共用同一划分，训练和验证无索引交集；像素 SHA-256 审计也未发现两者之间完全相同的图像。该检查不排除语义近重复。"
    )
    left.paragraph("1.1 预处理与数据特征", "sub")
    left.paragraph(
        "读入前逐文件核验官方 MD5。像素先除以 255，再按训练集统计量标准化：x′=(x−μ)/σ。μ=(0.491514, 0.482315, 0.446814)，σ=(0.247051, 0.243480, 0.261510)。验证和测试不参与统计量估计，也不做随机增强。标准化不保证所有像素都落在 [−1,1]。"
    )
    left.paragraph(
        "各类数量均衡，因此未使用类别权重；测试集每类恰有 1,000 张，总准确率等于十类召回率的平均。小尺寸图像会削弱耳部、车轮、肢体等细节，猫/狗、汽车/卡车可能保留相似的局部纹理与轮廓。需要同时观察每类指标和具体样本，不能仅从总体准确率判断问题是否解决。"
    )
    left.paragraph("1.2 实验思路", "sub")
    left.paragraph(
        "先检验基线能否拟合训练集，再处理泛化差距；以增强后的模型为起点逐项改变结构和损失，避免把增强收益算给网络结构。所有候选均从零训练，验证集用于选择配方与检查点。方案确定后统一测试，不再根据测试排序调参。"
    )

    right.paragraph("2 模型结构与设计依据", "heading")
    right.paragraph(
        "采用约 0.49M 参数的紧凑网络，以较低计算成本完成多组对照。32×32 输入不使用 ImageNet 式大步长 7×7 stem，而由 3×3 卷积逐级提取特征。所有 Conv 均无偏置，配合 BatchNorm；分类头使用全局平均池化，减少大规模全连接层带来的参数。"
    )
    right.table(
        [
            ["模块", "输出 C×H×W", "主要操作"],
            ["Stem", "32×32×32", "3×3 Conv+BN+ReLU"],
            ["Stage 1", "32×16×16", "步长 2 卷积；两层卷积"],
            ["Stage 2", "64×8×8", "步长 2 卷积；两层卷积"],
            ["Stage 3", "128×4×4", "步长 2 卷积；两层卷积"],
            ["分类头", "10", "GAP→Dropout(0.1)→Linear"],
        ],
        [0.20, 0.30, 0.50],
    )
    right.paragraph("2.1 普通网络与残差网络", "sub")
    right.paragraph(
        "每个 stage 先用步长 2 的 Conv-BN-ReLU 得到 x，再经两层 3×3 卷积得到 F(x)。普通 CNN 输出 ReLU(F(x))；残差 CNN 输出 ReLU(x+F(x))[2]。第二层 BN 后先不激活，完成相加后再 ReLU。两者层数、宽度和参数量完全相同（492,010），使对比主要反映 shortcut 在当前深度与预算下的作用，而非容量变化。"
    )
    right.paragraph("2.2 SE 通道注意力", "sub")
    right.paragraph(
        "对残差分支特征进行全局平均池化，得到通道描述 z，再经两层 1×1 卷积、ReLU 和 Sigmoid 生成通道权重 s，输出 s⊙F(x)，最后与 shortcut 相加[3]。压缩比为 8，隐藏维度至少为 4。SE 模型共 497,638 个参数，仅增加 5,628 个（1.14%）。"
    )
    right.paragraph(
        "SE 的动机是根据输入重标定通道，使判别特征获得较高权重。但额外参数不一定带来提升，浅网络中的残差收益也可能很小，因此需用固定预算比较，而不能根据结构复杂度预判优劣。SE 是通道机制，与后文的 Grad-CAM 空间热图并非同一概念。"
    )
    right.paragraph(
        "全局平均池化前保留 128×4×4 特征图，便于解释性分析。其空间分辨率较低，放大热图只能提供粗粒度区域信息，不能当作精确物体分割。"
    )
    c.showPage()

    # Page 2: optimization and controlled comparisons.
    left, right = page(2)
    left.paragraph("3 训练设置与优化策略", "heading")
    left.table(
        [
            ["项目", "设置"],
            ["优化器", "AdamW；lr 峰值 0.003；weight decay 0.0005"],
            ["主实验", "batch=256；最多 24 epoch；seed=2026"],
            ["学习率", "前 2 轮线性预热，随后 cosine 降至 0.00001"],
            ["检查点", "验证准确率严格提升时保存；同分保留较早轮次"],
            ["早停", "连续 8 轮无提升；扩展实验 patience=12"],
            ["环境", "Python 3.12.9；PyTorch 2.5.1；Apple MPS"],
        ],
        [0.23, 0.77],
    )
    left.paragraph("3.1 数据增强与正则化", "sub")
    left.paragraph(
        "增强先反射填充 4 像素，再为每张图独立随机裁剪 32×32，以 0.5 概率水平翻转。这样保留类别含义，同时减少对位置、方向的依赖；未使用易产生不自然姿态的垂直翻转。所有组均使用 BatchNorm、Dropout=0.1 和权重衰减，故 E0 的“无增强”不表示无正则化。"
    )
    left.paragraph("3.2 标签平滑与 Mixup", "sub")
    left.paragraph(
        "标签平滑令 q=(1−ε)onehot(y)+ε/10，ε=0.1，训练损失为 −Σq<sub>k</sub>log p<sub>k</sub>。它限制对单一标签过强的置信度，但也可能产生欠自信，故除准确率外还需检查 NLL 和校准误差。"
    )
    left.paragraph(
        "Mixup[4] 采样 λ~Beta(0.2,0.2)，对 batch 中的图像混合 x̃=λx<sub>i</sub>+(1−λ)x<sub>j</sub>，损失为 λL(p,y<sub>i</sub>)+(1−λ)L(p,y<sub>j</sub>)。E5 在已有 SE 与标签平滑的基础上加入 Mixup，检验其额外作用；并非 Mixup 的独立全因子对照。"
    )
    left.paragraph("3.3 预算与评估约束", "sub")
    left.paragraph(
        "预热控制初期更新幅度，cosine 在后期减小步长。主实验统一 24 轮上限，E0–E5 的早停均未触发；最终权重仍按最佳验证轮次选择。六组结束后，将验证最优配方从零重训 60 轮，作为独立预算实验，不与主消融混为等计算量比较。"
    )
    left.paragraph(
        "训练损失在普通交叉熵、平滑、Mixup 下数值基线不同，不能直接横比。Mixup 训练准确率按两标签加权，增强训练与干净验证的难度也不同。所有组的验证损失统一为无平滑的硬标签交叉熵。"
    )
    left.paragraph(
        "同构组使用相同初始化种子；SE 增加了初始化参数，Mixup 额外消耗排列随机数，所以不能声称各组更新路径完全匹配。加速器内核也可能非确定，固定种子不保证跨设备逐位一致。"
    )

    right.paragraph("4 对比实验与调优结果", "heading")
    right.table(
        [
            ["实验", "相对前组的变化", "验证 / %", "测试 / %"],
            ["E0", "普通 CNN，无增强", "82.40", "80.82"],
            ["E1", "+ crop、flip", "87.66", "87.18"],
            ["E2", "+ 残差连接", "87.82", "87.25"],
            ["E3", "+ SE", "87.70", "87.03"],
            ["E4", "+ 标签平滑", "88.20", "87.34"],
            ["E5", "+ Mixup", "88.06", "87.41"],
            ["E4-60", "E4 配方重训 60 轮", "89.58", "89.22"],
        ],
        [0.16, 0.40, 0.22, 0.22],
    )
    right.paragraph(
        "主消融均训练 24 轮，各耗时 2.06–2.56 分钟；扩展实验训练 60 轮，耗时 6.18 分钟。时间含验证与检查点存储，不含数据读入和统计量估计。",
        "small",
    )
    right.paragraph("4.1 增强收益最明确", "sub")
    right.paragraph(
        "E0→E1 的验证准确率提高 5.26 个百分点，是主消融中最大增益。E0 最后一轮训练准确率达到 100%，验证仅 82.38%，说明容量已经足以拟合训练样本；继续追求训练拟合不如增加合理变化来改善泛化。"
    )
    right.paragraph("4.2 结构改动的增量较小", "sub")
    right.paragraph(
        "E1→E2 验证提高 0.16 个百分点，E2→E3 下降 0.12 个百分点。残差与 SE 在这套浅层骨架和短预算下没有明显收益。SE 参数更多、训练更慢，也说明加入注意力不自动等于性能提升。差异很小，单种子实验不足以说明稳定优劣。"
    )
    right.paragraph("4.3 软标签收益需分指标看", "sub")
    right.paragraph(
        "E3→E4 验证提高 0.50 个百分点；再加入 Mixup 则下降 0.14 个百分点。E5 的训练目标更难，软标签叠加可能在短预算下加重欠拟合，但尚无单独实验确认原因。E5 的测试值略高于 E4，不能据此反向选择配方，因为测试结果是在选择完成后才查看。"
    )
    right.paragraph("4.4 延长训练仍有收益", "sub")
    right.paragraph(
        "E4 按验证结果被选为扩展配方。E4-60 在第 54 轮达到最佳验证 89.58%，较 E4 提高 1.38 个百分点，测试为 89.22%。延长总周期也改变了 cosine 曲线，并非直接从第 24 轮继续；所得收益归于额外训练预算及相应调度。"
    )
    right.paragraph(
        "学习率、weight decay 和批大小统一固定，未做各架构独立最优搜索。这有利于控制比较条件，却不意味着每种方法已经达到最优。调度与早停虽然已实现，但没有无调度消融，不能给出其独立提升幅度。"
    )
    c.showPage()

    # Page 3: learning curves, per-class scores and probability calibration.
    left, right = page(3)
    left.paragraph("5 收敛曲线与泛化差距", "heading")
    left.figure(
        "curves.png",
        "图 1  E4-60 的 loss/accuracy 曲线。虚线标记最佳轮次 54；训练 loss 为平滑交叉熵，验证为硬标签交叉熵。",
        maxheight=290,
    )
    left.paragraph(
        "后期步长减小后，验证准确率逐渐稳定在 89% 以上。第 60 轮训练准确率为 96.32%，验证为 89.54%，说明仍存在泛化差距；使用验证最优权重，而非默认最后一轮。"
    )
    left.paragraph(
        "训练图像经过随机裁剪/翻转，且 Dropout、BatchNorm 处于训练状态；验证采用干净图像和 eval 状态。因此训练和验证曲线的高低关系既受泛化能力影响，也受输入难度与模块状态影响，不能仅据某轮验证高于训练就推断数据泄漏。"
    )
    left.figure(
        "ablation_curves.png",
        "图 2  六组主实验的验证准确率。统一 24 轮预算下，增强组整体优于无增强组，结构与损失差异相对较小。",
        maxheight=160,
    )
    left.paragraph(
        "无增强模型的训练误差迅速接近零，却没有持续改善验证；增强组训练拟合较慢，最终泛化更好。这一对照支持先排查过拟合，再考虑增加结构复杂度的调优顺序。"
    )

    right.paragraph("6 最终性能与概率校准", "heading")
    right.paragraph(
        "最终模型 E4-60 测试正确 8,922/10,000，准确率 89.22%，Macro-F1=0.8921。95% Wilson 区间为 [88.60%, 89.81%]；它仅描述固定模型在同分布测试样本抽样下的不确定性，不包含训练种子与分布变化。"
    )
    right.table(
        [["类别", "准确率/Recall", "Precision", "F1"]]
        + [
            [
                label,
                f'{metrics["recall"][i]*100:.1f}%',
                f'{metrics["precision"][i]*100:.1f}%',
                f'{metrics["f1"][i]:.3f}',
            ]
            for i, label in enumerate(
                [
                    "airplane",
                    "automobile",
                    "bird",
                    "cat",
                    "deer",
                    "dog",
                    "frog",
                    "horse",
                    "ship",
                    "truck",
                ]
            )
        ],
        [0.31, 0.29, 0.23, 0.17],
    )
    right.paragraph(
        "每类均为 1,000 张；“每类准确率”按真实类别计算，即召回率。汽车最好（94.9%），猫最弱（77.3%），相差 17.6 个百分点。数量均衡排除了类别样本数差异这一直接解释。"
    )
    right.figure(
        "calibration.png",
        "图 3  15 个等宽置信度区间的可靠性曲线；越接近对角线，分箱置信度与经验准确率越一致。",
        maxheight=151,
    )
    right.table(
        [["实验", "NLL ↓", "ECE ↓", "Brier ↓"]]
        + [
            [
                label,
                f'{tests[run]["nll"]:.3f}',
                f'{tests[run]["ece15"]:.3f}',
                f'{tests[run]["brier"]:.3f}',
            ]
            for run, label in [
                ("E2_residual", "E2"),
                ("E4_smooth", "E4"),
                ("E5_mixup", "E5"),
                (selected, "E4-60"),
            ]
        ],
        [0.28, 0.24, 0.24, 0.24],
    )
    right.paragraph(
        "平滑与 Mixup 并未同步改善校准：E4、E5 的 ECE 均高于 E2，E5 尤为明显。降低置信度不等于置信度更准确；在当前配方下出现欠自信倾向。E4-60 虽准确率最高，NLL 仍高于 E2，说明准确率与概率质量不能混为一谈。"
    )
    c.showPage()

    # Page 4: confusion structure and observed failures.
    left, right = page(4)
    left.paragraph("7 混淆矩阵与类别难点", "heading")
    left.figure(
        "confusion.png",
        "图 4  E4-60 测试混淆矩阵：行是真实类别，列是预测类别，格内为数量，每行总数 1,000。",
        maxheight=265,
    )
    left.table(
        [
            ["真实 → 预测", "数量", "占该真实类"],
            ["dog → cat", 102, "10.2%"],
            ["cat → dog", 89, "8.9%"],
            ["cat → bird", 37, "3.7%"],
            ["bird → cat", 37, "3.7%"],
            ["cat → frog", 35, "3.5%"],
        ],
        [0.58, 0.18, 0.24],
    )
    left.paragraph(
        "猫和狗的双向混淆共 191 张，占全部 1,078 张错误的 17.7%。猫还会被分成鸟、蛙，说明难点不只是两个类别之间的固定边界，也可能涉及姿态、遮挡和局部纹理。混淆是有向计数，不能假定 A→B 与 B→A 对称。"
    )
    left.paragraph(
        "汽车→卡车为 34 张，船→飞机为 31 张，鸟→飞机为 28 张。车辆类别共享道路、轮廓和车轮，鸟与飞机共享天空及展开的横向结构。这些是可检验的解释，混淆矩阵本身不能证明模型依赖背景捷径，仍需回到图像与扰动分析。"
    )
    left.paragraph("7.1 置信度不能替代正确性", "sub")
    left.paragraph(
        "正确样本的平均最大概率为 0.877，错误样本为 0.659；仍有 110 张错误的概率不低于 0.9，占错误数的 10.2%。因此分析错误时既要检查低置信边界样本，也要检查高置信失败，避免将 softmax 概率直接视为正确性保证。"
    )

    right.paragraph("8 具体误分类样本分析", "heading")
    right.figure(
        "errors.png",
        "图 5  置信度最高的四张错误。保留官方标签；图上 p 为原预测类别概率。",
        maxheight=265,
    )
    right.paragraph(
        "#7852：黑色长腿犬被判为鹿（0.965）。细长四肢、侧身轮廓与草地背景同时出现，使局部视觉线索具有混淆性；不能只因模型高置信就认为官方标签错误。"
    )
    right.paragraph(
        "#5177：鹿的近景被判为猫（0.963）。图像主要保留棕色动物局部纹理，整体体态信息缺失。此类局部视角需要利用更细的头部结构，而 32×32 分辨率限制了可用细节。"
    )
    right.paragraph(
        "#3867：黑白猫被判为狗（0.954）。局部姿态与颜色组合相似。Grad-CAM 的预测类响应偏向身体下部，真实类响应偏向左上，但这种差别只反映目标响应，不能单凭热图确认错误原因。"
    )
    right.paragraph(
        "#8576：展开翅膀的鸟被判为飞机（0.948）。蓝色背景中的横向轮廓与机翼相近，提示轮廓/背景组合可能压过了鸟类细节；后续可通过背景替换或局部遮挡进一步检验。"
    )
    right.paragraph("8.1 中等置信错误的补充观察", "sub")
    right.paragraph(
        "按置信度排序取中位附近错误，#3789 主要呈现红色车身与车轮局部，#2111 以卡车车头正面为主，两者均误判为汽车（约 0.669）。缺失完整车型信息时，模型容易依赖共享部件。#6656 的浅色蛙轮廓模糊，被判为马；#7813 的浅色猫与背景边界不清，被判为狗。可见错误也来自低分辨率、局部视角和主体边界不足。"
    )
    right.paragraph(
        "样本按事先固定规则选取：最高置信错误 4 张、中位附近错误 4 张，并取指定类别的正确样本作对照。少量样本用于解释现象，不代替全测试集统计。"
    )
    c.showPage()

    # Page 5: explanations, occlusion and conclusions.
    left, right = page(5)
    left.paragraph("9 可解释性分析与讨论", "heading")
    left.paragraph(
        "在最终 128×4×4 特征图 A 上，以目标类别 logit z<super>c</super> 求梯度，α<sub>k</sub><super>c</super>=mean<sub>hw</sub>(∂z<super>c</super>/∂A<sub>k</sub>)，再取 ReLU(Σ<sub>k</sub>α<sub>k</sub><super>c</super>A<sub>k</sub>)，得到 Grad-CAM[5]。不用 softmax 概率求梯度，以减少饱和影响。每图独立归一化后上采样到 32×32。"
    )
    left.figure(
        "cam.png",
        "图 6  两张错误与一张正确样本。各列自上而下为原图、预测类别热图、真实类别热图；#9922 为正确识别的狗。",
        maxheight=278,
    )
    left.paragraph(
        "#7852 的预测鹿响应主要在躯干附近；#8576 的飞机响应集中于鸟的中部/翅膀附近，而真实类别响应偏向其他区域。#9922 正确犬样本的响应覆盖面部附近。区域重叠与视觉解释相容，但不能据此确认因果；正确预测也不自动证明热图有效。"
    )
    left.paragraph(
        "4×4 特征图放大后形成平滑色块，难以分辨精确边界。不同图的颜色为各自相对强度，不可直接比较绝对贡献。由于网络采用 GAP+线性头，Grad-CAM 与解析 CAM 应相差正比例常数；实现检查验证了归一化后二者一致。"
    )
    left.paragraph("9.1 调优结论与后续方向", "heading")
    left.paragraph(
        "本实验最明确的改进来自裁剪/翻转，主消融验证提升 5.26 个百分点。残差与 SE 的增量很小；标签平滑略增准确率，却未改善校准；Mixup 在已有软标签基础上未增加验证收益。最终通过延长 E4 训练周期取得 89.22% 测试准确率，比 E0 高 8.40 个百分点。"
    )
    left.paragraph(
        "下一轮应先以至少三个种子复核关键差异，再在固定预算内搜索学习率 {0.001,0.003,0.006} 与权重衰减 {0.0001,0.0005}；针对动物混淆比较适度 CutMix/随机擦除。以上为后续方案，本次未运行。"
    )

    right.paragraph("9.2 遮挡敏感性检验", "sub")
    right.paragraph(
        "从测试集每类按索引取前 10 张，共 100 张，不按置信度或正确性筛选。将高响应、低响应或随机的 25% 像素设为训练通道均值，比较原预测类别的概率下降，遮挡后不切换目标类别。该诊断在方案冻结后进行，不用于调参。"
    )
    right.figure(
        "occlusion.png",
        "图 7  三种遮挡的平均预测概率下降（百分点），样本数均为 100。",
        maxheight=132,
    )
    right.table(
        [
            ["遮挡", "平均下降", "中位数下降"],
            ["高响应区", "52.39", "61.78"],
            ["低响应区", "10.44", "1.89"],
            ["随机区", "60.80", "76.67"],
        ],
        [0.34, 0.33, 0.33],
    )
    right.paragraph(
        "高响应遮挡影响大于低响应区，但不及随机遮挡。高响应减随机的平均差为 −8.41 个百分点，配对 bootstrap 95% 区间 [−17.74,0.76]，仅 37% 样本中高响应遮挡影响更大。因此不能声称 Grad-CAM 优于随机基线。"
    )
    right.paragraph(
        "随机像素更分散，可能更强地破坏局部纹理；高响应区相对集中，遮挡形状构成混杂因素。遮挡还可能产生分布外输入，100 张样本与样本内 bootstrap 也不覆盖训练随机性。后续应比较块状遮挡、填充值和更大样本量。"
    )
    right.paragraph("9.3 实验局限", "sub")
    right.paragraph(
        "每种配方只有一个种子，不能将小幅差异当作稳定或显著收益。顺序加入组件只估计当前上下文中的增量作用，不是完整全因子实验；统一超参数也未必适合每种方法。测试未用于阈值拟合或概率温度校准，后续应单独保留校准数据。"
    )
    right.paragraph("参考文献", "heading")
    for text in [
        "[1] Krizhevsky A. Learning Multiple Layers of Features from Tiny Images. 2009. CIFAR-10: www.cs.toronto.edu/~kriz/cifar.html.",
        "[2] He K, et al. Deep Residual Learning for Image Recognition. CVPR, 2016. arXiv:1512.03385.",
        "[3] Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks. CVPR, 2018. arXiv:1709.01507.",
        "[4] Zhang H, et al. mixup: Beyond Empirical Risk Minimization. ICLR, 2018. arXiv:1710.09412.",
        "[5] Selvaraju R R, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV, 2017. arXiv:1610.02391.",
    ]:
        right.paragraph(text, "small")
    c.showPage()
    c.save()
    print(out)


if __name__ == "__main__":
    main()
