# 仓库导航：学习、正式实验与历史结果分开

## 从哪里开始

- **了解项目和已核验结果**：根目录 [README](../README.md)。
- **运行手写 U-Net 实验**：[训练与续训操作](training.md)，正式入口 `python -m camus_segmentation.train`。
- **学习数据与网络**：[01 notebook](../notebooks/01_understand_camus_dataset.ipynb)。它包含逐步构建网络和单步更新的教学代码，不是正式 50 轮训练入口。
- **查看已完成的 nnU-Net 工作流**：[02 notebook](../notebooks/02_train_camus_with_nnunetv2.ipynb)。已有预处理、训练、预测时，不要为了查看后面的结果从头重新运行。
- **核对历史基线**：[实验登记表](experiments.md)。旧的 10 轮结果和新的续训结果必须分别标记。
- **验证代码**：` .venv/bin/python -m unittest discover -s tests -v `。

## 目录职责

| 目录 | 放什么 | 不放什么 |
|---|---|---|
| `camus_segmentation/` | 可重复调用的数据、模型、训练、评估、绘图代码 | notebook 草稿、训练日志 |
| `tests/` | CPU 小样本自动测试；测试数据为明确的合成 fixture | 正式医学研究结果 |
| `notebooks/` | 教学探索与 nnU-Net 的已有运行记录 | 唯一的正式手写训练入口 |
| `notebooks/reference/` | 参考 notebook，例如 Simpson 法 EF 示例 | 本项目已经验证的临床结论 |
| `examples/` | 独立的小型概念示例 | 自动测试或正式实验 |
| `data/raw/camus/` | 受条款约束的本地原始数据 | Git 跟踪文件 |
| `data/splits/` | 版本化的患者划分 | 随训练临时改变的划分 |
| `data/nnunet/` | 已有 nnU-Net 数据链接、预处理、权重、预测 | 本次手写续训结果 |
| `outputs/baselines/` | 冻结的历史基线副本及来源说明 | 被新训练覆盖的 checkpoint |
| `outputs/runs/<run-name>/` | 一个实验段的配置、日志、CSV、权重、曲线 | 多次无标识运行混写 |
| `outputs/checkpoints/` | 为旧 notebook 保留的历史路径 | 新实验默认写入位置 |
| `docs/` | 方法说明、操作指南、实验登记 | 未标注来源的数字 |
| `docs/assets/` | 经选择用于 README 的图 | 每次试验自动覆盖的图片 |

## 为什么没有把整个目录搬一遍

已有 notebook 和 nnU-Net 工作区依赖相对路径、符号链接和持久产物。为了“整齐”搬动 `data/`、重命名包或清空 notebook 输出，可能破坏已完成的训练。

本次整理保留这些路径和原始 notebook 的全部单元格、代码与输出，通过清晰的入口、导航与独立 run 目录区分职责。根目录原 `test.py` 实际只是 NumPy shape 示例，移到 `examples/numpy_shapes.py`，不把它误算为测试套件。

## Git 与大文件

`.gitignore` 已排除 `.venv/`、原始数据、nnU-Net 工作区与 `outputs/`。不要为了提交结果使用 `git add -f` 上传这些目录。可以提交经过筛选的指标 JSON/CSV 和图片，但先核对来源、CAMUS 条款及是否包含病例数据。

整理前已有未提交的 README 和 nnU-Net notebook。它们属于既有工作，不能用 `git reset --hard`、清空输出或重建 notebook 来“清理”。此轮操作不自动 commit 或 push。
