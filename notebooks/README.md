# Notebook 使用说明

本目录保留已有学习记录和昂贵运行的输出。正式手写 U-Net 训练入口在 Python 包，不要求重新运行 notebook。

| 文件 | 目的 | 什么时候运行 |
|---|---|---|
| `01_understand_camus_dataset.ipynb` | 数据、mask、tensor、卷积、逐步构建 U-Net、单步梯度更新 | 学习概念或检查数据时 |
| `02_train_camus_with_nnunetv2.ipynb` | nnU-Net v2 单折 2D 完整流程与已有输出 | 需要对应步骤时，先检查已有产物 |
| `reference/script_camus_ef.ipynb` | EF / Simpson 双平面法参考代码 | 学习参考，不代表本项目已完成 EF 临床验证 |

- Markdown 的 `Step n` 是语义步骤；`In[n]` 只是执行计数，重启会变化。
- kernel 重启会丢变量，但不会删除 `data/nnunet/` 和 checkpoint。不要因为变量未定义就重做预处理或训练。
- 02 的 Step 10 中手写 U-Net 一列是**原 10 epoch 的历史参考值**，不是自动读取最新续训模型。不要将它当成新 50 轮实验的结果。
- 原 notebook 输出暂不清除，也不在整理过程中批量重跑。
- notebook 含本机路径配置；换电脑后先检查并修改 `PROJECT_ROOT`，不要把历史输出中的绝对路径当作通用安装路径。
- 已保存的超声、标注和预测示例来自CAMUS的匿名研究病例，仅用于非商业研究展示；数据引用及CC BY-NC-SA 4.0等适用条款见[CAMUS_LICENSE.md](../CAMUS_LICENSE.md)。不随仓库分发完整原始数据或模型权重。

下一步：[正式训练操作](../docs/training.md) · [仓库导航](../docs/repository-guide.md) · [实验登记](../docs/experiments.md)
