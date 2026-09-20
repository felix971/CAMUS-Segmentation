# 手写 U-Net：训练、恢复与结果查看

## 先理解这次实验

我们不是重新初始化网络，而是把已完成的第 10 轮模型及 AdamW 状态加载回来，从第 11 轮继续，到**累计第 50 轮**停止。

```text
冻结旧的第10轮 checkpoint
          ↓ 恢复 model + optimizer
第11轮 → 验证 → 记录CSV/保存best与last
          ↓
        ……
          ↓
第50轮 → 固定验证最佳模型 → 分析曲线
```

旧 checkpoint 没有 RNG 状态，因此无法恢复当时下一轮本来会采用的数据打乱顺序。续训使用并记录 seed 42；新 checkpoint 保存 RNG，方便后续恢复。不要称它是从 epoch 1 固定 seed 的完整重现实验。

## 1. 必须保留的旧成果

- 原路径：`outputs/checkpoints/best_model.pt`。
- 冻结副本：`outputs/baselines/unet_10ep/best.pt`。
- 旧 notebook、数据和 `data/nnunet/` 不搬动、不重跑。
- 新 run 目录不能已存在；误用同名目录应报错，而非悄悄覆盖。

下面所有命令都在项目根目录执行。**先检查 run 是否已经在运行，不要重复启动正式命令。**

## 2. 短程检查（不作为实验结果）

先执行自动测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

用独立目录，对真实 checkpoint 做一个训练 batch 和一个验证 batch 的管线检查：

```bash
.venv/bin/python -u -m camus_segmentation.train \
  --resume outputs/baselines/unet_10ep/best.pt \
  --epochs 11 --seed 42 --device cuda \
  --run-dir outputs/runs/smoke_resume10 \
  --smoke-test
```

这不是完成一整轮正式训练。检查原权重未改、AdamW step 延续、文件能重新加载。`smoke_test` 产物不能作为正式续训起点；正式训练仍从冻结的第 10 轮开始。

## 3. 正式续训到总共 50 轮

```bash
.venv/bin/python -u -m camus_segmentation.train \
  --resume outputs/baselines/unet_10ep/best.pt \
  --epochs 50 --seed 42 --device cuda \
  --run-dir outputs/runs/unet_resume10_to50_seed42
```

`--epochs 50` 是目标总轮数，不是“再加 50”。模型、batch、输入尺寸、loss、AdamW 超参数保持 checkpoint 中的配置；不要同时改增强、scheduler 或 AMP。续训参数与源配置冲突时应拒绝。

训练仅使用 train/validation；不会每轮评估 test。正式运行会写 `train.log`，无需依赖 notebook 的输出保存。

## 4. 一个 run 中的文件

| 文件 | 含义 |
|---|---|
| `config.json` | 本段参数、来源 checkpoint、代码/环境/划分标识等 |
| `baseline.pt` | 冻结的原基线副本，不被本段训练替换 |
| `baseline.json` | 来源基线指标；旧 checkpoint 只提供一个保存时点，不能代替完整历史曲线 |
| `best.pt` | 以验证平均前景 Dice 选择；如果后面都没更好，就保持已有最佳模型 |
| `last.pt` | 最近完成的训练状态，包含优化器和 RNG；用于恢复 |
| `history.csv` | 真实记录的本段各轮 loss、各类 Dice、步数、学习率、耗时 |
| `train.log` | 人可读的运行日志 |
| `summary.json` | 正常完成后的汇总；不能只看它的目标 epoch 判断已完成 |
| `curves.png` | 运行绘图命令后生成的曲线 |

`history.csv` 不补造 epoch 1–10 的缺失训练曲线。第 10 轮已知的验证指标作为独立基线点绘制。

## 5. 看曲线，不反复用 test 调参

至少完成一个完整 epoch 后，或整个训练结束后：

```bash
.venv/bin/python -m camus_segmentation.plot_history \
  --run-dir outputs/runs/unet_resume10_to50_seed42
```

已有 `curves.png` 时，需要显式加 `--overwrite` 更新图片；此选项只用于绘图，不用于覆盖训练目录。

- 训练 loss 降、验证 Dice 升：仍可能受益于增加预算。
- 训练 loss 降、验证 Dice 持续跌：关注过拟合，使用验证最佳 checkpoint。
- 都进入平台期：再单独研究学习率或优化问题，不自动堆轮数。
- 单次运行的提升不等于多 seed 稳定收益；跨网格与 nnU-Net 的比较仍然是 pipeline 比较。

## 6. 如果中断，怎么恢复

1. 确认原训练进程已经退出。
2. 检查源 run 的 `last.pt` 对应最近**完成**的 epoch，而不是只看日志中正在执行哪轮。
3. 保留源 run 不动，以一个**新目录**恢复同一个总预算：

```bash
.venv/bin/python -u -m camus_segmentation.train \
  --resume outputs/runs/unet_resume10_to50_seed42/last.pt \
  --epochs 50 --device cuda \
  --run-dir outputs/runs/unet_resume10_to50_part2
```

完整 checkpoint 会恢复其随机状态，不能靠改 `--seed` 把恢复实验伪装成独立新 seed。不要孤立移动 `last.pt` 而丢掉同目录的历史最佳模型和来源文件。新目录记录新的实验段；查看整条曲线时要追溯父 run，不把缺失段插值成真实结果。

中断发生在一轮中途时，那轮尚未完成的更新不属于持久化的 `last.pt`；恢复会从最后完整轮次继续。已经完成 50 轮时不再执行同目标续训。

## 7. 显式评估指定模型

先查看验证集（不是 test）：

```bash
.venv/bin/python -m camus_segmentation.evaluate_test \
  --checkpoint outputs/runs/unet_resume10_to50_seed42/best.pt \
  --split validation \
  --output-json outputs/runs/unet_resume10_to50_seed42/validation.json
```

模型与比较方案固定后，才手动执行最终测试：

```bash
.venv/bin/python -m camus_segmentation.evaluate_test \
  --checkpoint outputs/runs/unet_resume10_to50_seed42/best.pt \
  --split test \
  --output-json outputs/runs/unet_resume10_to50_seed42/test.json
```

两者数值不能混称。评估沿用 checkpoint 输入尺寸与原 resized-grid 协议；它不会自动变成原始物理空间评估。JSON 与图片不要覆盖来源 checkpoint；已有输出使用新名字。

指定验证图可视化：

```bash
.venv/bin/python -m camus_segmentation.visualize_prediction \
  --checkpoint outputs/runs/unet_resume10_to50_seed42/best.pt \
  --sample-index 0 \
  --output outputs/runs/unet_resume10_to50_seed42/validation_example.png
```

不传 checkpoint 的旧推理命令仍指向历史 10 轮模型，不会猜测哪个 run 是最新。README 的 curated 图片不会被普通可视化命令自动替换。

## 8. nnU-Net 不在这次续训范围

已有 nnU-Net 的配置、预处理、权重与预测继续位于 `data/nnunet/`。参见 [notebook 导航](../notebooks/README.md)。本次不重训 nnU-Net、不清空其输出，也不把手写续训结果覆盖到旧比较表。
