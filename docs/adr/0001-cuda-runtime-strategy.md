# ADR-0001：使用 PyTorch 自带的 CUDA Runtime

- 状态：已接受
- 日期：2026-07-29

## 背景

开发机器运行 Ubuntu 26.04 LTS，配备 NVIDIA GeForce RTX 4090。NVIDIA
Driver 版本为 595.84，`nvidia-smi` 报告最高支持 CUDA 13.2。系统当前没有
安装 `nvcc`。

项目需要使用 GPU 训练 CAMUS 超声心动图分割模型，但目前不需要编写自定义
CUDA kernel 或从源码编译 PyTorch CUDA 扩展。

## 决策

普通模型开发和训练使用官方预编译的 PyTorch CUDA 构建，以及该构建携带的
CUDA Runtime 和相关数学库。

现阶段不安装系统级 CUDA Toolkit，也不因为缺少 `nvcc` 而安装
`nvidia-cuda-toolkit`。NVIDIA Driver 继续作为系统级共享依赖。

## 原因

- 预编译 PyTorch 已提供普通训练所需的 CUDA 实现。
- 避免引入一套当前不需要的系统级 CUDA 工具链。
- 减少 Toolkit、PyTorch Runtime 和 Driver 版本概念混淆。
- 项目依赖可以在隔离环境中明确记录和复现。

## 影响

- 可以进行普通 PyTorch GPU 训练和推理。
- Python 环境不隔离系统 NVIDIA Driver。
- 如果以后需要编译 `.cu` 文件或自定义 CUDA 扩展，需要重新评估并安装兼容
  的 CUDA Toolkit。
- 仍需在安装 PyTorch 后验证实际 Runtime 版本、GPU 可见性和基本 GPU 运算。
