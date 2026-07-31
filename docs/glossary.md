# 项目术语表

本文件只记录已经完成口头验收的概念。后续学习过程中持续补充。

## 计算环境与 CUDA

### GPU

用于大规模并行计算的硬件设备。本项目使用 NVIDIA GeForce RTX 4090，
具有约 24 GiB 显存。

### NVIDIA Driver

操作系统级组件，负责操作系统、CUDA 应用与 NVIDIA GPU 之间的通信。
`nvidia-smi` 显示的 CUDA 版本表示当前驱动支持的最高 CUDA 版本，不表示
机器安装了相同版本的 CUDA Toolkit。

### CUDA Toolkit

用于开发和编译 CUDA 程序的工具集合，通常包含 `nvcc`、头文件、开发库、
调试工具和性能分析工具。普通的预编译 PyTorch GPU 训练不要求安装完整
CUDA Toolkit。

### nvcc

CUDA Toolkit 中的 CUDA 编译器，主要用于编译 `.cu` 源文件和自定义 CUDA
扩展。`nvcc` 不存在并不代表 PyTorch 无法使用 GPU。

### CUDA Runtime

CUDA 程序运行时使用的库。PyTorch 的 CUDA 构建通常携带与其构建版本匹配
的 CUDA Runtime 和相关数学库，并通过系统中的 NVIDIA Driver 使用 GPU。

### PyTorch CUDA 构建

包含预编译 CUDA 实现的 PyTorch 发行包。普通训练使用这些已经编译好的
实现，不需要在训练时调用 `nvcc`。

### CUDA kernel

提交给 GPU 并由大量 GPU 线程执行的计算函数。启动 kernel 存在固定开销，
因此很小的计算任务不一定比 CPU 更快。

### RAM

系统内存，主要由 CPU 和普通进程使用。默认创建的 CPU Tensor 数据存放在
RAM 中。

### VRAM

GPU 显存。使用 `device="cuda"` 创建或移动到 CUDA 设备的 Tensor 数据存放
在 VRAM 中。

### `nvidia-smi`

用于查看 NVIDIA Driver、GPU、显存占用和进程状态的系统工具。其输出中的
CUDA Version 是驱动兼容能力，不是本机 Toolkit 的安装证明。

### `torch.cuda.is_available()`

用于判断当前 PyTorch 进程是否具备可用的 CUDA 后端并能发现 CUDA 设备。
返回 `False` 时应依次检查硬件、驱动、PyTorch 构建和运行库，而不是直接
安装 `nvcc`。

## 已确认的运行链路

```text
Python 代码
→ PyTorch
→ CUDA Runtime / CUDA 数学库
→ NVIDIA Driver
→ NVIDIA GPU
```
