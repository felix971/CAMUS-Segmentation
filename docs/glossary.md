# Project Glossary

This glossary records only concepts whose understanding has been confirmed through oral review. It will be expanded as learning continues.

## Computing Environment and CUDA

### GPU

A hardware device for massively parallel computation. This project uses an NVIDIA GeForce RTX 4090
with approximately 24 GiB of VRAM.

### NVIDIA Driver

An operating-system-level component that enables communication between the operating system, CUDA applications, and the NVIDIA GPU.
The CUDA version displayed by `nvidia-smi` is the highest CUDA version supported by the current driver;
it does not mean that the same version of the CUDA Toolkit is installed on the machine.

### CUDA Toolkit

A collection of tools for developing and compiling CUDA programs, typically including `nvcc`, headers, development libraries,
debuggers, and profiling tools. Standard GPU training with a precompiled PyTorch build does not require
the full CUDA Toolkit to be installed.

### nvcc

The CUDA compiler in the CUDA Toolkit, primarily used to compile `.cu` source files and custom CUDA
extensions. The absence of `nvcc` does not mean that PyTorch cannot use the GPU.

### CUDA Runtime

Libraries used by CUDA programs at runtime. CUDA builds of PyTorch typically bundle a matching
CUDA Runtime and related mathematical libraries, and access the GPU through the system's NVIDIA Driver.

### PyTorch CUDA Build

A PyTorch distribution containing precompiled CUDA implementations. Standard training uses these
precompiled implementations and does not need to invoke `nvcc` during training.

### CUDA kernel

A computational function submitted to the GPU and executed by many GPU threads. Launching a kernel has a fixed overhead,
so very small computational tasks are not necessarily faster on a GPU than on a CPU.

### RAM

System memory, primarily used by the CPU and ordinary processes. Data for CPU tensors created by default is stored
in RAM.

### VRAM

GPU memory. Data for tensors created with `device="cuda"` or moved to a CUDA device is stored
in VRAM.

### `nvidia-smi`

A system utility for inspecting the NVIDIA Driver, GPU, VRAM usage, and process status. The
CUDA Version in its output indicates driver compatibility, not proof of a locally installed Toolkit.

### `torch.cuda.is_available()`

Checks whether the current PyTorch process has a usable CUDA backend and can detect a CUDA device.
If it returns `False`, check the hardware, driver, PyTorch build, and runtime libraries in that order,
rather than immediately installing `nvcc`.

## Confirmed Execution Stack

```text
Python code
→ PyTorch
→ CUDA Runtime / CUDA mathematical libraries
→ NVIDIA Driver
→ NVIDIA GPU
```
