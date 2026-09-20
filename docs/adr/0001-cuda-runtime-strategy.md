# ADR-0001: Use the CUDA Runtime Bundled with PyTorch

- Status: Accepted
- Date: 2026-07-29

## Context

The development machine runs Ubuntu 26.04 LTS and has an NVIDIA GeForce RTX 4090. The NVIDIA
Driver version is 595.84, and `nvidia-smi` reports support for up to CUDA 13.2. At the time of this decision,
`nvcc` is not installed on the system.

The project requires GPU training for CAMUS echocardiography segmentation models, but does not currently require custom
CUDA kernels or compilation of PyTorch CUDA extensions from source.

## Decision

Use an official precompiled PyTorch CUDA build for standard model development and training, together with
its bundled CUDA Runtime and related mathematical libraries.

Do not install a system-wide CUDA Toolkit at this stage, or install
`nvidia-cuda-toolkit` merely because `nvcc` is absent. The NVIDIA Driver remains a shared system-level dependency.

## Rationale

- Precompiled PyTorch already provides the CUDA implementations needed for standard training.
- This avoids introducing a system-level CUDA toolchain that is not currently needed.
- It reduces confusion between Toolkit, PyTorch Runtime, and Driver versions.
- Project dependencies can be explicitly recorded and reproduced in an isolated environment.

## Consequences

- Standard PyTorch GPU training and inference are supported.
- The Python environment does not isolate the system NVIDIA Driver.
- If compiling `.cu` files or custom CUDA extensions becomes necessary, revisit this decision and install
  a compatible CUDA Toolkit.
- After installing PyTorch, the actual Runtime version, GPU visibility, and basic GPU operations must still be verified.
