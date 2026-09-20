"""Small file/RNG utilities for reproducible training runs."""

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager, redirect_stdout
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextmanager
def log_stdout(path):
    with path.open("x") as stream, redirect_stdout(Tee(sys.stdout, stream)):
        try:
            yield
        except BaseException:
            traceback.print_exc(file=sys.stdout)
            raise


def runtime_provenance(project_root, device):
    def git(*args):
        result = subprocess.run(["git", "-C", str(project_root), *args],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    try:
        git_info = {"head": git("rev-parse", "HEAD"),
                    "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        git_info = {"head": None, "dirty": None}
    return {
        "git": git_info,
        "source_hashes": {str(path.relative_to(project_root)): sha256(path)
                          for path in sorted((project_root / "camus_segmentation").glob("*.py"))},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("torch", "numpy", "nibabel")},
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "torch_threads": torch.get_num_threads(),
            "variables": {key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "PYTHONHASHSEED")},
        },
    }


def capture_rng(generators):
    numpy_state = np.random.get_state(legacy=True)
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "loaders": {key: generator.get_state() for key, generator in generators.items()},
    }


def restore_rng(state, generators):
    random.setstate(state["python"])
    name, keys, position, cached, value = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, cached, value))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])
    for key, generator in generators.items():
        generator.set_state(state["loaders"][key])


def atomic_save(value, path):
    """Publish only a fully serialized and flushed checkpoint on this filesystem."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_copy(source, path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
