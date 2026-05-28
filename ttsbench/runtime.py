"""Inspect the local runtime: platform, CPU, memory, acceleration backends.

Backend detection is import-guarded so the base install (no torch/mlx/etc.) still
reports honestly: only backends that are actually importable and usable now are
listed. On Apple Silicon, CUDA is never available; GPU means MPS/MLX/Core ML.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field

import psutil


@dataclass
class RuntimeInfo:
    platform: str
    machine: str
    processor: str
    python_version: str
    cpu_count_logical: int | None
    cpu_count_physical: int | None
    total_memory_gb: float
    apple_silicon: bool
    available_backends: list[str] = field(default_factory=list)


def _detect_backends() -> list[str]:
    backends = ["cpu"]

    try:
        import torch

        if torch.backends.mps.is_available():
            backends.append("mps")
        if torch.cuda.is_available():
            backends.append("cuda")
    except Exception:
        pass

    for module, name in (("mlx", "mlx"), ("coremltools", "coreml"), ("onnxruntime", "onnx")):
        try:
            __import__(module)
        except Exception:
            continue
        backends.append(name)

    return backends


def detect_runtime() -> RuntimeInfo:
    machine = platform.machine()
    total_bytes = psutil.virtual_memory().total
    return RuntimeInfo(
        platform=platform.platform(),
        machine=machine,
        processor=platform.processor() or "unknown",
        python_version=sys.version.split()[0],
        cpu_count_logical=psutil.cpu_count(logical=True),
        cpu_count_physical=psutil.cpu_count(logical=False),
        total_memory_gb=round(total_bytes / (1024**3), 2),
        apple_silicon=platform.system() == "Darwin" and machine == "arm64",
        available_backends=_detect_backends(),
    )


def format_runtime(info: RuntimeInfo) -> str:
    lines = [
        "TTSBench runtime",
        f"  Platform:          {info.platform}",
        f"  Machine:           {info.machine}",
        f"  Processor:         {info.processor}",
        f"  Python:            {info.python_version}",
        f"  CPU (logical):     {info.cpu_count_logical}",
        f"  CPU (physical):    {info.cpu_count_physical}",
        f"  Memory:            {info.total_memory_gb} GB",
        f"  Apple Silicon:     {info.apple_silicon}",
        f"  Backends available: {', '.join(info.available_backends)}",
    ]
    return "\n".join(lines)
