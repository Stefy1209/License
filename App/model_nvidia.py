"""
model_nvidia.py — Depth estimation backend for NVIDIA GPU hosts.

Uses DepthAnything3 via PyTorch + CUDA, with optional torch.compile()
and AMP (automatic mixed precision) for maximum throughput.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional


def load_model(model_id: str, device: str):
    """Load DepthAnything3, move to *device*, and optionally compile it."""
    import torch
    from depth_anything_3.api import DepthAnything3

    # Help PyTorch find CUDA libraries inside Conda environments.
    if "CONDA_PREFIX" in os.environ:
        conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
        os.environ["LD_LIBRARY_PATH"] = (
            conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )

    # Disable JIT profiling to reduce overhead on first inference.
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  [NVIDIA] Model compiled with torch.compile.")
        except Exception as exc:
            print(f"  [NVIDIA] torch.compile skipped: {exc}")

    return model


def estimate_depth(
    rgb_frame: np.ndarray,
    model,
    intrinsics: Optional[np.ndarray],
    device: str,
) -> np.ndarray:
    """Run inference on *rgb_frame*; return float32 depth map at model resolution."""
    import torch
    with torch.inference_mode():
        ctx = (
            torch.amp.autocast("cuda") if device == "cuda" else torch.no_grad()
        )
        with ctx:
            prediction = model.inference(
                image=[rgb_frame],
                intrinsics=[intrinsics] if intrinsics is not None else None,
            )
    return prediction.depth[0]