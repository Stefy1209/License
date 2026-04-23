"""
model_nvidia.py — Depth estimation backend for NVIDIA GPU hosts.

Uses DepthAnything3 via PyTorch + CUDA, with optional torch.compile()
and AMP (automatic mixed precision) for maximum throughput.

Metric depth scaling (DA3METRIC-LARGE)
---------------------------------------
The model outputs a raw unitless prediction. Per the official README:

    metric_depth = focal * net_output / 300.0

where focal = (fx + fy) / 2 from the camera intrinsic matrix K.

If no intrinsics are available the fallback focal length (estimated from
a ~70° FOV assumption in camera.py) is used, giving approximate metric
values instead of accurate ones.
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
    """
    Run inference on *rgb_frame*; return float32 metric depth map.

    Steps
    -----
    1. Run DA3METRIC-LARGE inference — no intrinsics passed to the model.
    2. Scale the raw output to metres: depth_m = focal * raw / 300.0 where focal = (fx + fy) / 2 from the camera matrix.
    """
    import torch
    with torch.inference_mode():
        ctx = (
            torch.amp.autocast("cuda") if device == "cuda" else torch.no_grad()
        )
        with ctx:
            # intrinsics are NOT passed to inference() — DA3METRIC-LARGE
            # does not use them; that parameter is for pose-conditioned mode.
            prediction = model.inference(image=[rgb_frame])

    raw = prediction.depth[0]   # unitless network output, float32

    # Scale to metric depth using the official formula.
    if intrinsics is not None:
        focal = (float(intrinsics[0, 0]) + float(intrinsics[1, 1])) / 2.0
    else:
        # Fallback: estimate focal from a ~70° horizontal FOV assumption.
        # This matches build_fallback_intrinsics() in camera.py.
        h, w = rgb_frame.shape[:2]
        import math
        focal = w / (2.0 * math.tan(math.radians(35)))

    metric_depth = focal * raw / 300.0
    return metric_depth.astype(np.float32)