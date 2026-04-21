"""
model.py — Hardware-aware factory for depth estimation.

Delegates to:
  * model_nvidia.py  — PyTorch + CUDA (NVIDIA GPU)
  * model_rpi.py     — hailort (Raspberry Pi 5 + AI HAT+)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any

from hardware import HardwareProfile


def load_model(model_id: str, device: str, hw: HardwareProfile, cfg: dict) -> Any:
    if hw.is_nvidia:
        from model_nvidia import load_model as _load
        return _load(model_id, device)

    if hw.is_rpi:
        from model_rpi import load_model as _load
        return _load(model_id, device, cfg.get("rpi", {}), hw.depth_mode)

    raise RuntimeError(f"No model backend for profile '{hw.profile}'.")


def estimate_depth(
    rgb_frame: np.ndarray,
    model: Any,
    intrinsics: Optional[np.ndarray],
    device: str,
    hw: HardwareProfile,
) -> np.ndarray:
    if hw.is_nvidia:
        from model_nvidia import estimate_depth as _infer
        return _infer(rgb_frame, model, intrinsics, device)

    if hw.is_rpi:
        from model_rpi import estimate_depth as _infer
        return _infer(rgb_frame, model, intrinsics, device)

    raise RuntimeError(f"No inference backend for profile '{hw.profile}'.")
