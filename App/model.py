"""
model.py — Hardware-aware factory for depth estimation.

Depending on the active HardwareProfile this module imports and delegates to:
  * model_nvidia.py  — PyTorch + CUDA (NVIDIA GPU)
  * model_rpi.py     — hailort (Raspberry Pi 5 + AI HAT+)

Usage (from main.py)
--------------------
    from model import load_model, estimate_depth

Both functions share the same external signature regardless of backend.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any

from hardware import HardwareProfile


def load_model(model_id: str, device: str, hw: HardwareProfile, cfg: dict) -> Any:
    """
    Load the depth model appropriate for *hw*.

    Parameters
    ----------
    model_id : HuggingFace model ID (used by the NVIDIA backend).
    device   : Torch device string (from hw.torch_device()).
    hw       : Active HardwareProfile.
    cfg      : Full config dict (RPi backend reads the [rpi] section).
    """
    if hw.is_nvidia:
        from model_nvidia import load_model as _load
        return _load(model_id, device)

    if hw.is_rpi:
        from model_rpi import load_model as _load
        rpi_cfg = cfg.get("rpi", {})
        return _load(model_id, device, rpi_cfg)

    raise RuntimeError(f"No model backend for profile '{hw.profile}'.")


def estimate_depth(
    rgb_frame: np.ndarray,
    model: Any,
    intrinsics: Optional[np.ndarray],
    device: str,
    hw: HardwareProfile,
) -> np.ndarray:
    """
    Run depth inference on *rgb_frame* using the backend selected by *hw*.

    Returns float32 depth map at model resolution.
    """
    if hw.is_nvidia:
        from model_nvidia import estimate_depth as _infer
        return _infer(rgb_frame, model, intrinsics, device)

    if hw.is_rpi:
        from model_rpi import estimate_depth as _infer
        return _infer(rgb_frame, model, intrinsics, device)

    raise RuntimeError(f"No inference backend for profile '{hw.profile}'.")
