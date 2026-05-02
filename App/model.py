from __future__ import annotations

import numpy as np
from typing import Optional, Any

from hardware import HardwareProfile
from config import AppConfig


class ModelRegistry:
    """Factory that loads and runs the appropriate depth model backend."""

    def __init__(self, cfg: AppConfig, hw: HardwareProfile) -> None:
        self._hw  = hw
        self._cfg = cfg
        device    = hw.torch_device()
        self._device = device
        self._model  = self._load(cfg.model.id, device)

    def _load(self, model_id: str, device: str) -> Any:
        if self._hw.is_nvidia:
            from model_nvidia import load_model as _load
            return _load(model_id, device)
        if self._hw.is_rpi:
            from model_rpi import load_model as _load
            rpi_cfg = {
                "hef_path":           self._cfg.rpi.hef_path,
                "model_input_width":  self._cfg.rpi.model_input_width,
                "model_input_height": self._cfg.rpi.model_input_height,
            }
            return _load(model_id, device, rpi_cfg, self._hw.depth_mode)
        raise RuntimeError(f"No model backend for profile '{self._hw.profile}'.")

    def estimate_depth(
        self,
        rgb_frame: np.ndarray,
        intrinsics: Optional[np.ndarray],
    ) -> np.ndarray:
        if self._hw.is_nvidia:
            from model_nvidia import estimate_depth as _infer
            return _infer(rgb_frame, self._model, intrinsics, self._device)
        if self._hw.is_rpi:
            from model_rpi import estimate_depth as _infer
            return _infer(rgb_frame, self._model, intrinsics, self._device)
        raise RuntimeError(f"No inference backend for profile '{self._hw.profile}'.")


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
