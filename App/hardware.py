"""
hardware.py — Central hardware profile selector.

Supported profiles
------------------
  nvidia   : x86/ARM host with a CUDA-capable NVIDIA GPU.
  rpi      : Raspberry Pi 5 + AI HAT+ (Hailo-8, 26 TOPS NPU).

Supported depth modes
---------------------
  metric   : Absolute depth in metres (DA3-metric on NVIDIA).
  relative : Unitless inverted relative depth (DA2 ViT-S on RPi/Hailo).

Both are set in config.toml:

    [hardware]
    profile    = "nvidia"   # or "rpi"
    depth_mode = "metric"   # or "relative"
"""

from __future__ import annotations

SUPPORTED_PROFILES    = ("nvidia", "rpi")
SUPPORTED_DEPTH_MODES = ("metric", "relative")


class HardwareProfile:
    """Immutable description of the active hardware target."""

    def __init__(self, profile: str, depth_mode: str) -> None:
        profile    = profile.strip().lower()
        depth_mode = depth_mode.strip().lower()

        if profile not in SUPPORTED_PROFILES:
            raise ValueError(
                f"Unknown hardware profile '{profile}'. "
                f"Must be one of: {SUPPORTED_PROFILES}"
            )
        if depth_mode not in SUPPORTED_DEPTH_MODES:
            raise ValueError(
                f"Unknown depth_mode '{depth_mode}'. "
                f"Must be one of: {SUPPORTED_DEPTH_MODES}"
            )

        self.profile    = profile
        self.depth_mode = depth_mode
        

    @property
    def is_nvidia(self) -> bool:
        return self.profile == "nvidia"

    @property
    def is_rpi(self) -> bool:
        return self.profile == "rpi"

    @property
    def is_metric(self) -> bool:
        return self.depth_mode == "metric"

    @property
    def is_relative(self) -> bool:
        return self.depth_mode == "relative"


    def torch_device(self) -> str:
        if self.is_nvidia:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        # RPi: inference runs on Hailo; torch only used for pre/post on CPU.
        return "cpu"

    def __repr__(self) -> str:
        return (
            f"HardwareProfile(profile={self.profile!r}, "
            f"depth_mode={self.depth_mode!r})"
        )


def get_profile(cfg: dict) -> HardwareProfile:
    """Extract and validate the hardware profile from the loaded config."""
    hw_cfg = cfg.get("hardware", {})

    profile = hw_cfg.get("profile")
    if profile is None:
        raise KeyError(
            "Missing [hardware] profile in config.toml.\n"
            "Add:\n\n    [hardware]\n    profile    = \"nvidia\"  # or \"rpi\"\n"
            "    depth_mode = \"metric\"  # or \"relative\""
        )

    depth_mode = hw_cfg.get("depth_mode", "metric")
    return HardwareProfile(profile, depth_mode)
