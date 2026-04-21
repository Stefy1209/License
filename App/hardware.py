"""
hardware.py — Central hardware profile selector.

Supported profiles
------------------
  nvidia   : x86/ARM host with a CUDA-capable NVIDIA GPU.
  rpi      : Raspberry Pi 5 + AI HAT+ (Hailo-8 NPU, 26 TOPS).

The active profile is read from config.toml:

    [hardware]
    profile = "nvidia"   # or "rpi"
"""

from __future__ import annotations

SUPPORTED_PROFILES = ("nvidia", "rpi")


class HardwareProfile:
    """Immutable description of the active hardware target."""

    def __init__(self, profile: str) -> None:
        profile = profile.strip().lower()
        if profile not in SUPPORTED_PROFILES:
            raise ValueError(
                f"Unknown hardware profile '{profile}'. "
                f"Must be one of: {SUPPORTED_PROFILES}"
            )
        self.profile = profile

    # ------------------------------------------------------------------ #
    #  Convenience predicates                                            #
    # ------------------------------------------------------------------ #

    @property
    def is_nvidia(self) -> bool:
        return self.profile == "nvidia"

    @property
    def is_rpi(self) -> bool:
        return self.profile == "rpi"

    # ------------------------------------------------------------------ #
    #  Torch device string                                               #
    # ------------------------------------------------------------------ #

    def torch_device(self) -> str:
        """Return the torch device string appropriate for this profile."""
        if self.is_nvidia:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        # RPi: inference runs on Hailo via hailort; torch only used for
        # pre/post-processing on CPU.
        return "cpu"

    def __repr__(self) -> str:
        return f"HardwareProfile(profile={self.profile!r})"


def get_profile(cfg: dict) -> HardwareProfile:
    """Extract and validate the hardware profile from the loaded config."""
    try:
        profile_str = cfg["hardware"]["profile"]
    except KeyError:
        raise KeyError(
            "Missing [hardware] profile in config.toml.\n"
            "Add:\n\n    [hardware]\n    profile = \"nvidia\"  # or \"rpi\""
        )
    return HardwareProfile(profile_str)
