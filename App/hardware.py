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

    @classmethod
    def from_config(cls, cfg) -> "HardwareProfile":
        return cls(cfg.hardware.profile, cfg.hardware.depth_mode)

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
        return "cpu"

    def __repr__(self) -> str:
        return (
            f"HardwareProfile(profile={self.profile!r}, "
            f"depth_mode={self.depth_mode!r})"
        )
