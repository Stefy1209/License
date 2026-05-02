from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class CameraConfig:
    id: int
    width: int
    height: int
    max_read_retries: int


@dataclass(frozen=True)
class CalibrationConfig:
    file: str
    cols: int
    rows: int
    square_mm: float
    min_frames: int


@dataclass(frozen=True)
class ModelConfig:
    id: str


@dataclass(frozen=True)
class RpiConfig:
    hef_path: str
    model_input_width: int
    model_input_height: int


@dataclass(frozen=True)
class DepthConfig:
    depth_map_save_location: str


@dataclass(frozen=True)
class GroundConfig:
    seed_region: float
    ransac_iterations: int
    plane_smoothing: float
    normal_threshold: float
    ground_map_save_location: str
    ransac_threshold_metric: float
    ransac_threshold_relative: float


@dataclass(frozen=True)
class VisualizationConfig:
    window_title: str
    ground_overlay_alpha: float
    ground_colour_bgr: Tuple[int, int, int]
    colorbar_width: int


@dataclass(frozen=True)
class HardwareConfig:
    profile: str
    depth_mode: str


@dataclass(frozen=True)
class AppConfig:
    hardware: HardwareConfig
    camera: CameraConfig
    calibration: CalibrationConfig
    model: ModelConfig
    rpi: RpiConfig
    depth: DepthConfig
    ground: GroundConfig
    visualization: VisualizationConfig

    @staticmethod
    def load(path: str = "config.toml") -> "AppConfig":
        raw = _read_toml(path)

        hw  = raw.get("hardware", {})
        cam = raw.get("camera", {})
        cal = raw.get("calibration", {})
        mdl = raw.get("model", {})
        rpi = raw.get("rpi", {})
        dep = raw.get("depth", {})
        gnd = raw.get("ground", {})
        vis = raw.get("visualization", {})

        return AppConfig(
            hardware=HardwareConfig(
                profile=hw.get("profile", "nvidia"),
                depth_mode=hw.get("depth_mode", "metric"),
            ),
            camera=CameraConfig(
                id=cam.get("id", 0),
                width=cam.get("width", 640),
                height=cam.get("height", 480),
                max_read_retries=cam.get("max_read_retries", 5),
            ),
            calibration=CalibrationConfig(
                file=cal.get("file", "calibration.npz"),
                cols=cal.get("cols", 9),
                rows=cal.get("rows", 6),
                square_mm=cal.get("square_mm", 18.0),
                min_frames=cal.get("min_frames", 20),
            ),
            model=ModelConfig(id=mdl.get("id", "depth-anything/da3metric-large")),
            rpi=RpiConfig(
                hef_path=rpi.get("hef_path", "models/depth_anything_v2_vits.hef"),
                model_input_width=int(rpi.get("model_input_width", 224)),
                model_input_height=int(rpi.get("model_input_height", 224)),
            ),
            depth=DepthConfig(
                depth_map_save_location=dep.get("depth_map_save_location", "depth_map.npy"),
            ),
            ground=GroundConfig(
                seed_region=gnd.get("seed_region", 0.5),
                ransac_iterations=gnd.get("ransac_iterations", 200),
                plane_smoothing=gnd.get("plane_smoothing", 0.85),
                normal_threshold=gnd.get("normal_threshold", 0.91),
                ground_map_save_location=gnd.get("ground_map_save_location", "ground_mask.npy"),
                ransac_threshold_metric=gnd.get("ransac_threshold_metric", 0.05),
                ransac_threshold_relative=gnd.get("ransac_threshold_relative", 0.03),
            ),
            visualization=VisualizationConfig(
                window_title=vis.get("window_title", "DA3 Calibrated Live Feed"),
                ground_overlay_alpha=vis.get("ground_overlay_alpha", 0.45),
                ground_colour_bgr=tuple(vis.get("ground_colour_bgr", [0, 220, 80])),
                colorbar_width=vis.get("colorbar_width", 70),
            ),
        )


def _read_toml(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: Configuration file '{path}' not found.\n"
            "       Create it or point to one with --config <path>."
        )
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            sys.exit(
                "ERROR: TOML support is unavailable.\n"
                "       Python >= 3.11 includes tomllib automatically.\n"
                "       For Python 3.9/3.10 run:  pip install tomli"
            )
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:
        sys.exit(f"ERROR: Could not parse '{path}':\n  {exc}")
