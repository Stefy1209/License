"""
model_rpi.py — Depth estimation backend for Raspberry Pi 5 + AI HAT+ (Hailo-8).

Pipeline
--------
1. Pre-process the RGB frame into the tensor format expected by the HEF.
2. Run inference via hailort (the official Hailo Python SDK).
3. Post-process the raw output:
     - depth_mode = "relative": invert the disparity map so that closer pixels have SMALLER values (consistent with the metric convention used by the rest of the pipeline).
     - depth_mode = "metric":   use the output as-is (requires a metric-calibrated HEF, e.g. a custom DA3 compile).

DA2 ViT-S from the Hailo Model Zoo outputs *inverse* relative depth (disparity): higher value = closer. We invert it to get a depth-like map where higher value = farther, then normalise to [0, 1].

Requirements (installed on the Pi)
------------------------------------
    pip install hailort
    pip install numpy opencv-python

HEF configuration in config.toml
----------------------------------
    [rpi]
    hef_path          = "models/depth_anything_v2_vits.hef"
    model_input_width  = 518
    model_input_height = 518
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

class HailoDepthModel:
    """Thin wrapper around a loaded Hailo HEF network group."""

    def __init__(
        self,
        hef_path: str,
        input_width: int,
        input_height: int,
        depth_mode: str,
    ) -> None:
        self.input_width  = input_width
        self.input_height = input_height
        self.hef_path     = hef_path
        self.depth_mode   = depth_mode   # "metric" | "relative"

        try:
            from hailo_platform import (
                HEF,
                VDevice,
                HailoStreamInterface,
                InferVStreams,
                ConfigureParams,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType,
            )
        except ImportError as exc:
            raise ImportError(
                "hailort Python package not found.\n"
                "On the Raspberry Pi run:  pip install hailort\n"
                f"Original error: {exc}"
            ) from exc

        self._InferVStreams       = InferVStreams
        self._FormatType          = FormatType
        self._InputVStreamParams  = InputVStreamParams
        self._OutputVStreamParams = OutputVStreamParams

        hef           = HEF(hef_path)
        self._vdevice = VDevice()
        params        = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._vdevice.configure(hef, params)[0]
        self._ng_params     = self._network_group.create_params()

        input_params  = InputVStreamParams.make(
            self._network_group, quantized=False,
            format_type=FormatType.FLOAT32,
        )
        output_params = OutputVStreamParams.make(
            self._network_group, quantized=False,
            format_type=FormatType.FLOAT32,
        )
        self._input_params  = input_params
        self._output_params = output_params
        self._input_name    = list(input_params.keys())[0]
        self._output_name   = list(output_params.keys())[0]

        print(
            f"  [RPi/Hailo] HEF loaded: {hef_path} "
            f"({input_width}x{input_height})  depth_mode={depth_mode}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str, rpi_cfg: dict, depth_mode: str) -> HailoDepthModel:
    """
    Load the Hailo HEF model described in *rpi_cfg*.

    Parameters
    ----------
    model_id   : kept for API symmetry; not used on RPi (HEF is pre-compiled).
    device     : kept for API symmetry; always "cpu" on RPi.
    rpi_cfg    : the [rpi] section from config.toml.
    depth_mode : "metric" | "relative" — controls post-processing.
    """
    hef_path     = rpi_cfg.get("hef_path", "models/depth_anything_v2_vits.hef")
    input_width  = int(rpi_cfg.get("model_input_width",  518))
    input_height = int(rpi_cfg.get("model_input_height", 518))
    return HailoDepthModel(hef_path, input_width, input_height, depth_mode)


def estimate_depth(
    rgb_frame: np.ndarray,
    model: HailoDepthModel,
    intrinsics: Optional[np.ndarray],  # kept for API symmetry; not used here
    device: str,                        # kept for API symmetry
) -> np.ndarray:
    """
    Run depth estimation on *rgb_frame* via the Hailo NPU.

    Returns
    -------
    depth : (H_model, W_model) float32 array.
        - relative mode: values in [0, 1], closer pixels have SMALLER values.
        - metric mode  : values in metres (requires a metric HEF).
    """
    ih, iw = model.input_height, model.input_width

    # ---- Pre-processing ----
    resized      = cv2.resize(rgb_frame, (iw, ih), interpolation=cv2.INTER_LINEAR)
    norm         = resized.astype(np.float32) / 255.0
    mean         = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std          = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm         = (norm - mean) / std
    input_tensor = norm[np.newaxis, ...]    # (1, H, W, 3) NHWC

    # ---- Inference ----
    with model._network_group.activate(model._ng_params):
        with model._InferVStreams(
            model._network_group,
            model._input_params,
            model._output_params,
        ) as infer_pipeline:
            infer_pipeline.send({model._input_name: input_tensor})
            output = infer_pipeline.recv()

    raw   = output[model._output_name]
    depth = raw.squeeze().astype(np.float32)

    # ---- Post-processing ----
    if model.depth_mode == "relative":
        # DA2 ViT-S outputs inverse depth (disparity): closer = higher value.
        # Invert so that closer = smaller value, matching the metric convention
        # expected by ground.py and visualization.py.
        # Guard against divide-by-zero on flat outputs.
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (d_max - depth) / (d_max - d_min)   # invert + normalise to [0, 1]
        else:
            depth = np.zeros_like(depth)
    else:
        # Metric mode: clip to a safe positive range.
        depth = np.clip(depth, 1e-3, None)

    return depth
