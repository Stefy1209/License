"""
model_rpi.py — Depth estimation backend for Raspberry Pi 5 + AI HAT+ (Hailo-8).

Pipeline
--------
1. Pre-process the RGB frame into the tensor format expected by the HEF.
2. Run inference via hailort (the official Hailo Python SDK).
3. Post-process raw output logits back to a float32 metric depth map.

Requirements (installed on the Pi)
-----------------------------------
    pip install hailort          # Hailo runtime Python bindings
    pip install numpy opencv-python

The compiled HEF model is configured via config.toml:

    [model]
    id           = "depth-anything/da3metric-large"   # kept for reference
    hef_path     = "models/depth_anything_v2.hef"     # Hailo HEF file
    input_width  = 518
    input_height = 518

Notes
-----
* Hailo runs inference asynchronously; we use the synchronous wrapper `InferVStreams` for simplicity and deterministic latency.
* If `hef_path` is missing or hailort is not installed the module raises a clear error rather than silently falling back.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional


# ---------------------------------------------------------------------------
# Model container — carries everything the estimate_depth() call needs.
# ---------------------------------------------------------------------------

class HailoDepthModel:
    """Thin wrapper around a loaded Hailo HEF network group."""

    def __init__(
        self,
        hef_path: str,
        input_width: int,
        input_height: int,
    ) -> None:
        self.input_width  = input_width
        self.input_height = input_height
        self.hef_path     = hef_path

        # Imports deferred so the rest of the project stays importable on
        # machines that do not have hailort installed.
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

        hef            = HEF(hef_path)
        self._vdevice  = VDevice()
        params         = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group   = self._vdevice.configure(hef, params)[0]
        self._ng_params       = self._network_group.create_params()

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

        # Retrieve input/output layer names once.
        self._input_name  = list(input_params.keys())[0]
        self._output_name = list(output_params.keys())[0]

        print(
            f"  [RPi/Hailo] HEF loaded: {hef_path}  "
            f"({input_width}×{input_height})"
        )


# ---------------------------------------------------------------------------
# Public API — mirrors model_nvidia.py so main.py can call identically.
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str, rpi_cfg: dict) -> HailoDepthModel:
    """
    Load the Hailo HEF model described in *rpi_cfg*.

    Parameters
    ----------
    model_id : kept for API symmetry; not used on RPi (HEF is pre-compiled).
    device   : kept for API symmetry; always "cpu" on RPi.
    rpi_cfg  : the [rpi] section from config.toml.
    """
    hef_path     = rpi_cfg.get("hef_path", "models/depth_anything_v2.hef")
    input_width  = int(rpi_cfg.get("model_input_width",  518))
    input_height = int(rpi_cfg.get("model_input_height", 518))
    return HailoDepthModel(hef_path, input_width, input_height)


def estimate_depth(
    rgb_frame: np.ndarray,
    model: HailoDepthModel,
    intrinsics: Optional[np.ndarray],  # kept for API symmetry; not used here
    device: str,                        # kept for API symmetry
) -> np.ndarray:
    """
    Run metric depth estimation on *rgb_frame* via the Hailo NPU.

    Returns
    -------
    depth : (H_model, W_model) float32 array in metres.
    """
    ih, iw = model.input_height, model.input_width

    # ---- Pre-processing ----
    resized   = cv2.resize(rgb_frame, (iw, ih), interpolation=cv2.INTER_LINEAR)
    norm      = resized.astype(np.float32) / 255.0
    # ImageNet normalisation (DA2/DA3 standard)
    mean      = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std       = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm      = (norm - mean) / std
    # Hailo expects NHWC
    input_tensor = norm[np.newaxis, ...]     # (1, H, W, 3)

    # ---- Inference ----
    with model._network_group.activate(model._ng_params):
        with model._InferVStreams(
            model._network_group,
            model._input_params,
            model._output_params,
        ) as infer_pipeline:
            input_data = {model._input_name: input_tensor}
            infer_pipeline.send(input_data)
            output     = infer_pipeline.recv()

    raw = output[model._output_name]   # (1, H, W) or (1, 1, H, W)

    # ---- Post-processing ----
    depth = raw.squeeze().astype(np.float32)

    # DA3-metric output is already in metres (positive = farther).
    # Ensure minimum positive depth.
    depth = np.clip(depth, 1e-3, None)

    return depth
