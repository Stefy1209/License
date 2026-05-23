from __future__ import annotations

import os
from typing import Any


class CalibrationValidationError(ValueError):
    """Raised when calibration input fails validation."""


class CalibrationValidator:

    def validate_rms(self, rms: float) -> None:
        """rms must be a non-negative number not exceeding 10.0 px."""
        if not isinstance(rms, (int, float)):
            raise CalibrationValidationError("RMS must be a number.")
        if rms < 0:
            raise CalibrationValidationError("RMS cannot be negative.")
        if rms > 10.0:
            raise CalibrationValidationError(
                f"RMS {rms:.4f} is unreasonably high (> 10 px)."
            )

    def validate_npz_file(self, path: str) -> Any:
        """Checks the .npz file exists and contains the required calibration arrays.

        Returns the opened NpzFile so the caller can read values without reopening.
        """
        import numpy as np

        if not os.path.exists(path):
            raise CalibrationValidationError(f"File not found: {path}")
        if not path.endswith(".npz"):
            raise CalibrationValidationError("File must have the .npz extension.")
        try:
            data = np.load(path)
        except Exception as exc:
            raise CalibrationValidationError(f"Cannot read .npz file: {exc}") from exc

        cam_key  = next((k for k in ("camera_matrix", "mtx", "K")            if k in data), None)
        dist_key = next((k for k in ("distortion_coefficients", "dist", "D") if k in data), None)

        if cam_key is None:
            raise CalibrationValidationError(
                "Missing camera_matrix (accepted keys: camera_matrix, mtx, K)."
            )
        if dist_key is None:
            raise CalibrationValidationError(
                "Missing distortion coefficients (accepted keys: distortion_coefficients, dist, D)."
            )

        mtx = data[cam_key]
        if mtx.shape != (3, 3):
            raise CalibrationValidationError(
                f"camera_matrix must be 3×3, but has shape {mtx.shape}."
            )

        return data

    def validate_name(self, name: str) -> None:
        """name is optional but must not exceed 255 characters."""
        if name and len(name) > 255:
            raise CalibrationValidationError("Name must not exceed 255 characters.")

    def validate_notes(self, notes: str) -> None:
        """notes is optional but must not exceed 2048 characters."""
        if notes and len(notes) > 2048:
            raise CalibrationValidationError("Notes must not exceed 2048 characters.")
