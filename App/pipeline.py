from __future__ import annotations

import os
import sys
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import cv2

from config import AppConfig
from hardware import HardwareProfile
from calibration import CalibrationService
from camera import CameraFactory, build_undistort_maps, build_fallback_intrinsics, undistort
from model import ModelRegistry
from ground import detect_ground_mask
from path import find_starting_point, find_ending_point, find_path
from visualization import save_depth_map, save_ground_mask
from profiling import FrameProfiler, NullProfiler, make_profiler, build_latency_profiler


@dataclass
class FrameResult:
    """All computed outputs for a single processed frame."""
    rgb_frame:    np.ndarray
    depth_map:    Optional[np.ndarray]     = None
    ground_mask:  Optional[np.ndarray]     = None
    path:         np.ndarray               = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    start_point:  Optional[Tuple[int,int]] = None
    end_point:    Optional[Tuple[int,int]] = None
    plane:        Optional[np.ndarray]     = None


class DepthPipeline:
    """
    Manages the full inference + ground + path pipeline.

    Runs camera capture in a background thread; the main thread (or GUI)
    calls process_next_frame() to get the latest result.
    """

    def __init__(self, cfg: AppConfig, hw: HardwareProfile) -> None:
        self._cfg    = cfg
        self._hw     = hw
        self._device: Optional[str] = None

        self._mtx:           Optional[np.ndarray] = None
        self._dist:          Optional[np.ndarray] = None
        self._undistort_maps = None
        self._fallback_mtx:  Optional[np.ndarray] = None
        self._prev_plane:    Optional[np.ndarray] = None

        self._model = None
        self._cap   = None

        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event:  threading.Event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        self._profiler: FrameProfiler = make_profiler(cfg)
        self._last_capture_ms: float = 0.0

    def load_calibration(self) -> bool:
        """Try to load calibration; returns True on success."""
        try:
            svc = CalibrationService(self._cfg)
            self._mtx, self._dist = svc.load()
            print("Calibration loaded.")
            return True
        except Exception as exc:
            print(f"No calibration: {exc}. Using fallback intrinsics.")
            return False

    def load_model(self) -> None:
        """Load the depth model (blocking, may take several seconds)."""
        self._device = self._hw.torch_device()
        print(f"Loading depth model ({self._hw.profile} backend)…")
        self._model = ModelRegistry(self._cfg, self._hw)

    def start_capture(self) -> None:
        """Open the camera and start the background capture thread."""
        self._cap = CameraFactory.open(self._cfg, self._hw)
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_worker,
            daemon=True,
            name="capture",
        )
        self._capture_thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and release resources."""
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._profiler.close()

    def set_profiling_enabled(self, enabled: bool) -> None:
        """Turn latency profiling on/off at runtime (e.g. from the GUI toggle).

        Enabling starts a fresh CSV run; disabling flushes and closes the active
        one. Swapping the strategy reference is atomic under the GIL, so it is
        safe against the inference thread that reads ``self._profiler``.
        """
        is_active = not isinstance(self._profiler, NullProfiler)
        if enabled == is_active:
            return
        if enabled:
            self._profiler = build_latency_profiler(self._cfg)
        else:
            old, self._profiler = self._profiler, NullProfiler()
            old.close()

    def process_next_frame(self, timeout: float = 0.5) -> Optional[FrameResult]:
        """
        Pull the latest camera frame, run the full pipeline, and return a FrameResult.
        Returns None if no frame is available within *timeout* seconds.
        """
        try:
            frame = self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

        h, w = frame.shape[:2]

        # Local reference: a runtime profiling toggle can swap self._profiler,
        # but a single frame's stages must all land in the same profiler.
        profiler = self._profiler

        if self._mtx is not None and self._dist is not None and self._undistort_maps is None:
            self._undistort_maps = build_undistort_maps(self._mtx, self._dist, (w, h))
        if self._mtx is None and self._fallback_mtx is None:
            self._fallback_mtx = build_fallback_intrinsics(w, h)

        with profiler.stage("undistort"):
            if self._undistort_maps is not None:
                frame = undistort(frame, self._undistort_maps)
            effective_mtx = self._mtx if self._mtx is not None else self._fallback_mtx
            rgb_frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = FrameResult(rgb_frame=frame)

        if self._model is None:
            return result

        try:
            with profiler.stage("depth"):
                sys.stdout = open(os.devnull, "w")
                raw_depth  = self._model.estimate_depth(rgb_frame, effective_mtx)
                sys.stdout = sys.__stdout__
                depth_map = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)
            result.depth_map = depth_map

            if effective_mtx is not None:
                gnd_cfg = self._cfg.ground
                with profiler.stage("ground"):
                    ground_mask, self._prev_plane = detect_ground_mask(
                        depth_map, effective_mtx,
                        seed_region               = gnd_cfg.seed_region,
                        ransac_threshold_metric   = gnd_cfg.ransac_threshold_metric,
                        ransac_threshold_relative = gnd_cfg.ransac_threshold_relative,
                        ransac_iterations         = gnd_cfg.ransac_iterations,
                        plane_smoothing           = gnd_cfg.plane_smoothing,
                        normal_threshold          = gnd_cfg.normal_threshold,
                        prev_plane                = self._prev_plane,
                        depth_mode                = self._hw.depth_mode,
                    )
                result.ground_mask = ground_mask
                result.plane       = self._prev_plane

                try:
                    with profiler.stage("path"):
                        start = find_starting_point(ground_mask)
                        end   = find_ending_point(ground_mask)
                        result.path        = find_path(ground_mask, start, end)
                        result.start_point = start
                        result.end_point   = end
                except ValueError:
                    pass
            else:
                result.ground_mask = np.zeros((h, w), dtype=bool)

        except RuntimeError as exc:
            sys.stdout = sys.__stdout__
            print(f"Inference error: {exc}")

        profiler.record("capture", self._last_capture_ms)
        profiler.commit_frame()
        return result

    def save_outputs(self, result: FrameResult) -> None:
        """Persist depth map and ground mask from *result* to disk."""
        if result.depth_map is not None:
            save_depth_map(result.depth_map, self._cfg.depth.depth_map_save_location)
        if result.ground_mask is not None:
            save_ground_mask(result.ground_mask, self._cfg.ground.ground_map_save_location)

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _capture_worker(self) -> None:
        max_retries = self._cfg.camera.max_read_retries
        consecutive_failures = 0

        while not self._stop_event.is_set():
            _t0 = time.perf_counter()
            ret, frame = self._cap.read()
            self._last_capture_ms = (time.perf_counter() - _t0) * 1000.0
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_retries:
                    print("Capture thread: too many camera failures. Stopping.")
                    self._stop_event.set()
                continue
            consecutive_failures = 0

            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)
