"""
profiling.py — Per-frame, per-stage latency measurement for the pipeline.

The pipeline always talks to a :class:`FrameProfiler`; which concrete strategy
it holds decides whether anything is actually measured:

* :class:`NullProfiler`    — used when profiling is disabled. Every method is a
  no-op and ``stage()`` hands back a single reusable ``nullcontext``, so the hot
  path pays essentially nothing (Null Object pattern).
* :class:`LatencyProfiler` — times each stage with ``time.perf_counter()``,
  buffers one row per frame in memory and flushes them to a CSV in batches, so
  there is no disk I/O on the per-frame hot path.

Use :func:`make_profiler` to build the right one from an ``AppConfig``.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional


class FrameProfiler(ABC):
    """Interface for measuring the latency of pipeline stages, frame by frame."""

    @abstractmethod
    def stage(self, name: str):
        """Context manager that times the wrapped block as stage *name*."""

    @abstractmethod
    def record(self, name: str, ms: float) -> None:
        """Record an already-measured duration (ms) for stage *name*."""

    @abstractmethod
    def commit_frame(self) -> None:
        """Close the current frame, turning its stage timings into one sample."""

    @abstractmethod
    def close(self) -> None:
        """Flush any buffered samples to their destination and release resources."""


class NullProfiler(FrameProfiler):
    """No-op profiler used when profiling is disabled — negligible overhead."""

    def __init__(self) -> None:
        # A single reusable context manager so stage() allocates nothing per call.
        self._noop = nullcontext()

    def stage(self, name: str):
        return self._noop

    def record(self, name: str, ms: float) -> None:
        pass

    def commit_frame(self) -> None:
        pass

    def close(self) -> None:
        pass


class LatencyProfiler(FrameProfiler):
    """Times pipeline stages and writes one CSV row per frame.

    Per-frame work is cheap: a ``perf_counter()`` pair per stage and an in-memory
    row append. Rows are written to disk in batches (every *flush_every* frames
    and on :meth:`close`), so the per-frame hot path never blocks on disk I/O.
    """

    # Synchronous stages whose durations sum into total_ms, in CSV column order.
    SYNC_STAGES = ("undistort", "depth", "ground", "path")
    # Extra informational stages measured but excluded from total_ms.
    INFO_STAGES = ("capture",)

    _COLUMNS = ("frame", "timestamp", "capture_ms",
                "undistort_ms", "depth_ms", "ground_ms", "path_ms", "total_ms")

    def __init__(self, output_path: str, flush_every: int = 50) -> None:
        self._output_path = output_path
        self._flush_every = max(1, flush_every)

        self._rows:    List[List] = []
        self._current: Dict[str, float] = {}
        self._frame_idx = 0
        self._closed = False

        self._lock = threading.Lock()
        self._header_written = False

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            # Recorded even if the wrapped block raised, so partial work counts.
            self._current[name] = (time.perf_counter() - t0) * 1000.0

    def record(self, name: str, ms: float) -> None:
        self._current[name] = ms

    def commit_frame(self) -> None:
        cur = self._current
        self._current = {}

        total = 0.0
        for stage in self.SYNC_STAGES:
            if stage in cur:
                total += cur[stage]

        row = [
            self._frame_idx,
            time.time(),
            cur.get("capture", ""),
            cur.get("undistort", ""),
            cur.get("depth", ""),
            cur.get("ground", ""),
            cur.get("path", ""),
            total,
        ]
        self._frame_idx += 1

        with self._lock:
            self._rows.append(row)
            if len(self._rows) >= self._flush_every:
                self._flush_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            self._closed = True

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush_locked(self) -> None:
        """Append buffered rows to the CSV. Caller must hold ``self._lock``."""
        if not self._rows:
            return
        rows, self._rows = self._rows, []
        try:
            with open(self._output_path, "a", newline="") as fh:
                writer = csv.writer(fh)
                if not self._header_written:
                    writer.writerow(self._COLUMNS)
                    self._header_written = True
                writer.writerows(rows)
        except OSError as exc:
            # Profiling must never crash the pipeline; drop the batch and warn.
            print(f"LatencyProfiler: could not write '{self._output_path}': {exc}")


def build_latency_profiler(cfg, *, flush_every: int = 50) -> LatencyProfiler:
    """Build a :class:`LatencyProfiler` writing to a timestamped CSV.

    The file name encodes the hardware profile and depth mode so samples stay
    tied to the conditions they were measured under, e.g.
    ``profiling/latency_nvidia_metric_20260604_153012.csv``.
    """
    profile    = getattr(cfg.hardware, "profile", "unknown")
    depth_mode = getattr(cfg.hardware, "depth_mode", "unknown")
    stamp      = time.strftime("%Y%m%d_%H%M%S")
    filename   = f"latency_{profile}_{depth_mode}_{stamp}.csv"
    output_path = os.path.join(cfg.profiling.output_dir, filename)
    return LatencyProfiler(output_path, flush_every=flush_every)


def make_profiler(cfg, *, flush_every: int = 50) -> FrameProfiler:
    """Build the profiler selected by *cfg*.

    Returns a :class:`NullProfiler` when profiling is disabled, otherwise a
    :class:`LatencyProfiler` (see :func:`build_latency_profiler`).
    """
    prof = getattr(cfg, "profiling", None)
    if prof is None or not prof.enabled:
        return NullProfiler()
    return build_latency_profiler(cfg, flush_every=flush_every)
