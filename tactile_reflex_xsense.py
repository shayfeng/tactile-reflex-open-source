#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dual-Xense tactile reflex controller for a Piper gripper.

This is the XenseSDK counterpart of ``tactile_reflex_v3.py``.  It keeps the
IDLE -> CLOSING -> HOLDING policy and Piper command convention, while replacing
image-intensity proxies with calibrated Xense force and marker outputs.

Metric mapping (after an unloaded baseline is removed):

* Fn: compressive component of ForceResultant Fz, in SDK force units.
* Sx/Sy: force-weighted static Marker2D displacement from Marker2DInit, in
  rectified pixels.  For this mounting physical X(horizontal) follows image y,
  while physical Y(vertical) follows image x.
* SlipX/SlipY: force-weighted frame-to-frame marker speed, in rectified
  pixels/source-frame.  The vertical reflex uses SlipY.
* CoP: normal-force-weighted position on the fixed Marker2DInit grid, in
  rectified image pixels; physical vertical CoP follows image x.

Piper motion is opt-in.  Without ``--enable-piper`` this program is sense-only.
Keys in GUI mode: s=start, r=reset/recalibrate, q/Esc=quit.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# CUDA's eager module loading can spend minutes compiling the Xense models on
# Blackwell GPUs.  Set this before xensesdk/onnxruntime can initialize CUDA.
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import cv2
import numpy as np

try:
    from xensesdk import Sensor
except Exception:  # pragma: no cover - exercised only outside the SDK env
    Sensor = None

try:
    from piper_sdk import C_PiperInterface_V2
except Exception:  # pragma: no cover - Piper is optional in sense-only mode
    C_PiperInterface_V2 = None


# OG000692 was replaced.  Current USB enumeration (2026-08-30) reports
# OG000656 and OG000691; the replacement is installed on the left side.
LEFT_SERIAL = "OG000656"
RIGHT_SERIAL = "OG000691"
LEFT_SX_SIGN = 1.0
LEFT_SY_SIGN = -1.0
RIGHT_SX_SIGN = 1.0
RIGHT_SY_SIGN = 1.0
STATE_NAMES = ("IDLE", "CLOSING", "HOLDING")


def bootstrap_onnxruntime_gpu(infer_device: str) -> bool:
    """Preload pip-installed CUDA/cuDNN libraries before Xense creates sessions.

    Returns whether ONNX Runtime advertises the CUDA provider.  This is only a
    preliminary check; ``Sensor.infer_engine_using_gpu`` is verified after each
    real sensor is opened.
    """
    if infer_device not in ("auto", "cuda"):
        raise ValueError(f"unsupported inference device: {infer_device}")
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    try:
        import onnxruntime as ort
    except Exception as error:
        message = f"could not import ONNX Runtime for GPU preload: {error}"
        if infer_device == "cuda":
            raise RuntimeError(message) from error
        print(f"WARNING: {message}; Xense may use CPU", file=sys.stderr)
        return False

    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        print(
            "WARNING: this ONNX Runtime has no preload_dlls(); CUDA/cuDNN must "
            "already be on the dynamic-library path",
            file=sys.stderr,
        )
    else:
        try:
            # Empty string tells ORT to load NVIDIA libraries installed by pip.
            preload(directory="")
        except Exception as error:
            print(
                f"WARNING: could not preload pip CUDA/cuDNN libraries: {error}",
                file=sys.stderr,
            )

    providers = tuple(ort.get_available_providers())
    cuda_available = "CUDAExecutionProvider" in providers
    print(
        f"ONNX Runtime {getattr(ort, '__version__', 'unknown')}: "
        f"providers={providers}; CUDA_MODULE_LOADING=LAZY"
    )
    if infer_device == "cuda" and not cuda_available:
        raise RuntimeError(
            "--infer-device cuda requested, but ONNX Runtime does not advertise "
            "CUDAExecutionProvider"
        )
    if infer_device == "auto" and not cuda_available:
        print(
            "WARNING: CUDAExecutionProvider is unavailable; Xense inference will "
            "likely use CPU",
            file=sys.stderr,
        )
    return cuda_available


def verify_sensor_inference_devices(
    infer_device: str,
    sensor_states: Sequence[tuple[str, bool]],
) -> bool:
    """Check the SDK's actual engine choice for every opened sensor."""
    non_gpu = [serial for serial, using_gpu in sensor_states if not using_gpu]
    if non_gpu and infer_device == "cuda":
        raise RuntimeError(
            "--infer-device cuda requested, but XenseSDK reports CPU inference for: "
            + ", ".join(non_gpu)
        )
    return not non_gpu


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def clamp(value: float, lower: float, upper: float) -> float:
    return lower if value < lower else upper if value > upper else value


@dataclass(frozen=True)
class XenseFrame:
    """One internally synchronized group returned by selectSensorInfo()."""

    serial: str
    rectify: np.ndarray
    depth: np.ndarray
    force: np.ndarray
    resultant: np.ndarray
    marker2d: np.ndarray
    marker2d_init: np.ndarray
    sensor_timestamp: float
    captured_ns: int
    sequence: int = 0
    # Sensor-native SDK coordinates retained even when the application-facing
    # arrays below are vertically normalized for a right-hand installation.
    marker2d_sdk: np.ndarray | None = None
    marker2d_init_sdk: np.ndarray | None = None


def _copy_array(value: Any, name: str, ndim: int) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"XenseSDK returned no {name}")
    array = np.asarray(value)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise RuntimeError(
            f"invalid Xense {name}: shape={array.shape}, finite={np.isfinite(array).all()}"
        )
    return np.array(array, copy=True)


def _validate_marker_coordinates(marker: np.ndarray, image_shape: tuple[int, int]) -> None:
    """Reject a displacement field accidentally interpreted as absolute coordinates."""
    image_h, image_w = image_shape
    x, y = marker[..., 0], marker[..., 1]
    if np.ptp(x) < image_w * 0.25 or np.ptp(y) < image_h * 0.25:
        raise RuntimeError(
            "Marker2D does not span the rectified image; expected absolute marker "
            "coordinates compatible with XenseSDK 2.1.2"
        )
    if (
        x.min() < -image_w or x.max() > 2 * image_w
        or y.min() < -image_h or y.max() > 2 * image_h
    ):
        raise RuntimeError("Marker2D coordinates are outside the rectified image scale")


def read_xense_frame(sensor: Any, serial: str, flip_y: bool = False) -> XenseFrame:
    """Read all reflex inputs in one SDK call so they belong to one frame."""
    if Sensor is None:
        raise RuntimeError("xensesdk is not installed")
    (
        rectify,
        depth,
        force,
        resultant,
        marker2d,
        marker2d_init,
        timestamp,
    ) = sensor.selectSensorInfo(
        Sensor.OutputType.Rectify,
        Sensor.OutputType.Depth,
        Sensor.OutputType.Force,
        Sensor.OutputType.ForceResultant,
        Sensor.OutputType.Marker2D,
        Sensor.OutputType.Marker2DInit,
        Sensor.OutputType.TimeStamp,
    )
    rectify_a = _copy_array(rectify, "Rectify", 3)
    depth_a = _copy_array(depth, "Depth", 2)
    force_a = _copy_array(force, "Force", 3)
    resultant_a = _copy_array(resultant, "ForceResultant", 1)
    marker_a = _copy_array(marker2d, "Marker2D", 3)
    marker_init_a = _copy_array(marker2d_init, "Marker2DInit", 3)
    if (
        force_a.shape[-1] != 3
        or resultant_a.shape != (6,)
        or marker_a.shape[-1] != 2
        or marker_init_a.shape != marker_a.shape
    ):
        raise RuntimeError(
            "unexpected Xense shapes: "
            f"Force={force_a.shape}, ForceResultant={resultant_a.shape}, "
            f"Marker2D={marker_a.shape}, Marker2DInit={marker_init_a.shape}"
        )
    _validate_marker_coordinates(marker_a, rectify_a.shape[:2])
    _validate_marker_coordinates(marker_init_a, rectify_a.shape[:2])
    marker_sdk_a = marker_a
    marker_init_sdk_a = marker_init_a

    if flip_y:
        image_height = rectify_a.shape[0]
        rectify_a = np.ascontiguousarray(rectify_a[::-1])
        depth_a = np.ascontiguousarray(depth_a[::-1])
        force_a = np.ascontiguousarray(force_a[::-1])
        marker_a = np.ascontiguousarray(marker_a[::-1])
        marker_init_a = np.ascontiguousarray(marker_init_a[::-1])
        marker_a[..., 1] = (image_height - 1) - marker_a[..., 1]
        marker_init_a[..., 1] = (image_height - 1) - marker_init_a[..., 1]

    return XenseFrame(
        serial=serial,
        rectify=rectify_a,
        depth=depth_a,
        force=force_a,
        resultant=resultant_a,
        marker2d=marker_a,
        marker2d_init=marker_init_a,
        sensor_timestamp=float(timestamp),
        captured_ns=time.monotonic_ns(),
        marker2d_sdk=marker_sdk_a,
        marker2d_init_sdk=marker_init_sdk_a,
    )


def validate_frame_pair(
    left: XenseFrame,
    right: XenseFrame,
    now_ns: int,
    stale_timeout_s: float,
    max_pair_skew_s: float,
) -> tuple[float, float, float]:
    """Validate timing before metrics, policy state, or robot commands change."""
    left_age_ms = (now_ns - left.captured_ns) / 1e6
    right_age_ms = (now_ns - right.captured_ns) / 1e6
    if min(left_age_ms, right_age_ms) < 0:
        raise RuntimeError("Xense frame has a future host timestamp")
    if max(left_age_ms, right_age_ms) > stale_timeout_s * 1000.0:
        raise RuntimeError(
            f"stale tactile input: left={left_age_ms:.1f}ms right={right_age_ms:.1f}ms"
        )
    pair_skew_ms = abs(left.captured_ns - right.captured_ns) / 1e6
    if pair_skew_ms > max_pair_skew_s * 1000.0:
        raise RuntimeError(
            f"Xense pair skew is too large: {pair_skew_ms:.1f}ms "
            f"> {max_pair_skew_s * 1000.0:.1f}ms"
        )
    return left_age_ms, right_age_ms, pair_skew_ms


class XenseSensorWorker:
    """Own one Sensor's entire lifecycle and publish its latest complete frame."""

    def __init__(
        self,
        sensor_factory: Callable[[], Any],
        serial: str,
        flip_y: bool = False,
    ):
        self.sensor_factory = sensor_factory
        self.serial = serial
        self.flip_y = flip_y
        self._condition = threading.Condition()
        self._latest: Optional[XenseFrame] = None
        self._sequence = 0
        self._running = False
        self._ready = False
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._infer_engine_using_gpu = False
        self._calibration_request: Optional[int] = None
        self._calibration_result: Optional[list[XenseFrame]] = None

    def start(self, timeout_s: float = 10.0) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
            self._ready = False
            self._error = None
            self._latest = None
            self._infer_engine_using_gpu = False
            self._thread = threading.Thread(
                target=self._loop,
                name=f"xense-{self.serial}",
                daemon=True,
            )
            self._thread.start()
            deadline = time.monotonic() + timeout_s
            while not self._ready:
                if self._error is not None:
                    raise RuntimeError(
                        f"could not open Xense {self.serial}: {self._error}"
                    ) from self._error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out opening Xense {self.serial}")
                self._condition.wait(remaining)

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError(f"Xense worker {self.serial} did not stop")
            self._thread = None

    def _loop(self) -> None:
        sensor = None
        try:
            sensor = self.sensor_factory()
            with self._condition:
                self._infer_engine_using_gpu = bool(
                    getattr(sensor, "infer_engine_using_gpu", False)
                )
                self._ready = True
                self._condition.notify_all()
            while True:
                with self._condition:
                    if not self._running:
                        break
                    calibration_frames = self._calibration_request
                    if calibration_frames is not None:
                        self._calibration_request = None
                if calibration_frames is not None:
                    sensor.calibrateSensor()
                    frames = [
                        read_xense_frame(sensor, self.serial, self.flip_y)
                        for _ in range(calibration_frames)
                    ]
                    with self._condition:
                        self._calibration_result = frames
                        self._condition.notify_all()
                    continue
                frame = read_xense_frame(sensor, self.serial, self.flip_y)
                with self._condition:
                    self._sequence += 1
                    self._latest = XenseFrame(
                        **{**frame.__dict__, "sequence": self._sequence}
                    )
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._running = False
                self._condition.notify_all()
        finally:
            if sensor is not None:
                try:
                    sensor.release()
                except Exception as error:
                    with self._condition:
                        if self._error is None:
                            self._error = error
            with self._condition:
                self._ready = False
                self._running = False
                self._condition.notify_all()

    def calibrate_and_collect(
        self,
        frame_count: int,
        timeout_s: float,
    ) -> list[XenseFrame]:
        if frame_count < 2:
            raise ValueError("calibration frame count must be at least 2")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            if not self._running or not self._ready:
                raise RuntimeError(f"Xense worker {self.serial} is not running")
            if self._calibration_request is not None:
                raise RuntimeError(f"Xense worker {self.serial} is already calibrating")
            self._calibration_result = None
            self._calibration_request = frame_count
            self._condition.notify_all()
            while self._calibration_result is None:
                if self._error is not None:
                    raise RuntimeError(
                        f"Xense calibration {self.serial} failed: {self._error}"
                    ) from self._error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out calibrating Xense {self.serial}")
                self._condition.wait(remaining)
            result = self._calibration_result
            self._calibration_result = None
            return result

    def wait_for_new(self, after_sequence: int, timeout_s: float) -> XenseFrame:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError(
                        f"Xense worker {self.serial} failed: {self._error}"
                    ) from self._error
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for Xense {self.serial}")
                self._condition.wait(remaining)

    @property
    def sequence(self) -> int:
        with self._condition:
            return self._sequence

    @property
    def infer_engine_using_gpu(self) -> bool:
        with self._condition:
            return self._infer_engine_using_gpu

@dataclass
class XenseMetricParams:
    ema_alpha: float = 0.3
    baseline_frames: int = 30
    normal_sign: float = -1.0
    contact_force_threshold: float = 0.01
    noise_multiplier: float = 5.0
    min_contact_cells: int = 3
    adaptive_baseline_rate: float = 0.002
    output_zero_epsilon: float = 1e-5


class XenseMetricsComputer:
    """Convert native Xense outputs into force, shear, and slip metrics.

    Xense image axes are ``u=image x`` (right-positive) and ``v=image y``
    (down-positive).  For this installation the user's physical axes are
    ``Y(vertical) <- u`` and ``X(horizontal) <- v``.  ``sx_sign`` and
    ``sy_sign`` make the signed physical directions explicit per sensor.
    """

    def __init__(
        self,
        params: XenseMetricParams,
        sx_sign: float = 1.0,
        sy_sign: float = 1.0,
    ):
        if sx_sign not in (-1.0, 1.0) or sy_sign not in (-1.0, 1.0):
            raise ValueError("Sx/Sy axis signs must be -1 or +1")
        self.params = params
        self.sx_sign = float(sx_sign)
        self.sy_sign = float(sy_sign)
        self.baseline_force: Optional[np.ndarray] = None
        self.baseline_resultant: Optional[np.ndarray] = None
        self.prev_marker: Optional[np.ndarray] = None
        self.prev_sequence: Optional[int] = None
        self.force_cell_threshold = 0.0
        self.fn_noise = 0.0
        self.slip_y_noise = 0.0
        self.slip_y_cell_noise = 0.0
        self.marker_velocity_noise: Optional[np.ndarray] = None
        self.sy_noise = 0.0  # compatibility alias used by older callers
        self.effective_contact_threshold = params.contact_force_threshold
        self.is_calibrated = False
        self._ema = {
            "Fn": 0.0,
            "Sx": 0.0,
            "Sy": 0.0,
            "SlipX": 0.0,
            "SlipY": 0.0,
            "SlipMag": 0.0,
            "CoP_x": 0.0,
            "CoP_y": 0.0,
        }

    @staticmethod
    def _marker_on_force_grid(marker: np.ndarray, force_shape: tuple[int, int]) -> np.ndarray:
        if marker.shape[:2] == force_shape:
            return marker.astype(np.float64, copy=False)
        width, height = force_shape[1], force_shape[0]
        x = cv2.resize(marker[..., 0], (width, height), interpolation=cv2.INTER_LINEAR)
        y = cv2.resize(marker[..., 1], (width, height), interpolation=cv2.INTER_LINEAR)
        return np.stack((x, y), axis=-1).astype(np.float64)

    def calibrate(self, frames: Sequence[XenseFrame]) -> None:
        if len(frames) < 2:
            raise ValueError("at least two unloaded Xense frames are required")
        force_shapes = {frame.force.shape for frame in frames}
        if len(force_shapes) != 1:
            raise ValueError(f"inconsistent Force shapes during calibration: {force_shapes}")

        self.baseline_force = np.mean(
            np.stack([frame.force for frame in frames]).astype(np.float64), axis=0
        )
        self.baseline_resultant = np.mean(
            np.stack([frame.resultant for frame in frames]).astype(np.float64), axis=0
        )
        normal_samples = []
        fn_samples = []
        marker_velocity_samples = []
        previous_marker = None
        for frame in frames:
            normal = self.params.normal_sign * (
                frame.force[..., 2] - self.baseline_force[..., 2]
            )
            normal_samples.append(np.abs(normal).ravel())
            fn_samples.append(
                abs(
                    self.params.normal_sign
                    * (frame.resultant[2] - self.baseline_resultant[2])
                )
            )
            marker = self._marker_on_force_grid(frame.marker2d, frame.force.shape[:2])
            if previous_marker is not None:
                marker_velocity_samples.append(np.abs(marker - previous_marker))
            previous_marker = marker

        self.force_cell_threshold = float(
            np.percentile(np.concatenate(normal_samples), 99.5)
            * self.params.noise_multiplier
        )
        self.fn_noise = float(np.percentile(fn_samples, 99.0))
        if marker_velocity_samples:
            velocity_samples = np.stack(marker_velocity_samples)
            # Runtime pressure weighting can concentrate almost all weight on
            # one cell.  A whole-grid mean during unloaded calibration would
            # hide that cell's jitter, so estimate and remove a deadband for
            # every marker component before any pressure aggregation.
            self.marker_velocity_noise = np.percentile(
                velocity_samples, 99.0, axis=0
            )
            deadband = self.marker_velocity_noise * self.params.noise_multiplier
            residual = np.maximum(velocity_samples - deadband[None, ...], 0.0)
            residual_vertical_max = np.max(residual[..., 0], axis=(1, 2))
            self.slip_y_noise = float(
                np.percentile(residual_vertical_max, 99.0)
            )
            self.slip_y_cell_noise = float(
                np.percentile(self.marker_velocity_noise[..., 0], 99.5)
            )
        else:
            self.marker_velocity_noise = np.zeros(
                (*frames[-1].force.shape[:2], 2), dtype=np.float64
            )
            self.slip_y_noise = 0.0
            self.slip_y_cell_noise = 0.0
        self.sy_noise = self.slip_y_noise
        self.effective_contact_threshold = max(
            self.params.contact_force_threshold,
            self.fn_noise * self.params.noise_multiplier,
        )
        self.prev_marker = self._marker_on_force_grid(
            frames[-1].marker2d, frames[-1].force.shape[:2]
        ).copy()
        self.prev_sequence = None
        h, w = frames[-1].rectify.shape[:2]
        self._ema = {
            "Fn": 0.0,
            "Sx": 0.0,
            "Sy": 0.0,
            "SlipX": 0.0,
            "SlipY": 0.0,
            "SlipMag": 0.0,
            "CoP_x": w / 2.0,
            "CoP_y": h / 2.0,
        }
        self.is_calibrated = True

    def compute(
        self,
        frame: XenseFrame,
        allow_baseline_update: bool = True,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if not self.is_calibrated or self.baseline_force is None or self.baseline_resultant is None:
            raise RuntimeError("Xense metrics must be calibrated before compute()")
        for name, array in (
            ("Rectify", frame.rectify), ("Depth", frame.depth),
            ("Force", frame.force), ("ForceResultant", frame.resultant),
            ("Marker2D", frame.marker2d), ("Marker2DInit", frame.marker2d_init),
        ):
            if not np.isfinite(array).all():
                raise RuntimeError(f"Xense {name} is not finite")
        if frame.force.shape != self.baseline_force.shape:
            raise RuntimeError(
                f"Force shape changed from {self.baseline_force.shape} to {frame.force.shape}"
            )

        corrected_resultant = frame.resultant.astype(np.float64) - self.baseline_resultant
        corrected_force = frame.force.astype(np.float64) - self.baseline_force
        normal = self.params.normal_sign * corrected_force[..., 2]
        pressure = np.maximum(normal, 0.0)
        force_mask = pressure > self.force_cell_threshold
        pressure_clean = np.where(force_mask, pressure, 0.0)
        area = int(np.count_nonzero(force_mask))

        fn_raw = max(0.0, float(self.params.normal_sign * corrected_resultant[2]))
        contact = bool(
            fn_raw >= self.effective_contact_threshold
            and area >= self.params.min_contact_cells
        )

        marker = self._marker_on_force_grid(frame.marker2d, frame.force.shape[:2])
        marker_init = self._marker_on_force_grid(
            frame.marker2d_init, frame.force.shape[:2]
        )
        if self.prev_marker is None or self.prev_sequence is None:
            marker_delta = np.zeros_like(marker)
            sequence_gap = 1
        else:
            sequence_gap = max(1, frame.sequence - self.prev_sequence)
            marker_delta = (marker - self.prev_marker) / sequence_gap
        marker_displacement = marker - marker_init
        marker_speed_raw = np.linalg.norm(marker_delta, axis=2)
        if self.marker_velocity_noise is None:
            raise RuntimeError("marker velocity noise map is not calibrated")
        marker_deadband = (
            self.marker_velocity_noise * self.params.noise_multiplier
        )
        marker_abs_velocity = np.maximum(
            np.abs(marker_delta) - marker_deadband, 0.0
        )
        marker_delta_clean = np.sign(marker_delta) * marker_abs_velocity
        marker_speed = np.linalg.norm(marker_abs_velocity, axis=2)

        image_h, image_w = frame.rectify.shape[:2]
        cop_x_raw, cop_y_raw = image_w / 2.0, image_h / 2.0
        total_pressure = float(np.sum(pressure_clean))
        sx_raw = sy_raw = 0.0
        slip_x_raw = slip_y_raw = slip_mag_raw = 0.0
        slip_x_signed = slip_y_signed = 0.0
        marker_nonzero_fraction = 0.0
        marker_velocity_image = np.zeros(2, dtype=np.float64)
        shear_image_vec = np.zeros(2, dtype=np.float64)
        if contact and total_pressure > 0.0:
            # Force-weighting prevents a large zero/deadband region from making
            # the old median estimator identically zero, while limiting the
            # influence of isolated marker jitter to its pressure contribution.
            weights = pressure_clean / total_pressure
            weighted = lambda values: float(np.sum(values * weights))
            velocity_u = weighted(marker_delta_clean[..., 0])
            velocity_v = weighted(marker_delta_clean[..., 1])
            displacement_u = weighted(marker_displacement[..., 0])
            displacement_v = weighted(marker_displacement[..., 1])
            marker_velocity_image[:] = (velocity_u, velocity_v)
            shear_image_vec[:] = (displacement_u, displacement_v)

            # Physical X is horizontal and follows image v/y.  Physical Y is
            # vertical and follows image u/x, as specified for this mounting.
            sx_raw = self.sx_sign * displacement_v
            sy_raw = self.sy_sign * displacement_u
            slip_x_signed = self.sx_sign * velocity_v
            slip_y_signed = self.sy_sign * velocity_u
            slip_x_raw = weighted(marker_abs_velocity[..., 1])
            slip_y_raw = weighted(marker_abs_velocity[..., 0])
            slip_mag_raw = weighted(marker_speed)
            marker_nonzero_fraction = float(
                np.count_nonzero(force_mask & (marker_speed > 1e-9)) / max(area, 1)
            )
            # CoP belongs to the fixed pressure grid.  Using current marker
            # coordinates would make pure tangential shear look like pressure
            # redistribution and could falsely trigger the CoP release policy.
            cop_x_raw = float(
                np.sum(marker_init[..., 0] * pressure_clean) / total_pressure
            )
            cop_y_raw = float(
                np.sum(marker_init[..., 1] * pressure_clean) / total_pressure
            )

        alpha = self.params.ema_alpha
        if contact:
            self._ema["Fn"] = alpha * fn_raw + (1.0 - alpha) * self._ema["Fn"]
            self._ema["Sx"] = alpha * sx_raw + (1.0 - alpha) * self._ema["Sx"]
            self._ema["Sy"] = alpha * sy_raw + (1.0 - alpha) * self._ema["Sy"]
            self._ema["SlipX"] = (
                alpha * slip_x_raw + (1.0 - alpha) * self._ema["SlipX"]
            )
            self._ema["SlipY"] = (
                alpha * slip_y_raw + (1.0 - alpha) * self._ema["SlipY"]
            )
            self._ema["SlipMag"] = (
                alpha * slip_mag_raw + (1.0 - alpha) * self._ema["SlipMag"]
            )
            self._ema["CoP_x"] = alpha * cop_x_raw + (1.0 - alpha) * self._ema["CoP_x"]
            self._ema["CoP_y"] = alpha * cop_y_raw + (1.0 - alpha) * self._ema["CoP_y"]
        else:
            for name in ("Fn", "Sx", "Sy", "SlipX", "SlipY", "SlipMag"):
                self._ema[name] *= 1.0 - alpha
            if allow_baseline_update:
                rate = self.params.adaptive_baseline_rate
                self.baseline_force = (
                    (1.0 - rate) * self.baseline_force + rate * frame.force
                )
                self.baseline_resultant = (
                    (1.0 - rate) * self.baseline_resultant + rate * frame.resultant
                )

        # EMA values approach zero asymptotically and otherwise leave tails such
        # as 3.9e-35 in console output, CSV files, and Stage1 traces.  Clamp only
        # derived scalar channels; the original resultant/force arrays remain
        # untouched for diagnostics.  This floor is three orders of magnitude
        # below the 0.01 Fn contact gate and four below the 0.1 SlipY gate.
        zero_epsilon = float(self.params.output_zero_epsilon)
        if zero_epsilon < 0.0 or not math.isfinite(zero_epsilon):
            raise ValueError("output_zero_epsilon must be finite and non-negative")
        for name in ("Fn", "Sx", "Sy", "SlipX", "SlipY", "SlipMag"):
            if abs(self._ema[name]) < zero_epsilon:
                self._ema[name] = 0.0

        def zero_small(value: float) -> float:
            value = float(value)
            return 0.0 if abs(value) < zero_epsilon else value

        fn_raw = zero_small(fn_raw)
        sx_raw = zero_small(sx_raw)
        sy_raw = zero_small(sy_raw)
        slip_x_raw = zero_small(slip_x_raw)
        slip_y_raw = zero_small(slip_y_raw)
        slip_mag_raw = zero_small(slip_mag_raw)
        slip_x_signed = zero_small(slip_x_signed)
        slip_y_signed = zero_small(slip_y_signed)
        marker_velocity_image[np.abs(marker_velocity_image) < zero_epsilon] = 0.0
        shear_image_vec[np.abs(shear_image_vec) < zero_epsilon] = 0.0

        self.prev_marker = marker.copy()
        self.prev_sequence = frame.sequence
        result: dict[str, Any] = {
            "Fn_raw": fn_raw,
            "Fn": float(self._ema["Fn"]),
            "Sx_raw": sx_raw,
            "Sx": float(self._ema["Sx"]),
            "Sy_raw": sy_raw,
            "Sy": float(self._ema["Sy"]),
            "SlipX_raw": slip_x_raw,
            "SlipX": float(self._ema["SlipX"]),
            "SlipY_raw": slip_y_raw,
            "SlipY": float(self._ema["SlipY"]),
            "SlipMag_raw": slip_mag_raw,
            "SlipMag": float(self._ema["SlipMag"]),
            "SlipX_signed": slip_x_signed,
            "SlipY_signed": slip_y_signed,
            "marker_velocity_image": marker_velocity_image.astype(np.float32),
            "shear_image_vec_raw": shear_image_vec.astype(np.float32),
            "shear_image_vec": np.asarray(
                (
                    self.sy_sign * self._ema["Sy"],
                    self.sx_sign * self._ema["Sx"],
                ),
                dtype=np.float32,
            ),
            "marker_nonzero_fraction": marker_nonzero_fraction,
            "CoP_x": float(self._ema["CoP_x"]),
            "CoP_y": float(self._ema["CoP_y"]),
            "CoP_horizontal": float(self.sx_sign * self._ema["CoP_y"]),
            "CoP_vertical": float(self.sy_sign * self._ema["CoP_x"]),
            "Area": area,
            "contact": contact,
            "tau": float(self.effective_contact_threshold),
            "force_resultant": corrected_resultant.astype(np.float32),
            "force_resultant_raw": frame.resultant.astype(np.float32),
            "depth_max": float(np.max(frame.depth)),
            "sensor_timestamp": frame.sensor_timestamp,
            "sequence": frame.sequence,
            "sequence_gap": sequence_gap,
        }
        vis = {
            "pressure": pressure_clean.astype(np.float32),
            "mask": force_mask,
            "marker_speed": marker_speed.astype(np.float32),
            "marker_speed_raw": marker_speed_raw.astype(np.float32),
            "marker_displacement": marker_displacement.astype(np.float32),
            "depth": frame.depth,
        }
        return result, vis


@dataclass
class ReflexParams:
    hz: float = 50.0
    max_stroke_um: int = 100_000
    min_stroke_um: int = 20_000
    close_step_um: int = 150
    init_effort: int = 1200
    max_effort: int = 5000
    fn_stop_threshold: float = 0.10
    slip_y_threshold: float = 0.20
    slip_effort_step: int = 300
    slip_pos_step_um: int = 80
    max_pos_adjust_um: int = 3000
    slip_y_quiet_threshold: float = 0.10
    cop_vertical_decrease_th_px: float = -10.0
    squeeze_effort_step: int = 200
    squeeze_pos_step_um: int = 60
    fn_protect_limit: float = 5.0
    protect_effort_step: int = 400
    protect_pos_step_um: int = 100

    def validate(self) -> None:
        if not math.isfinite(self.hz) or self.hz <= 0:
            raise ValueError("hz must be positive")
        if not 0 <= self.min_stroke_um < self.max_stroke_um:
            raise ValueError("stroke limits must satisfy 0 <= min < max")
        if not 0 < self.init_effort <= self.max_effort:
            raise ValueError("effort limits must satisfy 0 < init <= max")
        if not 0 < self.fn_stop_threshold < self.fn_protect_limit:
            raise ValueError("force thresholds must satisfy 0 < stop < protect")
        if (
            not math.isfinite(self.slip_y_threshold)
            or not math.isfinite(self.slip_y_quiet_threshold)
            or not 0 <= self.slip_y_quiet_threshold < self.slip_y_threshold
        ):
            raise ValueError(
                "slip thresholds must satisfy 0 <= quiet < trigger"
            )
        if (
            not math.isfinite(self.cop_vertical_decrease_th_px)
            or self.cop_vertical_decrease_th_px >= 0
        ):
            raise ValueError("vertical CoP decrease threshold must be negative")
        for name in (
            "close_step_um", "slip_effort_step", "slip_pos_step_um",
            "max_pos_adjust_um", "squeeze_effort_step", "squeeze_pos_step_um",
            "protect_effort_step", "protect_pos_step_um",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    command: bool
    stop_l: bool
    stop_r: bool
    stop_any: bool
    slip_l: bool
    slip_r: bool
    slip_any: bool
    dual_contact: bool
    fn_min: float
    fn_max: float
    fn_raw_max: float
    slip_y_max: float
    cop_vertical_avg: float
    delta_cop_vertical: float


class ReflexPolicy:
    """Pure v3-style state machine, testable without a sensor or robot."""

    STATE_IDLE = 0
    STATE_CLOSING = 1
    STATE_HOLDING = 2

    def __init__(self, params: ReflexParams):
        params.validate()
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.state = self.STATE_IDLE
        self.current_pos = self.params.max_stroke_um
        self.current_effort = self.params.init_effort
        self.cop_baseline_l: Optional[float] = None
        self.cop_baseline_r: Optional[float] = None
        self.hold_start_pos: Optional[int] = None
        self.current_adjusted_um = 0
        self.hold_fn_baseline = 0.0
        self.stable_cop_vertical_ref_l: Optional[float] = None
        self.stable_cop_vertical_ref_r: Optional[float] = None
        self.last_action = ""

    def start(self) -> bool:
        if self.state != self.STATE_IDLE:
            return False
        self.state = self.STATE_CLOSING
        return True

    def step(self, left: dict[str, Any], right: dict[str, Any]) -> PolicyDecision:
        stop_l = bool(
            left["contact"]
            and left["Fn"] >= self.params.fn_stop_threshold
        )
        stop_r = bool(
            right["contact"]
            and right["Fn"] >= self.params.fn_stop_threshold
        )
        stop_any = stop_l or stop_r
        slip_l = bool(
            left["contact"]
            and left["SlipY_raw"] > self.params.slip_y_threshold
        )
        slip_r = bool(
            right["contact"]
            and right["SlipY_raw"] > self.params.slip_y_threshold
        )
        slip_any = slip_l or slip_r
        dual_contact = bool(left["contact"] and right["contact"])
        fn_min = float(min(left["Fn"], right["Fn"]))
        fn_max = float(max(left["Fn"], right["Fn"]))
        fn_raw_max = float(max(left["Fn_raw"], right["Fn_raw"]))
        slip_y_max = float(max(left["SlipY"], right["SlipY"]))

        if left["contact"] and self.cop_baseline_l is None:
            self.cop_baseline_l = float(left["CoP_vertical"])
        if right["contact"] and self.cop_baseline_r is None:
            self.cop_baseline_r = float(right["CoP_vertical"])

        cop_vertical_avg = (
            (float(left["CoP_vertical"]) + float(right["CoP_vertical"])) / 2.0
            if dual_contact
            else 0.0
        )
        have_cop_refs = (
            self.stable_cop_vertical_ref_l is not None
            and self.stable_cop_vertical_ref_r is not None
        )
        delta_cop_vertical = (
            (
                float(left["CoP_vertical"]) - self.stable_cop_vertical_ref_l
                + float(right["CoP_vertical"]) - self.stable_cop_vertical_ref_r
            ) / 2.0
            if dual_contact and have_cop_refs
            else 0.0
        )
        action = ""
        command = False
        state_at_start = self.state

        # Safety correction versus v3: protection is evaluated before every
        # active-state action, including CLOSING and asymmetric contact.
        protected = (
            self.state != self.STATE_IDLE
            and fn_raw_max >= self.params.fn_protect_limit
        )
        if protected:
            self.state = self.STATE_HOLDING
            self.current_effort = max(
                self.params.init_effort,
                self.current_effort - self.params.protect_effort_step,
            )
            self._emergency_release_position(self.params.protect_pos_step_um)
            action = "PROTECT"
            command = True

        if not protected and self.state == self.STATE_CLOSING:
            if stop_any:
                self.state = self.STATE_HOLDING
                action = "LOCK"
            elif self.current_pos <= self.params.min_stroke_um:
                self.state = self.STATE_HOLDING
                action = "STROKE_LIMIT"
            else:
                self.current_pos = max(
                    self.params.min_stroke_um,
                    self.current_pos - self.params.close_step_um,
                )
                self.current_effort = self.params.init_effort
                action = "CLOSE"
                command = True

        if (
            not protected
            and state_at_start == self.STATE_HOLDING
            and self.state == self.STATE_HOLDING
        ):
            if self.hold_start_pos is None and dual_contact:
                self.hold_start_pos = self.current_pos
                self.current_adjusted_um = 0
                self.stable_cop_vertical_ref_l = float(left["CoP_vertical"])
                self.stable_cop_vertical_ref_r = float(right["CoP_vertical"])
                self.hold_fn_baseline = fn_min
                delta_cop_vertical = 0.0

            if slip_any and self.hold_start_pos is not None:
                self.current_effort = int(
                    clamp(
                        self.current_effort + self.params.slip_effort_step,
                        self.params.init_effort,
                        self.params.max_effort,
                    )
                )
                if self.current_adjusted_um < self.params.max_pos_adjust_um:
                    step = min(
                        self.params.slip_pos_step_um,
                        self.params.max_pos_adjust_um - self.current_adjusted_um,
                    )
                    old_pos = self.current_pos
                    self.current_pos = max(self.params.min_stroke_um, old_pos - step)
                    actual_step = old_pos - self.current_pos
                    self.current_adjusted_um += actual_step
                action = "SLIP"
                command = True
            elif (
                dual_contact
                and have_cop_refs
                and delta_cop_vertical < self.params.cop_vertical_decrease_th_px
                and slip_y_max < self.params.slip_y_quiet_threshold
            ):
                self.current_effort = max(
                    self.params.init_effort,
                    self.current_effort - self.params.squeeze_effort_step,
                )
                self._release_position(self.params.squeeze_pos_step_um)
                self.stable_cop_vertical_ref_l += 0.3 * (
                    float(left["CoP_vertical"]) - self.stable_cop_vertical_ref_l
                )
                self.stable_cop_vertical_ref_r += 0.3 * (
                    float(right["CoP_vertical"]) - self.stable_cop_vertical_ref_r
                )
                action = "RELEASE"
                command = True

            if have_cop_refs and dual_contact:
                self.stable_cop_vertical_ref_l = (
                    0.98 * self.stable_cop_vertical_ref_l
                    + 0.02 * float(left["CoP_vertical"])
                )
                self.stable_cop_vertical_ref_r = (
                    0.98 * self.stable_cop_vertical_ref_r
                    + 0.02 * float(right["CoP_vertical"])
                )
            if not action and self.hold_fn_baseline > 0:
                self.hold_fn_baseline = 0.95 * self.hold_fn_baseline + 0.05 * fn_min

        self.last_action = action
        return PolicyDecision(
            action=action,
            command=command,
            stop_l=stop_l,
            stop_r=stop_r,
            stop_any=stop_any,
            slip_l=slip_l,
            slip_r=slip_r,
            slip_any=slip_any,
            dual_contact=dual_contact,
            fn_min=fn_min,
            fn_max=fn_max,
            fn_raw_max=fn_raw_max,
            slip_y_max=slip_y_max,
            cop_vertical_avg=cop_vertical_avg,
            delta_cop_vertical=delta_cop_vertical,
        )

    def _release_position(self, requested_step: int) -> None:
        if self.hold_start_pos is None or self.current_pos >= self.hold_start_pos:
            return
        step = min(requested_step, self.hold_start_pos - self.current_pos)
        self.current_pos += step
        self.current_adjusted_um = max(0, self.current_adjusted_um - step)

    def _emergency_release_position(self, requested_step: int) -> None:
        """Open on over-force even before HOLD has established a baseline."""
        old_pos = self.current_pos
        self.current_pos = min(
            self.params.max_stroke_um,
            self.current_pos + requested_step,
        )
        released = self.current_pos - old_pos
        self.current_adjusted_um = max(0, self.current_adjusted_um - released)


class DualXenseReflexController:
    """Hardware integration, logging, and UI around ReflexPolicy."""

    CSV_COLUMNS = (
        "t_wall", "state", "pos_um", "effort",
        "FnL", "FnR", "Fn_min", "FnRawL", "FnRawR", "Fn_raw_max",
        "SxRawL", "SxL", "SyRawL", "SyL",
        "SxRawR", "SxR", "SyRawR", "SyR",
        "SlipXRawL", "SlipXL", "SlipYRawL", "SlipYL",
        "SlipMagRawL", "SlipMagL", "SlipXSignedL", "SlipYSignedL",
        "SlipXRawR", "SlipXR", "SlipYRawR", "SlipYR",
        "SlipMagRawR", "SlipMagR", "SlipXSignedR", "SlipYSignedR",
        "SlipY_max", "marker_nonzero_fraction_L", "marker_nonzero_fraction_R",
        "CoP_Lx", "CoP_Ly", "CoP_Rx", "CoP_Ry",
        "CoP_horizontal_L", "CoP_horizontal_R",
        "CoP_vertical_L", "CoP_vertical_R", "CoP_vertical_avg",
        "AreaL", "AreaR", "contactL", "contactR",
        "stopL", "stopR", "stop_any",
        "slipL", "slipR", "slip_any", "tauL", "tauR",
        "cop_vertical_shift_max", "delta_cop_vertical", "adjusted_um", "action",
        "hold_fn_base", "frame_ms", "freq_hz_avg",
        "left_serial", "right_serial", "left_sequence", "right_sequence",
        "left_sequence_gap", "right_sequence_gap",
        "left_sensor_time", "right_sensor_time", "left_age_ms", "right_age_ms",
        "pair_skew_ms", "L_Fx", "L_Fy", "L_Fz", "L_Tx", "L_Ty", "L_Tz",
        "R_Fx", "R_Fy", "R_Fz", "R_Tx", "R_Ty", "R_Tz",
        "depth_max_L", "depth_max_R",
    )

    def __init__(
        self,
        left_serial: str,
        right_serial: str,
        can_iface: str,
        enable_piper: bool,
        params: ReflexParams,
        metric_params: XenseMetricParams,
        out_csv: Path,
        infer_mode: str = "normal",
        infer_device: str = "auto",
        flip_right: bool = True,
        headless: bool = False,
        frame_timeout_s: float = 1.0,
        stale_timeout_s: float = 0.5,
        max_pair_skew_s: float = 0.2,
        sensor_open_timeout_s: float = 60.0,
        piper_enable_timeout_s: float = 10.0,
        left_sx_sign: float = LEFT_SX_SIGN,
        left_sy_sign: float = LEFT_SY_SIGN,
        right_sx_sign: float = RIGHT_SX_SIGN,
        right_sy_sign: float = RIGHT_SY_SIGN,
    ):
        if left_serial == right_serial:
            raise ValueError("left and right Xense serials must be different")
        if (
            frame_timeout_s <= 0
            or stale_timeout_s <= 0
            or max_pair_skew_s <= 0
            or sensor_open_timeout_s <= 0
        ):
            raise ValueError("sensor-open, frame, stale, and pair-skew timeouts must be positive")
        if infer_device not in ("auto", "cuda"):
            raise ValueError(f"unsupported inference device: {infer_device}")
        if Sensor is None:
            raise RuntimeError(
                "xensesdk is unavailable; install xensesdk in the active Python environment"
            )
        params.validate()
        self.params = params
        self.policy = ReflexPolicy(params)
        self._configured_slip_y_threshold = params.slip_y_threshold
        self.left_serial = left_serial
        self.right_serial = right_serial
        self.headless = headless
        self.frame_timeout_s = frame_timeout_s
        self.stale_timeout_s = stale_timeout_s
        self.max_pair_skew_s = max_pair_skew_s
        self.window_name = "tactile_reflex_xsense"
        self._last_left_sequence = 0
        self._last_right_sequence = 0
        self._last_tick_ns: Optional[int] = None
        self._frequency = deque(maxlen=20)
        self._last_action_printed = ""
        self._display_pressure_scale: dict[str, float] = {}
        self._cleaned = False
        self.left_worker: Optional[XenseSensorWorker] = None
        self.right_worker: Optional[XenseSensorWorker] = None
        self.piper = None
        self.enable_piper = False
        self._csv_file = None
        self._csv_writer = None

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            discovered = Sensor.scanSerialNumber()
            missing = [serial for serial in (left_serial, right_serial) if serial not in discovered]
            if missing:
                raise RuntimeError(f"Xense sensor(s) not found: {', '.join(missing)}")
            print(f"Xense devices: {discovered}")
            print(
                f"Opening LEFT={left_serial}, RIGHT={right_serial}, "
                f"infer_mode={infer_mode}, infer_device={infer_device}"
            )
            self.left_worker = XenseSensorWorker(
                lambda: Sensor.create(left_serial, infer_mode=infer_mode),
                left_serial,
                flip_y=False,
            )
            self.right_worker = XenseSensorWorker(
                lambda: Sensor.create(right_serial, infer_mode=infer_mode),
                right_serial,
                flip_y=flip_right,
            )
            self.left_worker.start(timeout_s=sensor_open_timeout_s)
            self.right_worker.start(timeout_s=sensor_open_timeout_s)
            sensor_states = (
                (left_serial, self.left_worker.infer_engine_using_gpu),
                (right_serial, self.right_worker.infer_engine_using_gpu),
            )
            if verify_sensor_inference_devices(infer_device, sensor_states):
                print("Xense inference device: GPU (verified for both sensors)")
            else:
                cpu_serials = ", ".join(
                    serial for serial, using_gpu in sensor_states if not using_gpu
                )
                print(
                    f"WARNING: XenseSDK is using CPU inference for: {cpu_serials}",
                    file=sys.stderr,
                )
            self.metrics_l = XenseMetricsComputer(
                metric_params,
                sx_sign=left_sx_sign,
                sy_sign=left_sy_sign,
            )
            self.metrics_r = XenseMetricsComputer(
                metric_params,
                sx_sign=right_sx_sign,
                sy_sign=right_sy_sign,
            )
            self._csv_file = out_csv.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self.CSV_COLUMNS)
            self._csv_file.flush()
            if enable_piper:
                self._init_piper(can_iface, piper_enable_timeout_s)
        except Exception:
            self.cleanup(open_gripper=False)
            raise

    @property
    def state(self) -> int:
        return self.policy.state

    def _init_piper(self, can_iface: str, timeout_s: float) -> None:
        if C_PiperInterface_V2 is None:
            raise RuntimeError("--enable-piper requested but piper_sdk is unavailable")
        print(f"Initializing Piper on {can_iface} (motion explicitly enabled) ...")
        self.piper = C_PiperInterface_V2(can_iface)
        self.piper.ConnectPort()
        deadline = time.monotonic() + timeout_s
        while not self.piper.EnablePiper():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Piper did not enable within {timeout_s:.1f}s")
            time.sleep(0.01)
        try:
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0xAD)
        except Exception as error:
            print(f"WARNING: MotionCtrl_2 failed: {error}")
        self.enable_piper = True
        self.piper.GripperCtrl(self.params.max_stroke_um, 1000, 0x02, 0)
        time.sleep(0.6)
        self.piper.GripperCtrl(self.params.max_stroke_um, 1000, 0x01, 0)
        time.sleep(0.8)
        print("Piper enabled and gripper opened")

    def _send_gripper(self) -> None:
        if not self.enable_piper or self.piper is None:
            return
        position = int(
            clamp(self.policy.current_pos, self.params.min_stroke_um, self.params.max_stroke_um)
        )
        effort = int(
            clamp(self.policy.current_effort, self.params.init_effort, self.params.max_effort)
        )
        try:
            self.piper.GripperCtrl(abs(position), effort, 0x01, 0)
        except Exception as error:
            raise RuntimeError(f"Piper GripperCtrl failed: {error}") from error

    def auto_init(self, wait_sec: float = 1.5) -> None:
        assert self.left_worker is not None and self.right_worker is not None
        print("\nEnsure BOTH Xense surfaces are unloaded.")
        print(f"Calibrating in {wait_sec:.1f}s ...")
        time.sleep(wait_sec)
        frame_count = self.metrics_l.params.baseline_frames
        calibration_timeout = max(10.0, frame_count * 0.5)
        left_frames = self.left_worker.calibrate_and_collect(
            frame_count, calibration_timeout
        )
        right_frames = self.right_worker.calibrate_and_collect(
            frame_count, calibration_timeout
        )
        self.metrics_l.calibrate(left_frames)
        self.metrics_r.calibrate(right_frames)
        max_contact_threshold = max(
            self.metrics_l.effective_contact_threshold,
            self.metrics_r.effective_contact_threshold,
        )
        if max_contact_threshold >= self.params.fn_stop_threshold:
            raise RuntimeError(
                "calibrated contact threshold is not below --fn-stop "
                f"({max_contact_threshold:.6g} >= {self.params.fn_stop_threshold:.6g}); "
                "ensure both sensors are unloaded or retune thresholds"
            )
        self.params.slip_y_threshold = max(
            self._configured_slip_y_threshold,
            self.metrics_l.slip_y_noise,
            self.metrics_r.slip_y_noise,
        )
        # Discard frames published before/during the other sensor's sequential
        # calibration.  The first update must wait for a post-calibration frame
        # from each worker instead of accepting an old cached frame.
        self._last_left_sequence = self.left_worker.sequence
        self._last_right_sequence = self.right_worker.sequence
        self._last_tick_ns = None
        self._frequency.clear()
        self._display_pressure_scale.clear()
        print(
            "Calibration complete: "
            f"contact_threshold L={self.metrics_l.effective_contact_threshold:.6g}, "
            f"R={self.metrics_r.effective_contact_threshold:.6g}; "
            f"vertical_cell_noise(p99.5) L={self.metrics_l.slip_y_cell_noise:.4g}, "
            f"R={self.metrics_r.slip_y_cell_noise:.4g}px/source-frame; "
            f"post-deadband_noise L={self.metrics_l.slip_y_noise:.4g}, "
            f"R={self.metrics_r.slip_y_noise:.4g}px/source-frame; "
            f"effective_slip_y_threshold={self.params.slip_y_threshold:.4g}px/source-frame"
        )

    def reset(self, wait_sec: float = 1.5) -> None:
        print("\nRESET: open, recalibrate, return to IDLE")
        self.policy.reset()
        self._send_gripper()
        self.auto_init(wait_sec)

    def start_grasping(self) -> None:
        if self.policy.start():
            if self.enable_piper:
                print("\nStart grasping: IDLE -> CLOSING (Piper commands enabled)")
            else:
                print(
                    "\nSENSE ONLY: IDLE -> CLOSING policy simulation; "
                    "no command is sent to Piper"
                )
        else:
            print(f"\nCannot start while state={STATE_NAMES[self.policy.state]}")

    def _measure_frequency(self, tick_ns: int) -> tuple[float, float]:
        if self._last_tick_ns is None:
            self._last_tick_ns = tick_ns
            return 0.0, 0.0
        elapsed_s = max((tick_ns - self._last_tick_ns) / 1e9, 1e-9)
        self._last_tick_ns = tick_ns
        self._frequency.append(1.0 / elapsed_s)
        return float(sum(self._frequency) / len(self._frequency)), elapsed_s * 1000.0

    def update(self) -> tuple[Optional[np.ndarray], dict[str, Any], tuple[float, float]]:
        loop_started = time.monotonic()
        assert self.left_worker is not None and self.right_worker is not None
        left_frame = self.left_worker.wait_for_new(
            self._last_left_sequence, self.frame_timeout_s
        )
        right_frame = self.right_worker.wait_for_new(
            self._last_right_sequence, self.frame_timeout_s
        )
        self._last_left_sequence = left_frame.sequence
        self._last_right_sequence = right_frame.sequence
        now_ns = time.monotonic_ns()
        left_age_ms, right_age_ms, pair_skew_ms = validate_frame_pair(
            left_frame,
            right_frame,
            now_ns,
            self.stale_timeout_s,
            self.max_pair_skew_s,
        )

        allow_baseline_update = self.policy.state == ReflexPolicy.STATE_IDLE
        result_l, vis_l = self.metrics_l.compute(
            left_frame, allow_baseline_update=allow_baseline_update
        )
        result_r, vis_r = self.metrics_r.compute(
            right_frame, allow_baseline_update=allow_baseline_update
        )
        decision = self.policy.step(result_l, result_r)
        if decision.command:
            self._send_gripper()
        if decision.action and decision.action != self._last_action_printed:
            print(
                f"{decision.action}: Fn=[{result_l['Fn']:.4g}, {result_r['Fn']:.4g}] "
                f"FnRawMax={decision.fn_raw_max:.4g} "
                f"Stop=L{int(decision.stop_l)}/R{int(decision.stop_r)} "
                f"SlipYRaw=[{result_l['SlipY_raw']:.4f}, "
                f"{result_r['SlipY_raw']:.4f}] "
                f"pos={self.policy.current_pos / 1000:.1f}mm "
                f"effort={self.policy.current_effort}"
            )
        self._last_action_printed = decision.action

        tick_ns = time.monotonic_ns()
        avg_hz, frame_ms = self._measure_frequency(tick_ns)
        cop_vertical_shift_l = (
            abs(result_l["CoP_vertical"] - self.policy.cop_baseline_l)
            if self.policy.cop_baseline_l is not None else 0.0
        )
        cop_vertical_shift_r = (
            abs(result_r["CoP_vertical"] - self.policy.cop_baseline_r)
            if self.policy.cop_baseline_r is not None else 0.0
        )
        self._write_csv(
            result_l, result_r, decision, avg_hz, frame_ms,
            left_age_ms, right_age_ms, pair_skew_ms,
            max(cop_vertical_shift_l, cop_vertical_shift_r),
        )

        visual = None
        if not self.headless:
            visual = self._draw_ui(
                left_frame, result_l, vis_l, right_frame, result_r, vis_r,
                decision, avg_hz, frame_ms,
            )
        remaining = 1.0 / self.params.hz - (time.monotonic() - loop_started)
        if remaining > 0:
            time.sleep(remaining)
        result = {
            "L": result_l,
            "R": result_r,
            "Fn_min": decision.fn_min,
            "Fn_max": decision.fn_max,
            "Fn_raw_max": decision.fn_raw_max,
            "SlipY_max": decision.slip_y_max,
            "CoP_vertical_avg": decision.cop_vertical_avg,
            "delta_CoP_vertical": decision.delta_cop_vertical,
            "stopL": decision.stop_l,
            "stopR": decision.stop_r,
            "stop_any": decision.stop_any,
            "slip_any": decision.slip_any,
            "state": self.policy.state,
            "pos_um": self.policy.current_pos,
            "effort": self.policy.current_effort,
            "action": decision.action,
        }
        return visual, result, (avg_hz, frame_ms)

    def _write_csv(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        decision: PolicyDecision,
        avg_hz: float,
        frame_ms: float,
        left_age_ms: float,
        right_age_ms: float,
        pair_skew_ms: float,
        cop_vertical_shift_max: float,
    ) -> None:
        if self._csv_writer is None:
            return
        resultant_l = left["force_resultant"]
        resultant_r = right["force_resultant"]
        row = (
            time.time(), STATE_NAMES[self.policy.state], self.policy.current_pos,
            self.policy.current_effort, left["Fn"], right["Fn"], decision.fn_min,
            left["Fn_raw"], right["Fn_raw"], decision.fn_raw_max,
            left["Sx_raw"], left["Sx"], left["Sy_raw"], left["Sy"],
            right["Sx_raw"], right["Sx"], right["Sy_raw"], right["Sy"],
            left["SlipX_raw"], left["SlipX"], left["SlipY_raw"], left["SlipY"],
            left["SlipMag_raw"], left["SlipMag"],
            left["SlipX_signed"], left["SlipY_signed"],
            right["SlipX_raw"], right["SlipX"], right["SlipY_raw"], right["SlipY"],
            right["SlipMag_raw"], right["SlipMag"],
            right["SlipX_signed"], right["SlipY_signed"],
            decision.slip_y_max,
            left["marker_nonzero_fraction"], right["marker_nonzero_fraction"],
            left["CoP_x"], left["CoP_y"], right["CoP_x"], right["CoP_y"],
            left["CoP_horizontal"], right["CoP_horizontal"],
            left["CoP_vertical"], right["CoP_vertical"],
            decision.cop_vertical_avg,
            left["Area"], right["Area"], int(left["contact"]),
            int(right["contact"]),
            int(decision.stop_l), int(decision.stop_r), int(decision.stop_any),
            int(decision.slip_l), int(decision.slip_r),
            int(decision.slip_any), left["tau"], right["tau"],
            cop_vertical_shift_max, decision.delta_cop_vertical,
            self.policy.current_adjusted_um, decision.action,
            self.policy.hold_fn_baseline, frame_ms, avg_hz,
            self.left_serial, self.right_serial, left["sequence"], right["sequence"],
            left["sequence_gap"], right["sequence_gap"],
            left["sensor_timestamp"], right["sensor_timestamp"], left_age_ms,
            right_age_ms, pair_skew_ms, *resultant_l.tolist(), *resultant_r.tolist(),
            left["depth_max"], right["depth_max"],
        )
        self._csv_writer.writerow(row)
        if self._csv_file is not None and int(time.monotonic() * 2) % 2 == 0:
            self._csv_file.flush()

    @staticmethod
    def _heatmap(
        values: np.ndarray,
        size: tuple[int, int],
        high: Optional[float] = None,
    ) -> np.ndarray:
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size == 0:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if high is None:
            # Percentile only over active cells.  Including hundreds of zero
            # cells makes P98 equal zero for small contacts/noise speckles.
            high = float(np.percentile(positive, 98.0))
        scaled = np.zeros_like(values, dtype=np.uint8)
        if high > 0:
            scaled = np.clip(values * (255.0 / high), 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        return cv2.resize(heat, size, interpolation=cv2.INTER_NEAREST)

    def _stable_pressure_scale(self, title: str, pressure: np.ndarray) -> float:
        positive = pressure[np.isfinite(pressure) & (pressure > 0)]
        if positive.size == 0:
            return 0.0
        target = float(np.percentile(positive, 98.0))
        previous = self._display_pressure_scale.get(title)
        if previous is None or previous <= 0:
            scale = target
        else:
            # Raise the ceiling reasonably quickly to avoid saturation, but
            # lower it slowly so stationary contact does not pulse brighter.
            alpha = 0.25 if target > previous else 0.08
            scale = previous + alpha * (target - previous)
        self._display_pressure_scale[title] = max(scale, np.finfo(float).eps)
        return self._display_pressure_scale[title]

    def _sensor_overlay(
        self,
        frame: XenseFrame,
        result: dict[str, Any],
        vis: dict[str, np.ndarray],
        title: str,
    ) -> np.ndarray:
        image = frame.rectify.copy()
        h, w = image.shape[:2]
        # Leave Rectify untouched for sub-contact force noise.  Keep the last
        # scale across short contact dropouts so the next overlay cannot jump
        # merely because its color ceiling was reinitialized.
        if result["contact"]:
            scale = self._stable_pressure_scale(title, vis["pressure"])
            heat = self._heatmap(vis["pressure"], (w, h), high=scale)
            mask = cv2.resize(
                vis["mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            image[mask] = cv2.addWeighted(image, 0.35, heat, 0.65, 0)[mask]
            cop = (
                int(round(result["CoP_x"])),
                int(round(result["CoP_y"])),
            )
            cv2.drawMarker(
                image, cop,
                (0, 0, 255), cv2.MARKER_CROSS, 18, 2,
            )
            # Marker displacement is stored in image coordinates (u, v), so
            # this arrow remains visually truthful even though physical axes
            # are mapped as Y<-u and X<-v.
            shear_u, shear_v = np.asarray(result["shear_image_vec"], dtype=float)
            if math.hypot(shear_u, shear_v) > 1e-3:
                arrow_scale = 6.0
                endpoint = (
                    int(round(clamp(cop[0] + arrow_scale * shear_u, 0, w - 1))),
                    int(round(clamp(cop[1] + arrow_scale * shear_v, 0, h - 1))),
                )
                cv2.arrowedLine(
                    image, cop, endpoint, (0, 255, 255), 2, cv2.LINE_AA,
                    tipLength=0.25,
                )
        cv2.rectangle(image, (0, 0), (w, 86), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"{title} {frame.serial} Fn={result['Fn']:.4g}",
            (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (240, 240, 240), 1, cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Sx(H/imgY)={result['Sx']:+.4f} Sy(V/imgX)={result['Sy']:+.4f}",
            (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"SlipY abs/signed={result['SlipY']:.4f}/{result['SlipY_signed']:+.4f}",
            (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA,
        )
        resultant = result["force_resultant"]
        cv2.putText(
            image,
            "F=" + np.array2string(resultant[:3], precision=3, suppress_small=True),
            (8, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (180, 180, 180), 1, cv2.LINE_AA,
        )
        return image

    def _draw_ui(
        self,
        left_frame: XenseFrame,
        left: dict[str, Any],
        vis_l: dict[str, np.ndarray],
        right_frame: XenseFrame,
        right: dict[str, Any],
        vis_r: dict[str, np.ndarray],
        decision: PolicyDecision,
        avg_hz: float,
        frame_ms: float,
    ) -> np.ndarray:
        panels = (
            self._sensor_overlay(left_frame, left, vis_l, "LEFT"),
            self._sensor_overlay(right_frame, right, vis_r, "RIGHT"),
        )
        canvas = np.hstack(panels)
        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, h - 58), (w, h), (0, 0, 0), -1)
        line1 = (
            f"{'PIPER' if self.enable_piper else 'SIM/SENSE-ONLY'} "
            f"{STATE_NAMES[self.policy.state]} {avg_hz:.1f}Hz/{frame_ms:.1f}ms "
            f"pos={self.policy.current_pos / 1000:.1f}mm effort={self.policy.current_effort} "
            f"action={decision.action or '---'}"
        )
        line2 = (
            f"Fn min/max={decision.fn_min:.4g}/{decision.fn_max:.4g} "
            f"SlipY max={decision.slip_y_max:.4f}px/src-frame "
            f"dCoP_V={decision.delta_cop_vertical:+.1f}px "
            f"adjust={self.policy.current_adjusted_um}um"
        )
        cv2.putText(canvas, line1, (8, h - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (190, 190, 190), 1, cv2.LINE_AA)
        border = (0, 0, 255) if decision.action in ("SLIP", "PROTECT") else (0, 200, 0)
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), border, 3)
        return canvas

    def cleanup(self, open_gripper: bool = True) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        # Do not wait for a potentially blocked SDK read before issuing the
        # same exit action as tactile_reflex_v3: open the gripper first.
        if open_gripper and self.enable_piper and self.piper is not None:
            try:
                self.policy.current_pos = self.params.max_stroke_um
                self.policy.current_effort = self.params.init_effort
                self._send_gripper()
                time.sleep(0.2)
            except Exception as error:
                print(f"WARNING: could not open gripper during cleanup: {error}")
        for worker in (self.left_worker, self.right_worker):
            if worker is not None:
                try:
                    worker.stop()
                except Exception as error:
                    print(f"WARNING: worker stop failed: {error}")
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
        if self.piper is not None:
            try:
                self.piper.DisconnectPort()
            except Exception as error:
                print(f"WARNING: Piper disconnect failed: {error}")
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XenseSDK dual tactile reflex controller (sense-only by default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--left-sn", default=LEFT_SERIAL)
    parser.add_argument("--right-sn", default=RIGHT_SERIAL)
    parser.add_argument("--no-flip-right", action="store_true")
    parser.add_argument("--infer-mode", choices=("normal", "small", "fast"), default="normal")
    parser.add_argument(
        "--infer-device", choices=("auto", "cuda"), default="auto",
        help="prefer CUDA or require both Xense engines to actually use CUDA",
    )
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--can", default="can0")
    parser.add_argument(
        "--enable-piper", action="store_true",
        help="explicitly enable Piper motion; omitted means sense-only",
    )
    parser.add_argument(
        "--confirm-force-thresholds",
        action="store_true",
        help="confirm fn-stop/fn-protect were calibrated before enabling Piper",
    )
    parser.add_argument(
        "--confirm-axis-mapping",
        action="store_true",
        help="confirm physical X/Y mapping and all four per-sensor axis signs",
    )
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--calibration-wait", type=float, default=1.5)
    parser.add_argument("--baseline-frames", type=int, default=30)
    parser.add_argument("--frame-timeout", type=float, default=1.0)
    parser.add_argument("--sensor-open-timeout", type=float, default=60.0)
    parser.add_argument("--stale-timeout", type=float, default=0.5)
    parser.add_argument("--max-pair-skew", type=float, default=0.2)
    parser.add_argument("--normal-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument(
        "--left-sx-sign", type=float, choices=(-1.0, 1.0), default=LEFT_SX_SIGN
    )
    parser.add_argument(
        "--left-sy-sign", type=float, choices=(-1.0, 1.0), default=LEFT_SY_SIGN
    )
    parser.add_argument(
        "--right-sx-sign", type=float, choices=(-1.0, 1.0), default=RIGHT_SX_SIGN
    )
    parser.add_argument(
        "--right-sy-sign", type=float, choices=(-1.0, 1.0), default=RIGHT_SY_SIGN
    )
    parser.add_argument("--contact-force", type=float, default=0.01)
    parser.add_argument("--noise-multiplier", type=float, default=5.0)
    parser.add_argument("--fn-stop", type=float, default=0.10)
    parser.add_argument("--fn-protect", type=float, default=5.0)
    parser.add_argument(
        "--slip-y-th", "--sy-th", dest="slip_y_th", type=float, default=0.20,
        help="physical-vertical marker-speed trigger; --sy-th is a legacy alias",
    )
    parser.add_argument(
        "--slip-y-quiet", "--sy-quiet", dest="slip_y_quiet", type=float, default=0.10,
        help="quiet vertical speed for CoP release; --sy-quiet is a legacy alias",
    )
    parser.add_argument(
        "--cop-vertical-dec-th", "--cop-dec-th", dest="cop_vertical_dec_th",
        type=float, default=-10.0,
        help="physical-vertical CoP decrease threshold; --cop-dec-th is a legacy alias",
    )
    parser.add_argument("--init-effort", type=int, default=1200)
    parser.add_argument("--max-effort", type=int, default=5000)
    parser.add_argument("--min-stroke-um", type=int, default=20_000)
    parser.add_argument("--max-stroke-um", type=int, default=100_000)
    parser.add_argument("--close-step-um", type=int, default=150)
    parser.add_argument("--slip-effort-step", type=int, default=300)
    parser.add_argument("--slip-pos-step", type=int, default=80)
    parser.add_argument("--max-pos-adjust-um", type=int, default=3000)
    parser.add_argument("--squeeze-effort-step", type=int, default=200)
    parser.add_argument("--squeeze-pos-step", type=int, default=60)
    parser.add_argument("--protect-effort-step", type=int, default=400)
    parser.add_argument("--protect-pos-step", type=int, default=100)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.enable_piper and not args.confirm_force_thresholds:
        raise ValueError(
            "--enable-piper requires --confirm-force-thresholds after unloaded/load calibration"
        )
    if args.enable_piper and not args.confirm_axis_mapping:
        raise ValueError(
            "--enable-piper requires --confirm-axis-mapping after checking both sensor axes"
        )
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.baseline_frames < 2:
        raise ValueError("--baseline-frames must be at least 2")
    if not math.isfinite(args.calibration_wait) or args.calibration_wait < 0:
        raise ValueError("--calibration-wait must be finite and non-negative")
    for name in (
        "frame_timeout", "sensor_open_timeout", "stale_timeout", "max_pair_skew", "contact_force",
        "noise_multiplier", "fn_stop", "fn_protect", "slip_y_th", "slip_y_quiet",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.contact_force >= args.fn_stop:
        raise ValueError("--contact-force must be smaller than --fn-stop")
    if args.slip_y_quiet >= args.slip_y_th:
        raise ValueError("--slip-y-quiet must be smaller than --slip-y-th")
    if not math.isfinite(args.cop_vertical_dec_th) or args.cop_vertical_dec_th >= 0:
        raise ValueError("--cop-vertical-dec-th must be finite and negative")
    if not args.headless and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise ValueError("no graphical display is available; use --headless")


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    bootstrap_onnxruntime_gpu(args.infer_device)
    params = ReflexParams(
        hz=args.hz,
        max_stroke_um=args.max_stroke_um,
        min_stroke_um=args.min_stroke_um,
        close_step_um=args.close_step_um,
        init_effort=args.init_effort,
        max_effort=args.max_effort,
        fn_stop_threshold=args.fn_stop,
        slip_y_threshold=args.slip_y_th,
        slip_effort_step=args.slip_effort_step,
        slip_pos_step_um=args.slip_pos_step,
        max_pos_adjust_um=args.max_pos_adjust_um,
        slip_y_quiet_threshold=args.slip_y_quiet,
        cop_vertical_decrease_th_px=args.cop_vertical_dec_th,
        squeeze_effort_step=args.squeeze_effort_step,
        squeeze_pos_step_um=args.squeeze_pos_step,
        fn_protect_limit=args.fn_protect,
        protect_effort_step=args.protect_effort_step,
        protect_pos_step_um=args.protect_pos_step,
    )
    metric_params = XenseMetricParams(
        baseline_frames=args.baseline_frames,
        normal_sign=args.normal_sign,
        contact_force_threshold=args.contact_force,
        noise_multiplier=args.noise_multiplier,
    )
    out_csv = args.out_csv or Path("logs") / f"tactile_reflex_xsense_{now_str()}.csv"
    print("=" * 80)
    print("tactile_reflex_xsense: native Xense force + marker reflex")
    print(
        f"LEFT={args.left_sn} RIGHT={args.right_sn} infer={args.infer_mode} "
        f"device={args.infer_device}"
    )
    print(f"mode={'PIPER ENABLED' if args.enable_piper else 'SENSE ONLY'}")
    if not args.enable_piper:
        print("NOTE: [s] runs CLOSING as a simulation; it cannot move the Piper gripper")
    print(
        f"Fn stop/protect={params.fn_stop_threshold:g}/{params.fn_protect_limit:g} "
        f"SDK units; SlipY trigger/quiet={params.slip_y_threshold:g}/"
        f"{params.slip_y_quiet_threshold:g} px/source-frame"
    )
    print(
        "axes: physical X(horizontal)<-image y, physical Y(vertical)<-image x; "
        f"signs L(X,Y)=({args.left_sx_sign:+.0f},{args.left_sy_sign:+.0f}) "
        f"R(X,Y)=({args.right_sx_sign:+.0f},{args.right_sy_sign:+.0f})"
    )
    print(f"CSV={out_csv.resolve()}")
    print("=" * 80)
    controller: Optional[DualXenseReflexController] = None
    try:
        controller = DualXenseReflexController(
            left_serial=args.left_sn,
            right_serial=args.right_sn,
            can_iface=args.can,
            enable_piper=args.enable_piper,
            params=params,
            metric_params=metric_params,
            out_csv=out_csv,
            infer_mode=args.infer_mode,
            infer_device=args.infer_device,
            flip_right=not args.no_flip_right,
            headless=args.headless,
            frame_timeout_s=args.frame_timeout,
            stale_timeout_s=args.stale_timeout,
            max_pair_skew_s=args.max_pair_skew,
            sensor_open_timeout_s=args.sensor_open_timeout,
            left_sx_sign=args.left_sx_sign,
            left_sy_sign=args.left_sy_sign,
            right_sx_sign=args.right_sx_sign,
            right_sy_sign=args.right_sy_sign,
        )
        controller.auto_init(args.calibration_wait)
        if args.auto_start:
            controller.start_grasping()
        elif not args.headless:
            print("\nKeys: [s]=start  [r]=reset/recalibrate  [q/Esc]=quit")
        frame_count = 0
        last_print = 0.0
        while args.max_frames is None or frame_count < args.max_frames:
            visual, result, (avg_hz, frame_ms) = controller.update()
            frame_count += 1
            if not args.headless and visual is not None:
                cv2.imshow(controller.window_name, visual)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    controller.reset(args.calibration_wait)
                elif key in (ord("s"), ord("S")):
                    controller.start_grasping()
            now = time.monotonic()
            if args.headless and now - last_print >= 0.5:
                last_print = now
                print(
                    f"[{STATE_NAMES[result['state']]:8s}] {avg_hz:5.1f}Hz "
                    f"Fn L/R={result['L']['Fn']:.4g}/{result['R']['Fn']:.4g} "
                    f"Sy(V/imgX) L/R={result['L']['Sy']:+.4f}/{result['R']['Sy']:+.4f} "
                    f"SlipY={result['L']['SlipY']:.4f}/{result['R']['SlipY']:.4f} "
                    f"signed={result['L']['SlipY_signed']:+.4f}/"
                    f"{result['R']['SlipY_signed']:+.4f} "
                    f"pos={result['pos_um']/1000:.1f}mm effort={result['effort']} "
                    f"action={result['action'] or '---'}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if controller is not None:
            controller.cleanup(open_gripper=True)
    return 0


def main() -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args())
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
