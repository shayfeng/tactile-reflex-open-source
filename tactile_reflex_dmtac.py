"""
DM-Tac realtime tactile monitor v1
==================================

This is a DM-Tac version of tactile_realtime_v3.py.

It opens two DM-Tac sensors, reads:
  - infer image
  - depth
  - deformation flow
  - shear flow
  - 6D force / torque
  - contact area

Hotkeys:
  r - reset both sensors, keep sensor surfaces free of contact
  c - recalibrate force/depth/flow baseline, keep surfaces free of contact
  s - start gripper closing (if enabled)
  k - screenshot
  p - toggle time-series panel
  q / Esc - quit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import suppress
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from platform import system
from typing import Any


def configure_vendor_dll_paths() -> None:
    """Make bundled CUDA/cuDNN DLLs visible before importing dmrobotics."""
    script_dir = Path(__file__).resolve().parent
    configured_root = os.environ.get("DMTAC_SDK_ROOT", "").strip()
    candidates = [
        Path(configured_root) if configured_root else Path("/does/not/exist"),
        script_dir / "icra" / "icra" / "DM-Tac",
        script_dir / "DM-Tac",
        Path(r"D:\DM_research\icra\icra\DM-Tac"),
    ]
    dm_tac_root = next((p for p in candidates if p.exists()), candidates[0])

    dll_dirs = [
        dm_tac_root / "V1.2.11.N_20260421111033" / "win" / "sdk" / "depends" / "CUDA" / "bin",
        dm_tac_root / "V1.2.11.N_20260421111033" / "win" / "sdk" / "dmpython310",
        dm_tac_root / "V1.2.11.N_20260421111033" / "win" / "sdk" / "dmpython310" / "Library" / "bin",
    ]

    existing = [str(path) for path in dll_dirs if path.exists()]
    for path in existing:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(existing + [old_path])


def configure_python_paths() -> None:
    """Make the bundled dmrobotics package importable in a local conda env."""
    script_dir = Path(__file__).resolve().parent
    configured_root = os.environ.get("DMTAC_SDK_ROOT", "").strip()
    root = Path(configured_root) if configured_root else script_dir
    sdk_candidates = (
        root / "DM-Tac-SDK" / "SDK_Publish_1.2.10",
        root / "SDK_Publish_1.2.10",
        root,
    )
    sdk_root = next((path for path in sdk_candidates if path.exists()), sdk_candidates[0])
    bundled_site = (
        Path(os.environ["DMTAC_PYTHON_SITE"])
        if os.environ.get("DMTAC_PYTHON_SITE")
        else script_dir / "linux" / "sdk" / "dmpython310" / "lib" / "python3.10" / "site-packages"
    )
    paths = [sdk_root]
    if sys.version_info[:2] == (3, 10):
        paths.append(bundled_site)
    for path in paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def configure_repo_paths() -> None:
    """Make the sibling pyAgxArm package importable from the DM-Tac script."""
    script_dir = Path(__file__).resolve().parent
    repo_root = Path(os.environ.get("DMTAC_REPO_ROOT", script_dir.parent))
    package_root = repo_root / "pyAgxArm"
    for path in (package_root, repo_root):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


configure_vendor_dll_paths()
configure_python_paths()
configure_repo_paths()

import cv2
import numpy as np

try:
    from dmrobotics import Mode, Sensor, SensorOptions, listConnectedDevIDs
    from dmrobotics.utils import put_arrows_on_image
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
    _DMTAC_IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - depends on vendor SDK
    Mode = Sensor = SensorOptions = listConnectedDevIDs = None
    put_arrows_on_image = None
    AgxArmFactory = ArmModel = create_agx_arm_config = None
    _DMTAC_IMPORT_ERROR = error


HISTORY_LEN = 300
PLOT_W = 420
PLOT_H = 220
DEPTH_VIEW_W = 384
DEPTH_VIEW_H = 520
INFO_W = 430

GRIPPER_PERCENT_MIN = 2
GRIPPER_PERCENT_MAX = 100
GRIPPER_DEFAULT_FORCE_N = 1.0
GRIPPER_DEFAULT_MAX_WIDTH = 0.1
GRIPPER_DEFAULT_PUBLISH_HZ = 20.0
GRIPPER_DEFAULT_SETTLE_TIMEOUT = 2.0
GRIPPER_DEFAULT_TOLERANCE_PERCENT = 1.5
GRIPPER_DEFAULT_STEP_PERCENT = 1


def parse_dev_id(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def payload_img(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    if hasattr(payload, "img"):
        return payload.img
    return payload


def as_gray_u8(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def gray_to_bgr(img: np.ndarray | None, fallback_shape=(288, 384)) -> np.ndarray:
    gray = as_gray_u8(img)
    if gray is None:
        h, w = fallback_shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(out, "NO IMAGE", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
        return out
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def depth_to_heat(depth: np.ndarray | None, fallback_shape=(288, 384), scale=80.0) -> np.ndarray:
    if depth is None:
        h, w = fallback_shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    d = np.asarray(depth, dtype=np.float32)
    d_u8 = (d * scale).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)


def flow_stats(flow: np.ndarray | None) -> tuple[float, float]:
    if flow is None:
        return 0.0, 0.0
    arr = np.asarray(flow, dtype=np.float32)
    if arr.ndim < 3 or arr.shape[-1] < 2:
        return 0.0, 0.0
    mag = np.linalg.norm(arr[..., :2], axis=2)
    return float(np.mean(mag)), float(np.percentile(mag, 95))


def shear_component_stats(shear: np.ndarray | None) -> dict[str, float]:
    if shear is None:
        return {
            "x_mean": 0.0,
            "y_mean": 0.0,
            "x_abs_mean": 0.0,
            "y_abs_mean": 0.0,
            "x_p95": 0.0,
            "y_p95": 0.0,
            "mag_mean": 0.0,
            "mag_p95": 0.0,
        }
    arr = np.asarray(shear, dtype=np.float32)
    if arr.ndim < 3 or arr.shape[-1] < 2:
        return shear_component_stats(None)
    sx = arr[..., 0]
    sy = arr[..., 1]
    mag = np.linalg.norm(arr[..., :2], axis=2)
    return {
        "x_mean": float(np.mean(sx)),
        "y_mean": float(np.mean(sy)),
        "x_abs_mean": float(np.mean(np.abs(sx))),
        "y_abs_mean": float(np.mean(np.abs(sy))),
        "x_p95": float(np.percentile(np.abs(sx), 95)),
        "y_p95": float(np.percentile(np.abs(sy), 95)),
        "mag_mean": float(np.mean(mag)),
        "mag_p95": float(np.percentile(mag, 95)),
    }


def flow_to_vis(flow: np.ndarray | None, fallback_shape=(288, 384), step=16, scale=20.0) -> np.ndarray:
    if flow is None:
        h, w = fallback_shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.asarray(flow)
    canvas = np.zeros(arr.shape[:2] + (3,), dtype=np.uint8)
    return put_arrows_on_image(canvas, arr, step=step, scale=scale)


def titled_tile(title: str, image: np.ndarray, title_h: int = 24) -> np.ndarray:
    title_bar = np.zeros((title_h, image.shape[1], 3), dtype=np.uint8)
    title_bar[:] = (18, 18, 18)
    cv2.putText(title_bar, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([title_bar, image])


def centered_tile(tile: np.ndarray, width: int, fill=(0, 0, 0)) -> np.ndarray:
    if tile.shape[1] >= width:
        return tile
    out = np.zeros((tile.shape[0], width, 3), dtype=np.uint8)
    out[:] = fill
    x0 = (width - tile.shape[1]) // 2
    out[:, x0:x0 + tile.shape[1]] = tile
    return out


def put_label_value(
    img: np.ndarray,
    label: str,
    value: str,
    y: int,
    color=(225, 225, 225),
    value_color=(255, 255, 255),
) -> None:
    cv2.putText(img, label, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(img, value, (145, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, value_color, 1, cv2.LINE_AA)


def normalize_vector(value: Any, n: int | None = None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = value[1]
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if n is not None:
        arr = arr[:n]
    return [float(x) for x in arr]


def safe_scalar(value: Any) -> float:
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            value = value[1]
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(arr[0])
    except Exception:
        return 0.0


def resolve_default_comm() -> tuple[str, str]:
    plat = system()
    if plat == "Linux":
        return "socketcan", "can0"
    if plat == "Darwin":
        return "slcan", "/dev/ttyACM0"
    if plat == "Windows":
        return "agx_cando", "0"
    raise RuntimeError(f"Unsupported platform: {plat}")


def create_cfg(robot_name: str, firmware: str, interface: str, channel: str, bitrate: int):
    return create_agx_arm_config(
        robot=getattr(ArmModel, robot_name.upper()),
        firmeware_version=firmware,
        interface=interface,
        channel=channel,
        bitrate=bitrate,
    )


def clamp_percent(value: int) -> int:
    return max(GRIPPER_PERCENT_MIN, min(GRIPPER_PERCENT_MAX, value))


def percent_to_width(percent: int, max_width: float) -> float:
    return max_width * clamp_percent(percent) / 100.0


def width_to_percent(width: float, max_width: float) -> float:
    if max_width <= 0:
        return 0.0
    return max(0.0, min(100.0, width / max_width * 100.0))


def read_feedback(end_effector):
    return end_effector.get_gripper_status(), end_effector.get_gripper_ctrl_states()


def wait_for_status(end_effector, timeout: float = 2.0, poll_interval: float = 0.1):
    start = time.monotonic()
    while True:
        status = end_effector.get_gripper_status()
        if status is not None:
            return status
        if time.monotonic() - start >= timeout:
            return None
        time.sleep(poll_interval)


def get_max_width(end_effector, fallback: float) -> float:
    param = None
    with suppress(Exception):
        param = end_effector.get_gripper_teaching_pendant_param(timeout=1.0, min_interval=0.0)
    if param is None:
        return fallback
    raw = float(getattr(param.msg, "max_range_config", fallback))
    if raw in (0.07, 0.1):
        return raw
    return fallback


def print_health(status) -> None:
    if status is None:
        print("gripper status unavailable")
        return
    foc = status.msg.foc_status
    if not foc.driver_enable_status:
        print("warning: driver_enable_status=False")
    if not foc.homing_status:
        print("warning: homing_status=False")


def format_status(status, ctrl, command_percent: int, max_width: float) -> str:
    if status is None and ctrl is None:
        return f"cmd={command_percent:>3d}% status=unknown ctrl=unknown"

    parts = [f"cmd={command_percent:>3d}%"]
    if status is not None:
        value = float(status.msg.value)
        force = float(getattr(status.msg, "force", 0.0))
        mode = getattr(status.msg, "mode", "width")
        foc = status.msg.foc_status
        parts.append(
            "status="
            f"{width_to_percent(value, max_width):>6.1f}%"
            f"(width={value:.4f}m, force={force:.2f}N, mode={mode}, "
            f"en={foc.driver_enable_status}, home={foc.homing_status})"
        )
    else:
        parts.append("status=unknown")

    if ctrl is not None:
        value = float(ctrl.msg.value)
        force = float(getattr(ctrl.msg, "force", 0.0))
        parts.append(
            f"ctrl={width_to_percent(value, max_width):>6.1f}%"
            f"(width={value:.4f}m, force={force:.2f}N)"
        )
    else:
        parts.append("ctrl=unknown")
    return " ".join(parts)


def wait_until_percent(
    end_effector,
    target_percent: int,
    max_width: float,
    tolerance_percent: float,
    timeout: float,
    publish_hz: float,
    force: float,
):
    deadline = time.monotonic() + timeout
    interval = 1.0 / publish_hz if publish_hz > 0 else 0.05
    last_status = None
    last_ctrl = None
    target_width = percent_to_width(target_percent, max_width)

    end_effector.move_gripper_m(value=target_width, force=force)
    while True:
        status, ctrl = read_feedback(end_effector)
        last_status, last_ctrl = status, ctrl
        if status is not None:
            actual_percent = width_to_percent(float(status.msg.value), max_width)
            if abs(actual_percent - target_percent) <= tolerance_percent:
                return last_status, last_ctrl, True

        if time.monotonic() >= deadline:
            return last_status, last_ctrl, False

        end_effector.move_gripper_m(value=target_width, force=force)
        time.sleep(interval)


def send_percent(
    end_effector,
    percent: int,
    max_width: float,
    force: float,
    tolerance_percent: float,
    timeout: float,
    publish_hz: float,
):
    target_percent = clamp_percent(percent)
    return wait_until_percent(
        end_effector=end_effector,
        target_percent=target_percent,
        max_width=max_width,
        tolerance_percent=tolerance_percent,
        timeout=timeout,
        publish_hz=publish_hz,
        force=force,
    )


@dataclass
class DMTacFrame:
    fid: int = -1
    status: int = -1
    raw: np.ndarray | None = None
    infer: np.ndarray | None = None
    depth: np.ndarray | None = None
    deformation: np.ndarray | None = None
    shear: np.ndarray | None = None
    force: list[float] | None = None
    contact_area: float = 0.0
    force_norm: float = 0.0
    shear_force: float = 0.0
    torque_norm: float = 0.0
    depth_mean: float = 0.0
    depth_max: float = 0.0
    depth_delta: np.ndarray | None = None
    depth_delta_mean: float = 0.0
    depth_delta_max: float = 0.0
    deformation_mean: float = 0.0
    shear_flow_mean: float = 0.0
    shear_flow_p95: float = 0.0
    shear_x_mean: float = 0.0
    shear_y_mean: float = 0.0
    shear_x_abs_mean: float = 0.0
    shear_y_abs_mean: float = 0.0
    shear_x_p95: float = 0.0
    shear_y_p95: float = 0.0
    contact: bool = False
    slip_like: bool = False


@dataclass
class DMTacCalibration:
    force_bias: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    force_noise_p95: float = 0.0
    depth_baseline: np.ndarray | None = None
    depth_noise_p95: float = 0.0
    deformation_noise_p95: float = 0.0
    shear_noise_p95: float = 0.0
    shear_x_noise_p95: float = 0.0
    shear_y_noise_p95: float = 0.0
    contact_area_bias: float = 0.0
    ready: bool = False


@dataclass
class DMTacRuntime:
    name: str
    dev_id: int | str
    sensor: Sensor
    last_fid: int = -1
    fps_count: int = 0
    fps: float = 0.0
    last_fps_ts: float = field(default_factory=time.time)
    last_frame: DMTacFrame | None = None
    calibration: DMTacCalibration = field(default_factory=DMTacCalibration)
    ema: dict[str, float] = field(default_factory=lambda: {
        "Fn": 0.0,
        "Sx": 0.0,
        "Sy": 0.0,
        "Sm": 0.0,
        "Depth": 0.0,
        "Area": 0.0,
    })
    contact_latched: bool = False
    slip_latched: bool = False
    contact_on_count: int = 0
    contact_off_count: int = 0
    slip_on_count: int = 0
    slip_off_count: int = 0
    contact_transitions: int = 0
    slip_transitions: int = 0
    hist: dict[str, deque] = field(default_factory=lambda: {
        "Fn": deque(maxlen=HISTORY_LEN),
        "Sx": deque(maxlen=HISTORY_LEN),
        "Sy": deque(maxlen=HISTORY_LEN),
        "Sm": deque(maxlen=HISTORY_LEN),
        "Depth": deque(maxlen=HISTORY_LEN),
    })


class DMTacRealtimeVis:
    def __init__(
        self,
        dev_ids: list[int | str] | None = None,
        backend: str = "cpu",
        mode: str = "standard",
        max_fps: int = 120,
        timeout_ms: int = 50,
        arrow_step: int = 16,
        depth_scale: float = 80.0,
        min_contact_area: float = 1.0,
        min_force_norm: float = 0.05,
        min_depth_delta: float = 0.004,
        slip_flow_threshold: float = 0.12,
        calibration_frames: int = 45,
        ema_alpha: float = 0.15,
        use_ema: bool = False,
        hysteresis_ratio: float = 0.55,
        stable_frames: int = 6,
        force_plot_max: float = 0.2,
        shear_plot_max: float = 0.05,
        depth_plot_max: float = 0.2,
        auto_plot_scale: bool = True,
        target_width: int = 1500,
        show_camera: bool = False,
        show_raw: bool = False,
        raw_depth_only: bool = False,
        show_flow_images: bool = False,
        initial_show_plot: bool = False,
        parallel_reads: bool = True,
        no_display: bool = False,
        debug_metrics: bool = False,
        duration_s: float = 0.0,
        save_last_frame: str = "",
        with_gripper: bool = True,
        gripper_robot: str = "nero",
        gripper_firmware: str = "v120",
        gripper_interface: str | None = None,
        gripper_channel: str | None = None,
        gripper_bitrate: int = 1_000_000,
        gripper_max_width: float = GRIPPER_DEFAULT_MAX_WIDTH,
        gripper_force: float = GRIPPER_DEFAULT_FORCE_N,
        gripper_open_percent: int = GRIPPER_PERCENT_MAX,
        gripper_close_percent: int = GRIPPER_PERCENT_MIN,
        gripper_publish_hz: float = GRIPPER_DEFAULT_PUBLISH_HZ,
        gripper_tolerance_percent: float = GRIPPER_DEFAULT_TOLERANCE_PERCENT,
        gripper_settle_timeout: float = GRIPPER_DEFAULT_SETTLE_TIMEOUT,
        gripper_step_percent: int = GRIPPER_DEFAULT_STEP_PERCENT,
        auto_start_gripper: bool = False,
        gripper_start_signal_file: str = "",
        slip_retry_step_percent: int = 2,
        slip_retry_force_scale: float = 1.15,
        slip_retry_max_attempts: int = 2,
        event_file: str = "",
    ):
        self.backend = backend
        self.mode = Mode.HIGH if mode == "high" else Mode.STANDARD
        self.max_fps = max_fps
        self.timeout_ms = timeout_ms
        self.arrow_step = arrow_step
        self.depth_scale = depth_scale
        self.min_contact_area = min_contact_area
        self.min_force_norm = min_force_norm
        self.min_depth_delta = min_depth_delta
        self.slip_flow_threshold = slip_flow_threshold
        self.calibration_frames = calibration_frames
        self.ema_alpha = ema_alpha
        self.use_ema = use_ema
        self.hysteresis_ratio = hysteresis_ratio
        self.stable_frames = stable_frames
        self.plot_max = {
            "Fn": force_plot_max,
            "Sx": shear_plot_max,
            "Sy": shear_plot_max,
            "Sm": shear_plot_max,
            "Depth": depth_plot_max,
        }
        self.auto_plot_scale = auto_plot_scale
        self.target_width = target_width
        self.raw_depth_only = raw_depth_only
        self.show_raw = show_raw or raw_depth_only
        self.show_camera = show_camera or self.show_raw or show_flow_images
        self.show_flow_images = show_flow_images
        self.parallel_reads = parallel_reads
        self.no_display = no_display
        self.debug_metrics = debug_metrics
        self.duration_s = duration_s
        self.save_last_frame = save_last_frame
        self.show_plot = initial_show_plot
        self.last_final = None
        self.last_debug_ts = time.time()
        self.with_gripper = with_gripper
        self.gripper_robot_name = gripper_robot
        self.gripper_firmware = gripper_firmware
        self.gripper_interface = gripper_interface
        self.gripper_channel = gripper_channel
        self.gripper_bitrate = gripper_bitrate
        self.gripper_max_width = gripper_max_width
        self.gripper_force = gripper_force
        self.gripper_open_percent = clamp_percent(gripper_open_percent)
        self.gripper_close_percent = clamp_percent(gripper_close_percent)
        self.gripper_publish_hz = gripper_publish_hz
        self.gripper_tolerance_percent = gripper_tolerance_percent
        self.gripper_settle_timeout = gripper_settle_timeout
        self.gripper_step_percent = max(1, int(gripper_step_percent))
        self.auto_start_gripper = auto_start_gripper
        self.gripper_start_signal_file = Path(gripper_start_signal_file).expanduser() if gripper_start_signal_file else None
        self.gripper_start_pending = bool(auto_start_gripper or self.gripper_start_signal_file is not None)
        self.slip_retry_step_percent = max(1, int(slip_retry_step_percent))
        self.slip_retry_force_scale = max(1.0, float(slip_retry_force_scale))
        self.slip_retry_max_attempts = max(0, int(slip_retry_max_attempts))
        self.event_file = Path(event_file).expanduser() if event_file else None
        self.gripper_robot = None
        self.gripper_effector = None
        self.gripper_target_percent = self.gripper_open_percent
        self.gripper_active = False
        self.gripper_contact_hold = False
        self.gripper_retry_active = False
        self.gripper_retry_count = 0
        self._last_slip_seen = False
        self.gripper_last_command_ts = 0.0
        self.gripper_last_status = None
        self.gripper_last_ctrl = None
        self.gripper_last_known_percent = float(self.gripper_open_percent)
        self._last_event_payload: dict[str, Any] | None = None

        if dev_ids is None:
            detected = listConnectedDevIDs() or []
            print(f"[Detect] {detected}")
            dev_ids = [d["device_index"] for d in detected if "device_index" in d]

        if len(dev_ids) < 2:
            raise RuntimeError(f"Need two DM-Tac sensors, got {dev_ids}")

        self.runtimes = [
            self._open_sensor("S1", dev_ids[0]),
            self._open_sensor("S2", dev_ids[1]),
        ]
        self.executor = ThreadPoolExecutor(max_workers=len(self.runtimes)) if self.parallel_reads else None

        if self.with_gripper:
            self._init_gripper()

        self.calibrate_all(reset_sensor=False)

    def _open_sensor(self, name: str, dev_id: int | str) -> DMTacRuntime:
        opt = SensorOptions(
            dev_id=dev_id,
            backend=self.backend,
            mode=self.mode,
            show_fps=False,
            max_fps=self.max_fps,
            enable_raw=self.show_raw,
            enable_deformation=self.show_flow_images,
            enable_depth=True,
            enable_shear=not self.raw_depth_only,
            enable_force=not self.raw_depth_only,
        )
        sensor = Sensor(opt)
        print(f"[{name}] opened dev_id={dev_id}, id={sensor.getDevID()}, status={sensor.getDevStatus()}")
        return DMTacRuntime(name=name, dev_id=dev_id, sensor=sensor)

    def _init_gripper(self) -> None:
        interface, channel = resolve_default_comm()
        if self.gripper_interface:
            interface = self.gripper_interface
        if self.gripper_channel:
            channel = self.gripper_channel

        cfg = create_cfg(
            self.gripper_robot_name,
            self.gripper_firmware,
            interface,
            channel,
            self.gripper_bitrate,
        )
        print(
            f"[Gripper] config robot={self.gripper_robot_name} firmware={self.gripper_firmware} "
            f"{interface}:{channel}"
        )
        self.gripper_robot = AgxArmFactory.create_arm(cfg)
        self.gripper_effector = self.gripper_robot.init_effector(self.gripper_robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.gripper_robot.connect()
        time.sleep(0.2)

        enabled = False
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self.gripper_robot.enable():
                enabled = True
                break
            time.sleep(0.1)
        print(
            f"[Gripper] robot_ok={self.gripper_robot.is_ok()} effector_ok={self.gripper_effector.is_ok()} "
            f"enabled={enabled}"
        )
        if not enabled:
            print("[Gripper] warning: robot.enable() did not confirm within timeout")

        self.gripper_max_width = get_max_width(self.gripper_effector, self.gripper_max_width)
        print(
            f"[Gripper] max_width={self.gripper_max_width:.3f}m "
            f"range={GRIPPER_PERCENT_MIN}-{GRIPPER_PERCENT_MAX}"
        )
        startup_status = wait_for_status(self.gripper_effector, timeout=2.0)
        print_health(startup_status)

        status, ctrl, settled = send_percent(
            self.gripper_effector,
            self.gripper_open_percent,
            self.gripper_max_width,
            self.gripper_force,
            self.gripper_tolerance_percent,
            self.gripper_settle_timeout,
            self.gripper_publish_hz,
        )
        self.gripper_last_status = status
        self.gripper_last_ctrl = ctrl
        if status is not None:
            self.gripper_last_known_percent = width_to_percent(float(status.msg.value), self.gripper_max_width)
        print(format_status(status, ctrl, self.gripper_open_percent, self.gripper_max_width))
        if not settled:
            print("[Gripper] warning: startup open did not settle before timeout")
        self.gripper_target_percent = self.gripper_open_percent
        self.gripper_active = False
        self.gripper_contact_hold = False

    def _gripper_send_target(self, target_percent: int, force: float | None = None) -> None:
        if self.gripper_effector is None:
            return
        target_percent = clamp_percent(target_percent)
        self.gripper_target_percent = target_percent
        self.gripper_effector.move_gripper_m(
            value=percent_to_width(target_percent, self.gripper_max_width),
            force=self.gripper_force if force is None else force,
        )
        self.gripper_last_command_ts = time.monotonic()

    def _gripper_start_closing(self) -> None:
        if self.gripper_effector is None:
            print("[Gripper] disabled; ignore start close")
            return
        self.gripper_active = True
        self.gripper_contact_hold = False
        self.gripper_retry_active = False
        self.gripper_retry_count = 0
        self._last_slip_seen = False
        status, ctrl = read_feedback(self.gripper_effector)
        self.gripper_last_status = status
        self.gripper_last_ctrl = ctrl
        if status is not None:
            self.gripper_last_known_percent = width_to_percent(float(status.msg.value), self.gripper_max_width)
            self.gripper_target_percent = clamp_percent(int(round(self.gripper_last_known_percent)))
        else:
            self.gripper_target_percent = self.gripper_open_percent
        self.gripper_last_command_ts = 0.0
        print(
            f"[Gripper] start closing: step {self.gripper_step_percent}% toward {self.gripper_close_percent}% "
            f"(stop on tactile contact)"
        )

    def _maybe_start_gripper(self) -> None:
        if not self.gripper_start_pending or self.gripper_effector is None:
            return
        if self.gripper_start_signal_file is not None and not self.gripper_start_signal_file.exists():
            return
        self.gripper_start_pending = False
        self._gripper_start_closing()

    def _gripper_hold_current(self, reason: str) -> None:
        if self.gripper_effector is None:
            return
        status, ctrl = read_feedback(self.gripper_effector)
        current_percent = self.gripper_target_percent
        if status is not None:
            current_percent = clamp_percent(int(round(width_to_percent(float(status.msg.value), self.gripper_max_width))))
            self.gripper_last_known_percent = width_to_percent(float(status.msg.value), self.gripper_max_width)
        self._gripper_send_target(current_percent)
        self.gripper_active = False
        self.gripper_contact_hold = True
        self.gripper_retry_active = False
        self.gripper_last_status = status
        self.gripper_last_ctrl = ctrl
        print(f"[Gripper] {reason}: hold at {current_percent}%")
        if status is not None and ctrl is not None:
            print(format_status(status, ctrl, current_percent, self.gripper_max_width))

    def _gripper_start_retry(self, reason: str) -> None:
        if self.gripper_effector is None:
            return
        if self.gripper_retry_count >= self.slip_retry_max_attempts:
            print(f"[Gripper] {reason}: retry limit reached ({self.gripper_retry_count})")
            return
        status, ctrl = read_feedback(self.gripper_effector)
        current_percent = self.gripper_target_percent
        if status is not None:
            current_percent = clamp_percent(int(round(width_to_percent(float(status.msg.value), self.gripper_max_width))))
            self.gripper_last_known_percent = width_to_percent(float(status.msg.value), self.gripper_max_width)
        retry_percent = max(self.gripper_close_percent, current_percent - self.slip_retry_step_percent)
        retry_force = self.gripper_force * self.slip_retry_force_scale
        self.gripper_retry_active = True
        self.gripper_contact_hold = False
        self.gripper_active = False
        self.gripper_retry_count += 1
        self._gripper_send_target(retry_percent, force=retry_force)
        self.gripper_last_status = status
        self.gripper_last_ctrl = ctrl
        print(
            f"[Gripper] {reason}: retry {self.gripper_retry_count}/{self.slip_retry_max_attempts} "
            f"-> {retry_percent}% force={retry_force:.2f}"
        )
        if status is not None and ctrl is not None:
            print(format_status(status, ctrl, retry_percent, self.gripper_max_width))

    def _gripper_update_from_contact(self, contact_detected: bool) -> None:
        if self.gripper_effector is None:
            return

        if self.gripper_retry_active:
            now = time.monotonic()
            interval = 1.0 / self.gripper_publish_hz if self.gripper_publish_hz > 0 else 0.05
            if now - self.gripper_last_command_ts >= interval:
                self._gripper_send_target(
                    self.gripper_target_percent,
                    force=self.gripper_force * self.slip_retry_force_scale,
                )
            status, ctrl = read_feedback(self.gripper_effector)
            self.gripper_last_status = status
            self.gripper_last_ctrl = ctrl
            if status is not None:
                actual_percent = width_to_percent(float(status.msg.value), self.gripper_max_width)
                self.gripper_last_known_percent = actual_percent
                if contact_detected and abs(actual_percent - self.gripper_target_percent) <= self.gripper_tolerance_percent:
                    self.gripper_retry_active = False
                    self.gripper_contact_hold = True
                    print(f"[Gripper] retry settled at {actual_percent:.1f}%")
                    if status is not None and ctrl is not None:
                        print(format_status(status, ctrl, self.gripper_target_percent, self.gripper_max_width))
            return

        # AgxGripper requires move_gripper_m() to be published continuously.
        # A single command at contact is not enough while the arm is lifting;
        # keep re-sending the latched target until a retry or a new close starts.
        if self.gripper_contact_hold:
            status, ctrl = read_feedback(self.gripper_effector)
            self.gripper_last_status = status
            self.gripper_last_ctrl = ctrl
            actual_percent = self.gripper_last_known_percent
            if status is not None:
                actual_percent = width_to_percent(float(status.msg.value), self.gripper_max_width)
                self.gripper_last_known_percent = actual_percent
            # During an arm lift the object load can make the jaws open even
            # though the previous command is still being repeated. Treat a
            # lost contact or measurable opening drift as a tightening event.
            if (not contact_detected) or (
                actual_percent > self.gripper_target_percent + self.gripper_tolerance_percent
            ):
                self._gripper_start_retry("contact lost/width drift")
                return
            now = time.monotonic()
            interval = 1.0 / self.gripper_publish_hz if self.gripper_publish_hz > 0 else 0.05
            if now - self.gripper_last_command_ts >= interval:
                self._gripper_send_target(self.gripper_target_percent, force=self.gripper_force)
            return

        if not self.gripper_active:
            return

        if contact_detected:
            self._gripper_hold_current("contact")
            return

        now = time.monotonic()
        interval = 1.0 / self.gripper_publish_hz if self.gripper_publish_hz > 0 else 0.05
        if now - self.gripper_last_command_ts >= interval:
            next_target = max(self.gripper_close_percent, self.gripper_target_percent - self.gripper_step_percent)
            self._gripper_send_target(next_target)
            status, ctrl = read_feedback(self.gripper_effector)
            self.gripper_last_status = status
            self.gripper_last_ctrl = ctrl
            if status is not None:
                actual_percent = width_to_percent(float(status.msg.value), self.gripper_max_width)
                self.gripper_last_known_percent = actual_percent
                if abs(actual_percent - self.gripper_close_percent) <= self.gripper_tolerance_percent:
                    self.gripper_active = False
                    self.gripper_contact_hold = False
                    print(
                        f"[Gripper] reached minimum {self.gripper_close_percent}% "
                        f"without tactile contact"
                    )
                    print(format_status(status, ctrl, self.gripper_close_percent, self.gripper_max_width))

    def _gripper_update_from_slip(self, slip_detected: bool) -> None:
        if self.gripper_effector is None:
            return
        if slip_detected and not self._last_slip_seen:
            if self.gripper_contact_hold or self.gripper_active:
                self._gripper_start_retry("slip")
        if not slip_detected:
            self._last_slip_seen = False
        else:
            self._last_slip_seen = True

    def _gripper_summary_text(self) -> str:
        if self.gripper_effector is None:
            return "gripper=disabled"
        status = self.gripper_last_status
        ctrl = self.gripper_last_ctrl
        if status is None:
            status, ctrl = read_feedback(self.gripper_effector)
            self.gripper_last_status = status
            self.gripper_last_ctrl = ctrl
        state = "closing" if self.gripper_active else ("contact-hold" if self.gripper_contact_hold else "idle")
        actual_percent = self.gripper_last_known_percent
        force = 0.0
        if status is not None:
            actual_percent = width_to_percent(float(status.msg.value), self.gripper_max_width)
            self.gripper_last_known_percent = actual_percent
            force = float(getattr(status.msg, "force", 0.0))
        return (
            f"gripper={state} cmd={self.gripper_target_percent}% "
            f"actual={actual_percent:5.1f}% force={force:.2f}N"
        )

    def _gripper_short_text(self) -> str:
        return self._gripper_summary_text().replace("gripper=", "")

    def _gripper_shutdown(self) -> None:
        if self.gripper_effector is None:
            return
        with suppress(Exception):
            self._gripper_hold_current("shutdown")
        with suppress(Exception):
            self.gripper_robot.disconnect()

    def _event_state(self) -> dict[str, Any]:
        gripper_status = None
        gripper_force = 0.0
        if self.gripper_last_status is not None:
            with suppress(Exception):
                gripper_status = float(self.gripper_last_status.msg.value)
                gripper_force = float(getattr(self.gripper_last_status.msg, "force", 0.0))
        return {
            "ts": time.time(),
            "contact": None if not self.runtimes else any(rt.contact_latched for rt in self.runtimes),
            "slip": None if not self.runtimes else any(rt.slip_latched for rt in self.runtimes),
            "gripper_state": (
                "retry"
                if self.gripper_retry_active
                else ("closing" if self.gripper_active else ("hold" if self.gripper_contact_hold else "idle"))
            ),
            "gripper_cmd_percent": self.gripper_target_percent,
            "gripper_actual_percent": self.gripper_last_known_percent,
            "gripper_width_m": gripper_status,
            "gripper_force": gripper_force,
            "retry_count": self.gripper_retry_count,
        }

    def _write_event_file(self) -> None:
        if self.event_file is None:
            return
        payload = self._event_state()
        if payload == self._last_event_payload:
            return
        self.event_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.event_file.with_suffix(self.event_file.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        tmp_path.replace(self.event_file)
        self._last_event_payload = payload

    def _capture_raw(self, rt: DMTacRuntime) -> DMTacFrame | None:
        sensor = rt.sensor
        status = sensor.getDevStatus()
        if status != 0:
            return DMTacFrame(status=status)

        if not sensor.wait_for_new(rt.last_fid, timeout_ms=self.timeout_ms):
            return None

        frame = DMTacFrame(status=status)

        if self.show_raw:
            try:
                raw_fid, raw = sensor.getRawImg()
                frame.raw = payload_img(raw)
                frame.fid = raw_fid
            except Exception as exc:
                print(f"[{rt.name}] getRawImg failed: {exc}")

        if not self.raw_depth_only:
            try:
                infer_fid, infer = sensor.getInferImg()
                if frame.fid < 0:
                    frame.fid = infer_fid
                frame.infer = payload_img(infer)
            except Exception as exc:
                print(f"[{rt.name}] getInferImg failed: {exc}")

        try:
            _, frame.depth = sensor.getDepth()
        except Exception as exc:
            print(f"[{rt.name}] getDepth failed: {exc}")

        if self.show_flow_images:
            try:
                _, frame.deformation = sensor.getDeformation2D()
            except Exception as exc:
                print(f"[{rt.name}] getDeformation2D failed: {exc}")

        if not self.raw_depth_only:
            try:
                _, frame.shear = sensor.getShear()
            except Exception as exc:
                print(f"[{rt.name}] getShear failed: {exc}")

            try:
                frame.force = normalize_vector(sensor.getForce(), n=6)
            except Exception:
                frame.force = None

            try:
                frame.contact_area = safe_scalar(sensor.getContactArea())
            except Exception:
                frame.contact_area = 0.0

        if frame.fid >= 0:
            rt.last_fid = frame.fid
        return frame

    def _enrich_frame(self, rt: DMTacRuntime, frame: DMTacFrame) -> DMTacFrame:
        calib = rt.calibration

        if frame.force is not None and len(frame.force) >= 6:
            raw_force = np.asarray(frame.force[:6], dtype=np.float32)
            corrected = raw_force - calib.force_bias
            frame.force = [float(x) for x in corrected]
            fx, fy, fz, mx, my, mz = corrected[:6]
            frame.force_norm = float(np.linalg.norm([fx, fy, fz]))
            frame.shear_force = float(np.linalg.norm([fx, fy]))
            frame.torque_norm = float(np.linalg.norm([mx, my, mz]))

        frame.contact_area = max(0.0, frame.contact_area - calib.contact_area_bias)

        if frame.depth is not None:
            d = np.asarray(frame.depth, dtype=np.float32)
            frame.depth_mean = float(np.mean(d))
            frame.depth_max = float(np.max(d))
            if calib.depth_baseline is not None and calib.depth_baseline.shape == d.shape:
                frame.depth_delta = np.abs(d - calib.depth_baseline)
                frame.depth_delta_mean = float(np.mean(frame.depth_delta))
                frame.depth_delta_max = float(np.max(frame.depth_delta))
            else:
                frame.depth_delta = d
                frame.depth_delta_mean = frame.depth_mean
                frame.depth_delta_max = frame.depth_max

        frame.deformation_mean, _ = flow_stats(frame.deformation)
        shear_stats = shear_component_stats(frame.shear)
        frame.shear_x_mean = shear_stats["x_mean"]
        frame.shear_y_mean = shear_stats["y_mean"]
        frame.shear_x_abs_mean = shear_stats["x_abs_mean"]
        frame.shear_y_abs_mean = shear_stats["y_abs_mean"]
        frame.shear_x_p95 = shear_stats["x_p95"]
        frame.shear_y_p95 = shear_stats["y_p95"]
        frame.shear_flow_mean = shear_stats["mag_mean"]
        frame.shear_flow_p95 = shear_stats["mag_p95"]

        signals = {
            "Fn": frame.force_norm,
            "Sx": max(0.0, frame.shear_x_p95 - calib.shear_x_noise_p95),
            "Sy": max(0.0, frame.shear_y_p95 - calib.shear_y_noise_p95),
            "Sm": max(0.0, frame.shear_flow_p95 - calib.shear_noise_p95),
            "Depth": max(0.0, frame.depth_delta_max - calib.depth_noise_p95),
            "Area": frame.contact_area,
        }
        for key, value in signals.items():
            if self.use_ema:
                rt.ema[key] = (1.0 - self.ema_alpha) * rt.ema[key] + self.ema_alpha * value
            else:
                rt.ema[key] = value

        force_on = max(self.min_force_norm, calib.force_noise_p95 * 4.0)
        depth_on = max(self.min_depth_delta, calib.depth_noise_p95 * 4.0)
        area_on = max(self.min_contact_area, calib.contact_area_bias + 1.0)
        slip_on = max(self.slip_flow_threshold, calib.shear_noise_p95 * 4.0)

        contact_signal = (
            rt.ema["Area"] >= area_on
            or rt.ema["Fn"] >= force_on
            or rt.ema["Depth"] >= depth_on
        )
        contact_release = (
            rt.ema["Area"] < area_on * self.hysteresis_ratio
            and rt.ema["Fn"] < force_on * self.hysteresis_ratio
            and rt.ema["Depth"] < depth_on * self.hysteresis_ratio
        )
        old_contact = rt.contact_latched
        if contact_signal:
            rt.contact_on_count += 1
            rt.contact_off_count = 0
        elif contact_release:
            rt.contact_off_count += 1
            rt.contact_on_count = 0
        else:
            rt.contact_on_count = 0
            rt.contact_off_count = 0

        if rt.contact_on_count >= self.stable_frames:
            rt.contact_latched = True
        elif rt.contact_off_count >= self.stable_frames:
            rt.contact_latched = False
        if rt.contact_latched != old_contact:
            rt.contact_transitions += 1

        slip_signal = rt.contact_latched and rt.ema["Sm"] >= slip_on
        slip_release = rt.ema["Sm"] < slip_on * self.hysteresis_ratio
        old_slip = rt.slip_latched
        if slip_signal:
            rt.slip_on_count += 1
            rt.slip_off_count = 0
        elif slip_release:
            rt.slip_off_count += 1
            rt.slip_on_count = 0
        else:
            rt.slip_on_count = 0
            rt.slip_off_count = 0

        if rt.slip_on_count >= self.stable_frames:
            rt.slip_latched = True
        elif rt.slip_off_count >= self.stable_frames:
            rt.slip_latched = False
        if rt.slip_latched != old_slip:
            rt.slip_transitions += 1

        frame.contact = rt.contact_latched
        frame.slip_like = rt.slip_latched
        if frame.contact:
            frame.force_norm = rt.ema["Fn"]
            frame.shear_x_p95 = rt.ema["Sx"]
            frame.shear_y_p95 = rt.ema["Sy"]
            frame.shear_flow_p95 = rt.ema["Sm"]
            frame.depth_delta_max = rt.ema["Depth"]
        else:
            frame.force_norm = 0.0
            frame.shear_force = 0.0
            frame.torque_norm = 0.0
            frame.shear_x_mean = 0.0
            frame.shear_y_mean = 0.0
            frame.shear_x_abs_mean = 0.0
            frame.shear_y_abs_mean = 0.0
            frame.shear_x_p95 = 0.0
            frame.shear_y_p95 = 0.0
            frame.shear_flow_p95 = 0.0
            frame.depth_delta_max = 0.0
        return frame

    def calibrate_all(self, reset_sensor: bool = False) -> None:
        print()
        print("=" * 60)
        print("  DM-Tac calibration: keep both sensor surfaces free of contact")
        print("=" * 60)
        for rt in self.runtimes:
            if reset_sensor:
                try:
                    rt.sensor.reset()
                    time.sleep(0.2)
                except Exception as exc:
                    print(f"[{rt.name}] reset before calibration failed: {exc}")
            self._calibrate_one(rt)
        self._update_plot_scale_from_calibration()
        print("=" * 60)
        print()

    def _update_plot_scale_from_calibration(self) -> None:
        if not self.auto_plot_scale:
            return
        force_max = self.min_force_norm
        shear_max = self.slip_flow_threshold
        depth_max = self.min_depth_delta
        for rt in self.runtimes:
            calib = rt.calibration
            force_max = max(force_max, calib.force_noise_p95 * 4.0)
            shear_max = max(
                shear_max,
                calib.shear_noise_p95 * 4.0,
                calib.shear_x_noise_p95 * 4.0,
                calib.shear_y_noise_p95 * 4.0,
            )
            depth_max = max(depth_max, calib.depth_noise_p95 * 4.0)

        self.plot_max["Fn"] = max(force_max * 2.5, 0.12)
        self.plot_max["Sx"] = max(shear_max * 2.5, 0.04)
        self.plot_max["Sy"] = max(shear_max * 2.5, 0.04)
        self.plot_max["Sm"] = max(shear_max * 2.5, 0.04)
        self.plot_max["Depth"] = max(depth_max * 2.0, 0.02)
        print(
            f"[PlotScale] F={self.plot_max['Fn']:.3f} "
            f"shear={self.plot_max['Sm']:.3f} depth={self.plot_max['Depth']:.3f}"
        )

    def _calibrate_one(self, rt: DMTacRuntime) -> None:
        force_samples = []
        depth_samples = []
        deformation_samples = []
        shear_samples = []
        shear_x_samples = []
        shear_y_samples = []
        area_samples = []

        for hist in rt.hist.values():
            hist.clear()
        rt.ema = {k: 0.0 for k in rt.ema}
        rt.last_frame = None
        rt.contact_latched = False
        rt.slip_latched = False
        rt.contact_on_count = 0
        rt.contact_off_count = 0
        rt.slip_on_count = 0
        rt.slip_off_count = 0
        rt.contact_transitions = 0
        rt.slip_transitions = 0

        print(f"[{rt.name}] collecting {self.calibration_frames} no-contact frames...")
        for _ in range(self.calibration_frames):
            frame = self._capture_raw(rt)
            if frame is None or frame.status != 0:
                continue
            if frame.force is not None and len(frame.force) >= 6:
                force_samples.append(np.asarray(frame.force[:6], dtype=np.float32))
            if frame.depth is not None:
                depth_samples.append(np.asarray(frame.depth, dtype=np.float32))
            if frame.deformation is not None:
                _, p95 = flow_stats(frame.deformation)
                deformation_samples.append(p95)
            if frame.shear is not None:
                stats = shear_component_stats(frame.shear)
                shear_samples.append(stats["mag_p95"])
                shear_x_samples.append(stats["x_p95"])
                shear_y_samples.append(stats["y_p95"])
            area_samples.append(frame.contact_area)

        calib = DMTacCalibration()
        if force_samples:
            forces = np.vstack(force_samples)
            calib.force_bias = np.median(forces, axis=0).astype(np.float32)
            corrected = forces - calib.force_bias
            force_norms = np.linalg.norm(corrected[:, :3], axis=1)
            calib.force_noise_p95 = float(np.percentile(force_norms, 95))
        if depth_samples:
            depth_stack = np.stack(depth_samples, axis=0)
            calib.depth_baseline = np.median(depth_stack, axis=0).astype(np.float32)
            depth_noise = np.max(np.abs(depth_stack - calib.depth_baseline), axis=(1, 2))
            calib.depth_noise_p95 = float(np.percentile(depth_noise, 95))
        if deformation_samples:
            calib.deformation_noise_p95 = float(np.percentile(deformation_samples, 95))
        if shear_samples:
            calib.shear_noise_p95 = float(np.percentile(shear_samples, 95))
        if shear_x_samples:
            calib.shear_x_noise_p95 = float(np.percentile(shear_x_samples, 95))
        if shear_y_samples:
            calib.shear_y_noise_p95 = float(np.percentile(shear_y_samples, 95))
        if area_samples:
            calib.contact_area_bias = float(np.median(area_samples))
        calib.ready = True
        rt.calibration = calib

        print(
            f"[{rt.name}] force_bias={np.round(calib.force_bias, 4).tolist()} "
            f"force_noise95={calib.force_noise_p95:.4f} "
            f"depth_noise95={calib.depth_noise_p95:.5f} "
            f"shear_noise95={calib.shear_noise_p95:.4f} "
            f"shear_x_noise95={calib.shear_x_noise_p95:.4f} "
            f"shear_y_noise95={calib.shear_y_noise_p95:.4f} "
            f"area_bias={calib.contact_area_bias:.2f}"
        )

    def _read_frame(self, rt: DMTacRuntime) -> DMTacFrame | None:
        frame = self._capture_raw(rt)
        if frame is None:
            return None
        if frame.status != 0:
            return frame
        frame = self._enrich_frame(rt, frame)
        rt.last_frame = frame

        now = time.time()
        rt.fps_count += 1
        if now - rt.last_fps_ts >= 1.0:
            rt.fps = rt.fps_count / max(now - rt.last_fps_ts, 1e-6)
            rt.fps_count = 0
            rt.last_fps_ts = now

        rt.hist["Fn"].append(frame.force_norm)
        rt.hist["Sx"].append(frame.shear_x_p95)
        rt.hist["Sy"].append(frame.shear_y_p95)
        rt.hist["Sm"].append(frame.shear_flow_p95)
        rt.hist["Depth"].append(frame.depth_delta_max)
        return frame

    def _read_all_frames(self) -> list[DMTacFrame | None]:
        if self.executor is None:
            return [self._read_frame(rt) for rt in self.runtimes]

        futures = [self.executor.submit(self._read_frame, rt) for rt in self.runtimes]
        frames: list[DMTacFrame | None] = []
        for rt, fut in zip(self.runtimes, futures):
            try:
                frames.append(fut.result())
            except Exception as exc:
                print(f"[{rt.name}] read failed: {exc}")
                frames.append(None)
        return frames

    def _draw_sensor_panel(self, rt: DMTacRuntime, frame: DMTacFrame | None) -> np.ndarray:
        if frame is None:
            frame = rt.last_frame
        if frame is None:
            panel_w = DEPTH_VIEW_W + INFO_W
            panel_h = DEPTH_VIEW_H + 48
            if self.show_camera or self.show_flow_images:
                rows = 2 if self.show_flow_images else 1
                cols = 2 if self.show_camera else 1
                panel_w = 384 * cols
                panel_h = 104 + (288 + 24) * rows
            panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            cv2.putText(panel, f"{rt.name}: waiting", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
            return panel

        fallback = (288, 384)
        depth_vis = depth_to_heat(frame.depth_delta, fallback, self.depth_scale)
        if not self.show_camera:
            depth_vis = cv2.resize(depth_vis, (DEPTH_VIEW_W, DEPTH_VIEW_H))
        if self.show_camera:
            camera_img = frame.raw if self.show_raw else frame.infer
            camera_title = "raw" if self.show_raw else "infer"
            camera_vis = gray_to_bgr(camera_img, fallback)
            if camera_vis.shape[:2] != depth_vis.shape[:2]:
                camera_vis = cv2.resize(camera_vis, (depth_vis.shape[1], depth_vis.shape[0]))

        if frame.slip_like:
            border = (0, 0, 255)
            state = "SLIP-LIKE"
        elif frame.contact:
            border = (0, 220, 80)
            state = "CONTACT"
        else:
            border = (90, 90, 90)
            state = "FREE"

        depth_tile = titled_tile("normal deformation", depth_vis)
        if not self.show_camera and not self.show_flow_images:
            info_panel = self._draw_value_panel(rt, frame, state, depth_tile.shape[0])
            panel = np.hstack([depth_tile, info_panel])
            cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), border, 3)
            return panel

        if self.show_camera:
            top = np.hstack([titled_tile(camera_title, camera_vis), depth_tile])
        else:
            top = centered_tile(depth_tile, INFO_W, fill=(0, 0, 0))
        if self.show_flow_images:
            deformation_vis = flow_to_vis(frame.deformation, fallback, self.arrow_step)
            shear_vis = flow_to_vis(frame.shear, fallback, self.arrow_step)
            bottom = np.hstack([titled_tile("deformation", deformation_vis), titled_tile("shear", shear_vis)])
            if bottom.shape[1] != top.shape[1]:
                bottom = cv2.resize(bottom, (top.shape[1], bottom.shape[0]))
            image_grid = np.vstack([top, bottom])
        else:
            image_grid = top

        info_h = 104
        info_bar = np.zeros((info_h, image_grid.shape[1], 3), dtype=np.uint8)
        info_bar[:] = (12, 12, 12)
        info_1 = f"{rt.name} {state}  fid={frame.fid}  fps={rt.fps:.1f}"
        info_2 = (
            f"F={frame.force_norm:.3f}  shearF={frame.shear_force:.3f}  torque={frame.torque_norm:.3f}  "
            f"area={frame.contact_area:.1f}  dDepth={frame.depth_delta_max:.4f}"
        )
        info_3 = (
            f"sx={frame.shear_x_p95:.3f}  sy={frame.shear_y_p95:.3f}  "
            f"sm={frame.shear_flow_p95:.3f}  "
            f"sx_mean={frame.shear_x_mean:.3f}  sy_mean={frame.shear_y_mean:.3f}"
        )
        info_4 = self._gripper_summary_text()
        cv2.putText(info_bar, info_1, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(info_bar, info_2, (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(info_bar, info_3, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(info_bar, info_4, (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 120), 1, cv2.LINE_AA)

        panel = np.vstack([info_bar, image_grid])
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), border, 3)
        return panel

    def _draw_value_panel(self, rt: DMTacRuntime, frame: DMTacFrame, state: str, height: int) -> np.ndarray:
        panel = np.zeros((height, INFO_W, 3), dtype=np.uint8)
        panel[:] = (12, 12, 12)

        state_color = (90, 90, 90)
        if frame.slip_like:
            state_color = (0, 0, 255)
        elif frame.contact:
            state_color = (0, 220, 80)

        cv2.putText(panel, rt.name, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(panel, state, (86, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2, cv2.LINE_AA)
        cv2.putText(panel, f"fid {frame.fid}", (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(panel, f"fps {rt.fps:.1f}", (170, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)

        y = 118
        step = 38
        put_label_value(panel, "F", f"{frame.force_norm:.3f}", y)
        put_label_value(panel, "shearF", f"{frame.shear_force:.3f}", y + step)
        put_label_value(panel, "torque", f"{frame.torque_norm:.3f}", y + step * 2)
        put_label_value(panel, "area", f"{frame.contact_area:.1f}", y + step * 3)
        put_label_value(panel, "dDepth", f"{frame.depth_delta_max:.4f}", y + step * 4, value_color=(200, 220, 255))
        put_label_value(panel, "sx", f"{frame.shear_x_p95:.3f}", y + step * 5)
        put_label_value(panel, "sy", f"{frame.shear_y_p95:.3f}", y + step * 6)
        put_label_value(panel, "sm", f"{frame.shear_flow_p95:.3f}", y + step * 7)
        put_label_value(panel, "sx_mean", f"{frame.shear_x_mean:.3f}", y + step * 8)
        put_label_value(panel, "sy_mean", f"{frame.shear_y_mean:.3f}", y + step * 9)
        if self.gripper_effector is not None:
            cv2.putText(panel, "gripper", (18, y + step * 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1, cv2.LINE_AA)
            cv2.putText(panel, self._gripper_short_text(), (145, y + step * 10), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (220, 220, 120), 1, cv2.LINE_AA)

        return panel

    def _draw_timeseries(self, rt: DMTacRuntime) -> np.ndarray:
        canvas = np.zeros((PLOT_H, PLOT_W, 3), dtype=np.uint8)
        canvas[:] = (28, 28, 28)

        configs = [
            ("Fn", (255, 200, 0), "F"),
            ("Sx", (0, 210, 255), "sx"),
            ("Sy", (0, 255, 120), "sy"),
            ("Sm", (255, 180, 80), "sm"),
            ("Depth", (200, 80, 255), "depth"),
        ]
        ml, mr, mt, mb = 55, 10, 20, 28
        pw = PLOT_W - ml - mr
        ph = PLOT_H - mt - mb
        cv2.rectangle(canvas, (ml, mt), (ml + pw, mt + ph), (70, 70, 70), 1)

        for idx, (key, color, label) in enumerate(configs):
            data = list(rt.hist[key])
            if len(data) < 2:
                continue
            arr = np.asarray(data, dtype=np.float32)
            y_min = 0.0
            y_max = max(self.plot_max.get(key, 1.0), 1e-4)
            pts = []
            for i, v in enumerate(arr):
                x = ml + int(i / max(len(arr) - 1, 1) * pw)
                y = mt + ph - int(np.clip((v - y_min) / (y_max - y_min), 0, 1) * ph)
                pts.append((x, y))
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{label}={arr[-1]:.3f}", (ml + 6, mt + 15 + idx * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        cv2.putText(canvas, rt.name, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        return canvas

    def _compose(self, frames: list[DMTacFrame | None]) -> np.ndarray:
        sensor_panels = [self._draw_sensor_panel(rt, fr) for rt, fr in zip(self.runtimes, frames)]
        main_panel = np.vstack(sensor_panels)

        if self.show_plot:
            plots = [self._draw_timeseries(rt) for rt in self.runtimes]
            plot_panel = np.vstack(plots)
            if plot_panel.shape[0] != main_panel.shape[0]:
                plot_panel = cv2.resize(plot_panel, (plot_panel.shape[1], main_panel.shape[0]))
            final = np.hstack([main_panel, plot_panel])
        else:
            final = main_panel

        scale = self.target_width / final.shape[1]
        if scale != 1:
            final = cv2.resize(final, (self.target_width, int(final.shape[0] * scale)))
        return final

    def reset_all(self) -> None:
        print("[Reset] keep both sensor surfaces free of contact.")
        if self.gripper_effector is not None and self.gripper_active:
            self._gripper_hold_current("reset")
            self.gripper_contact_hold = False
        for rt in self.runtimes:
            try:
                rt.sensor.reset()
                for hist in rt.hist.values():
                    hist.clear()
                rt.ema = {k: 0.0 for k in rt.ema}
                rt.contact_latched = False
                rt.slip_latched = False
                rt.contact_on_count = 0
                rt.contact_off_count = 0
                rt.slip_on_count = 0
                rt.slip_off_count = 0
            except Exception as exc:
                print(f"[{rt.name}] reset failed: {exc}")
        time.sleep(0.2)
        self.calibrate_all(reset_sensor=False)

    def _print_debug(self, frames: list[DMTacFrame | None]) -> None:
        now = time.time()
        if not self.debug_metrics or now - self.last_debug_ts < 1.0:
            return
        self.last_debug_ts = now
        for rt, frame in zip(self.runtimes, frames):
            if frame is None:
                print(f"[{rt.name}] waiting")
                continue
            print(
                f"[{rt.name}] fps={rt.fps:.1f} contact={frame.contact} slip={frame.slip_like} "
                f"F={rt.ema['Fn']:.4f} sx={rt.ema['Sx']:.4f} sy={rt.ema['Sy']:.4f} sm={rt.ema['Sm']:.4f} "
                f"emaDepth={rt.ema['Depth']:.5f} area={rt.ema['Area']:.2f} "
                f"transC={rt.contact_transitions} transS={rt.slip_transitions}"
            )

    def _print_summary(self) -> None:
        print()
        print("=" * 60)
        print("  Runtime summary")
        print("=" * 60)
        for rt in self.runtimes:
            print(
                f"[{rt.name}] contact_transitions={rt.contact_transitions} "
                f"slip_transitions={rt.slip_transitions} "
                f"last_emaF={rt.ema['Fn']:.4f} "
                f"last_sx={rt.ema['Sx']:.4f} "
                f"last_sy={rt.ema['Sy']:.4f} "
                f"last_sm={rt.ema['Sm']:.4f} "
                f"last_emaDepth={rt.ema['Depth']:.5f}"
            )
        print("=" * 60)

    def run(self) -> None:
        print("=" * 60)
        print("  DM-Tac realtime tactile monitor v1")
        print("=" * 60)
        if self.gripper_effector is not None:
            print("  r reset | c recalibrate | s start gripper close | k screenshot | p plot on/off | q/Esc quit")
        else:
            print("  r reset | c recalibrate | k screenshot | p plot on/off | q/Esc quit")
        print("  default lightweight UI: normal deformation image + tactile values")
        if self.raw_depth_only:
            print("  raw-depth-only mode: raw + depth only; force/shear metrics disabled")
        print(f"  sensor reads: {'parallel' if self.executor is not None else 'serial'}")
        if self.gripper_start_signal_file is not None and not self.auto_start_gripper:
            print(f"  gripper starts after signal file appears: {self.gripper_start_signal_file}")
        print("=" * 60)
        if self.auto_start_gripper and self.gripper_effector is not None:
            self._gripper_start_closing()
        if not self.no_display:
            cv2.namedWindow("DM-Tac Realtime v1", cv2.WINDOW_NORMAL)

        t_start = time.time()
        try:
            while True:
                frames = self._read_all_frames()
                self._print_debug(frames)
                contact_detected = any(frame is not None and frame.contact for frame in frames)
                slip_detected = any(frame is not None and frame.slip_like for frame in frames)
                self._gripper_update_from_contact(contact_detected)
                self._gripper_update_from_slip(slip_detected)
                self._write_event_file()

                if self.no_display:
                    key = 255
                else:
                    final = self._compose(frames)
                    self.last_final = final
                    cv2.imshow("DM-Tac Realtime v1", final)
                    key = cv2.waitKey(1) & 0xFF

                if self.duration_s > 0 and time.time() - t_start >= self.duration_s:
                    break
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    self.reset_all()
                elif key == ord("c"):
                    self.calibrate_all(reset_sensor=False)
                elif key == ord("s"):
                    if self.gripper_effector is not None:
                        self._gripper_start_closing()
                    else:
                        fname = f"dmtac_realtime_v1_{int(time.time())}.png"
                        cv2.imwrite(fname, final)
                        print(f"[Screenshot] {fname}")
                elif key == ord("k"):
                    fname = f"dmtac_realtime_v1_{int(time.time())}.png"
                    cv2.imwrite(fname, final)
                    print(f"[Screenshot] {fname}")
                elif key == ord("p"):
                    self.show_plot = not self.show_plot
        finally:
            if self.save_last_frame and self.last_final is not None:
                cv2.imwrite(self.save_last_frame, self.last_final)
                print(f"[Saved] {self.save_last_frame}")
            self._print_summary()
            if self.executor is not None:
                self.executor.shutdown(wait=True)
            self._gripper_shutdown()
            for rt in self.runtimes:
                with suppress(Exception):
                    rt.sensor.disconnect()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="DM-Tac realtime tactile monitor v1.")
    parser.add_argument("--dev_ids", nargs="*", help="Two device indices or serials. Example: --dev_ids 2 4")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "Flux", "flux"])
    parser.add_argument("--mode", default="standard", choices=["standard", "high"])
    parser.add_argument("--max_fps", type=int, default=120)
    parser.add_argument("--timeout_ms", type=int, default=50)
    parser.add_argument("--arrow_step", type=int, default=16)
    parser.add_argument("--depth_scale", type=float, default=80.0)
    parser.add_argument("--min_contact_area", type=float, default=1.0)
    parser.add_argument("--min_force_norm", type=float, default=0.05)
    parser.add_argument("--min_depth_delta", type=float, default=0.004)
    parser.add_argument("--slip_flow_threshold", type=float, default=0.12)
    parser.add_argument("--calibration_frames", type=int, default=45)
    parser.add_argument("--ema_alpha", type=float, default=0.15)
    parser.add_argument("--use_ema", action="store_true", help="Smooth displayed metrics. Off by default for lower latency.")
    parser.add_argument("--hysteresis_ratio", type=float, default=0.55)
    parser.add_argument("--stable_frames", type=int, default=6)
    parser.add_argument("--force_plot_max", type=float, default=0.2)
    parser.add_argument("--shear_plot_max", type=float, default=0.05)
    parser.add_argument("--depth_plot_max", type=float, default=0.2)
    parser.add_argument("--fixed_plot_scale", action="store_true", help="Use command-line plot limits instead of calibration-derived limits.")
    parser.add_argument("--target_width", type=int, default=1500)
    parser.add_argument("--show_camera", action="store_true", help="Show infer camera image next to the normal deformation map.")
    parser.add_argument("--show_raw", action="store_true", help="Show SDK raw camera image instead of infer image.")
    parser.add_argument("--raw_depth_only", action="store_true", help="Only read/show raw camera and depth. Disables infer/shear/force metrics for FPS testing.")
    parser.add_argument("--show_flow_images", action="store_true", help="Draw deformation/shear arrow images. Off by default for higher FPS.")
    parser.add_argument("--show_plot", action="store_true", help="Show time-series plots at startup. Off by default for higher FPS.")
    parser.add_argument("--serial_reads", action="store_true", help="Read the two sensors sequentially. Default is parallel for higher FPS.")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--debug_metrics", action="store_true")
    parser.add_argument("--duration_s", type=float, default=0.0)
    parser.add_argument("--save_last_frame", default="")
    parser.add_argument("--no_gripper", action="store_true", help="Disable AgxGripper integration.")
    parser.add_argument("--gripper_robot", default="nero")
    parser.add_argument("--gripper_firmware", default="v120")
    parser.add_argument("--gripper_interface", default=None)
    parser.add_argument("--gripper_channel", default=None)
    parser.add_argument("--gripper_bitrate", type=int, default=1_000_000)
    parser.add_argument("--gripper_max_width", type=float, default=GRIPPER_DEFAULT_MAX_WIDTH)
    parser.add_argument("--gripper_force", type=float, default=GRIPPER_DEFAULT_FORCE_N)
    parser.add_argument("--gripper_open_percent", type=int, default=GRIPPER_PERCENT_MAX)
    parser.add_argument("--gripper_close_percent", type=int, default=GRIPPER_PERCENT_MIN)
    parser.add_argument("--gripper_publish_hz", type=float, default=GRIPPER_DEFAULT_PUBLISH_HZ)
    parser.add_argument("--gripper_tolerance_percent", type=float, default=GRIPPER_DEFAULT_TOLERANCE_PERCENT)
    parser.add_argument("--gripper_settle_timeout", type=float, default=GRIPPER_DEFAULT_SETTLE_TIMEOUT)
    parser.add_argument("--gripper_step_percent", type=int, default=GRIPPER_DEFAULT_STEP_PERCENT)
    parser.add_argument("--auto_start_gripper", action="store_true", help="Automatically start gripper closing after init.")
    parser.add_argument(
        "--gripper_start_signal_file",
        default="",
        help="Optional file path. If set, the gripper waits until this file exists before starting to close.",
    )
    parser.add_argument("--slip_retry_step_percent", type=int, default=2, help="Extra tightening step on slip.")
    parser.add_argument("--slip_retry_force_scale", type=float, default=1.15, help="Force multiplier used on slip retry.")
    parser.add_argument("--slip_retry_max_attempts", type=int, default=2, help="Maximum slip retries before holding.")
    parser.add_argument("--event_file", default="", help="Optional JSON file to export contact/slip/gripper state.")
    args = parser.parse_args()

    if _DMTAC_IMPORT_ERROR is not None:
        raise SystemExit(
            "DM-Tac SDK is unavailable. Install dmrobotics/pyAgxArm and set "
            "DMTAC_SDK_ROOT (and, if needed, DMTAC_PYTHON_SITE) before running. "
            f"Import error: {_DMTAC_IMPORT_ERROR}"
        )

    dev_ids = [parse_dev_id(x) for x in args.dev_ids] if args.dev_ids else None
    vis = DMTacRealtimeVis(
        dev_ids=dev_ids,
        backend=args.backend,
        mode=args.mode,
        max_fps=args.max_fps,
        timeout_ms=args.timeout_ms,
        arrow_step=args.arrow_step,
        depth_scale=args.depth_scale,
        min_contact_area=args.min_contact_area,
        min_force_norm=args.min_force_norm,
        min_depth_delta=args.min_depth_delta,
        slip_flow_threshold=args.slip_flow_threshold,
        calibration_frames=args.calibration_frames,
        ema_alpha=args.ema_alpha,
        use_ema=args.use_ema,
        hysteresis_ratio=args.hysteresis_ratio,
        stable_frames=args.stable_frames,
        force_plot_max=args.force_plot_max,
        shear_plot_max=args.shear_plot_max,
        depth_plot_max=args.depth_plot_max,
        auto_plot_scale=not args.fixed_plot_scale,
        target_width=args.target_width,
        show_camera=args.show_camera,
        show_raw=args.show_raw,
        raw_depth_only=args.raw_depth_only,
        show_flow_images=args.show_flow_images,
        initial_show_plot=args.show_plot,
        parallel_reads=not args.serial_reads,
        no_display=args.no_display,
        debug_metrics=args.debug_metrics,
        duration_s=args.duration_s,
        save_last_frame=args.save_last_frame,
        with_gripper=not args.no_gripper,
        gripper_robot=args.gripper_robot,
        gripper_firmware=args.gripper_firmware,
        gripper_interface=args.gripper_interface,
        gripper_channel=args.gripper_channel,
        gripper_bitrate=args.gripper_bitrate,
        gripper_max_width=args.gripper_max_width,
        gripper_force=args.gripper_force,
        gripper_open_percent=args.gripper_open_percent,
        gripper_close_percent=args.gripper_close_percent,
        gripper_publish_hz=args.gripper_publish_hz,
        gripper_tolerance_percent=args.gripper_tolerance_percent,
        gripper_settle_timeout=args.gripper_settle_timeout,
        gripper_step_percent=args.gripper_step_percent,
        auto_start_gripper=args.auto_start_gripper,
        gripper_start_signal_file=args.gripper_start_signal_file,
        slip_retry_step_percent=args.slip_retry_step_percent,
        slip_retry_force_scale=args.slip_retry_force_scale,
        slip_retry_max_attempts=args.slip_retry_max_attempts,
        event_file=args.event_file,
    )
    vis.run()


if __name__ == "__main__":
    main()
