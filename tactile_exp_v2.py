#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tactile_exp_v2.py — Dual tactile reflex controller (optimized ~25 Hz)

基于 v2 闭环架构，核心控制律完全不变。
exp_v2 性能优化：
  1. 摄像头异步线程采集：后台线程持续抓帧，主循环零等待取最新帧
  2. 左右传感器并行计算：ThreadPoolExecutor 并行跑 compute()，利用 OpenCV C++ 释放 GIL
  预期帧率从 v2 的 12-14 FPS 提升到 25+ FPS

exp_v1 控制律 (完全保留):
  anti-slip:    Sy > sy_th              → effort += f(Sy), pos -= step
  减重/防挤压: CoP负向变化 + Sy低               → effort -= step, pos += step
  Fn硬保护:   Fn > fn_protect_limit              → 强制 effort -= step, pos += step
  otherwise:    维持当前状态

Keys:  s=start  r=reset  q=quit
"""

import argparse
import csv
import os
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

# =============================================================================
# Piper SDK (optional)
# =============================================================================
try:
    from piper_sdk import C_PiperInterface_V2
except Exception:
    C_PiperInterface_V2 = None


# =============================================================================
# Utilities
# =============================================================================

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# ---- USB 摄像头自动检测 ----
DEFAULT_USB_MAP = {
    "left":  "10.4",      # usb-0000:00:14.0-10.4
    "right": "11.3.3",    # usb-0000:00:14.0-11.3.3
}

def find_cam_by_usb_port(usb_port_suffix: str) -> Optional[int]:
    sys_base = "/sys/class/video4linux"
    if not os.path.isdir(sys_base):
        return None
    for dev_name in sorted(os.listdir(sys_base)):
        dev_path = os.path.join(sys_base, dev_name, "device")
        if not os.path.exists(dev_path):
            continue
        try:
            real = os.path.realpath(dev_path)
            if usb_port_suffix in real:
                num = int(dev_name.replace("video", ""))
                index_path = os.path.join(sys_base, dev_name, "index")
                if os.path.exists(index_path):
                    with open(index_path) as f:
                        if f.read().strip() == "0":
                            return num
                else:
                    return num
        except Exception:
            continue
    return None

def auto_detect_cameras() -> Tuple[Optional[int], Optional[int]]:
    cam_l = find_cam_by_usb_port(DEFAULT_USB_MAP["left"])
    cam_r = find_cam_by_usb_port(DEFAULT_USB_MAP["right"])
    return cam_l, cam_r

def lock_camera_settings(cap: cv2.VideoCapture, name: str = "CAM",
                         exposure: Optional[float] = 166.0,
                         width: int = 640, height: int = 480) -> None:
    try:
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=manual, 3=auto for V4L2
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if exposure is not None and exposure >= 0:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    except Exception:
        pass
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    width_rb = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_rb = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"  [{name}] camera locked: {width_rb:.0f}x{height_rb:.0f}, exposure={exp:.1f}, WB={wb:.0f}")


# =============================================================================
# Tactile metrics (same as v2, unchanged)
# =============================================================================

class TactileMetricsComputer:
    """
    Single tactile sensor metrics computer.
      Fn = Σ ΔI^γ / (H×W)   over contact mask
      Sy = median(|v_y|) over contact mask; v_y is Farneback's vertical component
      CoP = weighted centroid
    """

    def __init__(
        self,
        gamma: float = 1.5,
        min_contact_px: int = 200,
        flow_params: Optional[dict] = None,
        ema_alpha: float = 0.3,
        baseline_avg_frames: int = 30,
        adaptive_baseline_rate: float = 0.002,
        noise_percentile: float = 98,
        morph_kernel_size: int = 5,
    ):
        self.gamma = gamma
        self.min_contact_px = min_contact_px
        self.ema_alpha = ema_alpha
        self.baseline_avg_frames = baseline_avg_frames
        self.adaptive_rate = adaptive_baseline_rate
        self.noise_percentile = noise_percentile
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.flow_params = flow_params or dict(
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        self.baseline: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.tau: float = 20.0
        self.noise_floor: float = 0.0
        self.is_calibrated: bool = False

        self._ema = {'Fn': 0.0, 'Sy': 0.0, 'CoP_x': 0.0, 'CoP_y': 0.0}

    def reset_baseline_single(self, gray_frame: np.ndarray) -> None:
        self.baseline = gray_frame.astype(np.float32)
        self.prev_gray = gray_frame.copy()
        self._ema = {'Fn': 0.0, 'Sy': 0.0, 'CoP_x': 0.0, 'CoP_y': 0.0}

    def reset_baseline_multi(self, cap_func: Callable[[], Optional[np.ndarray]], n_frames: Optional[int] = None) -> bool:
        n = n_frames or self.baseline_avg_frames
        acc = None
        count = 0
        for _ in range(n):
            gray = cap_func()
            if gray is None:
                continue
            if acc is None:
                acc = gray.astype(np.float64)
            else:
                acc += gray.astype(np.float64)
            count += 1

        if count <= 0:
            return False

        self.baseline = (acc / count).astype(np.float32)
        self.prev_gray = self.baseline.astype(np.uint8).copy()
        self._ema = {'Fn': 0.0, 'Sy': 0.0, 'CoP_x': 0.0, 'CoP_y': 0.0}
        return True

    def auto_calibrate_noise(self, cap_func: Callable[[], Optional[np.ndarray]], n_frames: Optional[int] = None) -> bool:
        n = n_frames or self.baseline_avg_frames
        if self.baseline is None:
            return False

        all_deltas = []
        for _ in range(n):
            gray = cap_func()
            if gray is None:
                continue
            dI = np.abs(gray.astype(np.float32) - self.baseline)
            all_deltas.append(dI.flatten())

        if not all_deltas:
            return False

        combined = np.concatenate(all_deltas)
        self.tau = float(np.percentile(combined, self.noise_percentile))
        self.noise_floor = float(np.mean(combined))
        self.is_calibrated = True
        return True

    def _adaptive_baseline_update(self, curr_gray: np.ndarray, has_contact: bool) -> None:
        if (not has_contact) and (self.baseline is not None):
            r = self.adaptive_rate
            self.baseline = (1 - r) * self.baseline + r * curr_gray.astype(np.float32)

    def compute(self, curr_gray: np.ndarray) -> Tuple[Dict, Dict]:
        if self.baseline is None:
            self.reset_baseline_single(curr_gray)
            return self._empty_result(curr_gray.shape), self._empty_vis(curr_gray.shape)

        h, w = curr_gray.shape
        curr_f32 = curr_gray.astype(np.float32)

        delta_I = np.abs(curr_f32 - self.baseline)
        mask_raw = delta_I > self.tau

        mask_u8 = mask_raw.astype(np.uint8) * 255
        mask_cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, self.morph_kernel)
        mask = mask_cleaned > 0

        weight = np.zeros_like(delta_I)
        weight[mask] = delta_I[mask] ** self.gamma

        contact_px = int(np.sum(mask))
        has_contact = contact_px >= self.min_contact_px

        Fn_raw = float(np.sum(weight)) / (h * w) if has_contact else 0.0

        flow = None
        flow_mag = np.zeros((h, w), dtype=np.float32)
        Sy_raw = 0.0
        sy_vec = np.array([0.0, 0.0], dtype=np.float32)

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, curr_gray, None, **self.flow_params
            )
            fx, fy = flow[..., 0], flow[..., 1]
            # Paper Eq. (6): Sy is the median absolute vertical flow
            # component.  Keep the 2-D magnitude for visualization only.
            flow_mag = np.sqrt(fx**2 + fy**2)
            if has_contact:
                Sy_raw = float(np.median(np.abs(fy[mask])))
                sy_vec[0] = float(np.median(fx[mask]))
                sy_vec[1] = float(np.median(fy[mask]))

        CoP_x_raw, CoP_y_raw = w / 2.0, h / 2.0
        if has_contact:
            total_w = float(np.sum(weight))
            if total_w > 1e-6:
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                CoP_x_raw = float(np.sum(x_coords * weight) / total_w)
                CoP_y_raw = float(np.sum(y_coords * weight) / total_w)

        a = self.ema_alpha
        if has_contact:
            self._ema['Fn']    = a * Fn_raw    + (1 - a) * self._ema['Fn']
            self._ema['Sy']    = a * Sy_raw    + (1 - a) * self._ema['Sy']
            self._ema['CoP_x'] = a * CoP_x_raw + (1 - a) * self._ema['CoP_x']
            self._ema['CoP_y'] = a * CoP_y_raw + (1 - a) * self._ema['CoP_y']
        else:
            self._ema['Fn'] *= (1 - a)
            self._ema['Sy'] *= (1 - a)

        self._adaptive_baseline_update(curr_gray, has_contact)
        self.prev_gray = curr_gray.copy()

        result = {
            'Fn_raw': Fn_raw,
            'Fn':     float(self._ema['Fn']),
            'Sy_raw': Sy_raw,
            'Sy':     float(self._ema['Sy']),
            'Sy_vec': sy_vec,
            # Raw weighted centroid is the paper's CoP signal.  EMA values
            # remain available for diagnostics but are not used for release.
            'CoP_x':  CoP_x_raw,
            'CoP_y':  CoP_y_raw,
            'CoP_x_ema': float(self._ema['CoP_x']),
            'CoP_y_ema': float(self._ema['CoP_y']),
            'Area':   contact_px,
            'contact': bool(has_contact),
            'tau':    float(self.tau),
        }

        vis_data = {
            'delta_I':  delta_I,
            'mask':     mask,
            'weight':   weight,
            'flow':     flow,
            'flow_mag': flow_mag,
        }
        return result, vis_data

    def _empty_result(self, shape):
        h, w = shape[:2]
        return {
            'Fn_raw': 0.0, 'Fn': 0.0,
            'Sy_raw': 0.0, 'Sy': 0.0, 'Sy_vec': np.array([0.0, 0.0], dtype=np.float32),
            'CoP_x': w/2.0, 'CoP_y': h/2.0,
            'Area': 0, 'contact': False, 'tau': float(self.tau),
        }

    def _empty_vis(self, shape):
        h, w = shape[:2]
        return {
            'delta_I': np.zeros((h, w), dtype=np.float32),
            'mask': np.zeros((h, w), dtype=bool),
            'weight': np.zeros((h, w), dtype=np.float32),
            'flow': None,
            'flow_mag': np.zeros((h, w), dtype=np.float32),
        }


# =============================================================================
# Threaded camera capture (exp_v2)
# =============================================================================

class ThreadedCamera:
    """
    后台线程持续抓帧，主循环通过 get_latest() 零等待获取最新灰度帧。
    标定时需先 stop()，用 read_direct() 同步读取，再 start()。
    """

    def __init__(self, cap: cv2.VideoCapture, preprocess_fn: Optional[Callable] = None):
        self.cap = cap
        self.preprocess_fn = preprocess_fn
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            ret, raw = self.cap.read()
            if not ret:
                continue
            processed = self.preprocess_fn(raw) if self.preprocess_fn else raw
            with self._lock:
                self._frame = processed

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def read_direct(self) -> Optional[np.ndarray]:
        """同步读取（仅在线程停止时调用，用于标定阶段）"""
        ret, raw = self.cap.read()
        if not ret:
            return None
        return self.preprocess_fn(raw) if self.preprocess_fn else raw

    @property
    def is_running(self) -> bool:
        return self._running


# =============================================================================
# Reflex controller (exp_v2: threaded capture + parallel compute)
# =============================================================================

@dataclass
class ReflexParams:
    hz: float = 25

    # ---- 夹爪行程 ----
    max_stroke_um: int = 100 * 1000
    min_stroke_um: int = 20 * 1000
    close_step_um: int = 150

    # ---- effort 范围 ----
    init_effort: int = 1200
    max_effort: int = 5000

    # ---- CLOSING → HOLDING 触发 ----
    fn_stop_threshold: float = 1.3

    # ---- anti-slip (Sy 驱动, 闭环) ----
    sy_threshold: float = 0.20
    slip_effort_step: int = 300
    slip_pos_step_um: int = 80
    max_pos_adjust_um: int = 3000

    # ---- 减重/防挤压 (CoP 驱动 + Sy 排除) ----
    sy_quiet_threshold: float = 0.12
    cop_decrease_th_px: float = -10.0
    squeeze_effort_step: int = 200
    squeeze_pos_step_um: int = 60

    # ---- Fn 硬保护 (最高优先级) ----
    fn_protect_limit: float = 45
    protect_effort_step: int = 400
    protect_pos_step_um: int = 100

    # ---- 传感器参数 ----
    gamma: float = 1.5
    ema_alpha: float = 0.3
    baseline_avg_frames: int = 30
    noise_percentile: float = 98
    adaptive_baseline_rate: float = 0.002
    min_contact_px: int = 200


class DualTactileReflexController:
    STATE_IDLE = 0
    STATE_CLOSING = 1
    STATE_HOLDING = 2

    TACTILE_ROI = (40, 0, 640, 450)

    def __init__(
        self,
        cam_left: int,
        cam_right: int,
        can_iface: str,
        use_piper: bool,
        params: ReflexParams,
        out_csv: str,
        flip_right: bool = True,
        window_name: str = "tactile_reflex_exp_v2",
        tactile_roi: tuple | None = None,
        headless: bool = False,
        exposure: Optional[float] = 166.0,
        width: int = 640,
        height: int = 480,
    ):
        self.params = params
        self.flip_right = flip_right
        self.window_name = window_name
        self.headless = headless
        self.tactile_roi = tactile_roi if tactile_roi is not None else self.TACTILE_ROI

        # ---- 打开摄像头 ----
        self.cap_l = cv2.VideoCapture(cam_left)
        self.cap_r = cv2.VideoCapture(cam_right)
        if not self.cap_l.isOpened():
            raise RuntimeError(f"Cannot open left tactile camera id={cam_left}")
        if not self.cap_r.isOpened():
            raise RuntimeError(f"Cannot open right tactile camera id={cam_right}")

        lock_camera_settings(self.cap_l, "LEFT", exposure=exposure, width=width, height=height)
        lock_camera_settings(self.cap_r, "RIGHT", exposure=exposure, width=width, height=height)

        # ---- exp_v2: 构建预处理函数 + 线程化摄像头 ----
        roi = self.tactile_roi

        def _preprocess_left(frame: np.ndarray) -> np.ndarray:
            if roi:
                x1, y1, x2, y2 = roi
                frame = frame[y1:y2, x1:x2]
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def _preprocess_right(frame: np.ndarray) -> np.ndarray:
            if flip_right:
                frame = cv2.flip(frame, 0)
            if roi:
                x1, y1, x2, y2 = roi
                frame = frame[y1:y2, x1:x2]
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.tcam_l = ThreadedCamera(self.cap_l, _preprocess_left)
        self.tcam_r = ThreadedCamera(self.cap_r, _preprocess_right)

        # ---- exp_v2: 并行计算线程池 ----
        self._executor = ThreadPoolExecutor(max_workers=2)

        # ---- 传感器 ----
        s_kwargs = dict(
            gamma=params.gamma,
            min_contact_px=params.min_contact_px,
            ema_alpha=params.ema_alpha,
            baseline_avg_frames=params.baseline_avg_frames,
            adaptive_baseline_rate=params.adaptive_baseline_rate,
            noise_percentile=params.noise_percentile,
            morph_kernel_size=5,
        )
        self.sensor_l = TactileMetricsComputer(**s_kwargs)
        self.sensor_r = TactileMetricsComputer(**s_kwargs)

        # ---- Piper 夹爪 ----
        self.use_piper = bool(use_piper and (C_PiperInterface_V2 is not None))
        self.piper = None
        if self.use_piper:
            print(f"🔧 Initializing Piper gripper on {can_iface} ...")
            self.piper = C_PiperInterface_V2(can_iface)
            self.piper.ConnectPort()
            while not self.piper.EnablePiper():
                time.sleep(0.01)
            print("✅ Piper enabled")

            MOTION_VEL_ALL = 50
            MOTION_MIT_FLAG = 0xAD
            GRIP_SPEED = 1000

            try:
                self.piper.MotionCtrl_2(0x01, 0x01, int(MOTION_VEL_ALL), int(MOTION_MIT_FLAG))
                print("✅ Joint mode ensured (MotionCtrl_2)")
            except Exception as e:
                print(f"⚠️ MotionCtrl_2 failed (non-fatal): {e}")

            max_pos_um = self.params.max_stroke_um
            print("🔧 Initializing gripper mode & force open...")
            self.piper.GripperCtrl(abs(int(max_pos_um)), GRIP_SPEED, 0x02, 0)
            time.sleep(0.6)
            self.piper.GripperCtrl(abs(int(max_pos_um)), GRIP_SPEED, 0x01, 0)
            time.sleep(0.8)
            print("✅ Gripper initialized (fully open, mode ready)")
        else:
            print("⚠️ Piper control disabled. Running sense-only.")

        # ---- 状态 ----
        self.state = self.STATE_IDLE
        self.current_pos = params.max_stroke_um
        self.current_effort = params.init_effort
        self.cop_baseline_l = None
        self.cop_baseline_r = None

        self.hold_start_pos: Optional[int] = None
        self.current_adjusted_um: int = 0
        self.hold_fn_baseline: float = 0.0
        self.stable_cop_y_ref: Optional[float] = None
        self._last_action: str = ""
        self._prev_slip_any = False
        self._prev_action_state = "NONE"

        self.freq_history = deque(maxlen=20)
        self.last_tick_time = time.time()

        # ---- CSV ----
        self.csv_path = out_csv
        self._csv_f = open(self.csv_path, "w", newline="")
        self._csv_w = csv.writer(self._csv_f)
        self._csv_w.writerow([
            "t_wall","state","pos_um","effort",
            "FnL","FnR","Fn_min","Fn_raw_L","Fn_raw_R","Fn_raw_min","Fn_raw_max",
            "SyL","SyR","Sy_max","Sy_raw_L","Sy_raw_R","Sy_raw_max",
            "CoP_Lx","CoP_Ly","CoP_Rx","CoP_Ry","CoP_y_avg",
            "AreaL","AreaR","contactL","contactR",
            "slipL","slipR","slip_any","slip_on","slip_off","release_on","protect_on","tauL","tauR",
            "cop_dy_max","delta_cop_y","adjusted_um","action",
            "action_state","hold_fn_base",
            "frame_ms","freq_hz_avg",
        ])
        self._csv_f.flush()

        self._send_gripper(self.current_pos, self.current_effort)

    # ---- 标定用同步读取（线程停止时调用） ----

    def _grab_gray_l(self) -> Optional[np.ndarray]:
        return self.tcam_l.read_direct()

    def _grab_gray_r(self) -> Optional[np.ndarray]:
        return self.tcam_r.read_direct()

    # ---- 线程管理 ----

    def _start_capture_threads(self) -> None:
        self.tcam_l.start()
        self.tcam_r.start()
        time.sleep(0.05)

    def _stop_capture_threads(self) -> None:
        self.tcam_l.stop()
        self.tcam_r.stop()

    # ---- 标定 ----

    def auto_init(self, wait_sec: float = 2.0) -> bool:
        self._stop_capture_threads()

        print("\n" + "="*70)
        print("🤖 Dual tactile reflex init (exp_v2 threaded)")
        print("Ensure both tactile sensors are NOT in contact.")
        print(f"Auto-calibrating in {wait_sec:.1f}s ...")
        print("="*70)
        time.sleep(wait_sec)

        for _ in range(15):
            self.cap_l.read()
            self.cap_r.read()
        time.sleep(0.3)

        ok1 = self.sensor_l.reset_baseline_multi(self._grab_gray_l, n_frames=self.params.baseline_avg_frames)
        ok2 = self.sensor_r.reset_baseline_multi(self._grab_gray_r, n_frames=self.params.baseline_avg_frames)
        if not (ok1 and ok2):
            print("❌ baseline failed")
            return False

        ok1 = self.sensor_l.auto_calibrate_noise(self._grab_gray_l, n_frames=self.params.baseline_avg_frames)
        ok2 = self.sensor_r.auto_calibrate_noise(self._grab_gray_r, n_frames=self.params.baseline_avg_frames)
        if not (ok1 and ok2):
            print("❌ noise calibration failed")
            return False

        self.cop_baseline_l = None
        self.cop_baseline_r = None
        print(f"✅ init done: tauL={self.sensor_l.tau:.1f} tauR={self.sensor_r.tau:.1f}")

        self._start_capture_threads()
        return True

    def reset(self) -> None:
        print("\n♻️ RESET: open + re-calibrate + IDLE")
        self.state = self.STATE_IDLE
        self.current_pos = self.params.max_stroke_um
        self.current_effort = self.params.init_effort

        self.hold_start_pos = None
        self.current_adjusted_um = 0
        self.hold_fn_baseline = 0.0
        self.stable_cop_y_ref = None
        self._last_action = ""
        self._prev_slip_any = False
        self._prev_action_state = "NONE"

        self._send_gripper(self.current_pos, self.current_effort)
        self.auto_init(wait_sec=1.5)

    def start_grasping(self) -> None:
        if self.state != self.STATE_IDLE:
            print(f"\n⚠️ cannot start, state={self.state}")
            return
        print("\n▶️ Start grasping: IDLE -> CLOSING")
        self.state = self.STATE_CLOSING

    def _send_gripper(self, pos_um: int, effort: int) -> None:
        if not self.use_piper or self.piper is None:
            return
        try:
            self.piper.GripperCtrl(abs(int(pos_um)), int(effort), 0x01, 0)
        except Exception as e:
            print(f"\n❌ GripperCtrl error: {e}")

    def _measure_frequency(self) -> Tuple[float, float]:
        now = time.time()
        elapsed = now - self.last_tick_time
        self.last_tick_time = now
        freq = 1.0 / max(elapsed, 1e-6)
        self.freq_history.append(freq)
        avg_freq = float(sum(self.freq_history) / len(self.freq_history))
        frame_ms = float(elapsed * 1000.0)
        return avg_freq, frame_ms

    def _overlay(self, gray: np.ndarray, res: Dict, vis: Dict, title: str) -> np.ndarray:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        h, w = gray.shape
        mask = vis.get("mask", None)
        if mask is not None and res["contact"]:
            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bgr, contours, -1, (0, 255, 0), 1)
            cx, cy = int(res["CoP_x"]), int(res["CoP_y"])
            cv2.drawMarker(bgr, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

        cv2.rectangle(bgr, (0, 0), (w, 38), (0, 0, 0), -1)
        cv2.putText(bgr, f"{title}  Fn={res['Fn']:.3f}  Sy={res['Sy']:.2f}  Area={res['Area']}  tau={res['tau']:.1f}",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return bgr

    def _draw_ui(self, gray_l, res_l, vis_l, gray_r, res_r, vis_r, avg_freq, frame_ms, slip_any, cop_dy_max, cop_y_avg, delta_cop_y) -> np.ndarray:
        left = self._overlay(gray_l, res_l, vis_l, "LEFT")
        right = self._overlay(gray_r, res_r, vis_r, "RIGHT")

        combo = np.hstack([left, right])
        h, w = combo.shape[:2]

        cv2.rectangle(combo, (0, 0), (w, 58), (0, 0, 0), -1)

        state_str = ["IDLE", "CLOSING", "HOLDING"][self.state]
        color = (0, 0, 255) if slip_any else (0, 255, 0) if (res_l["contact"] and res_r["contact"]) else (120, 120, 120)

        info1 = f"STATE: {state_str} | Freq: {avg_freq:.1f}Hz ({frame_ms:.1f}ms) | Pos: {self.current_pos/1000:.1f}mm | Effort: {self.current_effort}"
        info2 = (f"Fn_min={min(res_l['Fn'], res_r['Fn']):.3f} max:{max(res_l['Fn'], res_r['Fn']):.3f}  Sy_max={max(res_l['Sy'], res_r['Sy']):.2f}  "
                 f"CoP_y={cop_y_avg:.1f}px  delta_CoP_y={delta_cop_y:+.1f}px  "
                 f"FnBase={self.hold_fn_baseline:.2f}  Adj:{self.current_adjusted_um}um")
        cv2.putText(combo, info1, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(combo, info2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.rectangle(combo, (0, 0), (w - 1, h - 1), color, 3)

        if self._last_action == "SLIP":
            cv2.putText(combo, "SLIP!", (w - 120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        elif self._last_action == "RELEASE":
            cv2.putText(combo, "RELEASE", (w - 180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
        elif self._last_action == "PROTECT":
            cv2.putText(combo, "Fn PROTECT!", (w - 250, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        return combo

    # ==================================================================
    # update() — exp_v2: 线程取帧 + 并行传感器计算, 控制律完全不变
    # ==================================================================

    def update(self) -> Tuple[Optional[np.ndarray], Optional[Dict], Tuple[float, float]]:
        tick_start = time.time()

        # ---- exp_v2: 从线程缓冲区零等待取最新帧 ----
        gray_l = self.tcam_l.get_latest()
        gray_r = self.tcam_r.get_latest()
        if (gray_l is None) or (gray_r is None):
            time.sleep(0.01)
            return None, None, self._measure_frequency()

        # ---- exp_v2: 左右传感器并行计算 ----
        fut_l = self._executor.submit(self.sensor_l.compute, gray_l)
        fut_r = self._executor.submit(self.sensor_r.compute, gray_r)
        res_l, vis_l = fut_l.result()
        res_r, vis_r = fut_r.result()

        # ==== 以下控制律与 v2 完全一致 ====

        slip_l = bool(res_l["contact"] and (res_l["Sy_raw"] > self.params.sy_threshold))
        slip_r = bool(res_r["contact"] and (res_r["Sy_raw"] > self.params.sy_threshold))
        slip_any = bool(slip_l or slip_r)

        if res_l["contact"] and self.cop_baseline_l is None:
            self.cop_baseline_l = (res_l["CoP_x"], res_l["CoP_y"])
        if res_r["contact"] and self.cop_baseline_r is None:
            self.cop_baseline_r = (res_r["CoP_x"], res_r["CoP_y"])

        cop_dy_l = abs(res_l["CoP_y"] - self.cop_baseline_l[1]) if (self.cop_baseline_l is not None) else 0.0
        cop_dy_r = abs(res_r["CoP_y"] - self.cop_baseline_r[1]) if (self.cop_baseline_r is not None) else 0.0
        cop_dy_max = float(max(cop_dy_l, cop_dy_r))

        Fn_min = float(min(res_l["Fn"], res_r["Fn"]))
        Fn_raw_min = float(min(res_l["Fn_raw"], res_r["Fn_raw"]))
        Sy_max = float(max(res_l["Sy"], res_r["Sy"]))

        cop_y_avg = 0.0
        dual_contact = res_l["contact"] and res_r["contact"]
        if dual_contact:
            cop_y_avg = (res_l["CoP_y"] + res_r["CoP_y"]) / 2.0

        delta_cop_y = 0.0
        if self.stable_cop_y_ref is not None:
            delta_cop_y = cop_y_avg - self.stable_cop_y_ref

        if self.state == self.STATE_CLOSING:
            if Fn_raw_min >= self.params.fn_stop_threshold:
                self.state = self.STATE_HOLDING
                print(f"\n✅ tactile lock: Fn_raw_min={Fn_raw_min:.3f} -> HOLDING")
            elif self.current_pos <= self.params.min_stroke_um:
                self.state = self.STATE_HOLDING
                print(f"\n⚠️ stroke limit reached -> HOLDING")
            else:
                self.current_pos -= int(self.params.close_step_um)
                self.current_pos = max(self.current_pos, self.params.min_stroke_um)
                self._send_gripper(self.current_pos, self.params.init_effort)

        elif self.state == self.STATE_HOLDING:
            if self.hold_start_pos is None and dual_contact:
                self.hold_start_pos = self.current_pos
                self.current_adjusted_um = 0
                self.stable_cop_y_ref = cop_y_avg
                self.hold_fn_baseline = Fn_min
                print(f"\n✅ Enter HOLDING | Fn_base={Fn_min:.3f} | CoP_y={cop_y_avg:.1f} | pos={self.current_pos/1000:.1f}mm")

            if self.stable_cop_y_ref is not None and dual_contact:
                self.stable_cop_y_ref = 0.98 * self.stable_cop_y_ref + 0.02 * cop_y_avg

            # ============================================================
            #  控制律 (每帧独立决策, Sy/Fn 闭环) — 与 v2 完全一致
            # ============================================================
            action = ""

            # ---- 1) Anti-slip: Sy > threshold → 加 effort + 收紧 ----
            if slip_any:
                self.current_effort = int(clamp(
                    self.current_effort + self.params.slip_effort_step,
                    self.params.init_effort, self.params.max_effort))

                if self.current_adjusted_um < self.params.max_pos_adjust_um:
                    step = min(self.params.slip_pos_step_um,
                               self.params.max_pos_adjust_um - self.current_adjusted_um)
                    self.current_pos -= step
                    self.current_pos = max(self.current_pos, self.params.min_stroke_um)
                    self.current_adjusted_um += step

                action = "SLIP"
                if self._last_action != action:
                    print(f"🛡️ SLIP (Sy={Sy_max:.2f}) | effort→{self.current_effort} | adj={self.current_adjusted_um}um")

            # ---- 2) 减重 / 防挤压 ----
            elif (dual_contact and
                  self.stable_cop_y_ref is not None and
                  delta_cop_y < self.params.cop_decrease_th_px and
                  Sy_max < self.params.sy_quiet_threshold):

                self.current_effort = max(
                    self.current_effort - self.params.squeeze_effort_step,
                    self.params.init_effort)

                if self.hold_start_pos is not None and self.current_pos < self.hold_start_pos:
                    step = min(self.params.squeeze_pos_step_um,
                               self.hold_start_pos - self.current_pos)
                    self.current_pos += step
                    self.current_adjusted_um = max(0, self.current_adjusted_um - step)

                self.stable_cop_y_ref += 0.3 * (cop_y_avg - self.stable_cop_y_ref)

                action = "RELEASE"
                if self._last_action != action:
                    print(f"⚖️ RELEASE (delta_CoP_y={delta_cop_y:.1f}px, Sy={Sy_max:.2f}) | effort→{self.current_effort} | pos→{self.current_pos/1000:.1f}mm")

            if action == "" and self.hold_fn_baseline > 0:
                self.hold_fn_baseline = 0.95 * self.hold_fn_baseline + 0.05 * Fn_min

            # ---- Fn 硬保护 (最高优先级) ----
            Fn_max = float(max(res_l["Fn"], res_r["Fn"]))
            if Fn_max > self.params.fn_protect_limit:
                self.current_effort = max(
                    self.current_effort - self.params.protect_effort_step,
                    self.params.init_effort)
                if self.hold_start_pos is not None and self.current_pos < self.hold_start_pos:
                    step = min(self.params.protect_pos_step_um,
                               self.hold_start_pos - self.current_pos)
                    self.current_pos += step
                    self.current_adjusted_um = max(0, self.current_adjusted_um - step)
                action = "PROTECT"
                if self._last_action != action:
                    print(f"🛑 Fn PROTECT (Fn_max={Fn_max:.3f} > {self.params.fn_protect_limit}) | effort→{self.current_effort} | pos→{self.current_pos/1000:.1f}mm")

            self._last_action = action
            self._send_gripper(self.current_pos, self.current_effort)

        # ==== 以上控制律与 exp_v1 完全一致 ====

        action_state = self._last_action if self._last_action else "NONE"
        slip_on = int(slip_any and not self._prev_slip_any)
        slip_off = int((not slip_any) and self._prev_slip_any)
        release_on = int(action_state == "RELEASE" and self._prev_action_state != "RELEASE")
        protect_on = int(action_state == "PROTECT" and self._prev_action_state != "PROTECT")
        self._prev_slip_any = slip_any
        self._prev_action_state = action_state

        avg_freq, frame_ms = self._measure_frequency()

        t_wall = time.time()
        self._csv_w.writerow([
            f"{t_wall:.6f}",
            ["IDLE", "CLOSING", "HOLDING"][self.state],
            int(self.current_pos), int(self.current_effort),
            f"{res_l['Fn']:.6f}", f"{res_r['Fn']:.6f}", f"{Fn_min:.6f}",
            f"{res_l['Fn_raw']:.6f}", f"{res_r['Fn_raw']:.6f}", f"{Fn_raw_min:.6f}", f"{max(res_l['Fn_raw'], res_r['Fn_raw']):.6f}",
            f"{res_l['Sy']:.6f}", f"{res_r['Sy']:.6f}", f"{Sy_max:.6f}",
            f"{res_l['Sy_raw']:.6f}", f"{res_r['Sy_raw']:.6f}", f"{max(res_l['Sy_raw'], res_r['Sy_raw']):.6f}",
            f"{res_l['CoP_x']:.3f}", f"{res_l['CoP_y']:.3f}", f"{res_r['CoP_x']:.3f}", f"{res_r['CoP_y']:.3f}", f"{cop_y_avg:.3f}",
            int(res_l["Area"]), int(res_r["Area"]),
            int(res_l["contact"]), int(res_r["contact"]),
            int(slip_l), int(slip_r), int(slip_any), int(slip_on), int(slip_off),
            int(release_on), int(protect_on),
            f"{res_l['tau']:.3f}", f"{res_r['tau']:.3f}",
            f"{cop_dy_max:.3f}", f"{delta_cop_y:.3f}", int(self.current_adjusted_um),
            self._last_action,
            action_state,
            f"{self.hold_fn_baseline:.6f}",
            f"{frame_ms:.3f}", f"{avg_freq:.3f}",
        ])
        if (int(time.time() * 10) % 10) == 0:
            self._csv_f.flush()

        vis = None
        if not self.headless:
            vis = self._draw_ui(gray_l, res_l, vis_l, gray_r, res_r, vis_r, avg_freq, frame_ms, slip_any, cop_dy_max, cop_y_avg, delta_cop_y)

        target_dt = 1.0 / max(self.params.hz, 1e-6)
        used = time.time() - tick_start
        if used < target_dt:
            time.sleep(target_dt - used)

        return vis, {
            "L": res_l, "R": res_r,
            "Fn_min": Fn_min, "Sy_max": Sy_max,
            "slip_any": slip_any, "cop_dy_max": cop_dy_max,
            "state": self.state,
            "pos_um": self.current_pos,
            "effort": self.current_effort,
        }, (avg_freq, frame_ms)

    def cleanup(self):
        self._stop_capture_threads()

        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

        try:
            self.current_pos = self.params.max_stroke_um
            self.current_effort = self.params.init_effort
            self._send_gripper(self.current_pos, self.current_effort)
            time.sleep(0.2)
        except Exception:
            pass

        try:
            if self.cap_l.isOpened():
                self.cap_l.release()
            if self.cap_r.isOpened():
                self.cap_r.release()
        except Exception:
            pass

        try:
            self._csv_f.flush()
            self._csv_f.close()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# =============================================================================
# Main
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_l", type=int, default=-1, help="左摄像头 ID (-1=自动检测)")
    ap.add_argument("--cam_r", type=int, default=-1, help="右摄像头 ID (-1=自动检测)")
    ap.add_argument("--no_flip_right", action="store_true")
    ap.add_argument("--hz", type=float, default=25.0)
    ap.add_argument("--exposure", type=float, default=166.0,
                    help="manual exposure target; use -1 to skip (paper: 166)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--can", type=str, default="can0")
    ap.add_argument("--no_piper", action="store_true")
    ap.add_argument("--out_csv", type=str, default="")

    ap.add_argument("--fn_stop", type=float, default=1.3)
    ap.add_argument("--init_effort", type=int, default=1200)
    ap.add_argument("--max_effort", type=int, default=5000)
    ap.add_argument("--min_stroke_um", type=int, default=20*1000)
    ap.add_argument("--max_stroke_um", type=int, default=100*1000)
    ap.add_argument("--close_step_um", type=int, default=150)

    ap.add_argument("--sy_th", type=float, default=0.20)
    ap.add_argument("--slip_effort_step", type=int, default=300)
    ap.add_argument("--slip_pos_step", type=int, default=80)
    ap.add_argument("--max_pos_adj_um", type=int, default=3000)

    ap.add_argument("--sy_quiet", type=float, default=0.12)
    ap.add_argument("--cop_dec_th", type=float, default=-10.0, help="CoP辅助确认减重")
    ap.add_argument("--squeeze_effort_step", type=int, default=200)
    ap.add_argument("--squeeze_pos_step", type=int, default=60)
    ap.add_argument("--fn_protect_limit", type=float, default=45.0,
                    help="EMA Fn_max protection threshold; recalibrate per sensor condition")

    ap.add_argument("--headless", action="store_true",
                    help="无 GUI 模式 (跳过可视化, 终端输出状态)")
    ap.add_argument("--auto_start", action="store_true",
                    help="标定完成后自动开始抓取 (配合 --headless)")

    return ap.parse_args()

def main():
    args = parse_args()

    if args.cam_l < 0 or args.cam_r < 0:
        print("🔍 自动检测触觉摄像头...")
        auto_l, auto_r = auto_detect_cameras()
        if args.cam_l < 0:
            if auto_l is not None:
                args.cam_l = auto_l
                print(f"  ✅ 左摄像头: /dev/video{auto_l}  (USB port {DEFAULT_USB_MAP['left']})")
            else:
                print("  ❌ 左摄像头自动检测失败，请用 --cam_l 手动指定")
                sys.exit(1)
        if args.cam_r < 0:
            if auto_r is not None:
                args.cam_r = auto_r
                print(f"  ✅ 右摄像头: /dev/video{auto_r}  (USB port {DEFAULT_USB_MAP['right']})")
            else:
                print("  ❌ 右摄像头自动检测失败，请用 --cam_r 手动指定")
                sys.exit(1)

    params = ReflexParams(
        hz=args.hz,
        max_stroke_um=args.max_stroke_um,
        min_stroke_um=args.min_stroke_um,
        close_step_um=args.close_step_um,
        init_effort=args.init_effort,
        max_effort=args.max_effort,
        fn_stop_threshold=args.fn_stop,
        sy_threshold=args.sy_th,
        slip_effort_step=args.slip_effort_step,
        slip_pos_step_um=args.slip_pos_step,
        max_pos_adjust_um=args.max_pos_adj_um,
        sy_quiet_threshold=args.sy_quiet,
        cop_decrease_th_px=args.cop_dec_th,
        squeeze_effort_step=args.squeeze_effort_step,
        squeeze_pos_step_um=args.squeeze_pos_step,
        fn_protect_limit=args.fn_protect_limit,
    )

    if args.out_csv:
        out_csv = args.out_csv
    else:
        os.makedirs("logs", exist_ok=True)
        out_csv = os.path.join("logs", f"tactile_exp_v2_{now_str()}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    print("="*80)
    print("tactile_exp_v2 (threaded capture + parallel compute)")
    print(f"sy_th={params.sy_threshold}  cop_dec_th={params.cop_decrease_th_px}")
    print(f"slip_effort_step={params.slip_effort_step}  squeeze_effort_step={params.squeeze_effort_step}")
    print(f"fn_protect_limit={params.fn_protect_limit}")
    print(f"target_hz={params.hz}")
    print(f"csv: {out_csv}")
    print("="*80)

    controller = DualTactileReflexController(
        cam_left=args.cam_l,
        cam_right=args.cam_r,
        can_iface=args.can,
        use_piper=not args.no_piper,
        params=params,
        out_csv=out_csv,
        flip_right=not args.no_flip_right,
        headless=args.headless,
        exposure=None if args.exposure < 0 else args.exposure,
        width=args.width,
        height=args.height,
    )

    if not controller.auto_init(wait_sec=1.5):
        controller.cleanup()
        return

    if args.auto_start:
        controller.start_grasping()

    if args.headless:
        print("\n🖥️  Headless mode — Ctrl+C to quit\n")
        _run_headless(controller)
    else:
        print("\nKeys: [s]=start  [r]=reset  [q]=quit\n")
        _run_gui(controller)


def _run_gui(controller):
    try:
        while True:
            vis, result, (avg_freq, frame_ms) = controller.update()
            if vis is not None:
                cv2.imshow(controller.window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                controller.reset()
            elif key == ord('s'):
                controller.start_grasping()

            if (controller.state != controller.STATE_IDLE) and (int(time.time() * 2) % 2 == 0) and (result is not None):
                sys.stdout.write(
                    f"\r📊 Freq:{avg_freq:5.1f}Hz  Fn:{result['Fn_min']:6.3f}  Sy:{result['Sy_max']:4.2f}  "
                    f"pos:{result['pos_um']/1000:5.1f}mm  effort:{result['effort']:4d}  "
                    f"slip:{int(result['slip_any'])}  adj:{controller.current_adjusted_um:4d}um   "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n⚠️ interrupted")
    finally:
        controller.cleanup()
        print("\n👋 exit safe")


def _run_headless(controller):
    _frame_count = 0
    _last_print = 0.0
    PRINT_INTERVAL = 0.5

    try:
        while True:
            _, result, (avg_freq, frame_ms) = controller.update()
            _frame_count += 1

            now = time.time()
            if result is not None and (now - _last_print) >= PRINT_INTERVAL:
                _last_print = now
                state_str = ["IDLE", "CLOSING", "HOLDING"][controller.state]
                action_str = controller._last_action or "---"
                sys.stdout.write(
                    f"\r[{state_str:8s}] "
                    f"Freq:{avg_freq:5.1f}Hz ({frame_ms:5.1f}ms)  "
                    f"Fn:{result['Fn_min']:6.3f}  Sy:{result['Sy_max']:4.2f}  "
                    f"pos:{result['pos_um']/1000:5.1f}mm  eff:{result['effort']:4d}  "
                    f"slip:{int(result['slip_any'])}  adj:{controller.current_adjusted_um:4d}um  "
                    f"act:{action_str:7s}  "
                    f"frames:{_frame_count}   "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n⚠️ interrupted (total frames: {_frame_count})")
    finally:
        controller.cleanup()
        print("\n👋 exit safe")

if __name__ == "__main__":
    main()
