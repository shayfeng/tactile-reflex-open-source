#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tactile_exp_v1.py — Dual tactile reflex controller (Phase-1 experiment ready)

Based on: tactile_reflex_v3.py (your current v2 closed-loop architecture)

Phase-1 requirements covered:
1.1 Freeze + reproducibility:
    - --out_dir/--tag/--notes CLI
    - Dump config.json (args, params, system, git hash if available, camera locked readings)
1.2 Standard trial folder layout:
    - Writes metrics.csv + config.json into out_dir
1.3 CSV improvements:
    - Adds frame_id (monotonic)
    - Adds event edges: slip_on/slip_off/release_on/protect_on
    - Adds action_state (NONE/SLIP/RELEASE/PROTECT)
    - Logs Sy_raw (per side + max) in addition to Sy (EMA)

Keys:  s=start  r=reset  q=quit
"""

import argparse
import csv
import importlib.util
import sysconfig


import json
import os
import platform
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple
# --- FIX: handle local csv.py shadowing the stdlib csv ---
if not hasattr(csv, "DictWriter"):
    stdlib_dir = sysconfig.get_paths().get("stdlib")
    csv_path = os.path.join(stdlib_dir, "csv.py")
    spec = importlib.util.spec_from_file_location("_stdlib_csv", csv_path)
    _stdlib_csv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_stdlib_csv)  # type: ignore
    csv = _stdlib_csv
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

def safe_cmd(cmd: str, max_chars: int = 4000) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out[:max_chars]
    except Exception as e:
        return f"ERR: {e}"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ---- USB camera auto-detection ----
# 通过 USB 物理端口路径自动找到 /dev/videoN 编号，
# 不再受插拔顺序影响。

# 默认 USB 端口映射 (根据你的硬件，只需改这里)
DEFAULT_USB_MAP = {
    "left":  "10.4",      # usb-0000:00:14.0-10.4
    "right": "11.3.3",    # usb-0000:00:14.0-11.3.3
}

def find_cam_by_usb_port(usb_port_suffix: str) -> Optional[int]:
    """
    在 /sys/class/video4linux/ 中查找 USB 端口路径包含 usb_port_suffix 的
    主捕获设备，返回其 /dev/videoN 编号。仅 Linux 可用。
    """
    sys_base = "/sys/class/video4linux"
    if not os.path.isdir(sys_base):
        return None
    for dev_name in sorted(os.listdir(sys_base)):
        # dev_name = "video0", "video8", ...
        dev_path = os.path.join(sys_base, dev_name, "device")
        if not os.path.exists(dev_path):
            continue
        try:
            real = os.path.realpath(dev_path)
            if usb_port_suffix in real:
                num = int(dev_name.replace("video", ""))
                # 用 index 文件判断: index==0 表示主捕获节点
                index_path = os.path.join(sys_base, dev_name, "index")
                if os.path.exists(index_path):
                    with open(index_path) as f:
                        if f.read().strip() == "0":
                            return num
                else:
                    # fallback: 取同一 USB 口的第一个设备
                    return num
        except Exception:
            continue
    return None

def auto_detect_cameras() -> Tuple[Optional[int], Optional[int]]:
    """尝试自动检测左/右触觉摄像头，返回 (cam_l_id, cam_r_id)"""
    cam_l = find_cam_by_usb_port(DEFAULT_USB_MAP["left"])
    cam_r = find_cam_by_usb_port(DEFAULT_USB_MAP["right"])
    return cam_l, cam_r

def get_camera_lock_readback(cap: cv2.VideoCapture) -> Dict[str, float]:
    # Read back a few key props (may return 0/-1 depending on driver)
    return {
        "exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
        "auto_exposure": float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
        "wb_temperature": float(cap.get(cv2.CAP_PROP_WB_TEMPERATURE)),
        "auto_wb": float(cap.get(cv2.CAP_PROP_AUTO_WB)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

def lock_camera_settings(cap: cv2.VideoCapture, name: str = "CAM",
                         exposure: Optional[float] = 166.0,
                         width: int = 640, height: int = 480) -> Dict[str, float]:
    """
    Best-effort lock for V4L2. Not all drivers honor these.
    """
    try:
        # These are the paper settings.  Drivers may use different units or
        # reject them; the readback is always stored in config.json.
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # OpenCV + V4L2: AUTO_EXPOSURE often uses 1=manual, 3=auto (varies)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if exposure is not None and exposure >= 0:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    except Exception:
        pass

    rb = get_camera_lock_readback(cap)
    print(f"  [{name}] camera locked(readback): exposure={rb['exposure']:.1f}, WB={rb['wb_temperature']:.0f}, "
          f"auto_exp={rb['auto_exposure']:.1f}, auto_wb={rb['auto_wb']:.1f}")
    return rb

def dump_config(out_dir: str, args: argparse.Namespace, params: "ReflexParams",
                cam_l_rb: Dict[str, float], cam_r_rb: Dict[str, float]) -> None:
    ensure_dir(out_dir)
    cfg = {
        "tag": getattr(args, "tag", ""),
        "notes": getattr(args, "notes", ""),
        "time": now_str(),
        "args": vars(args),
        "params": asdict(params),
        "camera": {
            "left": cam_l_rb,
            "right": cam_r_rb,
        },
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "uname": safe_cmd("uname -a"),
            "git_rev": safe_cmd("git rev-parse HEAD").strip(),
            "git_status": safe_cmd("git status --porcelain").strip(),
        },
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# =============================================================================
# V3 tactile metrics
# =============================================================================

class TactileMetricsComputer:
    """
    Single tactile sensor metrics computer (V3 robust).
      Fn = Σ ΔI^γ / (H×W)   over contact mask
      Sy_raw = median(|v_y|) over contact mask; v_y is Farneback's vertical component
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

        # Fn
        Fn_raw = float(np.sum(weight)) / (h * w) if has_contact else 0.0

        # Sy (Farneback optical flow)
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

        # CoP
        CoP_x_raw, CoP_y_raw = w / 2.0, h / 2.0
        if has_contact:
            total_w = float(np.sum(weight))
            if total_w > 1e-6:
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                CoP_x_raw = float(np.sum(x_coords * weight) / total_w)
                CoP_y_raw = float(np.sum(y_coords * weight) / total_w)

        # EMA (only for display/slow control; Sy_raw kept for thresholds)
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
            # The paper defines CoP directly from the current weighted mask.
            # Keep the EMA internally for display/diagnostics, but expose the
            # raw centroid for the release controller and CSV semantics.
            'CoP_x':  CoP_x_raw,
            'CoP_y':  CoP_y_raw,
            'CoP_x_ema': float(self._ema['CoP_x']),
            'CoP_y_ema': float(self._ema['CoP_y']),
            'Area':   contact_px,
            'contact': bool(has_contact),
            'tau':    float(self.tau),
        }

        vis_data = {
            'delta_I': delta_I,
            'mask':    mask,
            'flow':    flow,
            'flow_mag': flow_mag,
        }
        return result, vis_data

    def _empty_result(self, shape) -> Dict:
        h, w = shape
        return {
            'Fn_raw': 0.0,
            'Fn': 0.0,
            'Sy_raw': 0.0,
            'Sy': 0.0,
            'Sy_vec': np.array([0.0, 0.0]),
            'CoP_x': w / 2.0,
            'CoP_y': h / 2.0,
            'Area': 0,
            'contact': False,
            'tau': self.tau,
        }

    def _empty_vis(self, shape) -> Dict:
        h, w = shape
        return {
            'delta_I': np.zeros((h, w), dtype=np.float32),
            'mask': np.zeros((h, w), dtype=bool),
            'flow': None,
            'flow_mag': np.zeros((h, w), dtype=np.float32),
        }


# =============================================================================
# Controller
# =============================================================================

@dataclass
class ReflexParams:
    hz: float = 25.0

    max_stroke_um: int = 100 * 1000
    min_stroke_um: int = 20 * 1000
    close_step_um: int = 150

    init_effort: int = 1200
    max_effort: int = 5000
    fn_stop_threshold: float = 1.3

    # anti-slip (Sy 驱动, 闭环)
    sy_threshold: float = 0.20              # θ_s  Sy > 此值 → 判定滑移
    slip_effort_step: int = 300
    slip_pos_step_um: int = 80
    max_pos_adjust_um: int = 3000

    # 减重/防挤压 (CoP 驱动 + Sy 排除)
    sy_quiet_threshold: float = 0.12        # θ_q  Sy < 此值 → "没在滑"
    cop_decrease_th_px: float = -10.0       # θ_c  ΔCoP_y < 此值 → 判定减重
    squeeze_effort_step: int = 200
    squeeze_pos_step_um: int = 60

    # Fn 硬保护 (最高优先级, 独立于其他逻辑)
    fn_protect_limit: float = 45            # F_lim  Fn > 此值 → 强制减力
    protect_effort_step: int = 400
    protect_pos_step_um: int = 100


class DualTactileReflexController:
    STATE_IDLE = 0
    STATE_CLOSING = 1
    STATE_HOLDING = 2

    def __init__(
        self,
        cam_left: int,
        cam_right: int,
        can_iface: str,
        use_piper: bool,
        params: ReflexParams,
        out_dir: str,
        flip_right: bool = True,
        slip_off_n: int = 5,
        exposure: Optional[float] = 166.0,
        width: int = 640,
        height: int = 480,
    ):
        self.window_name = "Tactile Reflex (Dual GelSight Mini)"
        self.params = params
        self.flip_right = flip_right
        self.out_dir = out_dir
        ensure_dir(self.out_dir)

        # detection
        self.slip_off_n = max(1, int(slip_off_n))

        # Cameras
        self.cap_l = cv2.VideoCapture(cam_left)
        self.cap_r = cv2.VideoCapture(cam_right)
        self.cam_l_rb = lock_camera_settings(self.cap_l, "L", exposure=exposure,
                                             width=width, height=height)
        self.cam_r_rb = lock_camera_settings(self.cap_r, "R", exposure=exposure,
                                             width=width, height=height)

        # Tactile computers
        self.comp_l = TactileMetricsComputer(gamma=1.5)
        self.comp_r = TactileMetricsComputer(gamma=1.5)

        # Gripper
        self.piper: Optional[C_PiperInterface_V2] = None
        self.use_piper = use_piper and C_PiperInterface_V2 is not None
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

        # State
        self.state = self.STATE_IDLE
        self.current_pos = self.params.max_stroke_um
        self.current_effort = self.params.init_effort
        self.hold_start_pos: Optional[int] = None
        self.current_adjusted_um: int = 0
        self.hold_fn_baseline: float = 0.0
        self.stable_cop_y_ref: Optional[float] = None
        self._last_action: str = ""

        # frequency meter
        self._freq_deque = deque(maxlen=20)
        self._last_tick = time.time()

        # CSV
        self.out_csv = os.path.join(self.out_dir, "metrics.csv")
        self._csv_f = open(self.out_csv, "w", newline="", encoding="utf-8")
        self.fieldnames = [
            # indexing / time
            "frame_id", "t_wall",
            # state & cmd
            "state", "pos_um", "effort", "adjusted_um",
            # tactile (EMA)
            "Fn_L", "Fn_R", "Fn_min", "Fn_max",
            "Sy_L", "Sy_R", "Sy_max",
            # tactile (raw)
            "Fn_raw_L", "Fn_raw_R", "Fn_raw_min", "Fn_raw_max",
            "Sy_raw_L", "Sy_raw_R", "Sy_raw_max",
            "CoP_x_L", "CoP_y_L", "CoP_x_R", "CoP_y_R", "cop_y_avg",
            "Area_L", "Area_R", "contact_L", "contact_R",
            "tau_L", "tau_R",
            # derived
            "delta_cop_y",
            # slip flags (level)
            "slip_L", "slip_R", "slip_any",
            # event edges
            "slip_on", "slip_off", "release_on", "protect_on",
            "action_state",
            # misc
            "hold_fn_baseline",
            "frame_ms", "avg_freq",
        ]
        self._csv_w = csv.DictWriter(self._csv_f, fieldnames=self.fieldnames)
        self._csv_w.writeheader()

        # event memory
        self.frame_id = 0
        self._prev_slip_any = False
        self._prev_action_state = "NONE"
        self._quiet_counter = 0

    def _send_gripper(self, pos_um: int, effort: int) -> None:
        if not self.use_piper or self.piper is None:
            return
        try:
            self.piper.GripperCtrl(abs(int(pos_um)), int(effort), 0x01, 0)
        except Exception as e:
            print(f"\n❌ GripperCtrl error: {e}")

    def auto_init(self, wait_sec: float = 1.5) -> bool:
        print("  📷 Initializing baselines...")

        def cap_l_gray() -> Optional[np.ndarray]:
            ret, frame = self.cap_l.read()
            if not ret:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def cap_r_gray() -> Optional[np.ndarray]:
            ret, frame = self.cap_r.read()
            if not ret:
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.flip(gray, 0) if self.flip_right else gray

        if not self.comp_l.reset_baseline_multi(cap_l_gray):
            print("    ❌ Left camera baseline failed")
            return False
        if not self.comp_r.reset_baseline_multi(cap_r_gray):
            print("    ❌ Right camera baseline failed")
            return False

        print("  🔧 Auto calibrating noise floors...")
        time.sleep(wait_sec)
        if not self.comp_l.auto_calibrate_noise(cap_l_gray):
            print("    ❌ Left noise calib failed")
            return False
        if not self.comp_r.auto_calibrate_noise(cap_r_gray):
            print("    ❌ Right noise calib failed")
            return False

        self._send_gripper(self.params.max_stroke_um, self.params.init_effort)
        print("  ✅ Initialized")
        return True

    def reset(self) -> None:
        self.state = self.STATE_IDLE
        self.current_pos = self.params.max_stroke_um
        self.current_effort = self.params.init_effort
        self.current_adjusted_um = 0
        self.hold_start_pos = None
        self.hold_fn_baseline = 0.0
        self.stable_cop_y_ref = None
        self._last_action = ""
        self._send_gripper(self.current_pos, self.current_effort)
        print("🔄 Reset to idle")

        # reset event memory (keep frame_id monotonic across run)
        self._prev_slip_any = False
        self._prev_action_state = "NONE"
        self._quiet_counter = 0

    def start_grasping(self) -> None:
        if self.state != self.STATE_IDLE:
            return
        self.state = self.STATE_CLOSING
        print("🤏 Starting grasp...")

    def _measure_frequency(self) -> Tuple[float, float]:
        now = time.time()
        dt = now - self._last_tick
        if dt > 1e-6:
            freq = 1.0 / dt
            self._freq_deque.append(freq)
        self._last_tick = now
        avg_freq = sum(self._freq_deque) / max(len(self._freq_deque), 1)
        frame_ms = 1000.0 / max(avg_freq, 1e-6)
        return avg_freq, frame_ms

    def update(self) -> Tuple[Optional[np.ndarray], Optional[Dict], Tuple[float, float]]:
        tick_start = time.time()

        ret_l, frame_l = self.cap_l.read()
        ret_r, frame_r = self.cap_r.read()
        if not ret_l or not ret_r:
            return None, None, (0.0, 0.0)

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        if self.flip_right:
            gray_r = cv2.flip(gray_r, 0)

        res_l, vis_l = self.comp_l.compute(gray_l)
        res_r, vis_r = self.comp_r.compute(gray_r)

        # Aggregates (aligned with v2)
        Fn_min = float(min(res_l["Fn"], res_r["Fn"]))
        Fn_raw_min = float(min(res_l["Fn_raw"], res_r["Fn_raw"]))
        Fn_max = float(max(res_l["Fn"], res_r["Fn"]))
        Sy_max = float(max(res_l["Sy"], res_r["Sy"]))
        Sy_raw_max = float(max(res_l["Sy_raw"], res_r["Sy_raw"]))

        # Slip detection (v2: Sy_raw + contact)
        slip_l = bool(res_l["contact"] and (res_l["Sy_raw"] > self.params.sy_threshold))
        slip_r = bool(res_r["contact"] and (res_r["Sy_raw"] > self.params.sy_threshold))
        slip_any = bool(slip_l or slip_r)

        # CoP (v2: only when dual_contact)
        dual_contact = res_l["contact"] and res_r["contact"]
        cop_y_avg = 0.0
        if dual_contact:
            cop_y_avg = (res_l["CoP_y"] + res_r["CoP_y"]) / 2.0

        delta_cop_y = 0.0
        if self.stable_cop_y_ref is not None:
            delta_cop_y = cop_y_avg - self.stable_cop_y_ref

        action_state = "NONE"

        # ============================================================
        #  State machine (aligned with v2)
        # ============================================================
        if self.state == self.STATE_CLOSING:
            if Fn_raw_min >= self.params.fn_stop_threshold:
                self.state = self.STATE_HOLDING
                print(f"🛑 Grasp stopped at Fn_raw_min={Fn_raw_min:.3f} | pos={self.current_pos/1000:.1f}mm | effort={self.current_effort}")
            elif self.current_pos <= self.params.min_stroke_um:
                self.state = self.STATE_HOLDING
                print(f"⚠️ stroke limit reached -> HOLDING")
            else:
                self.current_pos -= int(self.params.close_step_um)
                self.current_pos = max(self.current_pos, self.params.min_stroke_um)
                self._send_gripper(self.current_pos, self.params.init_effort)

        elif self.state == self.STATE_HOLDING:
            # ---- 初次进入 HOLDING: 记录基线 (v2: deferred, require dual_contact) ----
            if self.hold_start_pos is None and dual_contact:
                self.hold_start_pos = self.current_pos
                self.current_adjusted_um = 0
                self.stable_cop_y_ref = cop_y_avg
                self.hold_fn_baseline = Fn_min
                delta_cop_y = 0.0
                print(f"✅ Enter HOLDING | Fn_base={Fn_min:.3f} | CoP_y={cop_y_avg:.1f} | pos={self.current_pos/1000:.1f}mm")

            # 持续更新 CoP ref (慢速追踪, v2: 0.02 rate)
            if self.stable_cop_y_ref is not None and dual_contact:
                self.stable_cop_y_ref = 0.98 * self.stable_cop_y_ref + 0.02 * cop_y_avg

            # ============================================================
            #  控制律 (v2: Sy 闭环 + CoP 辅助)
            # ============================================================

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

                action_state = "SLIP"
                if self._last_action != action_state:
                    print(f"🚨 SLIP (Sy_raw={Sy_raw_max:.3f} > {self.params.sy_threshold}) | effort→{self.current_effort} | pos→{self.current_pos/1000:.1f}mm")

            # ---- 2) Release: CoP负向变化 + Sy安静 (v2: elif, dual_contact, Sy_ema for quiet) ----
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

                # CoP ref tracking (v2: 0.3 rate during release)
                self.stable_cop_y_ref += 0.3 * (cop_y_avg - self.stable_cop_y_ref)

                action_state = "RELEASE"
                if self._last_action != action_state:
                    print(f"⚖️ RELEASE (ΔCoP_y={delta_cop_y:.1f}px, Sy={Sy_max:.3f}) | effort→{self.current_effort} | pos→{self.current_pos/1000:.1f}mm")

            # Fn baseline tracking when quiet (v2)
            if action_state == "NONE" and self.hold_fn_baseline > 0:
                self.hold_fn_baseline = 0.95 * self.hold_fn_baseline + 0.05 * Fn_min

            # ============================================================
            #  Fn 硬保护 (最高优先级, 独立于上面所有逻辑, v2)
            # ============================================================
            if Fn_max > self.params.fn_protect_limit:
                self.current_effort = max(
                    self.current_effort - self.params.protect_effort_step,
                    self.params.init_effort)
                if self.hold_start_pos is not None and self.current_pos < self.hold_start_pos:
                    step = min(self.params.protect_pos_step_um,
                               self.hold_start_pos - self.current_pos)
                    self.current_pos += step
                    self.current_adjusted_um = max(0, self.current_adjusted_um - step)
                action_state = "PROTECT"
                if self._last_action != action_state:
                    print(f"🛑 PROTECT (Fn_max={Fn_max:.3f} > {self.params.fn_protect_limit}) | effort→{self.current_effort} | pos→{self.current_pos/1000:.1f}mm")

            self._last_action = action_state
            self._send_gripper(self.current_pos, self.current_effort)
        else:
            # IDLE
            pass

        # ============================================================
        # Events (edges)
        # ============================================================
        slip_on = int((not self._prev_slip_any) and slip_any)

        # slip_off: require "quiet for N frames" after being in slip_any
        slip_off = 0
        if self._prev_slip_any and (not slip_any):
            self._quiet_counter += 1
            if self._quiet_counter >= self.slip_off_n:
                slip_off = 1
                self._quiet_counter = 0
        else:
            self._quiet_counter = 0

        release_on = int((action_state == "RELEASE") and (self._prev_action_state != "RELEASE"))
        protect_on = int((action_state == "PROTECT") and (self._prev_action_state != "PROTECT"))

        self._prev_slip_any = slip_any
        self._prev_action_state = action_state

        # ============================================================
        # Logging
        # ============================================================
        avg_freq, frame_ms = self._measure_frequency()
        t_wall = time.time()

        row = {
            "frame_id": int(self.frame_id),
            "t_wall": f"{t_wall:.6f}",
            "state": ["IDLE", "CLOSING", "HOLDING"][self.state],
            "pos_um": int(self.current_pos),
            "effort": int(self.current_effort),
            "adjusted_um": int(self.current_adjusted_um),

            "Fn_L": f"{res_l['Fn']:.6f}",
            "Fn_R": f"{res_r['Fn']:.6f}",
            "Fn_min": f"{Fn_min:.6f}",
            "Fn_max": f"{Fn_max:.6f}",

            "Sy_L": f"{res_l['Sy']:.6f}",
            "Sy_R": f"{res_r['Sy']:.6f}",
            "Sy_max": f"{Sy_max:.6f}",

            "Fn_raw_L": f"{res_l['Fn_raw']:.6f}",
            "Fn_raw_R": f"{res_r['Fn_raw']:.6f}",
            "Fn_raw_min": f"{Fn_raw_min:.6f}",
            "Fn_raw_max": f"{max(res_l['Fn_raw'], res_r['Fn_raw']):.6f}",

            "Sy_raw_L": f"{res_l['Sy_raw']:.6f}",
            "Sy_raw_R": f"{res_r['Sy_raw']:.6f}",
            "Sy_raw_max": f"{Sy_raw_max:.6f}",

            "CoP_x_L": f"{res_l['CoP_x']:.3f}",
            "CoP_y_L": f"{res_l['CoP_y']:.3f}",
            "CoP_x_R": f"{res_r['CoP_x']:.3f}",
            "CoP_y_R": f"{res_r['CoP_y']:.3f}",
            "cop_y_avg": f"{cop_y_avg:.3f}",

            "Area_L": int(res_l["Area"]),
            "Area_R": int(res_r["Area"]),
            "contact_L": int(res_l["contact"]),
            "contact_R": int(res_r["contact"]),
            "tau_L": f"{res_l['tau']:.3f}",
            "tau_R": f"{res_r['tau']:.3f}",

            "delta_cop_y": f"{delta_cop_y:.3f}",

            "slip_L": int(slip_l),
            "slip_R": int(slip_r),
            "slip_any": int(slip_any),

            "slip_on": int(slip_on),
            "slip_off": int(slip_off),
            "release_on": int(release_on),
            "protect_on": int(protect_on),
            "action_state": action_state,

            "hold_fn_baseline": f"{self.hold_fn_baseline:.6f}",
            "frame_ms": f"{frame_ms:.3f}",
            "avg_freq": f"{avg_freq:.3f}",
        }
        self._csv_w.writerow(row)
        self.frame_id += 1

        # flush periodically
        if (int(time.time() * 10) % 10) == 0:
            self._csv_f.flush()

        # UI (optional)
        vis = self._draw_ui(gray_l, res_l, vis_l, gray_r, res_r, vis_r, avg_freq, frame_ms, slip_any, delta_cop_y, action_state)
        target_dt = 1.0 / max(self.params.hz, 1e-6)
        used = time.time() - tick_start
        if used < target_dt:
            time.sleep(target_dt - used)

        return vis, {
            "L": res_l, "R": res_r,
            "Fn_min": Fn_min, "Fn_max": Fn_max,
            "Sy_max": Sy_max, "Sy_raw_max": Sy_raw_max,
            "slip_any": slip_any,
            "state": self.state,
            "pos_um": self.current_pos,
            "effort": self.current_effort,
            "action": action_state,
            "delta_cop_y": delta_cop_y,
        }, (avg_freq, frame_ms)

    def _draw_ui(self, gray_l, res_l, vis_l, gray_r, res_r, vis_r,
                 avg_freq, frame_ms, slip_any: bool, delta_cop_y: float, action_state: str):
        frame_l = cv2.cvtColor(gray_l, cv2.COLOR_GRAY2BGR)
        frame_r = cv2.cvtColor(gray_r, cv2.COLOR_GRAY2BGR)
        frame_l = self._draw_vis(frame_l, res_l, vis_l, "L")
        frame_r = self._draw_vis(frame_r, res_r, vis_r, "R")
        vis = np.hstack([frame_l, frame_r])

        status = (f"Freq:{avg_freq:.1f}Hz ({frame_ms:.1f}ms)  "
                  f"State:{['IDLE','CLOSING','HOLDING'][self.state]}  "
                  f"pos:{self.current_pos/1000:.1f}mm effort:{self.current_effort} adj:{self.current_adjusted_um}um  "
                  f"Fn_max:{max(res_l['Fn'], res_r['Fn']):.3f}  "
                  f"Sy_raw_max:{max(res_l['Sy_raw'], res_r['Sy_raw']):.3f}  "
                  f"ΔCoP_y:{delta_cop_y:.1f}px  slip:{int(slip_any)}  act:{action_state}")
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def _draw_vis(self, frame_bgr: np.ndarray, res: Dict, vis: Dict, side: str) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if res['contact']:
            mask = vis['mask'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_bgr, contours, -1, (0, 255, 0), 1)
            cx, cy = int(res['CoP_x']), int(res['CoP_y'])
            cv2.drawMarker(frame_bgr, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

        # Flow arrows (optional)
        if vis.get('flow', None) is not None:
            step = 16
            ys, xs = np.mgrid[step//2:h:step, step//2:w:step]
            ys, xs = ys.flatten(), xs.flatten()
            fx = vis['flow'][ys, xs, 0]
            fy = vis['flow'][ys, xs, 1]
            mags = vis['flow_mag'][ys, xs]
            for i in range(len(xs)):
                if mags[i] < 0.3:
                    continue
                pt1 = (int(xs[i]), int(ys[i]))
                pt2 = (int(xs[i] + fx[i] * 4), int(ys[i] + fy[i] * 4))
                cv2.arrowedLine(frame_bgr, pt1, pt2, (0, 180, 255), 1, tipLength=0.3)

        border_color = (0, 220, 100) if res['contact'] else (80, 80, 80)
        cv2.rectangle(frame_bgr, (0, 0), (w-1, h-1), border_color, 2)

        info = (f"[{side}] Fn={res['Fn']:.4f} Sy_raw={res['Sy_raw']:.3f} "
                f"Sy={res['Sy']:.3f} CoP=({res['CoP_x']:.0f},{res['CoP_y']:.0f}) "
                f"Area={res['Area']} tau={res['tau']:.1f}")
        cv2.rectangle(frame_bgr, (0, h-30), (w, h), (0, 0, 0), -1)
        cv2.putText(frame_bgr, info, (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr

    def cleanup(self):
        try:
            self._send_gripper(self.params.max_stroke_um, self.params.init_effort)
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
    # cameras / runtime
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

    # phase1 reproducibility
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--notes", type=str, default="")

    # event/detection
    ap.add_argument("--slip_off_n", type=int, default=5, help="slip_off requires quiet for N frames")

    # gripper
    ap.add_argument("--fn_stop", type=float, default=1.3)
    ap.add_argument("--init_effort", type=int, default=1200)
    ap.add_argument("--max_effort", type=int, default=5000)
    ap.add_argument("--min_stroke_um", type=int, default=20*1000)
    ap.add_argument("--max_stroke_um", type=int, default=100*1000)
    ap.add_argument("--close_step_um", type=int, default=150)

    # anti-slip
    ap.add_argument("--sy_th", type=float, default=0.20)
    ap.add_argument("--sy_quiet", type=float, default=0.12)
    ap.add_argument("--slip_effort_step", type=int, default=300)
    ap.add_argument("--slip_pos_step", type=int, default=80)
    ap.add_argument("--max_pos_adj_um", type=int, default=3000)

    # release
    ap.add_argument("--cop_dec_th", type=float, default=-10.0, help="CoP辅助确认减重方向")
    ap.add_argument("--squeeze_effort_step", type=int, default=200)
    ap.add_argument("--squeeze_pos_step", type=int, default=60)

    # protect
    ap.add_argument("--fn_protect_limit", type=float, default=45)
    ap.add_argument("--protect_effort_step", type=int, default=400)
    ap.add_argument("--protect_pos_step", type=int, default=100)

    return ap.parse_args()

def main():
    args = parse_args()

    # ---- 摄像头自动检测 ----
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

    # out_dir default
    out_dir = args.out_dir.strip()
    if not out_dir:
        tag = args.tag.strip() or f"trial_{now_str()}"
        out_dir = os.path.join("logs", tag)
    ensure_dir(out_dir)

    params = ReflexParams(
        hz=args.hz,
        max_stroke_um=args.max_stroke_um,
        min_stroke_um=args.min_stroke_um,
        close_step_um=args.close_step_um,
        init_effort=args.init_effort,
        max_effort=args.max_effort,
        fn_stop_threshold=args.fn_stop,
        sy_threshold=args.sy_th,
        sy_quiet_threshold=args.sy_quiet,
        slip_effort_step=args.slip_effort_step,
        slip_pos_step_um=args.slip_pos_step,
        max_pos_adjust_um=args.max_pos_adj_um,
        cop_decrease_th_px=args.cop_dec_th,
        squeeze_effort_step=args.squeeze_effort_step,
        squeeze_pos_step_um=args.squeeze_pos_step,
        fn_protect_limit=args.fn_protect_limit,
        protect_effort_step=args.protect_effort_step,
        protect_pos_step_um=args.protect_pos_step,
    )

    print("="*80)
    print("tactile_exp_v1 (Phase-1 ready)")
    print(f"out_dir: {out_dir}")
    print(f"sy_th={params.sy_threshold} sy_quiet={params.sy_quiet_threshold} cop_dec_th={params.cop_decrease_th_px}")
    print(f"fn_protect_limit={params.fn_protect_limit}  slip_detect=Sy_raw  quiet_detect=Sy_ema")
    print("="*80)

    controller = DualTactileReflexController(
        cam_left=args.cam_l,
        cam_right=args.cam_r,
        can_iface=args.can,
        use_piper=not args.no_piper,
        params=params,
        out_dir=out_dir,
        flip_right=not args.no_flip_right,
        slip_off_n=args.slip_off_n,
        exposure=None if args.exposure < 0 else args.exposure,
        width=args.width,
        height=args.height,
    )

    # dump config AFTER camera lock readback exists
    dump_config(out_dir, args, params, controller.cam_l_rb, controller.cam_r_rb)

    if not controller.auto_init(wait_sec=1.5):
        controller.cleanup()
        return

    print("\nKeys: [s]=start  [r]=reset  [q]=quit\n")

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

            if result is not None and (int(time.time() * 2) % 2 == 0):
                sys.stdout.write(
                    f"\r📊 Freq:{avg_freq:5.1f}Hz  Fn_min:{result['Fn_min']:6.3f} Fn_max:{result['Fn_max']:6.3f}  "
                    f"Sy_raw:{result['Sy_raw_max']:5.3f}  pos:{result['pos_um']/1000:5.1f}mm  "
                    f"effort:{result['effort']:4d}  slip:{int(result['slip_any'])}  act:{result['action']:<7}   "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n⚠️ interrupted")
    finally:
        controller.cleanup()
        print("\n👋 exit safe")

if __name__ == "__main__":
    main()
