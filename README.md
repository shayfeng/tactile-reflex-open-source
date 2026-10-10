# TactileReflex：参数标定与 Reflex Controller

论文：[TactileReflex: Closed-Loop Tactile Reflex Control for Safe Deformable Object Manipulation via Vision-Based Tactile Sensing](https://arxiv.org/abs/2605.23568)
项目主页：[shayfeng.github.io/TactileReflex](https://shayfeng.github.io/TactileReflex/)

This repository includes subsequent improvements and sensor-specific extensions beyond the implementation evaluated in the IROS 2026 paper. Implementation details and calibrated thresholds may therefore differ from those reported in the paper.

本目录包含论文实验使用的 MC-TAC 控制器、约 25 Hz 的优化版本，以及 DM-TAC 和 Xense 后端：

- `tactile_exp_v1.py`：双相机触觉计算、启动基线标定、三通道 Reflex Controller、实时显示和 `metrics.csv/config.json` 记录。
- `tactile_exp_v2.py`：优化后的 MC-TAC Reflex Controller（异步相机线程 + 左右并行计算，目标约 25 Hz），输出事件增强版 CSV。
- `calibrate_thresholds.py`：从 E1/E2/E3 试验 CSV 计算阈值候选。

运行标定脚本会自动计算传感器相关阈值；`fn_stop` 是选定控制配置中的统一常量，由 `--fn_stop` 一并写入结果。输出仍应经过独立的可恢复实机试验确认。

MC-TAC 的 `Sy_raw` 按论文式（6）计算为接触区域内 `median(abs(v_y))`；二维光流幅值只用于可视化。修改这一统计量后，旧 CSV 中的 `Sy` 分布和 `sy_th`/`sy_quiet` 不再直接适用，应重新执行 E1–E3 标定。

## 1. 论文配置与适用范围

论文使用双 MC-Tac/GelSight 类传感器、`640×480` 图像、手动曝光 `166`、约 `11.5–12 Hz` 的带 UI 实际循环；左/右 USB 端口映射为 `10.4`/`11.3.3`。代码会尝试设置并回读这些条件，但不同 V4L2 驱动可能不接受同一数值，必须检查 `config.json`。

代码按“先读左相机，再读右相机”的顺序采集一对图像。若使用硬件触发同步，需要替换采集层并重新验证 `Sy`、CoP 和阈值。

`fn_stop` 使用选定控制配置中的固定接触/闭合停止常量；其余由传感器信号决定的阈值仍需随工况复核。更换传感器/胶体、分辨率或裁剪、曝光/白平衡/光源、安装预紧、相机位置、物体材料或载荷后，重新执行步骤 2–6。`exp_v1` 和 `exp_v2` 使用了不同的采集时序/ROI，阈值分别验证。

## 2. 安装

在本 README 所在目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install numpy opencv-python
python tactile_exp_v1.py --help
python tactile_exp_v2.py --help
python calibrate_thresholds.py --help
```

真实 Piper 运行需要与固件匹配的 `piper_sdk.C_PiperInterface_V2` 和已配置的 CAN 接口。其他夹爪需要适配初始化及 `_send_gripper()`。

## 3. 从零复现

每次试验使用独立目录，例如 `output/E1_static_soft_01`。

### 步骤 1：相机、方向和安全检查

```bash
v4l2-ctl --list-devices
```

确认左右 `/dev/videoN`，检查急停、夹爪行程和物体支撑。论文配置应在相机驱动中设为 `640×480`、手动曝光 `166`，并关闭自动曝光/白平衡；程序会再次尝试锁定并在 `config.json` 回读。若驱动不支持 166，记录实际值并按新工况重新标定。右图像若无需上下翻转，所有命令加 `--no_flip_right`。

先做不发机器人命令的检查：

```bash
python tactile_exp_v1.py --no_piper --cam_l 0 --cam_r 1 \
  --tag camera_check --out_dir output/camera_check
```

启动时两侧必须无接触。等待 `Initialized`，检查画面方向、`tau_L/tau_R`、`avg_freq` 和 `config.json` 的分辨率/曝光回读，按 `q` 退出。

### 步骤 2：E1 静态保持

用有支撑的物体建立双侧稳定接触：

```bash
python tactile_exp_v1.py --cam_l 0 --cam_r 1 --can can0 \
  --tag E1_static_soft_01 --out_dir output/E1_static_soft_01
```

启动后先保持无接触，等待 `Initialized`；按 `s` 低速闭合；进入 `HOLDING` 后保持 60–120 秒，不推、不拉、不抬起；按 `q` 退出。检查两侧接触、曝光和频率。双侧接触不成立或明显漂移时，重新采集。

程序启动会分别平均约 30 帧无接触图像，再用无接触差分的 P98 得到接触掩膜阈值 `tau`。控制阈值由后续 E1/E2/E3 数据分析得到；无接触噪声与稳定保持数据分开记录。

### 步骤 3：E2 小幅滑移/liftoff

重新启动并使用新目录：

```bash
python tactile_exp_v1.py --cam_l 0 --cam_r 1 --can can0 \
  --tag E2_slip_soft_01 --out_dir output/E2_slip_soft_01
```

按 `s` 建立稳定夹持后，在物体仍可被接住时移除支撑、轻微下拉或侧推。记录外部观察到的滑移，停止扰动后按 `q`。检查 `slip_on` 边沿与触觉峰值的对应关系。

### 步骤 4：E3 减载/CoP

```bash
python tactile_exp_v1.py --cam_l 0 --cam_r 1 --can can0 \
  --tag E3_unload_soft_01 --out_dir output/E3_unload_soft_01
```

按 `s` 后对物体做小幅、可恢复的加载再释放，重复 2–3 次，观察 `delta_cop_y` 的方向。论文 release 判据是负向 CoP 变化且 EMA `Sy_max` 安静；方向相反时检查相机翻转和坐标定义。

### 步骤 5：分析 CSV 得到候选值

```bash
python calibrate_thresholds.py \
  --static output/E1_static_soft_01/metrics.csv \
  --slip output/E2_slip_soft_01/metrics.csv \
  --unload output/E3_unload_soft_01/metrics.csv \
  --fn_stop 1.3 \
  --out output/soft_threshold_candidates.json
cat output/soft_threshold_candidates.json
```

默认只使用 `state == HOLDING`。三类输入分别对应静态保持、滑移和减载：

- `sy_quiet`：稳定保持期 **raw `Sy_raw_max` 的 P95 + 加性余量**；运行时 release 比较 EMA `Sy_max`。余量可用 `--sy_quiet_margin` 调整。
- `sy_th`：满足论文约束 `P99.9(静态 raw Sy) ≤ sy_th ≤ 最弱观测滑移` 的候选中点；分布重叠时输出 `null`。滑移记录应包含受控事件边沿，不能把普通静态帧冒充滑移。
- `cop_dec_th`：静态 CoP 漂移与受控减载事件的候选分界；必须检查负向方向和安静剪切门控。
- `fn_protect_limit`：稳定保持期 **raw `Fn_raw_max` 的 P99.9 × (1+余量)**；运行时保护比较 EMA `Fn_max`。余量可用 `--fn_protect_margin` 调整。
- `fn_stop`：统一的控制器接触/闭合停止常量（软杯历史值 `1.3`，硬杯历史值 `5.5`），脚本通过 `--fn_stop` 写入输出配置；运行时使用两侧 raw `Fn` 的最小值。运行标定脚本时用 `--fn_stop` 选择对应配置。

若 `sy_th` 或 `cop_dec_th` 为 `null`，重新采集对应试验。若静态噪声与滑移分布重叠，先改善成像、接触掩膜或采样。

### 步骤 6：人工确认后运行控制器

下面是论文软杯的历史起始值。新传感器使用步骤 5 的候选值，并通过物理试验确认：

```bash
python tactile_exp_v1.py --cam_l 0 --cam_r 1 --can can0 \
  --min_stroke_um 20000 --max_stroke_um 70000 \
  --fn_stop 1.3 --sy_th 0.20 --sy_quiet 0.12 \
  --cop_dec_th -10.0 --fn_protect_limit 45 \
  --init_effort 1200 --max_effort 5000 \
  --tag reflex_validated_01 --out_dir output/reflex_validated_01
```

按 `s` 开始闭合，观察 `CLOSING → HOLDING`、`SLIP`、`RELEASE`、`PROTECT` 和夹爪实际动作；按 `q` 退出。检查 `config.json` 保存的参数、相机回读、代码版本和 `metrics.csv`。

验证时先做小幅、可恢复试验，观察物体状态和保护响应延迟。HOLDING 表示达到 `fn_stop` 或最小行程。

### MC-TAC exp_v2（约 25 Hz 优化版）

`tactile_exp_v2.py` 使用与 exp_v1 相同的 Reflex 控制律，采用后台抓帧和左右并行计算，主循环目标频率默认为 25 Hz。实际频率以 CSV 的 `freq_hz_avg`、`frame_ms` 和物体观察为准。

exp_v2 默认使用论文平台上的 ROI `(x1=40, y1=0, x2=640, y2=450)`，所以处理图像是裁剪后的约 `600×450` 区域。改变 ROI、相机分辨率或帧率后，重新做 E1/E2/E3；exp_v1 和 exp_v2 的像素阈值分别验证。

先做不发 Piper 命令的感知检查：

```bash
python tactile_exp_v2.py --no_piper --headless \
  --cam_l 0 --cam_r 1 --hz 25 \
  --out_csv output/exp_v2_sense_check.csv
```

完成无接触基线后，使用有支撑物体做低 effort 试验：

```bash
python tactile_exp_v2.py --headless \
  --cam_l 0 --cam_r 1 --can can0 --hz 25 \
  --fn_stop 1.3 --sy_th 0.20 --sy_quiet 0.12 \
  --cop_dec_th -10.0 --fn_protect_limit 45 \
  --min_stroke_um 20000 --max_stroke_um 70000 \
  --out_csv output/exp_v2_reflex_01.csv
```

使用 `--auto_start` 才会在标定后自动闭合；GUI 模式按 `s` 开始、`r` 重新标定、`q` 退出。exp_v2 CSV 记录 `Fn_raw_*`、`Sy_raw_*`、`slip_on/off`、`release_on`、`protect_on` 和 `action_state`，可用于阈值分析。`Sy_raw` 是接触区域内相邻已处理帧竖直光流的 `median(abs(v_y))`，结合 `frame_ms` 读取。exp_v2 使用“只保留最新帧”的采集策略，可能跳过相机中间帧；因此 `frame_ms` 表示控制循环间隔，不保证每行都是新的硬件同步双帧。需要严格逐帧分析时，应在采集线程中加入帧序号和时间戳。

## 4. 控制逻辑与参数

1. **Anti-slip（R2）**：任一接触侧 raw `Sy > sy_th`，增加 effort 并收紧。
2. **Release（R3）**：R2 未触发，双侧接触，`delta_cop_y < cop_dec_th` 且 EMA `Sy_max < sy_quiet` 时降低 effort，并撤销额外收紧的位置。R2/R3 使用 `if/elif`，同周期互斥。
3. **Protect（R1）**：随后独立检查 EMA `Fn_max > fn_protect_limit`，执行减力和放松，可覆盖前两个动作。该通道配合硬件急停使用。

`Fn` 是图像代理量，不是牛顿；effort 是发送给 Piper 的命令，不是测得的夹持力。`fn_stop` 使用两侧 raw `Fn` 的最小值；保护使用 EMA `Fn_max`，两者不要混用。raw `Sy` 单位是像素/相邻处理帧，会随图像尺度和实际帧间隔变化。保护存在 EMA 和执行延迟。

论文软杯/硬杯历史值：

| CLI 参数 | 软杯 | 硬杯 | 运行时信号 |
|---|---:|---:|---|
| `--fn_stop` | 1.3 | 5.5 | 两侧 raw `Fn` 的最小值 |
| `--sy_th` | 0.20* | 0.20* | 接触侧 raw `Sy = median(abs(v_y))` |
| `--sy_quiet` | 0.12* | 0.10* | EMA `Sy_max` |
| `--cop_dec_th` | -10.0 | -3.5 | CoP 相对参考，px |
| `--fn_protect_limit` | 45 | 100 | EMA `Fn_max` |
| `gamma` | 1.5 | 1.5 | 源码固定 |
| `ema_alpha` | 0.3 | 0.3 | 源码固定 |

带 `*` 的 `sy_th` 和 `sy_quiet` 是原论文工况下的历史值；切换到当前 `abs(v_y)` 实现后，必须用新的 E1–E3 CSV 重新确认。

还需按工况复核 `min_contact_px=200`、5×5 morphology、Farneback 参数、`gamma`、EMA、参考跟踪率（背景 0.02、release 0.3）、控制增量、行程和实际频率。改变其中任一项后重新做 E1–E3。

## 5. DM-TAC 和 Xense 后端

原论文使用 MC-TAC；本仓库另外提供两个可选后端：

- `tactile_reflex_dmtac.py`：DM-TAC SDK 原生输出（infer/depth/deformation flow/shear flow/6D force/contact area），可选 AgxGripper。
- `tactile_reflex_xsense.py`：XenseSDK 原生 `Force`、`ForceResultant`、`Marker2D` 和 CoP，保留同一套 `CLOSING → HOLDING`、Anti-slip、Release、Protect 策略。

DM-TAC 和 Xense 的力、位移、剪切和接触面积采用各自 SDK 单位。两种后端分别进行无载标定、轴向确认和小幅可恢复物理试验；`calibrate_thresholds.py` 用于 MC-TAC 图像入口生成的 `metrics.csv`。

### 5.1 DM-TAC

DM-TAC 需要厂商 `dmrobotics`、`pyAgxArm` 和对应 CUDA/推理运行库。SDK 不随本仓库分发。安装后按实际路径设置：

```bash
export DMTAC_SDK_ROOT=/path/to/DM-Tac
export DMTAC_PYTHON_SITE=/path/to/dmpython310/lib/python3.10/site-packages
export DMTAC_REPO_ROOT=/path/to/repo-containing-pyAgxArm
python tactile_reflex_dmtac.py --help
```

先只检查传感器，不连接夹爪：

```bash
python tactile_reflex_dmtac.py \
  --dev_ids 2 4 --no_gripper --no_display --duration_s 30 \
  --event_file output/dmtac_sense_events.json
```

`--dev_ids` 替换为本机设备索引或序列号。启动后保持两侧 DM-TAC 表面无接触，脚本用 `--calibration_frames`（默认 45）估计深度、力和剪切噪声。`--slip_flow_threshold`、`--min_force_norm`、`--min_depth_delta`、`--min_contact_area` 随 SDK 模式、推理后端、采样率、安装预紧和接触材料调整。DM-TAC 的 flow 单位与 MC-TAC 的像素光流 `sy_th` 不同。

第一次运行建议始终使用 `--no_gripper`。确认设备编号、图像、深度、力和剪切正确后，再按 AgxGripper 文档填写接口、固件和 CAN 参数。`--auto_start_gripper` 才会自动闭合；否则用 `s` 开始，`r` 重置，`c` 重新无接触标定，`q`/Esc 退出：

```bash
python tactile_reflex_dmtac.py \
  --dev_ids 2 4 --backend cpu --mode standard \
  --gripper_robot nero --gripper_firmware v120 \
  --gripper_interface socketcan --gripper_channel can0 \
  --event_file output/dmtac_reflex_events.json
```

### 5.2 Xense

Xense 后端需要厂商 `xensesdk`、兼容的 ONNX Runtime/CUDA 依赖和两只已枚举的 Xense 传感器。默认是 sense-only，不会移动 Piper。先确认序列号并做无夹爪检查：

在 Python 3.9/3.10 环境按厂商版本安装 SDK（GPU/CUDA 依赖需与驱动匹配）：

```bash
python -m pip install "xensesdk[full]"
```

```bash
python tactile_reflex_xsense.py --help
python tactile_reflex_xsense.py \
  --left-sn OG000656 --right-sn OG000691 --infer-device auto \
  --headless --max-frames 300 --out-csv output/xense_sense_check.csv
```

启动后两侧表面必须无载荷。脚本默认采集 30 帧无载数据，估计力单元噪声死区、法向接触门槛和 Marker2D 速度噪声，并检查帧新鲜度和左右 pair skew。`--infer-device cuda` 要求两个传感器实际都使用 CUDA；`auto` 允许回退 CPU，但应阅读终端提示。

Xense 的 `Fn`/`fn-stop`/`fn-protect` 是 SDK 力代理单位；`SlipY` 是物理竖直方向 Marker2D 速度（校正像素/源帧）；CoP 阈值是物理竖直 CoP 像素变化。`--left/right-*-sign`、`--normal-sign` 和 `--no-flip-right` 必须用已知方向的小幅加载、卸载和滑移测试确认。

启用 Piper 前必须完成无载标定、轴向测试和保护阈值测试，并显式给出两个确认开关：

```bash
python tactile_reflex_xsense.py \
  --left-sn OG000656 --right-sn OG000691 --can can0 --enable-piper \
  --confirm-force-thresholds --confirm-axis-mapping \
  --fn-stop 0.10 --fn-protect 5.0 \
  --slip-y-th 0.20 --slip-y-quiet 0.10 \
  --cop-vertical-dec-th -10.0 \
  --out-csv output/xense_reflex_validated.csv
```

Xense 参数从本机无载和接触试验开始标定。Xense CSV 含传感器时间戳、帧序列、帧年龄和 pair skew。
