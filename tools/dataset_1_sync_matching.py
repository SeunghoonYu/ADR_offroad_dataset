#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6-Cam + 1-LiDAR Visualization (ROS CameraInfo YAML 지원)

- 카메라별로 ros_caminfo가 있으면 ROS 방식(rectify+P 투영), 없으면 OpenCV(K_new) 방식 사용
- LiDAR .bin: KITTI xyzi(float32) 가정
"""

import re
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json, time, datetime as dt
from typing import List

# =========================
# 0) 사용자 설정
# =========================
base_dir  = Path("/media/ysh/T7/snu_mt_0807/test0807_15_11")
yaml_path = Path("/home/ysh/off-road/tools/calib_matrix/matrix0801.yaml")

camera_dirs = [base_dir / "decoded_rgb" / f"camera_{i}" for i in range(1, 7)]
lidar_dir   = base_dir / "lidar_xyzi"

marks_dir = base_dir / "marks_json"
marks_dir.mkdir(parents=True, exist_ok=True)
run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
marks_json_path = marks_dir / f"sync_marks_{run_ts}.json"

# 시작 인덱스
idx_start = 50

# 거리 색칠 기준: "depth"(Z) 또는 "range"(sqrt(X^2+Y^2+Z^2))
color_metric = "range"
# 컬러맵 거리 범위(m): 자동(분위수) = None, 고정 = (vmin, vmax)
fixed_dist_range = (0.0, 24.0)  # 또는 None

# =========================
# 1) 회전/변환 유틸
# =========================
def rpy_matrix_ypr_zyx(yaw, pitch, roll):
    """yaw(Z) -> pitch(Y) -> roll(X) (ZYX)"""
    cy, sy = np.cos(yaw),   np.sin(yaw)   # Z
    cp, sp = np.cos(pitch), np.sin(pitch) # Y
    cr, sr = np.cos(roll),  np.sin(roll)  # X
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float64)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float64)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float64)
    return Rz @ Ry @ Rx

def make_T_parent_child(translation, ypr):
    R = rpy_matrix_ypr_zyx(*ypr)
    t = np.asarray(translation, dtype=np.float64).reshape(3,1)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t[:,0]
    return T

# =========================
# 2) ROS 스타일 보정 맵 생성
# =========================
def _rectify_maps_from_ros_caminfo(caminfo):
    """
    caminfo: YAML의 cameras[i].ros_caminfo 딕셔너리
      필수: width, height, K(3x3), D(<=14), R(3x3), P(3x4)
      선택: binning_x, binning_y, roi{x_offset,y_offset,width,height}
    반환:
      map1, map2: 보정(remap) 맵 (ROI 반영된 reduced 맵)
      P_rect: (3,4) 보정 후 투영행렬
      K_rect: (3,3) 보정 후 내참수 (P_rect[:3,:3])
      out_size: (w, h) 보정 결과 영상 크기(ROI 반영)
    """
    K = np.array(caminfo['K'], dtype=np.float64).reshape(3,3)
    D = np.array(caminfo['D'], dtype=np.float64).ravel()
    R = np.array(caminfo['R'], dtype=np.float64).reshape(3,3)
    P = np.array(caminfo['P'], dtype=np.float64).reshape(3,4)

    w_full = int(caminfo['width'])
    h_full = int(caminfo['height'])
    bx = int(caminfo.get('binning_x', 1))
    by = int(caminfo.get('binning_y', 1))
    assert bx >= 1 and by >= 1

    # binning 반영(ROS와 동일)
    Kb, Pb = K.copy(), P.copy()
    if bx > 1:
        sx = 1.0 / bx
        Kb[0,0] *= sx; Kb[0,2] *= sx
        Pb[0,0] *= sx; Pb[0,2] *= sx; Pb[0,3] *= sx
    if by > 1:
        sy = 1.0 / by
        Kb[1,1] *= sy; Kb[1,2] *= sy
        Pb[1,1] *= sy; Pb[1,2] *= sy; Pb[1,3] *= sy

    w_bin, h_bin = w_full // bx, h_full // by

    # full size 맵 생성 (binned 해상도)
    map1_full, map2_full = cv2.initUndistortRectifyMap(
        Kb, D, R, Pb, (w_bin, h_bin), cv2.CV_16SC2
    )

    # ROI → reduced 맵 (맵에서 원점 이동)
    roi = caminfo.get('roi', {}) or {}
    xoff = int(roi.get('x_offset', 0)) // bx
    yoff = int(roi.get('y_offset', 0)) // by
    rw   = int(roi.get('width',  w_full)) // bx
    rh   = int(roi.get('height', h_full)) // by

    if (xoff, yoff) != (0,0) or (rw, rh) != (w_bin, h_bin):
        reduced_map1 = map1_full[yoff:yoff+rh, xoff:xoff+rw].copy()
        reduced_map1 -= np.array([xoff, yoff], dtype=reduced_map1.dtype)
        reduced_map2 = map2_full[yoff:yoff+rh, xoff:xoff+rw]
        out_size = (rw, rh)
    else:
        reduced_map1, reduced_map2 = map1_full, map2_full
        out_size = (w_bin, h_bin)

    return {
        'map1': reduced_map1, 'map2': reduced_map2,
        'P_rect': Pb, 'K_rect': Pb[:3,:3],
        'out_size': out_size
    }

# =========================
# 3) YAML 로더 (ROS/OpenCV 자동 인식)
# =========================
def load_calib_yaml(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    img_w = int(cfg['image']['width'])
    img_h = int(cfg['image']['height'])
    alpha = float(cfg.get('undistort', {}).get('alpha', 0.0))
    proj_mode = cfg.get('projection', {}).get('mode', 'undistorted')

    cams = []
    for cam in cfg['cameras']:
        # extrinsic
        Trc_cam = make_T_parent_child(cam['rotcam_extrinsic']['translation'],
                                      cam['rotcam_extrinsic']['rotation_ypr'])
        L = cam['lidar_extrinsic']
        Trc_lidar = make_T_parent_child(L['translation'], L['rotation_ypr'])

        # cam <- lidar
        T_cam_rotcam = np.linalg.inv(Trc_cam)
        T_cam_lidar  = T_cam_rotcam @ Trc_lidar
        R_cam_lidar  = T_cam_lidar[:3,:3].copy()
        t_cam_lidar  = T_cam_lidar[:3, 3].copy()

        cam_dict = {
            'name': cam['name'],
            'frame_id': cam['frame_id'],
            'rotated_frame_id': cam['rotated_frame_id'],
            'T_cam_lidar': T_cam_lidar,
            'R_cam_lidar': R_cam_lidar,
            't_cam_lidar': t_cam_lidar,
            'projection_mode': proj_mode,
        }

        # rectification path 선택
        if 'ros_caminfo' in cam and cam['ros_caminfo']:
            rc = _rectify_maps_from_ros_caminfo(cam['ros_caminfo'])
            cam_dict.update({
                'rect_model': 'ros',
                'map1': rc['map1'], 'map2': rc['map2'],
                'P_rect': rc['P_rect'], 'K_rect': rc['K_rect'],
                'out_size': rc['out_size'],
            })
        else:
            # OpenCV 경로 (K_new)
            K    = np.array(cam['intrinsics']['K'], dtype=np.float64)
            dist = np.array(cam['distortion']['coeffs'], dtype=np.float64).reshape(1,-1)
            # alpha 적용
            K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (img_w,img_h), alpha, (img_w,img_h))
            map1, map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=K, distCoeffs=dist, R=None,
                newCameraMatrix=K_new, size=(img_w,img_h), m1type=cv2.CV_16SC2
            )
            cam_dict.update({
                'rect_model': 'opencv',
                'K': K, 'dist': dist,
                'K_new': K_new, 'roi': tuple(int(v) for v in roi),
                'map1': map1, 'map2': map2,
                'out_size': (img_w, img_h),
            })

        cams.append(cam_dict)

    return {'alpha': alpha, 'image_size': (img_w, img_h), 'cameras': cams}

# =========================
# 4) 투영(ROS/OpenCV 공용)
# =========================
def project_lidar_to_image(xyz_lidar, cam, image_shape=None, alpha=0.0):
    """
    반환:
      uv_int: (M,2), kept_idx: (M,), Z_in: (M,), pts_c_in: (M,3)
    """
    R, t = cam['R_cam_lidar'], cam['t_cam_lidar']
    pts_c = (R @ xyz_lidar.T + t.reshape(3,1)).T
    Z = pts_c[:,2]
    m = (Z > 0.1) & (Z < 200.0)
    if not np.any(m):
        return np.zeros((0,2), np.int32), np.array([], int), np.array([]), np.zeros((0,3))

    pts_c = pts_c[m]; Z = Z[m]

    rect_model = cam.get('rect_model', 'opencv')
    mode = cam.get('projection_mode', 'undistorted')

    if rect_model == 'ros':
        # 보정 영상에서 P로 직접 사영 (ROI 보정 불필요)
        P = cam['P_rect']  # (3,4)
        pts_h = np.hstack([pts_c, np.ones((pts_c.shape[0],1))])
        uvw   = (P @ pts_h.T).T
        uv    = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)

    else:
        # OpenCV 경로
        if mode == 'undistorted':
            K = cam['K_new']
            xs, ys = pts_c[:,0]/Z, pts_c[:,1]/Z
            uv1 = (K @ np.vstack([xs, ys, np.ones_like(xs)])).T
            uv  = uv1[:, :2] / uv1[:, 2:3]
            # alpha==0이면 나중에 이미지를 crop하므로 픽셀 원점 보정 필요
            if alpha == 0.0:
                x, y, w, h = cam['roi']
                if w > 0 and h > 0:
                    uv -= np.array([x, y], dtype=np.float64)
        elif mode == 'original':
            K, dist = cam['K'], cam['dist']
            rvec = np.zeros(3); tvec = np.zeros(3)
            uv, _ = cv2.projectPoints(pts_c, rvec, tvec, K, dist)
            uv = uv.reshape(-1, 2)
        else:
            raise ValueError(f"Unknown projection mode: {mode}")

    if image_shape is not None:
        H, W = image_shape[:2]
        in_img = (uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
        uv = uv[in_img]; pts_c_in = pts_c[in_img]; Z_in = Z[in_img]
        uv_int = np.rint(uv).astype(np.int32)
        kept_idx = np.nonzero(m)[0][in_img]
        return uv_int, kept_idx, Z_in, pts_c_in
    else:
        uv_int = np.rint(uv).astype(np.int32)
        kept_idx = np.nonzero(m)[0]
        return uv_int, kept_idx, Z, pts_c

# =========================
# 5) 기타 유틸
# =========================
def parse_ts(p):
    stem = Path(p).stem.split('_')
    sec  = int(stem[-2])
    nsec = int(re.match(r'(\d+)', stem[-1]).group(1))
    return sec + nsec * 1e-9

def _ts(p): return "_".join(Path(p).stem.split('_')[-2:])

def load_xyzi_bin(bin_path: Path):
    arr = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1,4)
    return arr[:, :3], arr[:, 3]  # xyz, intensity

def distance_colors_bgr(pts_c_in, vmin=None, vmax=None, metric="range"):
    if pts_c_in.shape[0] == 0:
        return np.zeros((0,3), np.uint8)
    if metric == "depth":
        vals = pts_c_in[:,2].copy()
    else:
        vals = np.linalg.norm(pts_c_in, axis=1)
    if vmin is None or vmax is None:
        vmin = max(0.1, np.percentile(vals, 5)) if vmin is None else vmin
        vmax = max(vmin + 1e-6, np.percentile(vals, 95)) if vmax is None else vmax
    norm = (vals - vmin) / (vmax - vmin + 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    norm_inv = 1.0 - norm
    vals_8u = (norm_inv * 255.0).astype(np.uint8).reshape(-1,1)
    colors_bgr = cv2.applyColorMap(vals_8u, cv2.COLORMAP_JET).reshape(-1,3)
    return colors_bgr

# =========================
# 6) 경로/입력 준비
# =========================
camera_files = [sorted(d.glob("*.jpg")) for d in camera_dirs]
lidar_files  = sorted(lidar_dir.glob("*.bin"))

# =========================
# 7) YAML 로드
# =========================
cfg = load_calib_yaml(yaml_path)
alpha_yaml = cfg['alpha']
img_w, img_h = cfg['image_size']

# =========================
# 8) 타임라인 (옵션)
# =========================
sensor_names = ['LiDAR'] + [f'Cam{i+1}' for i in range(6)]
raw_times = {s:[] for s in sensor_names}
raw_times['LiDAR'] = [parse_ts(p) for p in lidar_files]
for i in range(6):
    raw_times[f'Cam{i+1}'] = [parse_ts(p) for p in camera_files[i]]

if all(len(v)>0 for v in raw_times.values()):
    t0 = min(t for v in raw_times.values() for t in v)
    rel_times = {s: np.array(v) - t0 for s,v in raw_times.items()}

    plt.ion()
    fig, ax = plt.subplots(figsize=(12,4))
    ax.invert_yaxis()
    y = np.arange(len(sensor_names))
    blue_len, red_len = 0.7, 0.7
    blue = ax.vlines([],[],[], colors='tab:blue', lw=2, zorder=2)
    red  = ax.vlines([],[],[], colors='red', lw=3, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(sensor_names)
    ax.set_xlabel('Elapsed time (s)')
    ax.grid(axis='x', ls='--', alpha=.3); fig.tight_layout()
    view_span, blue_span = 0.7, 3.0

    def update_tl(cur_lidar_idx, img_idx_list):
        cur = [parse_ts(lidar_files[cur_lidar_idx]) - t0] + [parse_ts(camera_files[i][img_idx_list[i]]) - t0 for i in range(6)]
        red.set_segments([[(x,yy-red_len/2),(x,yy+red_len/2)] for x,yy in zip(cur,y)])
        center = cur[0]; seg = []
        for i,s in enumerate(sensor_names):
            t = rel_times[s]; m = (t >= center - blue_span) & (t <= center + blue_span)
            seg += [[(x, y[i] - blue_len/2), (x, y[i] + blue_len/2)] for x in t[m]]
        blue.set_segments(seg); ax.set_xlim(center - view_span, center + view_span)
        fig.canvas.draw_idle(); plt.pause(0.001)
else:
    update_tl = lambda *args, **kwargs: None  # 타임라인 비활성화

# =========================
# 9) 메인 그리기 함수
# =========================
def project_one_cam(i, img_idx_i, lidar_idx, draw_lidar=True):
    cam = cfg['cameras'][i]
    files = camera_files[i]
    if not files:
        return np.zeros((400,640,3), np.uint8)

    img_bgr = cv2.imread(str(files[img_idx_i]), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return np.zeros((400,640,3), np.uint8)

    # 보정(remap)
    und_bgr = cv2.remap(img_bgr, cam['map1'], cam['map2'], interpolation=cv2.INTER_LINEAR)

    # 출력 영상은:
    #  - ROS 경로: reduced_map이 이미 ROI 좌표계로 반환 → 추가 crop/오프셋 불필요
    #  - OpenCV 경로: alpha==0 이면 수동 crop + uv의 ROI 보정 필요
    img_disp = und_bgr
    if cam.get('rect_model','opencv') == 'opencv' and alpha_yaml == 0.0:
        x, y, w, h = cam['roi']
        if w > 0 and h > 0:
            img_disp = und_bgr[y:y+h, x:x+w].copy()

    # LiDAR 투영
    if draw_lidar and 0 <= lidar_idx < len(lidar_files):
        xyz, inten = load_xyzi_bin(lidar_files[lidar_idx])
        uv_int, kept_idx, Z, pts_c_in = project_lidar_to_image(
            xyz, cam, image_shape=img_disp.shape, alpha=alpha_yaml
        )

        if fixed_dist_range is None:
            colors_bgr = distance_colors_bgr(pts_c_in, vmin=None, vmax=None, metric=color_metric)
        else:
            vmin, vmax = fixed_dist_range
            colors_bgr = distance_colors_bgr(pts_c_in, vmin=vmin, vmax=vmax, metric=color_metric)

        for (u, v), c in zip(uv_int, colors_bgr):
            cv2.circle(img_disp, (int(u), int(v)), 2, (int(c[0]), int(c[1]), int(c[2])), -1, lineType=cv2.LINE_AA)

    # label = f"Cam{i+1} {_ts(files[img_idx_i])} LiDAR {lidar_idx}" + ("" if draw_lidar else " [LIDAR OFF]")
    label = f"Cam{i+1} {img_idx_i} LiDAR {lidar_idx}" + ("" if draw_lidar else " [LIDAR OFF]")
    cv2.putText(img_disp, label, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (120,255,0), 2)
    return cv2.resize(img_disp, (640,400))

# =========================
# 10) 타임싱크 후처리 json
# =========================
def _safe_read_json(p: Path):
    if not p.exists(): return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            x = json.load(f)
            return x if isinstance(x, list) else []
    except Exception:
        return []

def _append_json(p: Path, obj: dict):
    data = _safe_read_json(p)
    data.append(obj)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _build_snapshot(label: str, lidar_idx: int, cam_idx_list: List[int]):
    return {
        "label": label,  # "start1", "end1", ...
        "wall_time": dt.datetime.now().isoformat(),
        "base_dir": str(base_dir),
        "indices": {"lidar_idx": int(lidar_idx), "cam_idx": [int(x) for x in cam_idx_list]},
        "files": {
            "lidar": str(lidar_files[lidar_idx]) if 0 <= lidar_idx < len(lidar_files) else None,
            "cams": [str(camera_files[i][cam_idx_list[i]]) if (0 <= cam_idx_list[i] < len(camera_files[i])) else None for i in range(6)]
        }
    }

def overlay_segment_marks(canvas, phase, snap_start, snap_end, segment_id):
    H, W = canvas.shape[:2]

    # 헤더: 현재 진행 중인 세그먼트와 페이즈
    header = f"segment={segment_id}  phase={phase if phase else 'none'}"

    lines = [header]

    # ▼ start 라벨은 snap_start가 가진 seg_id로 표시
    if snap_start is not None:
        s_id = snap_start["seg_id"]
        cidx = snap_start["cam_idx"]; lidx = snap_start["lidar_idx"]
        lines.append(f"start{s_id}:  L={lidx}  C=[{cidx[0]}, {cidx[1]}, {cidx[2]}, {cidx[3]}, {cidx[4]}, {cidx[5]}]")

    # ▼ end 라벨도 snap_end의 seg_id로 표시
    if snap_end is not None:
        e_id = snap_end["seg_id"]
        cidx = snap_end["cam_idx"]; lidx = snap_end["lidar_idx"]
        lines.append(f"end{e_id}:    L={lidx}  C=[{cidx[0]}, {cidx[1]}, {cidx[2]}, {cidx[3]}, {cidx[4]}, {cidx[5]}]")

    # 글자/박스 스타일(현재처럼 작게)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.4
    th = 1
    pad_x, pad_y = 9, 6
    line_gap = 6

    sizes = [cv2.getTextSize(t, font, fs, th)[0] for t in lines]
    text_w = max(s[0] for s in sizes)
    text_hs = [s[1] for s in sizes]
    total_h = sum(text_hs) + line_gap * (len(lines) - 1)

    box_w = text_w + pad_x * 2
    box_h = total_h + pad_y * 2
    x1 = (W - box_w) // 2
    y1 = H - box_h - 12
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)

    cur_y = y1 + pad_y + text_hs[0]
    for i, t in enumerate(lines):
        tw, _ = sizes[i]
        tx = x1 + (box_w - tw) // 2
        if i == 0:       color = (220, 220, 220)  # header
        elif t.startswith("start"): color = (0, 255, 255)
        elif t.startswith("end"):   color = (255, 200, 120)
        else:             color = (220, 220, 220)
        cv2.putText(canvas, t, (tx, cur_y), font, fs, color, th, cv2.LINE_AA)
        cur_y += text_hs[i] + line_gap



# =========================
# 11) 메인 루프 (모자이크 구성)
# =========================
img_idx = [idx_start]*6
lidar_idx = idx_start
project_lidar = False
selected_cam = None          # 선택된 카메라 인덱스 (0~5)
control_mode = None          # 'cam' | 'lidar' | 'all' | None
global_step_size = 1

segment_id = 1 # 1부터 시작: start1/end1, start2/end2 ...
current_phase = None
snap_start = None
snap_end = None
allowed_next = "start" # 다음에 허용되는 키
last_msg = ""

def build_canvas(imgs):
    # 레이아웃 동일
    hs = [img.shape[0] for img in imgs]
    ws = [img.shape[1] for img in imgs]
    H_top = max(hs[0], hs[1], hs[5])
    H_bot = max(hs[2], hs[3], hs[4])
    W_left  = max(ws[1], ws[4])
    W_mid   = max(ws[0], ws[3])
    W_right = max(ws[5], ws[2])
    canvas_height = H_top + H_bot
    canvas_width  = W_left + W_mid + W_right
    can = np.zeros((canvas_height, canvas_width, 3), np.uint8)
    positions = [
        (0,         W_left),               # cam1
        (0,         0),                    # cam2
        (H_top,     W_left + W_mid),       # cam3
        (H_top,     W_left),               # cam4
        (H_top,     0),                    # cam5
        (0,         W_left + W_mid)        # cam6
    ]
    for img, (y0, x0) in zip(imgs, positions):
        h, w = img.shape[:2]
        can[y0:y0+h, x0:x0+w] = img
    return can


while True:
    imgs = [project_one_cam(i, img_idx[i], lidar_idx, draw_lidar=project_lidar) for i in range(6)]
    can  = build_canvas(imgs)
    overlay_segment_marks(can, current_phase, snap_start, snap_end, segment_id)
    cv2.imshow("Camera-LiDAR Projection tool", can)

    key = cv2.waitKey(0) & 0xFF

    def step_cam(i, delta):
        img_idx[i] = max(0, min(len(camera_files[i]) - 1, img_idx[i] + delta))

    def step_all(delta):
        for i in range(6):
            step_cam(i, delta)

    if key in (27, ord('q')):      # ESC, q
        break

    elif key == 32:                # Space: LiDAR on/off
        project_lidar = not project_lidar

    # --- step size 설정 ---
    elif key == ord('b'):
        global_step_size = 1
        print(f"[Step size] {global_step_size}")
    elif key == ord('n'):
        global_step_size = 5
        print(f"[Step size] {global_step_size}")
    elif key == ord('m'):
        global_step_size = 10
        print(f"[Step size] {global_step_size}")

    # --- 전체 앞뒤 이동 ---
    elif key == 44:  # ','
        lidar_idx = max(0, lidar_idx - global_step_size)
        step_all(-global_step_size)
    elif key == 46:  # '.'
        lidar_idx = min(len(lidar_files) - 1, lidar_idx + global_step_size)
        step_all(global_step_size)   

    # --- 모드 선택 ---
    elif 49 <= key <= 54:  # '1'~'6'
        selected_cam = key - 49
        control_mode = 'cam'
    elif key == ord('l'):
        control_mode = 'lidar'
        selected_cam = None
    elif key == ord('c'):
        control_mode = 'all'
        selected_cam = None

    # --- a/d 이동 ---
    elif key == ord('a'):
        if control_mode == 'cam' and selected_cam is not None:
            step_cam(selected_cam, -global_step_size)
        elif control_mode == 'lidar':
            lidar_idx = max(0, lidar_idx - global_step_size)
        elif control_mode == 'all':
            step_all(-global_step_size)

    elif key == ord('d'):
        if control_mode == 'cam' and selected_cam is not None:
            step_cam(selected_cam, global_step_size)
        elif control_mode == 'lidar':
            lidar_idx = min(len(lidar_files) - 1, lidar_idx + global_step_size)
        elif control_mode == 'all':
            step_all(global_step_size)

    elif key == ord('s'):
        if allowed_next != "start":
            ...
        else:
            label = f"start{segment_id}"
            snap = _build_snapshot(label, lidar_idx, img_idx)
            _append_json(marks_json_path, snap)

            # ▼ seg_id를 같이 저장
            snap_start = {"seg_id": segment_id, "lidar_idx": lidar_idx, "cam_idx": img_idx.copy()}
            current_phase = "start"
            allowed_next = "end"
            ...

    elif key == ord('e'):
        if allowed_next != "end":
            ...
        else:
            label = f"end{segment_id}"
            snap = _build_snapshot(label, lidar_idx, img_idx)
            _append_json(marks_json_path, snap)

            snap_end = {"seg_id": segment_id, "lidar_idx": lidar_idx, "cam_idx": img_idx.copy()}
            current_phase = "end"
            segment_id += 1
            allowed_next = "start"
            ...



    update_tl(lidar_idx, img_idx)
