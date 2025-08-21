#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6-Cam + 1-LiDAR Visualization Viewer (PyQt6)

Self-contained viewer application for SNU mountain dataset
"""

import sys
import os
import json
import re
import cv2
import yaml
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from PyQt6 import QtCore, QtGui, QtWidgets
import datetime as dt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6 import QtCore
# =========================
# 1) 데이터 로딩 및 설정
# =========================

def load_scene_meta():
    """Load scene metadata from scene_meta.json"""
    meta_path = Path("scene_meta.json")
    if not meta_path.exists():
        print(f"Error: {meta_path} not found!")
        return None
    
    with open(meta_path, 'r') as f:
        return json.load(f)

def load_camera_and_lidar_files():
    """Load camera and LiDAR file paths"""
    # Camera directories (상위 디렉토리 기준)
    camera_dirs = [Path("./decoded_rgb") / f"camera_{i}" for i in range(1, 7)]
    lidar_dir = Path("./lidar_xyzi")
    
    # Load camera files
    camera_files = []
    for cam_dir in camera_dirs:
        if cam_dir.exists():
            files = sorted(list(cam_dir.glob("*.jpg")))
            camera_files.append(files)
        else:
            print(f"Warning: {cam_dir} not found")
            camera_files.append([])
    
    # Load LiDAR files
    lidar_files = []
    if lidar_dir.exists():
        lidar_files = sorted(list(lidar_dir.glob("*.bin")))
    else:
        print(f"Warning: {lidar_dir} not found")
    
    return camera_files, lidar_files

def parse_ts(p):
    """Parse timestamp from filename"""
    stem = Path(p).stem.split('_')
    # 새로운 패턴: cam{i}_frame_{인덱스}_{timestamp}.jpg 또는 lidar_{인덱스}_{timestamp}xyzi.bin
    if 'cam' in stem[0]:  # 카메라 파일
        sec  = int(stem[-2])
        nsec = int(re.match(r'(\d+)', stem[-1]).group(1))
    else:  # LiDAR 파일
        sec  = int(stem[-2])
        nsec = int(re.match(r'(\d+)', stem[-1].replace('xyzi', '')).group(1))
    return sec + nsec * 1e-9

def _ts(p): 
    """Get timestamp string from filename"""
    stem = Path(p).stem.split('_')
    # 새로운 패턴에 맞게 timestamp 부분만 반환
    if 'cam' in stem[0]:  # 카메라 파일
        return "_".join(stem[-2:])
    else:  # LiDAR 파일
        return "_".join([stem[-2], stem[-1].replace('xyzi', '')])

# =========================
# 2) 보정 및 변환 유틸리티
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
# 3) 카메라 보정 및 투영
# =========================

def load_calib_yaml(yaml_path: Path):
    """Load camera calibration from YAML file"""
    if not yaml_path.exists():
        print(f"Warning: {yaml_path} not found, using default calibration")
        return None
    
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    img_w = int(cfg['image']['width'])
    img_h = int(cfg['image']['height'])
    alpha = float(cfg.get('undistort', {}).get('alpha', 0.0))
    proj_mode = cfg.get('projection', {}).get('mode', 'undistorted')

    cams = []
    for cam in cfg['cameras']:
        Trc_cam = make_T_parent_child(cam['rotcam_extrinsic']['translation'], 
                                    cam['rotcam_extrinsic']['rotation_ypr'])
        L = cam['lidar_extrinsic']
        Trc_lidar = make_T_parent_child(L['translation'], L['rotation_ypr'])

        T_cam_rotcam = np.linalg.inv(Trc_cam)
        T_cam_lidar = T_cam_rotcam @ Trc_lidar
        R_cam_lidar = T_cam_lidar[:3,:3].copy()
        t_cam_lidar = T_cam_lidar[:3, 3].copy()

        # Camera intrinsic - use intrinsics field
        K = np.array(cam['intrinsics']['K'], dtype=np.float64).reshape(3,3)
        D = np.array(cam['distortion']['coeffs'], dtype=np.float64).ravel()
        
        cams.append({
            'K': K, 'D': D, 'R_cam_lidar': R_cam_lidar, 't_cam_lidar': t_cam_lidar,
            'img_w': img_w, 'img_h': img_h, 'alpha': alpha, 'proj_mode': proj_mode
        })
    
    return cams

def _rectify_maps_from_ros_caminfo(caminfo):
    """Generate rectification maps from ROS camera info"""
    K = np.array(caminfo['K'], dtype=np.float64).reshape(3,3)
    D = np.array(caminfo['D'], dtype=np.float64).ravel()
    R = np.array(caminfo['R'], dtype=np.float64).reshape(3,3)
    P = np.array(caminfo['P'], dtype=np.float64).reshape(3,4)

    w_full = int(caminfo['width'])
    h_full = int(caminfo['height'])
    bx = int(caminfo.get('binning_x', 1))
    by = int(caminfo.get('binning_y', 1))
    
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
    map1_full, map2_full = cv2.initUndistortRectifyMap(Kb, D, R, Pb, (w_bin, h_bin), cv2.CV_16SC2)

    roi = caminfo.get('roi', {}) or {}
    xoff = int(roi.get('x_offset', 0)) // bx
    yoff = int(roi.get('y_offset', 0)) // by
    rw = int(roi.get('width', w_full)) // bx
    rh = int(roi.get('height', h_full)) // by

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
        'P_rect': Pb, 'K_rect': Pb[:3,:3], 'out_size': out_size
    }

# =========================
# 4) LiDAR 데이터 처리
# =========================

def load_lidar_points(lidar_path: Path) -> np.ndarray:
    """Load LiDAR points from binary file (KITTI format: x,y,z,intensity)"""
    if not lidar_path.exists():
        return np.empty((0, 4))
    
    points = np.fromfile(lidar_path, dtype=np.float32)
    points = points.reshape(-1, 4)  # x, y, z, intensity
    return points

def filter_lidar_points(points: np.ndarray, max_range: float = 50.0) -> np.ndarray:
    """Filter LiDAR points by range"""
    if len(points) == 0:
        return points
    
    ranges = np.linalg.norm(points[:, :3], axis=1)
    mask = ranges <= max_range
    return points[mask]

# =========================
# 5) 카메라 투영 함수
# =========================

def project_lidar_to_camera(points_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, 
                           img_w: int, img_h: int) -> tuple:
    """Project 3D points to camera image plane"""
    if len(points_3d) == 0:
        return np.empty((0, 2)), np.empty((0,))
    
    # Transform points to camera coordinate system
    points_cam = (R @ points_3d[:, :3].T).T + t
    
    # Filter points in front of camera
    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]
    points_3d = points_3d[mask]
    
    if len(points_cam) == 0:
        return np.empty((0, 2)), np.empty((0,))
    
    # Project to image plane
    points_2d = K @ points_cam.T
    points_2d = points_2d[:2] / points_2d[2]
    points_2d = points_2d.T
    
    # Filter points within image bounds
    mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_w) & 
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_h))
    
    return points_2d[mask], points_3d[mask]

def project_one_cam(cam_idx: int, img_idx: int, lidar_idx: int, 
                   draw_lidar: bool = True, point_radius: int = 2) -> np.ndarray:
    """Project LiDAR points to one camera image"""
    global camera_files, lidar_files, calib_data
    
    # Load camera image
    if cam_idx >= len(camera_files) or img_idx >= len(camera_files[cam_idx]):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    img_path = camera_files[cam_idx][img_idx]
    if not img_path.exists():
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Load LiDAR data
    if lidar_idx >= len(lidar_files):
        return img
    
    lidar_path = lidar_files[lidar_idx]
    points = load_lidar_points(lidar_path)
    points = filter_lidar_points(points, max_range=50.0)
    
    if len(points) == 0 or not draw_lidar:
        return img
    
    # Use calibration if available
    if calib_data and cam_idx < len(calib_data):
        calib = calib_data[cam_idx]
        K = calib['K']
        R = calib['R_cam_lidar']
        t = calib['t_cam_lidar']
        img_w, img_h = calib['img_w'], calib['img_h']
    else:
        # Default calibration (approximate) - adjusted for better projection
        img_w, img_h = img.shape[1], img.shape[0]
        K = np.array([[1000, 0, img_w//2], [0, 1000, img_h//2], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        # Move camera back to see LiDAR points in front
        t = np.array([0, 0, 5], dtype=np.float64)
    
    # Project points
    points_2d, points_3d = project_lidar_to_camera(points, K, R, t, img_w, img_h)
    
    if len(points_2d) == 0:
        return img
    
    # Color points by range
    ranges = np.linalg.norm(points_3d[:, :3], axis=1)
    colors = plt.cm.jet(ranges / ranges.max() if ranges.max() > 0 else 0)[:, :3]
    colors = (colors * 255).astype(np.uint8)
    
    # Draw points
    for i, (pt, color) in enumerate(zip(points_2d, colors)):
        cv2.circle(img, (int(pt[0]), int(pt[1])), point_radius, 
                  (int(color[2]), int(color[1]), int(color[0])), -1)
    
    # Add timestamp label
    if img_path.exists():
        timestamp_str = _ts(img_path)
        cv2.putText(img, f"Cam{cam_idx+1} {timestamp_str}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 255, 0), 2)
    
    return img

def build_canvas(imgs: List[np.ndarray]) -> np.ndarray:
    """Build a 2x3 grid canvas with order: top=2,1,6 / bottom=5,4,3"""
    # 정확히 6장 맞추기
    while len(imgs) < 6:
        imgs.append(np.zeros((480, 640, 3), dtype=np.uint8))
    imgs = imgs[:6]

    # 동일 크기로 리사이즈
    target_size = (640, 480)
    resized = []
    for img in imgs:
        if img is not None and img.size > 0:
            resized.append(cv2.resize(img, target_size))
        else:
            resized.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))

    # 원하는 순서로 재배치: [Cam2, Cam1, Cam6, Cam5, Cam4, Cam3]
    order = [1, 0, 5, 4, 3, 2]
    arranged = [resized[i] for i in order]

    # 2×3 그리드 생성
    row1 = np.hstack(arranged[:3])  # 2,1,6
    row2 = np.hstack(arranged[3:6]) # 5,4,3
    canvas = np.vstack([row1, row2])

    # 하단 타임라인 영역(300px)
    bottom_space = np.zeros((300, canvas.shape[1], 3), dtype=np.uint8)
    canvas = np.vstack([canvas, bottom_space])

    return canvas

# =========================
# 6) 스냅샷 및 JSON 유틸리티
# =========================

def _build_snapshot(label: str, lidar_idx: int, cam_idx_list: List[int]) -> dict:
    return {
        'label': label,
        'timestamp': dt.datetime.now().isoformat(),
        'lidar_idx': int(lidar_idx),
        'cam_indices': list(cam_idx_list)   # ← 리스트 복사!
    }

def _append_json(json_path: Path, obj: dict) -> None:
    """Append object to JSON file"""
    data = []
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                d = json.load(f)
                if isinstance(d, list):
                    data = d
            except:
                pass
    
    data.append(obj)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_timeline_matplotlib(current_lidar_idx, current_img_idx, control_mode,
                               snap_start, snap_end, segment_id, canvas_width, canvas_height):
    """Timeline (LiDAR 중심 고정, 창=±2s, 0.5s tick, START/END 센서별 표시)"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # === 픽셀/사이즈 ===
    fig_w_px = int(canvas_width)
    fig_h_px = int(canvas_height)
    dpi = 150
    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')

    base = max(8, int(fig_h_px * 0.06))
    tick = max(7, int(fig_h_px * 0.05))
    small = max(6, int(fig_h_px * 0.045))
    plt.rcParams['text.antialiased'] = True
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    sensor_names = ['LiDAR'] + [f'Cam{i+1}' for i in range(6)]
    lanes = len(sensor_names)
    lane_gap = fig_h_px / (lanes + 1)
    bar_len  = lane_gap * 0.45
    y = np.array([(i+1)*lane_gap for i in range(lanes)])
    ax.invert_yaxis()

    # === 타임스탬프 ===
    raw_times = {s: [] for s in sensor_names}
    if lidar_files:
        raw_times['LiDAR'] = [parse_ts(p) for p in lidar_files]
    for i in range(6):
        if i < len(camera_files) and camera_files[i]:
            raw_times[f'Cam{i+1}'] = [parse_ts(p) for p in camera_files[i]]

    if all(len(v) > 0 for v in raw_times.values()):
        t0 = min(t for v in raw_times.values() for t in v)
        rel_times = {s: np.array(v) - t0 for s, v in raw_times.items()}

        # 현재 LiDAR 기준시간
        ref_time = parse_ts(lidar_files[current_lidar_idx]) - t0 if current_lidar_idx < len(lidar_files) else 0.0

        # === 보기 창 설정: ±2초(총 4초) ===
        window_sec = 1.0
        x_left, x_right = ref_time - window_sec, ref_time + window_sec

        # 윈도우 안 모든 스탬프 선택
        selected_times = {}
        for sensor in sensor_names:
            t = rel_times.get(sensor, np.array([]))
            selected_times[sensor] = t[(t >= x_left) & (t <= x_right)] if t.size else np.array([])

        # 현재 포커스(빨간 막대) 시간들
        current_times = [ref_time]
        for i in range(6):
            if i < len(camera_files) and current_img_idx[i] < len(camera_files[i]):
                current_times.append(parse_ts(camera_files[i][current_img_idx[i]]) - t0)
            else:
                current_times.append(ref_time)

        # 배경(회색) 막대
        bg_len = 30
        for i, sensor in enumerate(sensor_names):
            ts = selected_times.get(sensor, np.array([]))
            if ts.size:
                ax.vlines(ts, y[i] - bg_len/2, y[i] + bg_len/2,
                          colors='lightgray', alpha=0.7, linewidth=4)

        # 현재 위치(빨간 막대)
        red_len = 30
        ax.vlines(current_times,
                  [y[i] - red_len/2 for i in range(len(y))],
                  [y[i] + red_len/2 for i in range(len(y))],
                  colors='red', linewidth=4, zorder=3)

        # === START/END: 센서별 y로 개별 표시 ===
        def _sensor_time_from_snapshot(snap, i_sensor):
            if snap is None: return None
            if i_sensor == 0:  # LiDAR
                lidx = snap.get("lidar_idx", 0)
                if 0 <= lidx < len(lidar_files):
                    return parse_ts(lidar_files[lidx]) - t0
                return None
            else:
                cam_i = i_sensor - 1
                cam_indices = snap.get("cam_indices", [0]*6)
                if 0 <= cam_i < 6 and 0 <= cam_indices[cam_i] < len(camera_files[cam_i]):
                    return parse_ts(camera_files[cam_i][cam_indices[cam_i]]) - t0
                return None

        def _draw_mark_per_sensor(snap, color, text):
            if snap is None:
                return
            mark_len = 30

            def _sensor_time_from_snapshot(snap, i_sensor):
                if i_sensor == 0:  # LiDAR
                    lidx = snap.get("lidar_idx", 0)
                    if 0 <= lidx < len(lidar_files):
                        return parse_ts(lidar_files[lidx]) - t0
                    return None
                else:
                    cam_i = i_sensor - 1
                    cam_indices = snap.get("cam_indices", [0]*6)
                    if 0 <= cam_i < 6 and 0 <= cam_indices[cam_i] < len(camera_files[cam_i]):
                        return parse_ts(camera_files[cam_i][cam_indices[cam_i]]) - t0
                    return None

            for i in range(len(sensor_names)):  # 0..6
                tval = _sensor_time_from_snapshot(snap, i)
                if tval is None:
                    continue

                # 윈도우 밖이면 가장자리로 클램프해서라도 보이게
                clamped = np.clip(tval, x_left, x_right)

                # 선
                ax.vlines([clamped], y[i] - mark_len/2, y[i] + mark_len/2,
                        colors=color, linewidth=4, zorder=5, alpha=0.95, clip_on=True)

                # 텍스트: 윈도우 밖에서 클램프된 경우엔 화살표로 표시
                label = text
                if tval < x_left:
                    label = f"← {text}"
                elif tval > x_right:
                    label = f"{text} →"

                ax.text(clamped, y[i] - mark_len/2 - 0.3, label,
                        ha='center', va='top', fontsize=8, color=color, fontweight='bold',
                        zorder=6, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.85, linewidth=0.0))

        _draw_mark_per_sensor(snap_start, 'blue',   'START')
        _draw_mark_per_sensor(snap_end,   'purple', 'END')

        # 컨트롤 모드 하이라이트/시간차
        if control_mode == "lidar":
            ax.scatter([current_times[0]], [y[0]], s=120, c='yellow',
                       marker='o', zorder=4, edgecolors='black', linewidth=1)
        elif control_mode.startswith("cam"):
            cam_num = int(control_mode[-1]) - 1
            if 0 <= cam_num < 6:
                ax.scatter([current_times[cam_num + 1]], [y[cam_num + 1]], s=120, c='yellow',
                           marker='o', zorder=4, edgecolors='black', linewidth=1)

        if control_mode != "all":
            ref = current_times[0] if control_mode == "lidar" else current_times[int(control_mode[-1])]
            skip_idx = 0 if control_mode == "lidar" else int(control_mode[-1])
            for i in range(len(sensor_names)):
                if (control_mode == "lidar" and i == 0) or (control_mode.startswith("cam") and i == skip_idx):
                    continue
                cur = current_times[i]
                diff = cur - ref
                color = 'green' if abs(diff) <= 0.1 else ('orange' if abs(diff) <= 0.5 else 'red')
                ax.vlines([cur], y[i] - red_len/2, y[i] + red_len/2, colors=color, linewidth=3, zorder=1, alpha=0.9)
                if abs(diff) > 0.01 and (x_left <= cur <= x_right):
                    ax.text(cur, y[i] + red_len/2 + 0.3, f'{diff:+.3f}s',
                            ha='center', va='bottom', fontsize=10, color=color, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.85))

        # === x축 고정 & 눈금 0.5s / 레이블 2자리 ===
        ax.set_xlim(x_left, x_right)
        xticks = np.arange(x_left, x_right + 1e-9, 0.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:.2f}" for t in xticks])

        # 제목/라벨
        current_timestamp = dt.datetime.fromtimestamp(parse_ts(lidar_files[current_lidar_idx])).strftime('%H:%M:%S.%f')[:-3]
        shown_count = sum(len(v) for v in selected_times.values())
        ax.set_title(f'Timeline - Mode: {control_mode.upper()} | Time: {current_timestamp} | Window: ±{window_sec:.1f}s | Shown: {shown_count}',
                     color='black', fontsize=13, fontweight='bold', pad=20)
        ax.set_xlabel('Elapsed time (s)', color='black', fontsize=11, fontweight='bold', labelpad=10)

    else:
        # 타임스탬프가 없을 때(프레임 인덱스)
        total_frames = max(len(lidar_files), max((len(cam_files) for cam_files in camera_files if cam_files), default=0))
        frame_range = np.arange(0, total_frames, 1)
        bg_len = 30
        for i, sensor in enumerate(sensor_names):
            ax.vlines(frame_range, y[i] - bg_len/2, y[i] + bg_len/2, colors='lightgray', alpha=0.6, linewidth=1.0)
        red_len = 30
        current_positions = [current_lidar_idx] + current_img_idx
        ax.vlines(current_positions, [y[i] - red_len/2 for i in range(len(y))],
                  [y[i] + red_len/2 for i in range(len(y))], colors='red', linewidth=4, zorder=3)
        ax.set_xlabel('Frame Index', color='black', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_title(f'Timeline - Mode: {control_mode.upper()} | LiDAR Frame: {current_lidar_idx}',
                     color='black', fontsize=13, fontweight='bold', pad=20)

    # === 스타일/범례 ===
    ax.set_yticks(y)
    ax.set_yticklabels(sensor_names, fontsize=base, color='black')
    ax.tick_params(axis='x', labelsize=tick, colors='black')
    ax.grid(axis='x', linestyle='--', alpha=0.35, color='gray')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color='red',    lw=3, label='Current'),
        Line2D([0],[0], color='blue',   lw=3, label='Start'),
        Line2D([0],[0], color='purple', lw=3, label='End'),
        Line2D([0],[0], color='lightgray', lw=3, alpha=0.8, label='Samples in window'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1),
              facecolor='white', edgecolor='black', fontsize=small, framealpha=1.0)
    plt.subplots_adjust(left=0.08, right=0.82, top=0.9, bottom=0.2)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = buf[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return img
def overlay_segment_marks(canvas, phase, snap_start, snap_end, segment_id,
                          current_lidar_idx, current_img_idx, control_mode):
    H, W = canvas.shape[:2]
    timeline_h = 300

    # ✅ 최종 붙일 크기 그대로 생성
    timeline_img = create_timeline_matplotlib(
        current_lidar_idx, current_img_idx, control_mode,
        snap_start, snap_end, segment_id,
        W, timeline_h
    )

    # ✅ 리사이즈 하지 말고 그대로 붙이기
    canvas[H - timeline_h:, :W] = timeline_img

# =========================
# 7) PyQt6 UI 클래스들
# =========================

class ImageLabel(QtWidgets.QLabel):
    """이미지(QImage)를 보여주는 라벨."""
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: black;")

    def show_ndarray(self, img_bgr: np.ndarray):
        if img_bgr is None:
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(
        self.size(),
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation
    ))

    def resizeEvent(self, e):
        # 리사이즈 시 현재 pixmap을 비율 유지로 다시 맞춤
        if self.pixmap():
            self.setPixmap(self.pixmap().scaled(
                self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
        super().resizeEvent(e)


class Viewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SNU Mountain Dataset Viewer (PyQt6)")
        self.resize(1500, 900)

        # ---- 상태 ----
        self.num_cams = 6
        self.img_idx: List[int] = [0] * self.num_cams
        self.lidar_idx: int = 0
        self.project_lidar: bool = True
        self.point_radius: int = 2
        self.step_size: int = 1
        self.allowed_next: str = "start"  # "start" -> "end" 토글
        self.segment_id: int = 1
        self.undo_stack: List[Dict[str, Any]] = []  # 최근 저장(s/e) 스냅샷 버퍼(되돌리기용 1단계 이상도 가능)
        self.cached_canvas: np.ndarray | None = None

        # ---- Segment marks ----
        self.current_phase: str = None
        self.snap_start: Dict[str, Any] = None
        self.snap_end: Dict[str, Any] = None
        
        # ---- Individual control mode ----
        self.control_mode: str = "all"  # "all", "lidar", "cam1", "cam2", ..., "cam6"
        self.individual_step: int = 1

        # 시작 인덱스 설정
        self.img_idx = [50] * self.num_cams
        self.lidar_idx = 50

        # ---- 중앙 이미지 ----
        self.view = ImageLabel()

        # ---- 오른쪽 패널 ----
        panel = self._build_right_panel()

        # ---- 레이아웃 ----
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)
        self._init_shortcuts()
        self._refresh()  # 초기 렌더

    # ========== UI 구성 ==========
    def _build_right_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        # 1) Frame Navigation 헤더
        title = QtWidgets.QLabel("<b>Frame Navigation</b>")
        v.addWidget(title)

        # 라이다 인덱스 라벨 (예: Frame: 2947 / 3061)
        self.lbl_idx = QtWidgets.QLabel("Frame: - / -")
        v.addWidget(self.lbl_idx)

        # 파일 경로 표시
        self.txt_files = QtWidgets.QTextEdit()
        self.txt_files.setReadOnly(True)
        self.txt_files.setMinimumHeight(120)
        v.addWidget(self.txt_files)

        # 네비게이션 슬라이더 (라이다 기준)
        self.sld = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld.setMinimum(0)
        self.sld.setMaximum(max(0, len(lidar_files) - 1))
        self.sld.setSingleStep(1)
        self.sld.setPageStep(1)
        self.sld.setValue(self.lidar_idx)
        self.sld.valueChanged.connect(self.on_slider_changed)
        v.addWidget(self.sld)

        # Prev/Next + step 크기
        h_nav = QtWidgets.QHBoxLayout()
        btn_prev = QtWidgets.QPushButton("◀ Prev")
        btn_next = QtWidgets.QPushButton("Next ▶")
        btn_prev.clicked.connect(lambda: self._relative_step(-self.step_size))
        btn_next.clicked.connect(lambda: self._relative_step(+self.step_size))
        h_nav.addWidget(btn_prev)
        h_nav.addWidget(btn_next)
        v.addLayout(h_nav)

        h_step = QtWidgets.QHBoxLayout()
        h_step.addWidget(QtWidgets.QLabel("Step:"))
        self.cmb_step = QtWidgets.QComboBox()
        self.cmb_step.addItems(["1", "5", "10"])
        self.cmb_step.currentIndexChanged.connect(self.on_step_changed)
        h_step.addWidget(self.cmb_step)
        h_step.addStretch(1)
        v.addLayout(h_step)

        v.addSpacing(8)
        v.addWidget(self._sep("Point / Visibility"))

        # LiDAR point size
        h_ps = QtWidgets.QHBoxLayout()
        h_ps.addWidget(QtWidgets.QLabel("LiDAR point size:"))
        self.spin_ps = QtWidgets.QSpinBox()
        self.spin_ps.setRange(1, 12)
        self.spin_ps.setValue(self.point_radius)
        self.spin_ps.valueChanged.connect(self.on_point_size_changed)
        h_ps.addWidget(self.spin_ps)
        v.addLayout(h_ps)

        # LiDAR on/off
        self.chk_lidar = QtWidgets.QCheckBox("Show LiDAR")
        self.chk_lidar.setChecked(self.project_lidar)
        self.chk_lidar.toggled.connect(self.on_lidar_toggle)
        v.addWidget(self.chk_lidar)

        v.addSpacing(8)
        v.addWidget(self._sep("Individual Control"))
        
        # Control mode selection
        h_mode = QtWidgets.QHBoxLayout()
        h_mode.addWidget(QtWidgets.QLabel("Control mode:"))
        self.cmb_control = QtWidgets.QComboBox()
        self.cmb_control.addItems(["All", "LiDAR", "Cam1", "Cam2", "Cam3", "Cam4", "Cam5", "Cam6"])
        self.cmb_control.currentTextChanged.connect(self.on_control_mode_changed)
        h_mode.addWidget(self.cmb_control)
        h_mode.addStretch(1)
        v.addLayout(h_mode)
        
        # Individual step controls
        h_ind_step = QtWidgets.QHBoxLayout()
        h_ind_step.addWidget(QtWidgets.QLabel("Individual step:"))
        self.spin_ind_step = QtWidgets.QSpinBox()
        self.spin_ind_step.setRange(1, 10)
        self.spin_ind_step.setValue(1)
        self.spin_ind_step.valueChanged.connect(self.on_individual_step_changed)
        h_ind_step.addWidget(self.spin_ind_step)
        h_ind_step.addStretch(1)
        v.addLayout(h_ind_step)
        
        # Individual navigation buttons
        h_ind_nav = QtWidgets.QHBoxLayout()
        self.btn_ind_prev = QtWidgets.QPushButton("◀ Prev Individual")
        self.btn_ind_next = QtWidgets.QPushButton("Next Individual ▶")
        self.btn_ind_prev.clicked.connect(lambda: self._individual_step(-self.individual_step))
        self.btn_ind_next.clicked.connect(lambda: self._individual_step(+self.individual_step))
        h_ind_nav.addWidget(self.btn_ind_prev)
        h_ind_nav.addWidget(self.btn_ind_next)
        v.addLayout(h_ind_nav)
        
        # Mode status label
        self.lbl_mode = QtWidgets.QLabel("Mode: All")
        v.addWidget(self.lbl_mode)

        v.addSpacing(8)
        v.addWidget(self._sep("Save Marks"))

        # Save/Undo 버튼
        h_save = QtWidgets.QHBoxLayout()
        self.btn_save_start = QtWidgets.QPushButton("Save START")
        self.btn_save_end   = QtWidgets.QPushButton("Save END")
        self.btn_undo       = QtWidgets.QPushButton("Undo")
        self.btn_save_start.clicked.connect(self.on_save_start)
        self.btn_save_end.clicked.connect(self.on_save_end)
        self.btn_undo.clicked.connect(self.on_undo)
        h_save.addWidget(self.btn_save_start)
        h_save.addWidget(self.btn_save_end)
        h_save.addWidget(self.btn_undo)
        v.addLayout(h_save)

        # 상태 라벨
        self.lbl_state = QtWidgets.QLabel("allowed_next: start")
        v.addWidget(self.lbl_state)

        v.addStretch(1)
        return w

    def _sep(self, text: str) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel(f"<b>{text}</b>"))
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        lay.addWidget(line)
        return box

    # ========== 이벤트 ==========
    def on_step_changed(self):
        self.step_size = int(self.cmb_step.currentText())
    
    def on_control_mode_changed(self, mode_text: str):
        self.control_mode = mode_text.lower()
        self.lbl_mode.setText(f"Mode: {mode_text}")
        self._refresh()
    
    def on_individual_step_changed(self, value: int):
        self.individual_step = value

    def on_point_size_changed(self, v: int):
        self.point_radius = int(v)
        self._refresh()

    def on_lidar_toggle(self, checked: bool):
        self.project_lidar = checked
        self._refresh()

    def on_slider_changed(self, value: int):
        # 슬라이더 이동은 "현재 조합에서 상대적으로 동일 step" 이동
        delta = value - self.lidar_idx
        if delta != 0:
            self._relative_step(delta, update_slider=False)
            # 내부에서 lidar_idx 갱신했으므로 슬라이더는 동기화만
            self.sld.blockSignals(True)
            self.sld.setValue(self.lidar_idx)
            self.sld.blockSignals(False)
    def _init_shortcuts(self):
    # 숫자 1~6 -> Cam1~Cam6
        for n in range(1, 7):
            sc = QShortcut(QKeySequence(str(n)), self)
            sc.activated.connect(lambda n=n: self._set_control_mode_cam(n))

        # L / l -> LiDAR
        sc_l = QShortcut(QKeySequence(QtCore.Qt.Key.Key_L), self)
        sc_l.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_l.activated.connect(self._set_control_mode_lidar)
        sc_comma  = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Comma),  self)
        sc_period = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Period), self)
        sc_comma.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_period.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_comma.activated.connect(lambda: self._individual_step(-self.individual_step))
        sc_period.activated.connect(lambda: self._individual_step(+self.individual_step))

        # 좌우 화살표 -> 전체 Prev/Next (현재 step 크기 사용)
        sc_left  = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Left),  self)
        sc_right = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Right), self)
        sc_left.activated.connect(lambda: self._relative_step(-self.step_size))
        sc_right.activated.connect(lambda: self._relative_step(+self.step_size))
        
        sc_space = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Space), self)
        sc_space.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_space.activated.connect(self._toggle_lidar_visibility)

    def _toggle_lidar_visibility(self):
        # 체크박스를 토글하면 on_lidar_toggle 신호가 자동으로 호출되어 화면 갱신됨
        self.chk_lidar.setChecked(not self.chk_lidar.isChecked())

    def _set_control_mode_cam(self, n: int):
        if 1 <= n <= 6:
            self.cmb_control.setCurrentText(f"Cam{n}")  # on_control_mode_changed가 자동 호출됨

    def _set_control_mode_lidar(self):
        self.cmb_control.setCurrentText("LiDAR")        # on_control_mode_changed가 자동 호출됨
    # 공통 이동 로직(라이다 + 6카메라 모두 동일 delta)
    def _relative_step(self, delta: int, update_slider: bool = True):
        if delta == 0: 
            return
        # lidar
        self.lidar_idx = int(np.clip(self.lidar_idx + delta, 0, len(lidar_files) - 1))
        # cams
        for i in range(self.num_cams):
            max_i = len(camera_files[i]) - 1 if len(camera_files[i]) > 0 else 0
            self.img_idx[i] = int(np.clip(self.img_idx[i] + delta, 0, max_i))
        if update_slider:
            self.sld.blockSignals(True)
            self.sld.setValue(self.lidar_idx)
            self.sld.blockSignals(False)
        self._refresh()

    # 개별 이동 로직
    def _individual_step(self, delta: int):
        if delta == 0:
            return
        
        if self.control_mode == "all":
            # 기존 전체 이동과 동일
            self._relative_step(delta)
        elif self.control_mode == "lidar":
            # LiDAR만 이동
            self.lidar_idx = int(np.clip(self.lidar_idx + delta, 0, len(lidar_files) - 1))
            self._refresh()
        elif self.control_mode.startswith("cam"):
            # 특정 카메라만 이동
            cam_num = int(self.control_mode[-1]) - 1  # cam1 -> 0, cam2 -> 1, ...
            if 0 <= cam_num < self.num_cams:
                max_i = len(camera_files[cam_num]) - 1 if len(camera_files[cam_num]) > 0 else 0
                self.img_idx[cam_num] = int(np.clip(self.img_idx[cam_num] + delta, 0, max_i))
        self._refresh()

    # 저장: START
    def on_save_start(self):
        if self.allowed_next != "start":
            self._toast("지금은 END만 저장할 수 있습니다.")
            return
        label = f"start{self.segment_id}"
        snap = _build_snapshot(label, self.lidar_idx, self.img_idx)
        _append_json(marks_json_path, snap)
        snap_fixed = copy.deepcopy(snap)  # ← 참조 끊기
        self.undo_stack.append({"op":"start", "segment_id": self.segment_id, "snapshot": snap_fixed})
        self.allowed_next = "end"
        self.current_phase = "start"
        self.snap_start = snap_fixed
        self._toast(f"Saved {label}")
        self._refresh_state_label()

    # 저장: END
    def on_save_end(self):
        if self.allowed_next != "end":
            self._toast("지금은 START만 저장할 수 있습니다.")
            return
        label = f"end{self.segment_id}"
        snap = _build_snapshot(label, self.lidar_idx, self.img_idx)
        _append_json(marks_json_path, snap)
        snap_fixed = copy.deepcopy(snap)  # ← 참조 끊기
        self.undo_stack.append({"op":"end", "segment_id": self.segment_id, "snapshot": snap_fixed})
        self.segment_id += 1
        self.allowed_next = "start"
        self.current_phase = "end"
        self.snap_end = snap_fixed
        self._toast(f"Saved {label}")
        self._refresh_state_label()

    # 되돌리기(최근 저장 1개 롤백)
    def on_undo(self):
        if not self.undo_stack:
            self._toast("되돌릴 저장이 없습니다.")
            return
        last = self.undo_stack.pop()
        # 파일에서 마지막 1개 entry 제거(간단한 방식: 모두 로드→pop→다시 저장)
        try:
            data = []
            if marks_json_path.exists():
                with open(marks_json_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, list):
                        data = d
            if data:
                data.pop()  # 마지막 항목 제거
                with open(marks_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._toast(f"Undo 실패: {e}")
            return

        # 상태 복구: START/END 순서 토글
        if last["op"] == "start":
            self.allowed_next = "start"  # 되돌렸으니 다시 START 가능
            self.current_phase = None
            self.snap_start = None
        else:
            # end를 되돌렸으면 segment_id도 하나 되돌림
            self.segment_id = max(1, self.segment_id - 1)
            self.allowed_next = "end"
            self.current_phase = "start"
            self.snap_end = None
        self._toast("되돌렸습니다.")
        self._refresh_state_label()

    def _toast(self, msg: str):
        self.statusBar().showMessage(msg, 2000)

    def _refresh_state_label(self):
        self.lbl_state.setText(f"allowed_next: {self.allowed_next}   (segment_id={self.segment_id})")

    # ========== 렌더/정보 ==========
    def _render_canvas(self) -> np.ndarray:
        imgs = []
        for i in range(self.num_cams):
            imgs.append(project_one_cam(
                i, self.img_idx[i], self.lidar_idx,
                draw_lidar=self.project_lidar,
                point_radius=self.point_radius
            ))
        can = build_canvas(imgs)
        
        # Add segment marks overlay
        overlay_segment_marks(can, self.current_phase, self.snap_start, self.snap_end, self.segment_id, 
                            self.lidar_idx, self.img_idx, self.control_mode)
        
        return can

    def _refresh(self):
        # 인덱스 라벨 - 모드에 따라 다르게 표시
        if self.control_mode == "lidar":
            self.lbl_idx.setText(f"LiDAR Frame: {self.lidar_idx+1} / {len(lidar_files)}")
        elif self.control_mode.startswith("cam"):
            cam_num = int(self.control_mode[-1]) - 1
            if 0 <= cam_num < self.num_cams:
                self.lbl_idx.setText(f"Cam{cam_num+1} Frame: {self.img_idx[cam_num]+1} / {len(camera_files[cam_num])}")
        else:
            self.lbl_idx.setText(f"Frame: {self.lidar_idx+1} / {len(lidar_files)}")
        
        # 파일 표시
        lid = str(lidar_files[self.lidar_idx]) if lidar_files else "N/A"
        if lidar_files:
            lidar_timestamp = _ts(lidar_files[self.lidar_idx])
            lidar_line = f">>> LiDAR: {lidar_timestamp} <<<" if self.control_mode == "lidar" else f"LiDAR: {lidar_timestamp}"
        else:
            lidar_line = "LiDAR: N/A"
        
        cam_lines = []
        for i in range(self.num_cams):
            if camera_files[i]:
                timestamp = _ts(camera_files[i][self.img_idx[i]])
                # 현재 제어 모드인 카메라는 강조 표시
                if self.control_mode == f"cam{i+1}":
                    cam_lines.append(f">>> Cam{i+1}: {timestamp} <<<")
                else:
                    cam_lines.append(f"Cam{i+1}: {timestamp}")
            else:
                cam_lines.append(f"Cam{i+1}: (no files)")
        
        self.txt_files.setPlainText(lidar_line + "\n" + "\n".join(cam_lines))

        # 렌더
        self.cached_canvas = self._render_canvas()
        self.view.show_ndarray(self.cached_canvas)
        self._refresh_state_label()


# =========================
# 8) 전역 변수 및 초기화
# =========================

# 전역 변수들
camera_files: List[List[Path]] = []
lidar_files: List[Path] = []
calib_data = None
marks_json_path: Path = None

def initialize_data():
    """Initialize all data structures"""
    global camera_files, lidar_files, calib_data, marks_json_path
    
    print("Loading scene metadata...")
    scene_meta = load_scene_meta()
    
    print("Loading camera and LiDAR files...")
    camera_files, lidar_files = load_camera_and_lidar_files()
    
    print(f"Found {len(lidar_files)} LiDAR files")
    for i, cam_files in enumerate(camera_files):
        print(f"Camera {i+1}: {len(cam_files)} files")
    
    # Try to load calibration data
    calib_path = Path("./calib_matrix/matrix0801.yaml")
    if calib_path.exists():
        print("Loading calibration data from calib_matrix/matrix0801.yaml...")
        calib_data = load_calib_yaml(calib_path)
    else:
        print("No calibration file found, using default parameters")
        calib_data = None
    
    # Setup marks JSON path
    marks_dir = Path("./marks_json")
    marks_dir.mkdir(parents=True, exist_ok=True)
    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    marks_json_path = marks_dir / f"sync_marks_{run_ts}.json"
    
    print(f"Marks will be saved to: {marks_json_path}")

# =========================
# 9) 메인 함수
# =========================

def main():
    # Initialize data
    initialize_data()
    
    # Check if we have data
    if len(lidar_files) == 0:
        print("Error: No LiDAR files found!")
        return
    
    if all(len(cam_files) == 0 for cam_files in camera_files):
        print("Error: No camera files found!")
        return
    
    # Start Qt application
    app = QtWidgets.QApplication(sys.argv)
    win = Viewer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()