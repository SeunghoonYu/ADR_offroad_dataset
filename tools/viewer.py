#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6-Cam + 1-LiDAR Visualization Viewer (PyQt6)

Self-contained viewer application for Iffroad dataset
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
import matplotlib as mpl
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
from PyQt6 import QtCore, QtGui, QtWidgets
import datetime as dt
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6 import QtCore
from PyQt6.QtWidgets import QProgressBar
from dataclasses import dataclass
from typing import Callable
import csv

# =========================
# 0) CONFIG (한 곳에서 수정)
# =========================
@dataclass
class AppConfig:
    # 데이터셋 루트 (읽기/marks_json 저장의 기준)
    base_dir: Path = Path("/mnt/e/off-road/data_0822/test0822_15_22")

    # 씬 export 네이밍 접두
    dataset_tag: str = "Gwangmyeong_Hagon"

    # 보정 파일 경로(없으면 기본 파라미터 사용)
    calib_yaml: Optional[Path] = Path("./calib_matrix/matrix0801.yaml")

    # 서브폴더/이름 규칙들
    marks_subdir: str = "marks_json"
    lidar_dirname: str = "lidar_xyzi"
    camera_root: str = "decoded_rgb"
    camera_prefix: str = "camera_"   # camera_1 ~ camera_6
    radar_dirs: Tuple[str, ...] = ("radar1", "radar2", "radar3")

    # 뷰어 디폴트
    camera_count: int = 6
    tile_w: int = 640
    tile_h: int = 480
    timeline_h: int = 300
    start_index_default: int = 50

    # --- LiDAR 표시/색상 ---
    lidar_cmap: str = "turbo_r"              # 예: "turbo", "viridis", "plasma", "jet", ...
    lidar_color_use_fixed_range: bool = True
    lidar_color_min_m: float = 0.0         
    lidar_color_max_m: float = 50.0        
    lidar_max_display_range_m: float = 200.0  

CFG = AppConfig()

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
    """Load camera and LiDAR file paths + return base_dir"""
    base_dir = CFG.base_dir
    camera_dirs = [base_dir / CFG.camera_root / f"{CFG.camera_prefix}{i}"
                   for i in range(1, CFG.camera_count + 1)]
    lidar_dir = base_dir / CFG.lidar_dirname
    
    camera_files = []
    for cam_dir in camera_dirs:
        if cam_dir.exists():
            files = sorted(list(cam_dir.glob("*.jpg")))
            camera_files.append(files)
        else:
            print(f"Warning: {cam_dir} not found")
            camera_files.append([])
    
    if lidar_dir.exists():
        lidar_files = sorted(list(lidar_dir.glob("*.bin")))
    else:
        print(f"Warning: {lidar_dir} not found")
        lidar_files = []

    return camera_files, lidar_files, base_dir



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
    
def _extract_sec_nsec(p: Path) -> Optional[Tuple[int, int]]:
    """
    파일명(stem)에서 끝의 숫자 2개를 (sec, nsec)으로 인식.
    예) cam3_frame_0123_1724141253_123456789.jpg
        lidar_000123_1724141253_123456789xyzi.bin
        radar1_000246_1724141253_123456789.bin
    """
    tokens = Path(p).stem.split('_')
    nums = []
    for tok in tokens:
        m = re.search(r'(\d+)', tok)  # '123456789xyzi' -> '123456789' 매칭
        if m:
            nums.append(int(m.group(1)))
    if len(nums) >= 2:
        return nums[-2], nums[-1]
    return None

def ts_float_from_path(p: Path) -> Optional[float]:
    ss = _extract_sec_nsec(p)
    if ss is None: 
        return None
    sec, nsec = ss
    return sec + nsec * 1e-9

def ts_str_from_path(p: Path) -> Optional[str]:
    ss = _extract_sec_nsec(p)
    if ss is None:
        return None
    sec, nsec = ss
    return f"{sec}_{nsec}"

def _build_ts_index(files: List[Path]) -> Dict[str, Path]:
    """파일들의 TS 문자열(sec_nsec) → Path 매핑 딕셔너리"""
    idx = {}
    for p in files:
        ts = ts_str_from_path(p)
        if ts:
            idx[ts] = p
    return idx


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
    points = filter_lidar_points(points, max_range=CFG.lidar_max_display_range_m)
    
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
    # 1) 사용할 컬러맵 선택 (matplotlib 3.6+: mpl.colormaps)
    try:
        cmap = mpl.colormaps.get(CFG.lidar_cmap, mpl.colormaps["jet"])
    except Exception:
        # 구버전 대비 안전장치
        cmap = getattr(plt.cm, CFG.lidar_cmap, plt.cm.jet)

    # 2) 정규화 스케일 결정
    if CFG.lidar_color_use_fixed_range:
        rng_min = CFG.lidar_color_min_m
        rng_max = CFG.lidar_color_max_m
        denom = max(rng_max - rng_min, 1e-6)
        norm = (ranges - rng_min) / denom
    else:
        rmax = float(ranges.max()) if ranges.size else 1.0
        norm = ranges / max(rmax, 1e-6)

    # 3) [0,1]로 클램프 후 색 변환
    norm = np.clip(norm, 0.0, 1.0)
    colors = (cmap(norm)[:, :3] * 255).astype(np.uint8)

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
    target_size = (CFG.tile_w, CFG.tile_h)
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
    bottom_space = np.zeros((CFG.timeline_h, canvas.shape[1], 3), dtype=np.uint8)
    canvas = np.vstack([canvas, bottom_space])

    return canvas

# Radar
# === add near _list_inputs_from_base ===
def _list_radars_from_base(base_dir: Path):
    radar_dirs = [base_dir / d for d in CFG.radar_dirs]
    radar_files = []
    for d in radar_dirs:
        radar_files.append(sorted(d.glob("*.bin")) if d.exists() else [])
    return radar_files, radar_dirs

def _precompute_radar_times(radar_files_lists):
    """각 레이더 리스트에 대해 float timestamp 배열을 미리 계산"""
    times = []
    for files in radar_files_lists:
        if files:
            arr = np.array([ts_float_from_path(p) or np.nan for p in files], dtype=np.float64)
        else:
            arr = np.array([], dtype=np.float64)
        times.append(arr)
    return times

def _nearest_index(times_arr: np.ndarray, t: float) -> Optional[int]:
    """정렬 가정 하에 t와 가장 가까운 인덱스 반환 (없으면 None)"""
    if times_arr.size == 0 or not np.isfinite(t):
        return None
    # NaN 제거
    valid = np.isfinite(times_arr)
    if not valid.any():
        return None
    a = times_arr[valid]
    # 이진탐색
    idx = np.searchsorted(a, t)
    cand = []
    if idx > 0: cand.append(idx - 1)
    if idx < a.size: cand.append(idx)
    if not cand:
        return None
    # 원본 인덱스로 되돌리기
    valid_idx = np.flatnonzero(valid)
    best_local = min(cand, key=lambda j: abs(a[j] - t))
    return int(valid_idx[best_local])


# =========================
# 6) 스냅샷 및 JSON 유틸리티
# =========================

def _build_snapshot(label: str, lidar_idx: int, cam_idx_list: List[int]) -> dict:
    return {
        'label': label,
        'timestamp': dt.datetime.now().isoformat(),
        'lidar_idx': int(lidar_idx),
        'cam_indices': list(cam_idx_list),   # ← 리스트 복사!
        "files": {
            "lidar": str(lidar_files[lidar_idx]) if 0 <= lidar_idx < len(lidar_files) else None,
            "cams": [str(camera_files[i][cam_idx_list[i]]) if (0 <= cam_idx_list[i] < len(camera_files[i])) else None for i in range(6)]
        }
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

# =========================
# X) Export (copy) helpers
# =========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    """dst에 동일명이 있으면 덮어씀."""
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except Exception:
            pass
    shutil.copy2(src, dst)

def copy_dir(src: Path, dst: Path, log=None):
    """src 디렉토리를 dst로 재귀 복사 (있으면 덮어씀)"""
    if not src.exists():
        if log: log(f"[warn] dir not found: {src}")
        return
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)  # py>=3.8
        if log: log(f"[info] copied dir: {src} -> {dst}")
    except Exception as e:
        if log: log(f"[error] copy dir failed: {src} -> {dst} ({e})")

def _get_imu_csv_path(base_dir: Path) -> Optional[Path]:
    p = base_dir / "imu" / "imu_data.csv"
    return p if p.exists() else None

def _infer_base_dir_from_marks(data: list) -> Optional[Path]:
    """
    marks_json 항목에 저장된 파일 경로(files.lidar / files.cams[0])로부터 base_dir 추정.
    파일이 실제로 존재하지 않아도 경로 문자열 패턴으로 폴백 추정한다.
    """
    for it in data:
        files = it.get("files", {}) or {}

        # 1) lidar 우선 시도
        lidar_path = files.get("lidar")
        if lidar_path:
            p = Path(lidar_path)
            # (A) 실제 경로가 있으면 기존 로직
            if p.exists() and p.parent.name == "lidar_xyzi":
                return p.parent.parent
            # (B) 폴백: 문자열 패턴으로 추정 (존재하지 않아도 됨)
            s = str(lidar_path)
            m = re.search(r"(.*?)(?:/|\\)lidar_xyzi(?:/|\\)", s)
            if m:
                cand = Path(m.group(1))
                return cand

        # 2) cams
        cams = files.get("cams", []) or []
        for c in cams:
            if not c:
                continue
            p = Path(c)
            # (A) 실제 경로가 있으면 기존 로직
            if p.exists():
                # .../decoded_rgb/camera_i/xxx.jpg
                if p.parent.parent.name == "decoded_rgb":
                    return p.parent.parent.parent
            # (B) 폴백: 문자열 패턴으로 추정
            s = str(c)
            m = re.search(r"(.*?)(?:/|\\)decoded_rgb(?:/|\\)camera_\d+(?:/|\\)", s)
            if m:
                cand = Path(m.group(1))
                return cand

    return None

def _list_inputs_from_base(base_dir: Path):
    lidar_xyzi_dir = base_dir / "lidar_xyzi"
    lidar_raw_dir  = base_dir / "lidar"          # ← 추가
    cam_dirs  = [base_dir / "decoded_rgb" / f"camera_{i}" for i in range(1,7)]

    lidar_xyzi = sorted(lidar_xyzi_dir.glob("*.bin"))
    lidar_raw  = sorted(lidar_raw_dir.glob("*.bin")) if lidar_raw_dir.exists() else []  # ← 추가
    cams       = [sorted(d.glob("*.jpg")) for d in cam_dirs]

    # 반환값을 (lidar_xyzi, cams, lidar_raw)로 확장
    return lidar_xyzi, cams, lidar_raw

def _pair_segments_from_marks(data: list):
    """
    현재 뷰어의 marks 포맷( top-level: lidar_idx / cam_indices )과
    사용자가 예전에 썼던 포맷( indices: {lidar_idx, cam_idx} ) 둘 다 지원.
    """
    def _get_sid(label: str) -> Optional[Tuple[str, int]]:
        if label.startswith("start"):
            return ("start", int(label.replace("start", "")))
        if label.startswith("end"):
            return ("end", int(label.replace("end", "")))
        return None

    starts, ends = {}, {}
    for it in data:
        label = it.get("label", "")
        kind_sid = _get_sid(label)
        if not kind_sid:
            continue
        kind, sid = kind_sid
        if kind == "start": starts[sid] = it
        else: ends[sid] = it

    scene_ids = sorted(set(starts) & set(ends))
    pairs = []
    for sid in scene_ids:
        st = starts[sid]; ed = ends[sid]
        # indices 포맷 우선
        if "indices" in st and "indices" in ed:
            l0 = int(st["indices"]["lidar_idx"])
            l1 = int(ed["indices"]["lidar_idx"])
            c0 = list(map(int, st["indices"]["cam_idx"]))
            c1 = list(map(int, ed["indices"]["cam_idx"]))
        else:
            # 현재 뷰어 포맷: top-level
            l0 = int(st["lidar_idx"])
            l1 = int(ed["lidar_idx"])
            c0 = list(map(int, st["cam_indices"]))
            c1 = list(map(int, ed["cam_indices"]))
        pairs.append((sid, (l0, l1), (c0, c1), st, ed))
    return pairs

def _derive_root_name(base_dir: Path, dataset_tag: Optional[str]) -> str:
    base_name = base_dir.name  # e.g., "test0807_15_11"
    if dataset_tag:
        m = re.search(r"(\d.*)$", base_name)
        suffix = m.group(1) if m else base_name
        return f"{dataset_tag}_{suffix}_scenes"
    else:
        return f"{base_name}_scenes"

def export_scenes_from_marks(marks_json_path: Path,
                             dataset_tag: Optional[str] = "test",
                             log_cb=None,
                             progress_cb: Optional[Callable[[int,int], None]] = None,
                             base_dir_override: Optional[Path] = None):
    """
    marks_json을 읽어 start/end 쌍 단위로 실제 파일을
    {OUT_ROOT}/{root_name}_{sid}/ 이하에 copy.
      - lidar_xyzi  : sec_nsec 기반 이름으로 copy
      - lidar (raw) : 같은 sec_nsec 매칭해 copy
      - decoded_rgb : LiDAR TS prefix로 jpg copy
      - radar1..3   : LiDAR TS에 가장 가까운 ts를 골라 copy
      - camera_info, tf_static 디렉토리 통째로 copy
      - 사용한 marks_json도 scene_dir/marks_json/ 에 함께 copy
      - IMU (base_dir/imu/imu_data.csv): LiDAR start/end 사이 구간만 잘라 scene_dir/imu/imu.csv로 저장
    진행률(progress_cb)은 프레임 단위로 호출.
    """

    # ------------ 내부 유틸 (로그/IMU 로딩) ------------
    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    def _get_imu_csv_path(base_dir: Path) -> Optional[Path]:
        p = base_dir / "imu" / "imu_data.csv"
        return p if p.exists() else None

    def _load_imu_csv(imu_csv: Path) -> Tuple[List[str], np.ndarray, List[List[str]]]:
        """
        CSV를 읽어 (헤더, times(float sec), rows(list-of-str)) 반환.
        지원 포맷(행당 timestamp 해석 우선순위):
          1) 'sec' + 'nsec'
          2) 'header.stamp.secs' + 'header.stamp.nsecs'
          3) 'timestamp' 또는 'time' (float sec)
          4) 첫 컬럼이 '123_456789000' 같은 'sec_nsec'
        위가 없으면 시도 중 실패한 행은 스킵.
        """
        import csv, math, re

        with imu_csv.open("r", newline="", encoding="utf-8") as f:
            sample = f.read(2048)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel  # ★ 폴백
            reader = csv.reader(f, dialect)
            try:
                header = next(reader)
            except StopIteration:
                return [], np.array([], dtype=np.float64), []

        # DictReader로 다시 읽기 (동일 dialect)
        with imu_csv.open("r", newline="", encoding="utf-8") as f2:
            reader = csv.DictReader(f2, fieldnames=header, dialect=dialect)
            rows = []
            times = []

            # 어떤 키로 timestamp를 만들지 미리 결정
            hdr = [h.strip() for h in header]

            def parse_time(row: Dict[str, str]) -> Optional[float]:
                # 1) sec/nsec
                if "sec" in row and "nsec" in row:
                    try:
                        return float(int(row["sec"])) + float(int(row["nsec"])) * 1e-9
                    except:
                        pass

                # 2) header.stamp.secs / header.stamp.nsecs
                if ("header.stamp.secs" in row) and ("header.stamp.nsecs" in row):
                    try:
                        return float(int(row["header.stamp.secs"])) + float(int(row["header.stamp.nsecs"])) * 1e-9
                    except:
                        pass

                # 3) timestamp or time (float sec)
                for key in ("timestamp", "time"):
                    if key in row:
                        try:
                            return float(row[key])
                        except:
                            pass

                # 4) 첫 컬럼이 "1724141253_123456789" 형태
                first_key = hdr[0] if hdr else None
                if first_key and first_key in row:
                    m = re.match(r"^\s*(\d+)\s*[_-]\s*(\d+)\s*$", str(row[first_key]))
                    if m:
                        try:
                            return float(int(m.group(1))) + float(int(m.group(2))) * 1e-9
                        except:
                            pass

                return None

            # 실제 읽기
            for i, row in enumerate(reader):
                # DictReader는 헤더행도 한 번 나오므로 스킵
                if i == 0 and list(row.values()) == header:
                    continue
                t = parse_time(row)
                if t is None or not np.isfinite(t):
                    # 파싱 실패한 행은 스킵
                    continue
                times.append(t)
                # 원본 순서를 유지한 채 문자열 리스트로 저장
                rows.append([row.get(h, "") for h in header])

        if not rows:
            return header, np.array([], dtype=np.float64), []

        times_arr = np.asarray(times, dtype=np.float64)
        # 혹시 시간 정렬 안되어 있으면 정렬
        if not np.all(np.diff(times_arr) >= 0):
            order = np.argsort(times_arr)
            times_arr = times_arr[order]
            rows = [rows[i] for i in order]

        return header, times_arr, rows

    # ------------ 1) marks 로드 ------------
    if not marks_json_path.exists():
        raise FileNotFoundError(f"Marks JSON not found: {marks_json_path}")
    with marks_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Marks JSON must be a list of entries.")

    # ------------ 2) base_dir 추정 ------------
    base_dir = base_dir_override if base_dir_override else _infer_base_dir_from_marks(data)
    if base_dir is None or not base_dir.exists():
        raise RuntimeError("Failed to infer base_dir from marks JSON. (files.lidar / files.cams 경로를 확인하세요)")

    # lidar_xyzi, cams, lidar_raw
    lidar_xyzi_files, cam_files, lidar_raw_files = _list_inputs_from_base(base_dir)
    lidar_raw_by_ts = _build_ts_index(lidar_raw_files)

    # radar
    radar_files, radar_dirs = _list_radars_from_base(base_dir)
    radar_times = _precompute_radar_times(radar_files)

    # IMU (규약 경로: base_dir/imu/imu_data.csv)
    imu_header: List[str] = []
    imu_times: np.ndarray = np.array([], dtype=np.float64)
    imu_rows: List[List[str]] = []
    imu_src = _get_imu_csv_path(base_dir)
    if imu_src:
        log(f"[info] IMU CSV     : {imu_src}")
        imu_header, imu_times, imu_rows = _load_imu_csv(imu_src)
        log(f"[info] IMU rows    : {len(imu_rows)} (times parsed)")
    else:
        log(f"[warn] IMU CSV not found at {base_dir / 'imu' / 'imu_data.csv'}")

    root_name = _derive_root_name(base_dir, dataset_tag)
    out_root = base_dir.parent / root_name
    ensure_dir(out_root)

    log(f"[info] Base dir     : {base_dir}")
    log(f"[info] Output root  : {out_root}")
    log(f"[info] Marks JSON   : {marks_json_path}")
    log(f"[info] Mode         : copy")
    log(f"[info] LiDAR xyzi   : {len(lidar_xyzi_files)} files")
    log(f"[info] LiDAR raw    : {len(lidar_raw_files)} files")
    for i, d in enumerate(radar_dirs, 1):
        log(f"[info] Radar{i} dir  : {d}  (files={len(radar_files[i-1])})")

    # ------------ 3) start/end 페어 만들기 ------------
    pairs_raw = _pair_segments_from_marks(data)
    if not pairs_raw:
        log("[error] No start/end pairs found.")
        return

    # 진행률 총 프레임 수
    pair_infos = []
    total_frames = 0
    for sid, (l0, l1), (c0, c1), st, ed in pairs_raw:
        L_len = max(0, l1 - l0 + 1)
        C_len = [max(0, c1[i] - c0[i] + 1) for i in range(6)]
        seg_len = min([L_len] + C_len)
        if seg_len > 0:
            pair_infos.append((sid, (l0, l1), (c0, c1), seg_len, st, ed))
            total_frames += seg_len

    done_frames = 0
    if progress_cb:
        progress_cb(done_frames, total_frames)

    # ------------ 4) 씬별 복사 ------------
    for sid, (l0, l1), (c0, c1), seg_len, st, ed in pair_infos:
        scene_dir = out_root / f"{root_name}_{sid}"
        out_lidar_xyzi = scene_dir / "lidar_xyzi"
        out_lidar_raw  = scene_dir / "lidar"
        out_cams  = [scene_dir / "decoded_rgb" / f"camera_{i}" for i in range(1,7)]
        out_radars = [scene_dir / f"radar{i}" for i in range(1,4)]

        ensure_dir(out_lidar_xyzi)
        ensure_dir(out_lidar_raw)
        for d in out_cams: ensure_dir(d)
        for d in out_radars: ensure_dir(d)

        # 부가 폴더/파일 (씬마다)
        copy_dir(base_dir / "camera_info", scene_dir / "camera_info", log)
        copy_dir(base_dir / "tf_static",   scene_dir / "tf_static",   log)
        dst_mj_dir = scene_dir / "marks_json"
        ensure_dir(dst_mj_dir)
        copy_file(marks_json_path, dst_mj_dir / marks_json_path.name)

        # 프레임 단위 복사
        for k in range(seg_len):
            lidar_src = lidar_xyzi_files[l0 + k]
            li_ts_str = ts_str_from_path(lidar_src) or _ts(lidar_src)
            li_ts_val = ts_float_from_path(lidar_src)

            # LiDAR xyzi
            dst_xyzi = out_lidar_xyzi / f"{li_ts_str}_{k:06d}.bin"
            copy_file(lidar_src, dst_xyzi)

            # LiDAR raw 매칭
            raw_src = lidar_raw_by_ts.get(li_ts_str)
            if raw_src is not None:
                dst_raw = out_lidar_raw / f"{li_ts_str}_{k:06d}.bin"
                copy_file(raw_src, dst_raw)
            else:
                log(f"[warn][scene {sid}] raw LiDAR not found for ts={li_ts_str}")

            # Cams
            for i in range(6):
                src_cam = cam_files[i][c0[i] + k]
                dst_cam = out_cams[i] / f"{li_ts_str}_{k:06d}.jpg"
                copy_file(src_cam, dst_cam)

            # Radars (가까운 ts 골라서)
            for r in range(3):
                if not radar_files[r]:
                    continue
                idx = _nearest_index(radar_times[r], li_ts_val)
                if idx is None:
                    continue
                src_radar = radar_files[r][idx]
                dst_radar = out_radars[r] / f"{li_ts_str}_{k:06d}.bin"
                copy_file(src_radar, dst_radar)

            done_frames += 1
            if progress_cb:
                progress_cb(done_frames, total_frames)

        # ---- IMU 잘라 저장 (LiDAR start/end 구간) ----
        try:
            if imu_rows and imu_times.size > 0 and seg_len > 0:
                t_start = ts_float_from_path(lidar_xyzi_files[l0])
                t_end   = ts_float_from_path(lidar_xyzi_files[l0 + seg_len - 1])
                if (t_start is not None) and (t_end is not None):
                    # t_start <= t <= t_end 구간 선택 (근접 인덱스)
                    def _nearest(a: np.ndarray, t: float) -> int:
                        if a.size == 0: return 0
                        i = np.searchsorted(a, t)
                        cand = []
                        if i > 0: cand.append(i-1)
                        if i < a.size: cand.append(i)
                        best = min(cand, key=lambda j: abs(a[j]-t)) if cand else 0
                        return int(best)
                    i0 = _nearest(imu_times, t_start)
                    i1 = _nearest(imu_times, t_end)
                    if i0 > i1: i0, i1 = i1, i0
                    # 여유를 조금 두고 포함(선택)
                    sel = imu_rows[i0:i1+1]

                    imu_out_dir = scene_dir / "imu"
                    ensure_dir(imu_out_dir)
                    imu_out_csv = imu_out_dir / "imu.csv"
                    # 쓰기
                    import csv
                    with imu_out_csv.open("w", newline="", encoding="utf-8") as fcsv:
                        writer = csv.writer(fcsv)
                        writer.writerow(imu_header if imu_header else [])
                        for row in sel:
                            writer.writerow(row)
                    log(f"[ok][scene {sid}] IMU slice saved: {imu_out_csv} (rows={len(sel)})")
                else:
                    log(f"[warn][scene {sid}] IMU slice skipped (LiDAR timestamps missing)")
            else:
                log(f"[info][scene {sid}] IMU not available, skipped")
        except Exception as e:
            log(f"[warn][scene {sid}] IMU slice failed: {e}")

        # scene_meta.json
        meta = {
            "scene_id": sid,
            "root_name": root_name,
            "source_base_dir": str(base_dir),
            "length": seg_len,
            "lidar_range": [l0, l0 + seg_len - 1],
            "cam_ranges": [[c0[i], c0[i] + seg_len - 1] for i in range(6)],
            "mode": "copy",
            "radars": {
                "radar1": len(radar_files[0]),
                "radar2": len(radar_files[1]),
                "radar3": len(radar_files[2]),
            },
            "imu": {
                "source": str(imu_src) if imu_src else None,
                "sliced": bool(imu_rows and imu_times.size > 0),
            }
        }
        with (scene_dir / "scene_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        log(f"[ok] {scene_dir}  (frames={seg_len})")

    # 100% 보장
    if progress_cb:
        progress_cb(total_frames, total_frames)
    log("[done] Export finished.")


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
    timeline_h = CFG.timeline_h

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
    clicked = pyqtSignal(int, int)  # (ix, iy) - 원본 캔버스 좌표

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: black;")
        # 원본 이미지 크기와 표시된 픽스맵 영역 기록
        self._img_w = None
        self._img_h = None
        self._disp_w = None
        self._disp_h = None
        self._disp_x0 = 0   # 라벨 내부에서 표시 시작 좌측상단 x
        self._disp_y0 = 0   # 라벨 내부에서 표시 시작 좌측상단 y

    def show_ndarray(self, img_bgr: np.ndarray):
        if img_bgr is None:
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        self._img_w, self._img_h = w, h

        qimg = QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        # 실제로 표시될 크기 계산 (비율 유지)
        scaled = pix.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)

        # 라벨 내부에서 가운데 정렬된 위치 기록
        lw, lh = self.width(), self.height()
        self._disp_w, self._disp_h = scaled.width(), scaled.height()
        self._disp_x0 = (lw - self._disp_w) // 2
        self._disp_y0 = (lh - self._disp_h) // 2

    def resizeEvent(self, e):
        if self.pixmap():
            scaled = self.pixmap().scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)
            # 리사이즈 시 표시 위치 갱신
            lw, lh = self.width(), self.height()
            self._disp_w, self._disp_h = scaled.width(), scaled.height()
            self._disp_x0 = (lw - self._disp_w) // 2
            self._disp_y0 = (lh - self._disp_h) // 2
        super().resizeEvent(e)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self._img_w is None or self._disp_w is None:
            return super().mousePressEvent(ev)
        x, y = ev.position().x(), ev.position().y()
        # 표시 영역 안인지 확인
        if (self._disp_x0 <= x <= self._disp_x0 + self._disp_w) and (self._disp_y0 <= y <= self._disp_y0 + self._disp_h):
            # 라벨 좌표 → 원본 이미지(캔버스) 좌표로 역변환
            rx = (x - self._disp_x0) * (self._img_w / self._disp_w)
            ry = (y - self._disp_y0) * (self._img_h / self._disp_h)
            self.clicked.emit(int(rx), int(ry))
        return super().mousePressEvent(ev)

class ExportWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)   # 0..100
    log = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, marks_path: Path, dataset_tag="SNU_mountain", base_dir_override=None):
        super().__init__()
        self.marks_path = Path(marks_path)
        self.dataset_tag = dataset_tag
        self.base_dir_override = base_dir_override

    @QtCore.pyqtSlot()
    def run(self):
        def _log(msg: str):
            self.log.emit(msg)
        def _prog(done: int, total: int):
            pct = int(done * 100 / max(1, total))
            self.progress.emit(pct)
        try:
            export_scenes_from_marks(
                self.marks_path,
                dataset_tag=self.dataset_tag,
                log_cb=_log,
                progress_cb=_prog,             # ← 진행률 콜백 전달
                base_dir_override=self.base_dir_override
            )
        except Exception as e:
            self.log.emit(f"[error] {e}")
        finally:
            self.finished.emit()

# def on_export_scenes(self):
#     try:
#         default_dir = str(((dataset_base_dir / "marks_json") if dataset_base_dir else Path("./marks_json")).resolve())
#         path, _ = QtWidgets.QFileDialog.getOpenFileName(
#             self, "Select marks JSON", default_dir, "JSON Files (*.json);;All Files (*)"
#         )
#         if not path:
#             return

#         self._log_export(f"[run] Export from: {path}")
#         self.btn_export.setEnabled(False)
#         self.prog.setValue(0)
#         QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

#         # 쓰레드 + 워커 구성
#         self._exp_thread = QtCore.QThread(self)
#         self._exp_worker = ExportWorker(Path(path), dataset_tag="SNU_mountain")
#         self._exp_worker.moveToThread(self._exp_thread)
#         self._exp_thread.started.connect(self._exp_worker.run)

#         # 신호 연결
#         self._exp_worker.log.connect(self._log_export)
#         self._exp_worker.progress.connect(self.prog.setValue)
#         self._exp_worker.finished.connect(self._exp_thread.quit)
#         self._exp_worker.finished.connect(self._exp_worker.deleteLater)
#         self._exp_thread.finished.connect(self._exp_thread.deleteLater)
#         self._exp_thread.finished.connect(lambda: self.btn_export.setEnabled(True))
#         self._exp_thread.finished.connect(lambda: QtWidgets.QApplication.restoreOverrideCursor())

#         self._exp_thread.start()

#     except Exception as e:
#         self._log_export(f"[error] {e}")
#         self.btn_export.setEnabled(True)
#         QtWidgets.QApplication.restoreOverrideCursor()


class Viewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offroad Dataset Viewer (PyQt6)")
        self.resize(1500, 900)

        self._busy: bool = False            
        self._pending_action = None         

        # ---- 상태 ----
        self.num_cams = CFG.camera_count
        self.img_idx: List[int] = [CFG.start_index_default] * self.num_cams
        self.lidar_idx: int = CFG.start_index_default
        self.project_lidar: bool = False
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
        self.view.clicked.connect(self.on_canvas_click)  # ← 추가

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

    def _run_or_queue(self, fn):
        """진행 중이면 첫 요청만 큐잉, 아니면 즉시 실행."""
        if self._busy:
            if self._pending_action is None:
                self._pending_action = fn
            return
        self._busy = True
        try:
            fn()
        finally:
            self._busy = False
            if self._pending_action:
                nxt = self._pending_action
                self._pending_action = None
                QtCore.QTimer.singleShot(0, nxt)

    def _start_export_dialog(self):
        default_dir = str(((dataset_base_dir / CFG.marks_subdir) if dataset_base_dir else Path("./marks_json")).resolve())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select marks JSON", default_dir, "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self._start_export(Path(path))

    def _start_export(self, marks_path: Path, base_dir_override: Optional[Path] = None):
        self._log_export(f"[run] Export from: {marks_path}")
        self.btn_export.setEnabled(False)
        self.prog.setValue(0)
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # QThread + Worker
        self._exp_thread = QtCore.QThread(self)
        self._exp_worker = ExportWorker(
            marks_path,
            dataset_tag=CFG.dataset_tag,
            base_dir_override=base_dir_override
        )
        self._exp_worker.moveToThread(self._exp_thread)
        self._exp_thread.started.connect(self._exp_worker.run)

        # signals
        self._exp_worker.log.connect(self._log_export)
        self._exp_worker.progress.connect(self.prog.setValue)
        self._exp_worker.finished.connect(self._on_export_finished)

        self._exp_thread.start()

    def _on_export_finished(self):
        try:
            # 100% 보장(혹시 못 올렸을 경우)
            self.prog.setValue(100)
        except Exception:
            pass
        self.btn_export.setEnabled(True)
        QtWidgets.QApplication.restoreOverrideCursor()
        if hasattr(self, "_exp_worker"):
            self._exp_worker.deleteLater()
            del self._exp_worker
        if hasattr(self, "_exp_thread"):
            self._exp_thread.quit()
            self._exp_thread.wait()
            self._exp_thread.deleteLater()
            del self._exp_thread


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
        self.cmb_step.addItems(["1", "5", "10", "20", "50"])
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
        
        v.addSpacing(8)
        v.addWidget(self._sep("Export Scenes"))

        # Export 버튼 + 로그창
        self.btn_export = QtWidgets.QPushButton("Export Scenes (copy)")
        self.btn_export.clicked.connect(self._start_export_dialog)
        v.addWidget(self.btn_export)

        self.txt_export = QtWidgets.QTextEdit()
        self.txt_export.setReadOnly(True)
        self.txt_export.setMinimumHeight(160)
        v.addWidget(self.txt_export)

        self.btn_export_latest = QtWidgets.QPushButton("Export Latest marks_json (copy)")
        self.btn_export_latest.clicked.connect(self.on_export_latest)
        v.addWidget(self.btn_export_latest)

        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        v.addWidget(self.prog)

        v.addStretch(1)
        return w
    def on_canvas_click(self, ix: int, iy: int):
        tile_w, tile_h = 640, 480
        grid_rows, grid_cols = 2, 3
        image_area_h = tile_h * grid_rows  # 960
        image_area_w = tile_w * grid_cols  # 1920

        # 타임라인 영역 클릭은 무시
        if ix < 0 or iy < 0 or ix >= image_area_w or iy >= image_area_h:
            return

        col = ix // tile_w
        row = iy // tile_h
        tile_idx = int(row * grid_cols + col)

        # build_canvas에서 사용한 순서
        order = [1, 0, 5, 4, 3, 2]  # [Cam2, Cam1, Cam6, Cam5, Cam4, Cam3]
        if 0 <= tile_idx < len(order):
            cam_orig = order[tile_idx]          # 0-based 카메라 인덱스
            self.cmb_control.setCurrentText(f"Cam{cam_orig+1}")  # on_control_mode_changed 트리거

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
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setAutoRepeat(False)  # ← 중요
            sc.activated.connect(lambda n=n: self._run_or_queue(lambda: self._set_control_mode_cam(n)))

        # L -> LiDAR
        sc_l = QShortcut(QKeySequence(QtCore.Qt.Key.Key_L), self)
        sc_l.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_l.setAutoRepeat(False)
        sc_l.activated.connect(lambda: self._run_or_queue(self._set_control_mode_lidar))

        # , / . -> 개별 이동(현 모드 유지)
        sc_comma  = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Comma),  self)
        sc_period = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Period), self)
        for sc in (sc_comma, sc_period):
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setAutoRepeat(False)
        sc_comma.activated.connect(lambda: self._run_or_queue(lambda: self._individual_step(-self.individual_step)))
        sc_period.activated.connect(lambda: self._run_or_queue(lambda: self._individual_step(+self.individual_step)))

        sc_a = QShortcut(QKeySequence(QtCore.Qt.Key.Key_A), self)
        sc_d = QShortcut(QKeySequence(QtCore.Qt.Key.Key_D), self)
        for sc in (sc_a, sc_d):
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setAutoRepeat(False)
        sc_a.activated.connect(lambda: self._run_or_queue(lambda: self._individual_step(-self.individual_step)))
        sc_d.activated.connect(lambda: self._run_or_queue(lambda: self._individual_step(+self.individual_step)))

        # ← / → -> 전역 이동 + All 모드로 전환(빨간 박스 제거)
        sc_left  = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Left),  self)
        sc_right = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Right), self)
        for sc in (sc_left, sc_right):
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setAutoRepeat(False)
        sc_left.activated.connect(lambda: self._run_or_queue(self._jump_global_prev))
        sc_right.activated.connect(lambda: self._run_or_queue(self._jump_global_next))

        # Space -> LiDAR 토글
        sc_space = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Space), self)
        sc_space.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_space.setAutoRepeat(False)
        sc_space.activated.connect(lambda: self._run_or_queue(self._toggle_lidar_visibility))

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

    def _jump_global_prev(self):
        if self.control_mode != "all":
            self.cmb_control.setCurrentText("All")  # 선택 해제(빨간 박스 사라짐)
        self._relative_step(-self.step_size)

    def _jump_global_next(self):
        if self.control_mode != "all":
            self.cmb_control.setCurrentText("All")
        self._relative_step(+self.step_size)


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

        # === 선택된 카메라 빨간 테두리 ===
        if self.control_mode.startswith("cam"):
            try:
                sel_cam = int(self.control_mode[-1]) - 1  # 0-based
            except Exception:
                sel_cam = None
            if sel_cam is not None and 0 <= sel_cam < 6:
                # build_canvas의 타일 순서와 역매핑
                order = [1, 0, 5, 4, 3, 2]
                # 타일 인덱스 찾기
                tile_idx = order.index(sel_cam)
                tile_w, tile_h = 640, 480
                x0 = (tile_idx % 3) * tile_w
                y0 = (tile_idx // 3) * tile_h
                # 테두리 (BGR: 빨강)
                cv2.rectangle(can, (x0+3, y0+3), (x0 + tile_w - 3, y0 + tile_h - 3),
                            (0, 0, 255), thickness=6, lineType=cv2.LINE_AA)

        # 타임라인 합성
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

    def _log_export(self, msg: str):
        self.txt_export.append(msg)
        self.txt_export.ensureCursorVisible()
        self.statusBar().showMessage(msg, 3000)

    def on_export_scenes(self):
        try:
            default_dir = str(((dataset_base_dir / CFG.marks_subdir) if dataset_base_dir else Path("./marks_json")).resolve())
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select marks JSON", default_dir, "JSON Files (*.json);;All Files (*)"
            )
            if not path:
                return

            self._log_export(f"[run] Export from: {path}")
            self.btn_export.setEnabled(False)
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            def _cb(msg):
                self._log_export(msg)

            # 1차 시도: 자동 추정
            try:
                export_scenes_from_marks(Path(path), dataset_tag=CFG.dataset_tag, log_cb=_cb)
            except RuntimeError as e:
                if "Failed to infer base_dir" not in str(e):
                    raise
                # 2차 시도: 사용자에게 base_dir 선택 받기
                self._log_export("[hint] Can't infer base dir from JSON. Select the dataset base directory (contains 'lidar_xyzi' and 'decoded_rgb').")
                base = QtWidgets.QFileDialog.getExistingDirectory(self, "Select dataset base directory")
                if not base:
                    self._log_export("[cancel] No base dir selected.")
                    return
                base_path = Path(base)
                if not (base_path / "lidar_xyzi").exists() or not (base_path / "decoded_rgb").exists():
                    self._log_export("[error] Selected directory does not contain 'lidar_xyzi' and 'decoded_rgb'.")
                    return
                export_scenes_from_marks(Path(path), dataset_tag=CFG.dataset_tag, log_cb=_cb, base_dir_override=base_path)

            self._log_export("[done] Export finished.")
        except Exception as e:
            self._log_export(f"[error] {e}")
        finally:
            self.btn_export.setEnabled(True)
            QtWidgets.QApplication.restoreOverrideCursor()


    def on_export_latest(self):
        try:
            mj = (dataset_base_dir / "marks_json") if dataset_base_dir else Path("./marks_json")
            if not mj.exists():
                self._log_export(f"[error] {mj} not found")
                return
            # 패턴을 *_syn_marks_*.json 으로 변경
            candidates = sorted(mj.glob("*_syn_marks_*.json"))
            if not candidates:
                self._log_export("[error] no *_syn_marks_*.json")
                return
            chosen = candidates[-1]
            self._log_export(f"[run] Export from latest: {chosen}")
            self.btn_export_latest.setEnabled(False)
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._start_export(chosen)
        except Exception as e:
            self._log_export(f"[error] {e}")
        finally:
            self.btn_export_latest.setEnabled(True)
            QtWidgets.QApplication.restoreOverrideCursor()



# =========================
# 8) 전역 변수 및 초기화
# =========================

# 전역 변수들
camera_files: List[List[Path]] = []
lidar_files: List[Path] = []
calib_data = None
marks_json_path: Path = None
dataset_base_dir: Optional[Path] = None  # ← 이미 선언돼 있으면 그대로 두세요

def initialize_data():
    """Initialize all data structures"""
    global camera_files, lidar_files, calib_data, marks_json_path, dataset_base_dir

    print("Loading scene metadata...")
    scene_meta = load_scene_meta()

    print("Loading camera and LiDAR files...")
    
    camera_files, lidar_files, dataset_base_dir = load_camera_and_lidar_files()
    print(f"Base dir: {dataset_base_dir}")
    print(f"Found {len(lidar_files)} LiDAR files")
    for i, cam_files_i in enumerate(camera_files):
        print(f"Camera {i+1}: {len(cam_files_i)} files")

    # Calibration
    calib_path = Path("./calib_matrix/matrix0801.yaml")
    if calib_path.exists():
        print(f"Loading calibration data from {calib_path}...")
        calib_data = load_calib_yaml(calib_path)
    else:
        print("No calibration file found, using default parameters")
        calib_data = None

    
    marks_dir = (dataset_base_dir / "marks_json") if (dataset_base_dir and dataset_base_dir.exists()) else Path("./marks_json")
    marks_dir.mkdir(parents=True, exist_ok=True)

    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = dataset_base_dir.name if dataset_base_dir else "dataset"
    
    marks_json_path = marks_dir / f"{base_name}_syn_marks_{run_ts}.json"

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