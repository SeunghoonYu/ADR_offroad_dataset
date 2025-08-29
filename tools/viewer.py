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
from functools import lru_cache

# =========================
# 0) CONFIG (한 곳에서 수정)
# =========================
@dataclass
class AppConfig:
    # Indexing MArks filename config
    # {sensor}_{origin_data_folder_name}_{postprocessing_time}_{worker_name}.json
    filename_prefix: str = "camera"
    worker_name: str = "TonyStark"

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
    overview_h: int = 120
    overview_base_color: tuple = (180, 120, 60)   # BGR (파란 베이스)
    overview_seg_color: tuple = (0, 160, 255)     # BGR (주황 박스)
    overview_gps_allow_color: tuple = (30, 180, 60)  # 초록(= GPS 불가의 여집합, 사용 가능)
    overview_gps_post_thick: int = 4                 # 초록 기둥/상단 두께
    overview_base_thick: int = 12                 # 베이스 라인 두께
    overview_post_thick: int = 6                  # 세그먼트 기둥/상단 두께
    overview_min_pix: int = 6                     # 세그민트 최소 픽셀 폭
    start_index_default: int = 100

    # --- LiDAR 표시/색상 ---
    lidar_cmap: str = "turbo_r"              # 예: "turbo", "viridis", "plasma", "jet", ...
    lidar_color_use_fixed_range: bool = True
    lidar_color_min_m: float = 2.0         
    lidar_color_max_m: float = 50.0        
    lidar_max_display_range_m: float = 150.0  

    # CLip params
    clip_min_frames: int = 200

    

CFG = AppConfig()

# =========================
# 1) 데이터 로딩 및 설정
# =========================

def _sanitize_token(s: str) -> str:
    """파일명 안전 토큰: 소문자/숫자/언더스코어만 남기고 비우면 'anon'."""
    return re.sub(r'[^a-z0-9_]+', '', (s or '').strip().lower()) or 'anon'


# ① 디코딩 캐시: 파일 이름(혹은 절대경로) 기준
@lru_cache(maxsize=64)
def _load_lidar_points_once(fname: str) -> np.ndarray:
    p = Path(fname)
    pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
    return pts

def load_lidar_points_cached(lidar_path: Path) -> np.ndarray:
    return _load_lidar_points_once(str(lidar_path))

# ② 범위 필터도 키에 포함 (max_range 바뀌면 다른 결과)
@lru_cache(maxsize=128)
def _filter_by_range(fname: str, max_range: float) -> np.ndarray:
    pts = load_lidar_points_cached(Path(fname))
    if pts.size == 0: 
        return pts
    rng = np.linalg.norm(pts[:, :3], axis=1)
    return pts[rng <= max_range]

def get_lidar_points(lidar_path: Path, max_range: float) -> np.ndarray:
    return _filter_by_range(str(lidar_path), float(max_range))

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

import json, re

def _merge_ranges(ranges: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not ranges: return []
    a = sorted((min(s,e), max(s,e)) for s,e in ranges)
    out = [a[0]]
    for s,e in a[1:]:
        ps,pe = out[-1]
        if s <= pe + 1:   # 인접/겹침 병합
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s,e))
    return out

def _complement_ranges(blocks: List[Tuple[int,int]], lo: int, hi: int) -> List[Tuple[int,int]]:
    """[lo,hi]에서 blocks를 뺀 여집합"""
    if lo > hi: return []
    blocks = _merge_ranges([(max(lo,a), min(hi,b)) for a,b in blocks if b>=lo and a<=hi])
    res = []
    cur = lo
    for a,b in blocks:
        if cur < a: res.append((cur, a-1))
        cur = max(cur, b+1)
    if cur <= hi: res.append((cur, hi))
    return res

def _read_gps_bad_ranges(json_path: Path) -> List[Tuple[int,int]]:
    """GNSS JSON에서 startN/endN 페어를 (a,b) 리스트로 추출"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    starts, ends = {}, {}
    for it in data:
        label = str(it.get("label",""))
        m = re.match(r"^(start|end)(\d+)$", label)
        if not m: continue
        kind, n = m.group(1), int(m.group(2))
        li = int(it.get("lidar_idx", 0))
        if kind == "start": starts[n] = li
        else: ends[n] = li
    pairs = []
    for n in sorted(set(starts) & set(ends)):
        a, b = int(starts[n]), int(ends[n])
        if a > b: a, b = b, a
        pairs.append((a,b))
    merged =  _merge_ranges(pairs)
    print(f"[GNSS:read] file={json_path.name} pairs={len(pairs)} merged={len(merged)}")
    if pairs:  print(f"[GNSS:read] pairs_sample={pairs[:8]}")
    if merged: print(f"[GNSS:read] merged_sample={merged[:8]}")

    return merged


def _extract_lidar_segments_from_marks(marks_path: Path) -> list[tuple[int, int]]:
    """
    현재 marks JSON에서 (startN, endN) 페어를 (l0, l1) 리스트로 추출.
    l0 <= l1 로 정렬, 오름차순 정렬 후 반환.
    """
    if not marks_path or (not marks_path.exists()):
        return []
    try:
        with marks_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = _pair_segments_from_marks(data)  # (sid, (l0,l1), (c0,c1), st, ed)
        segs = []
        for sid, (l0, l1), (_c0, _c1), _st, _ed in pairs:
            a, b = int(l0), int(l1)
            if a > b:
                a, b = b, a
            segs.append((a, b))
        segs.sort()
        return segs
    except Exception:
        return []

def _intersect_segments(a: list[tuple[int,int]], b: list[tuple[int,int]], N: int) -> list[tuple[int,int]]:
    """
    [0..N-1] 범위에서 두 세그먼트 집합의 교집합을 반환.
    각 세그먼트는 포함구간 [s,e].
    """
    A = _merge_segments(a, N)   # 이미 있는 함수 재사용
    B = _merge_segments(b, N)
    i = j = 0
    out = []
    while i < len(A) and j < len(B):
        s = max(A[i][0], B[j][0])
        e = min(A[i][1], B[j][1])
        if s <= e:
            out.append((s, e))
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out


def _merge_segments(segs: list[tuple[int,int]], N: int) -> list[tuple[int,int]]:
    """[l0,l1] 포함 구간들을 [0..N-1]에 클램프하고 겹침/인접을 병합."""
    if N <= 0 or not segs: 
        return []
    norm = []
    for a,b in segs:
        a,b = int(a), int(b)
        if a > b: a,b = b,a
        a = max(0, min(N-1, a))
        b = max(0, min(N-1, b))
        if a <= b:
            norm.append((a,b))
    if not norm: 
        return []
    norm.sort()
    merged = [norm[0]]
    for a,b in norm[1:]:
        pa,pb = merged[-1]
        if a <= pb + 1:               # 겹치거나 바로 닿으면 병합
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a,b))
    return merged

def _segments_complement(forbid: list[tuple[int,int]], N: int) -> list[tuple[int,int]]:
    """[0..N-1]에서 forbid(불가, 포함구간)의 여집합(사용 가능)을 반환."""
    if N <= 0: 
        return []
    merged = _merge_segments(forbid, N)
    if not merged:
        return [(0, N-1)]
    allow = []
    cur = 0
    for a,b in merged:
        if cur <= a-1:
            allow.append((cur, a-1))  # 표준적으로 양끝 제외(겹치기 싫으면 -1 사용)
        cur = b + 1
    if cur <= N-1:
        allow.append((cur, N-1))
    return allow

def _load_gps_bad_segments(gps_path: Path, total_lidar: int) -> list[tuple[int,int]]:
    """GPS팀 JSON에서 LiDAR 인덱스 구간을 읽어 불가구간(병합 후)으로 반환."""
    segs = _extract_lidar_segments_from_marks(gps_path)  # startN/endN 포맷 재사용
    return _merge_segments(segs, total_lidar)


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

def _paint_points_fast(img: np.ndarray, xy: np.ndarray, colors: np.ndarray):
    # 경계 체크(안전)
    H, W = img.shape[:2]
    if xy.size == 0: 
        return img
    iu = np.clip(xy[:,0], 0, W-1)
    iv = np.clip(xy[:,1], 0, H-1)
    img[iv, iu] = colors  # 벡터화 대입
    return img

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

def create_lidar_overview_bar(
    total_lidar: int,
    current_lidar_idx: int,
    segs: List[Tuple[int,int]],
    width: int,
    height: int,
    extra_segs: Optional[List[Tuple[int,int]]] = None,   # GNSS 허용 구간
    extra_color: Tuple[int,int,int] = (60, 200, 60),     # BGR (초록)
    merged_segs: Optional[List[Tuple[int,int]]] = None, # [ADD] 빨간(교집합)
    merged_color: Tuple[int,int,int] = (0, 0, 255),     # [ADD]
    *,
    font_shrink: float = 0.3,       # 글씨를 얼마나 줄일지 (1.0=기존, 0.85=조금 작게)
    green_raise_px: int = 30,        # 초록 ㄷ자 바의 높이를 얼마나 더 올릴지(+픽셀)
    green_post_th_add: int = 2       # 초록 ㄷ자 기둥/상단 두께를 얼마나 더 두껍게
) -> np.ndarray:
    """
    화살표 베이스(파란색) + 세그먼트 ㄷ자(주황색) + extra_segs ㄷ자(초록색) 오버뷰 바.
    초록 ㄷ자는 더 높고(=더 위쪽) 약간 더 두껍게, 글씨는 살짝 작게.
    """

    def _put_label(img, text, org, fs, color, thick=1, pad=3):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        x, y = int(org[0]), int(org[1])
        # 경계 안으로 클램프
        x = max(0, min(img.shape[1] - tw - 2, x))
        y = max(th + 2, min(img.shape[0] - 2, y))
        # 흰 배경 + 옅은 테두리로 가독성 확보
        cv2.rectangle(img, (x - pad, y - th - pad - 1), (x + tw + pad, y + pad), (255, 255, 255), -1)
        cv2.rectangle(img, (x - pad, y - th - pad - 1), (x + tw + pad, y + pad), (220, 220, 220), 1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick, cv2.LINE_AA)

    # 크기 & 폰트 스케일
    width = max(480, int(width))
    height = max(60, int(height))
    base_scale = height / 120.0
    fs_main  = max(0.6, 0.9 * base_scale * font_shrink)
    fs_small = max(0.5, 0.8 * base_scale * font_shrink)
    th_main  = max(1, int(round(2 * base_scale)))
    th_small = max(1, int(round(2 * base_scale)))

    

    img = np.full((height, width, 3), 255, np.uint8)
    red_drop = 40
    pad_l, pad_r = 24, 36
    pad_t, pad_b = 10, 8
    rail_y = height - pad_b - CFG.overview_base_thick // 2 - red_drop
    x0, x1 = pad_l, width - pad_r

    base_c = CFG.overview_base_color
    cv2.line(img, (x0, rail_y), (x1, rail_y), base_c, CFG.overview_base_thick, cv2.LINE_AA)
    head_w, head_h = 16, 12
    pts = np.array([[x1, rail_y], [x1 + head_w, rail_y], [x1, rail_y - head_h]], np.int32)
    cv2.fillConvexPoly(img, pts, base_c)

    if total_lidar <= 1:
        return img

    def ix_to_x(i: int) -> int:
        i = int(np.clip(i, 0, total_lidar - 1))
        return x0 + int((i / (total_lidar - 1)) * (x1 - x0))

    seg_c   = CFG.overview_seg_color
    post_th = CFG.overview_post_thick
    top_y_orange = rail_y - 22              # 주황 ㄷ자의 상단 y
    top_y_green  = top_y_orange - green_raise_px  # 초록 ㄷ자는 더 위로(=더 높게)
    min_px  = CFG.overview_min_pix

    def _draw_cap(xa, xb, y_top, color, thick):
        cv2.line(img, (xa, rail_y), (xa, y_top), color, thick, cv2.LINE_AA)
        cv2.line(img, (xb, rail_y), (xb, y_top), color, thick, cv2.LINE_AA)
        cv2.line(img, (xa, y_top), (xb, y_top), color, thick, cv2.LINE_AA)

    # ── 주황(내 JSON) 구간들 + 라벨 ──
    for a, b in (segs or []):
        if a > b: a, b = b, a
        xa, xb = ix_to_x(a), ix_to_x(b)
        if xb - xa < min_px:
            mid = (xa + xb) // 2
            xa, xb = mid - min_px // 2, mid + int(np.ceil(min_px / 2))
            xa = max(x0, xa); xb = min(x1, xb)

        # ㄷ자
        cv2.line(img, (xa, rail_y), (xa, top_y_orange), seg_c, post_th, cv2.LINE_AA)
        cv2.line(img, (xb, rail_y), (xb, top_y_orange), seg_c, post_th, cv2.LINE_AA)
        cv2.line(img, (xa, top_y_orange), (xb, top_y_orange), seg_c, post_th, cv2.LINE_AA)

        # 라벨
        if (xb - xa) >= (min_px + 12):
            _put_label(img, str(a), (xa - 6, top_y_orange - 6), fs_small, seg_c, th_small)
            _put_label(img, str(b), (xb - 6, top_y_orange - 6), fs_small, seg_c, th_small)
        else:
            _put_label(img, f"{a}-{b}", ((xa + xb) // 2 - 10, top_y_orange - 6), fs_small, seg_c, th_small)

    # ── 초록(GNSS 허용) 구간들 + 라벨 ──
    if extra_segs:
        c2 = extra_color
        post_th_green = post_th + int(green_post_th_add)
        for a, b in extra_segs:
            if a > b: a, b = b, a
            xa, xb = ix_to_x(a), ix_to_x(b)
            if xb - xa < min_px:
                mid = (xa + xb) // 2
                xa, xb = mid - min_px // 2, mid + int(np.ceil(min_px / 2))
                xa = max(x0, xa); xb = min(x1, xb)

            # 초록 ㄷ자 (더 높고, 더 두껍게)
            cv2.line(img, (xa, rail_y), (xa, top_y_green), c2, post_th_green, cv2.LINE_AA)
            cv2.line(img, (xb, rail_y), (xb, top_y_green), c2, post_th_green, cv2.LINE_AA)
            cv2.line(img, (xa, top_y_green), (xb, top_y_green), c2, post_th_green, cv2.LINE_AA)

            # 라벨 위치도 초록 상단 기준
            if (xb - xa) >= (min_px + 12):
                _put_label(img, str(a), (xa - 6, top_y_green - 6), fs_small, c2, th_small)
                _put_label(img, str(b), (xb - 6, top_y_green - 6), fs_small, c2, th_small)
            else:
                _put_label(img, f"{a}-{b}", ((xa + xb) // 2 - 10, top_y_green - 6), fs_small, c2, th_small)

    if merged_segs:
        bottom_y_red = rail_y + 22
        for a, b in merged_segs:
            if a > b: a, b = b, a
            xa, xb = ix_to_x(a), ix_to_x(b)
            if xb - xa < min_px:
                mid = (xa+xb)//2
                xa, xb = mid - min_px//2, mid + int(np.ceil(min_px/2))
                xa = max(x0, xa); xb = min(x1, xb)
            # 아래쪽으로 기둥/캡
            cv2.line(img, (xa, rail_y), (xa, bottom_y_red), merged_color, post_th+2, cv2.LINE_AA)
            cv2.line(img, (xb, rail_y), (xb, bottom_y_red), merged_color, post_th+2, cv2.LINE_AA)
            cv2.line(img, (xa, bottom_y_red), (xb, bottom_y_red), merged_color, post_th+2, cv2.LINE_AA)

    # 현재 인덱스 / 양끝
    cx = ix_to_x(current_lidar_idx)
    cv2.line(img, (cx, rail_y - 26), (cx, rail_y + 8), (0, 0, 255), 2, cv2.LINE_AA)
    _put_label(img, str(current_lidar_idx),
               (min(max(cx - 12, x0), x1 - 24), rail_y - 30), fs_small, (0, 0, 170), th_small)

    _put_label(img, "0", (x0 - 6, rail_y + 24), fs_small, (90, 90, 90), th_small)
    _put_label(img, f"{total_lidar - 1}", (x1 - 40, rail_y + 24), fs_small, (90, 90, 90), th_small)

    return img



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
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    img_w = int(cfg['image']['width'])
    img_h = int(cfg['image']['height'])
    alpha = float(cfg.get('undistort', {}).get('alpha', 0.0))
    proj_mode = cfg.get('projection', {}).get('mode', 'undistorted')

    cams = []
    for cam in cfg['cameras']:
        # === extrinsic: cam <- lidar ===
        Trc_cam = make_T_parent_child(cam['rotcam_extrinsic']['translation'],
                                      cam['rotcam_extrinsic']['rotation_ypr'])
        L = cam['lidar_extrinsic']
        Trc_lidar = make_T_parent_child(L['translation'], L['rotation_ypr'])

        T_cam_rotcam = np.linalg.inv(Trc_cam)
        T_cam_lidar  = T_cam_rotcam @ Trc_lidar
        R_cam_lidar  = T_cam_lidar[:3,:3].copy()
        t_cam_lidar  = T_cam_lidar[:3, 3].copy()

        cam_dict = {
            'name': cam['name'],
            'R_cam_lidar': R_cam_lidar,
            't_cam_lidar': t_cam_lidar,
            'img_w': img_w, 'img_h': img_h,
            'alpha': alpha,
            'proj_mode': proj_mode,
        }

        # === rectification path 선택 ===
        if 'ros_caminfo' in cam and cam['ros_caminfo']:
            rc = _rectify_maps_from_ros_caminfo(cam['ros_caminfo'])
            cam_dict.update({
                'rect_model': 'ros',
                'map1': rc['map1'], 'map2': rc['map2'],
                'P_rect': rc['P_rect'],          # (3,4)
                'K_rect': rc['K_rect'],          # (3,3)
                'out_size': rc['out_size'],      # (w,h)
            })
        else:
            # OpenCV 경로 (K_new)
            K    = np.array(cam['intrinsics']['K'], dtype=np.float64)
            dist = np.array(cam['distortion']['coeffs'], dtype=np.float64).reshape(1,-1)
            K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (img_w,img_h), alpha, (img_w,img_h))
            map1, map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=K, distCoeffs=dist, R=None,
                newCameraMatrix=K_new, size=(img_w,img_h), m1type=cv2.CV_16SC2
            )
            cam_dict.update({
                'rect_model': 'opencv',
                'K': K, 'D': dist,
                'K_new': K_new, 'roi': tuple(int(v) for v in roi),  # (x,y,w,h)
                'map1': map1, 'map2': map2,
                'out_size': (img_w, img_h),
            })

        cams.append(cam_dict)

    return cams  # ← 리스트로 반환(기존 calib_data 사용처와 호환)


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

def _merge_and_clip_segments(segs: list[tuple[int,int]], total_n: int) -> list[tuple[int,int]]:
    """구간들을 [0, total_n-1]로 클램프하고, 겹치거나 인접(바로 붙은) 구간을 병합."""
    if total_n <= 0:
        return []
    last = total_n - 1
    norm = []
    for a, b in segs:
        a = max(0, min(last, int(a)))
        b = max(0, min(last, int(b)))
        if a > b: a, b = b, a
        norm.append((a, b))
    norm.sort()

    merged = []
    for a, b in norm:
        if not merged or a > merged[-1][1] + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(a, b) for a, b in merged]

def _complement_segments(bad: list[tuple[int,int]], total_n: int) -> list[tuple[int,int]]:
    """[0..N-1]에서 bad의 여집합(= allow)을 반환."""
    if total_n <= 0:
        return []
    bad = _merge_and_clip_segments(bad, total_n)
    allow = []
    cur = 0
    last = total_n - 1
    for a, b in bad:
        if cur <= a - 1:
            allow.append((cur, a - 1))
        cur = b + 1
        if cur > last:
            break
    if cur <= last:
        allow.append((cur, last))
    return allow


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

    valid_mask = np.isfinite(points_2d).all(axis=1)
    points_2d = points_2d[valid_mask]
    points_3d = points_3d[valid_mask]
    
    # Filter points within image bounds
    mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_w) & 
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_h))
    
    return points_2d[mask], points_3d[mask]

# (lidar_file, cam_idx, color_range_key, point_radius_key) -> (xy_int[N,2], colors[N,3], z[N])
_projection_cache: dict[tuple, tuple] = {}

def _color_lut_256(name: str = None):
    key = f"__lut__{name or 'default'}"
    if not hasattr(_color_lut_256, key):
        lut_img = np.arange(256, dtype=np.uint8)[:, None]

        cmap_name = (name or "turbo").lower()
        reversed_flag = cmap_name.endswith("_r")
        base = cmap_name[:-2] if reversed_flag else cmap_name

        cv_map = {
            "turbo": getattr(cv2, "COLORMAP_TURBO", None),
            "jet": cv2.COLORMAP_JET,
        }.get(base, None)

        if cv_map is not None:
            lut_bgr = cv2.applyColorMap(lut_img, cv_map)
            if reversed_flag:
                lut_bgr = lut_bgr[::-1]
            lut = lut_bgr.reshape(256, 3).astype(np.uint8)
        else:
            # Matplotlib fallback
            try:
                cmap = mpl.colormaps.get(cmap_name)  # "turbo_r" 등 지원 시 바로 OK
            except Exception:
                cmap = None

            if cmap is None:
                # base만 시도하고, 필요하면 수동 reverse
                base_cmap = getattr(plt.cm, base, plt.cm.jet)
                cmap = base_cmap.reversed() if reversed_flag and hasattr(base_cmap, "reversed") else base_cmap

            lut_rgb = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            lut = lut_rgb[:, ::-1]  # RGB->BGR

        setattr(_color_lut_256, key, lut)

    return getattr(_color_lut_256, key)


def _project_and_color(points_xyz_i: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                       img_w: int, img_h: int,
                       vmin: float, vmax: float):
    """points_xyz_i: float32 [N,4], returns (xy_int, colors_bgr, z)"""
    # 1) world->cam
    X = points_xyz_i[:, :3].astype(np.float32, copy=False)
    # R (3x3), t (3,) 모두 float32
    Rc = R.astype(np.float32, copy=False)
    tc = t.astype(np.float32, copy=False)

    Xc = (X @ Rc.T) + tc  # [N,3]
    Z = Xc[:, 2]
    front = Z > 0.0
    if not np.any(front):
        return np.empty((0,2), np.int32), np.empty((0,3), np.uint8), Z

    Xc = Xc[front]; Z = Z[front]

    # 2) cam->pix (직접 수식이 K@ 나누기보다 더 빠름)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u = (Xc[:,0] * fx) / Z + cx
    v = (Xc[:,1] * fy) / Z + cy

    # 3) in-bounds
    iu = u.astype(np.int32); iv = v.astype(np.int32)
    inb = (iu >= 0) & (iu < img_w) & (iv >= 0) & (iv < img_h)
    if not np.any(inb):
        return np.empty((0,2), np.int32), np.empty((0,3), np.uint8), Z

    iu = iu[inb]; iv = iv[inb]
    xy = np.stack([iu, iv], axis=1)

    # 4) 색상 (고정 범위 정규화 → LUT 인덱스)
    rng = np.linalg.norm(X[front][inb], axis=1)
    if vmax <= vmin: vmax = vmin + 1e-3
    norm = (rng - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    idx = (norm * 255.0).astype(np.uint8)
    lut = _color_lut_256(CFG.lidar_cmap)
    colors = lut[idx]  # [N,3] BGR uint8
    return xy, colors, Z[inb]

def get_projection(lidar_path: Path, cam_idx: int, calib: dict, vmin: float, vmax: float, point_radius: int):
    key = (lidar_path.name, cam_idx, int(vmin*10), int(vmax*10), point_radius)
    if key in _projection_cache:
        return _projection_cache[key]

    # 준비
    points = get_lidar_points(lidar_path, CFG.lidar_max_display_range_m)
    if points.size == 0:
        _projection_cache[key] = (np.empty((0,2), np.int32), np.empty((0,3), np.uint8))
        return _projection_cache[key]

    K = calib['K'].astype(np.float32, copy=False)
    R = calib['R_cam_lidar'].astype(np.float32, copy=False)
    t = calib['t_cam_lidar'].astype(np.float32, copy=False)
    img_w, img_h = int(calib['img_w']), int(calib['img_h'])

    xy, colors, _ = _project_and_color(points, K, R, t, img_w, img_h, vmin, vmax)
    _projection_cache[key] = (xy, colors)
    return _projection_cache[key]


def project_one_cam(cam_idx: int, img_idx: int, lidar_idx: int, 
                   draw_lidar: bool = True, point_radius: int = 2) -> np.ndarray:
    global camera_files, lidar_files, calib_data

    # 1) 이미지 로드
    if cam_idx >= len(camera_files) or img_idx >= len(camera_files[cam_idx]):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img_path = camera_files[cam_idx][img_idx]
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # 2) (항상) 보정(remap)하여 표시용 이미지 만들기
    use_ros = False
    alpha = 0.0
    mode = 'undistorted'
    img_disp = img_bgr
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, np.float64)

    if calib_data and cam_idx < len(calib_data):
        c = calib_data[cam_idx]
        R = c['R_cam_lidar'].astype(np.float64)
        t = c['t_cam_lidar'].astype(np.float64)
        alpha = float(c.get('alpha', 0.0))
        use_ros = (c.get('rect_model','opencv') == 'ros')
        mode = c.get('proj_mode','undistorted')

        und = cv2.remap(img_bgr, c['map1'], c['map2'], interpolation=cv2.INTER_LINEAR)
        # OpenCV 경로에서 alpha==0이면 ROI 크롭
        if (not use_ros) and alpha == 0.0:
            x, y, w, h = c['roi']
            img_disp = und[y:y+h, x:x+w].copy()
        else:
            img_disp = und

    # 3) LiDAR가 꺼져 있으면 점만 생략하고 보정된 이미지만 반환
    if (not draw_lidar) or (lidar_idx >= len(lidar_files)):
        cv2.putText(img_disp, f"Cam{cam_idx+1} {_ts(img_path)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120,255,0), 2)
        return img_disp

    # 4) LiDAR 로드 및 필터링
    pts = load_lidar_points(lidar_files[lidar_idx])
    pts = filter_lidar_points(pts, max_range=120.0)
    if len(pts) == 0:
        cv2.putText(img_disp, f"Cam{cam_idx+1} {_ts(img_path)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120,255,0), 2)
        return img_disp

    # 5) LiDAR → 카메라 좌표
    pts_cam = (R @ pts[:, :3].T).T + t.reshape(1,3)
    Z = pts_cam[:,2]
    front = (Z > 0.1) & (Z < 200.0)
    if not np.any(front):
        cv2.putText(img_disp, f"Cam{cam_idx+1} {_ts(img_path)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120,255,0), 2)
        return img_disp
    pts_cam = pts_cam[front]; Z = Z[front]

    # 6) 사영
    if calib_data and cam_idx < len(calib_data):
        c = calib_data[cam_idx]
        if use_ros:
            # ROS: 보정 영상 좌표계에서 P로 직접 투영
            P = c['P_rect'].astype(np.float64)
            uvw = (P @ np.hstack([pts_cam, np.ones((pts_cam.shape[0],1))]).T).T
            uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-9, None)
        else:
            if mode == 'undistorted':
                Knew = c['K_new'].astype(np.float64)
                xs, ys = pts_cam[:,0]/Z, pts_cam[:,1]/Z
                uv1 = (Knew @ np.vstack([xs, ys, np.ones_like(xs)])).T
                uv  = uv1[:,:2] / uv1[:, 2:3]
                if alpha == 0.0:
                    x,y,_,_ = c['roi']
                    uv -= np.array([x, y], np.float64)  # ROI 오프셋 보정
            elif mode == 'original':
                K  = c['K'].astype(np.float64); D = c['D']
                uv, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), K, D)
                uv = uv.reshape(-1,2)
            else:
                raise ValueError(f"Unknown projection mode: {mode}")
    else:
        # fallback (캘리브 없음) : 그냥 중앙투영
        H, W = img_disp.shape[:2]
        K = np.array([[1000,0,W/2.0],[0,1000,H/2.0],[0,0,1]], np.float64)
        xs, ys = pts_cam[:,0]/Z, pts_cam[:,1]/Z
        uv1 = (K @ np.vstack([xs, ys, np.ones_like(xs)])).T
        uv  = uv1[:, :2] / uv1[:, 2:3]

    # 7) in-bounds & 그리기
    H, W = img_disp.shape[:2]
    m = (uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    if np.any(m):
        uv = uv[m].astype(np.int32)
        pts_vis = pts_cam[m]

        # ── 여기부터 색상 계산 교체 ───────────────────────────
        rng = np.linalg.norm(pts_vis[:, :3], axis=1)  # 거리

        if CFG.lidar_color_use_fixed_range:
            vmin, vmax = float(CFG.lidar_color_min_m), float(CFG.lidar_color_max_m)
        else:
            vmin, vmax = float(np.min(rng)), float(np.max(rng))
        if vmax <= vmin:
            vmax = vmin + 1e-3

        norm = (rng - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)
        idx = (norm * 255.0).astype(np.uint8)

        lut = _color_lut_256(CFG.lidar_cmap)   # ← turbo, turbo_r, jet, jet_r 등 지원
        col = lut[idx]                          # BGR uint8
        # ─────────────────────────────────────────────────────

        for (u, v), c3 in zip(uv, col):
            cv2.circle(img_disp, (int(u), int(v)), point_radius,
                       (int(c3[0]), int(c3[1]), int(c3[2])), -1)



    cv2.putText(img_disp, f"Cam{cam_idx+1} {_ts(img_path)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120,255,0), 2)
    return img_disp



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
    bottom_space = np.zeros((CFG.timeline_h + CFG.overview_h, canvas.shape[1], 3), dtype=np.uint8)
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

# === [ADD] segments helpers ================================================

def _intersect_segments(A: List[Tuple[int,int]], B: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """
    A, B는 [l0,l1] 포함구간 리스트. 교집합(모두 포함되는 구간들)을 돌려준다.
    반환은 정규화/병합하여 오름차순.
    """
    if not A or not B:
        return []
    a = _merge_segments(A, len(lidar_files))
    b = _merge_segments(B, len(lidar_files))
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        a0,a1 = a[i]; b0,b1 = b[j]
        s = max(a0,b0); e = min(a1,b1)
        if s <= e:
            out.append((s,e))
        if a1 < b1: i += 1
        else:       j += 1
    return _merge_segments(out, len(lidar_files))


# === [ADD] build merged (intersection) marks json ==========================

def write_merged_marks_json(
    cam_marks_path: Path,
    gps_bad_path: Optional[Path],   # GNSS JSON이 '불가' 구간(start/end)이라면 이걸 주고
    out_path: Optional[Path] = None # None이면 같은 폴더에 자동 네이밍
) -> Tuple[Path, List[Tuple[int,int]]]:
    """
    cam_marks_path의 (startN,endN) 쌍과, gps_bad_path에서 얻은 '불가' 구간의 여집합
    (= 허용구간)을 교집합하여 새로운 marks JSON을 생성한다.
    카메라 인덱스는 (ls - l0) 오프셋으로 재계산하여 넣는다.

    반환: (out_path, merged_lidar_segs)
    """
    # 1) 카메라 marks 로드
    with cam_marks_path.open("r", encoding="utf-8") as f:
        cam_data = json.load(f)
    pairs = _pair_segments_from_marks(cam_data)  # (sid,(l0,l1),(c0,c1),st,ed)

    # 2) GNSS 허용 구간 계산
    total = len(lidar_files)
    if gps_bad_path:
        bads = _read_gps_bad_ranges(gps_bad_path)           # [(a,b)] 불가
        allow = _complement_ranges(bads, 0, max(0,total-1)) # 허용
    else:
        allow = [(0, max(0,total-1))]  # GNSS 파일 없으면 전부 허용

    # 3) 교집합 만들면서 새 엔트리 빌드
    out_entries = []
    merged_segs = []
    seg_id = 1
    for _sid, (l0,l1), (c0,c1), st, ed in pairs:
        cams0 = list(map(int, c0))
        Lseg  = [(int(l0), int(l1))]
        inter = _intersect_segments(Lseg, allow)  # [(ls,le), ...]

        for (ls, le) in inter:
            # 카메라 시작/끝 오프셋 매핑
            length = le - ls  # inclusive 차이를 위해 아래 +1 보정 사용
            cam_start = []
            cam_end   = []
            for i in range(6):
                ci0 = int(cams0[i]) + (ls - int(l0))
                ci1 = ci0 + length
                cam_start.append(ci0)
                cam_end.append(ci1)

            # START 스냅
            start_snap = {
                "label": f"start{seg_id}",
                "timestamp": dt.datetime.now().isoformat(),
                "lidar_idx": int(ls),
                "cam_indices": cam_start,
                "files": {
                    "lidar": str(lidar_files[ls]) if 0 <= ls < len(lidar_files) else None,
                    "cams": [
                        str(camera_files[i][cam_start[i]]) if (0 <= cam_start[i] < len(camera_files[i])) else None
                        for i in range(6)
                    ]
                }
            }
            # END 스냅 (동일 길이로 맞춤)
            end_snap = {
                "label": f"end{seg_id}",
                "timestamp": dt.datetime.now().isoformat(),
                "lidar_idx": int(le),
                "cam_indices": cam_end,
                "files": {
                    "lidar": str(lidar_files[le]) if 0 <= le < len(lidar_files) else None,
                    "cams": [
                        str(camera_files[i][cam_end[i]]) if (0 <= cam_end[i] < len(camera_files[i])) else None
                        for i in range(6)
                    ]
                }
            }

            out_entries.append(start_snap)
            out_entries.append(end_snap)
            merged_segs.append((ls, le))
            seg_id += 1

    # 4) 저장 경로
    if out_path is None:
        marks_dir = cam_marks_path.parent
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = cam_marks_path.stem
        out_path = marks_dir / f"merged_{base}_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_entries, f, ensure_ascii=False, indent=2)

    return out_path, merged_segs


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

def _inject_reindex(header: List[str], rows: List[List[str]], colname: str = "index"):
    """
    header에 colname이 있으면 그 컬럼 값을 0..N-1로 덮어쓰기,
    없으면 맨 앞에 새 컬럼으로 추가.
    header가 비어 있으면(비정상 케이스) 원본을 그대로 반환.
    """
    if not header:
        return header, rows  # fallback

    header_out = header[:]
    if colname in header_out:
        j = header_out.index(colname)
        out_rows = []
        for i, r in enumerate(rows):
            rr = r[:]
            if j >= len(rr):
                rr += [""] * (j - len(rr) + 1)
            rr[j] = str(i)
            out_rows.append(rr)
        return header_out, out_rows
    else:
        header_out = [colname] + header_out
        out_rows = [[str(i)] + r for i, r in enumerate(rows)]
        return header_out, out_rows


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
        m = re.match(r"^(start|end)(\d+)$", label)
        if not m:
            continue
        kind, sid = m.group(1), int(m.group(2))
        if kind == "start": starts[sid] = it
        else: ends[sid] = it

    scene_ids = sorted(set(starts) & set(ends))
    pairs = []
    for sid in scene_ids:
        st = starts[sid]; ed = ends[sid]
        if "indices" in st and "indices" in ed:
            l0 = int(st["indices"]["lidar_idx"])
            l1 = int(ed["indices"]["lidar_idx"])
            c0 = list(map(int, st["indices"].get("cam_idx", [0]*6)))
            c1 = list(map(int, ed["indices"].get("cam_idx", [0]*6)))
        else:
            # ↓↓↓ GNSS JSON처럼 cam_indices가 없어도 안전하게 처리
            l0 = int(st["lidar_idx"])
            l1 = int(ed["lidar_idx"])
            def _safe_cam(obj, n=6):
                arr = obj.get("cam_indices")
                if isinstance(arr, list) and len(arr) == n:
                    return [int(x) for x in arr]
                return [0]*n
            c0 = _safe_cam(st)
            c1 = _safe_cam(ed)

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

    def _get_gps_csv_path(base_dir: Path) -> Optional[Path]:
        # 규약: base_dir/GPS/odom_data_synced.csv
        p = base_dir / "GPS" / "odom_data_synced.csv"
        return p if p.exists() else None

    def _load_gps_csv(gps_csv: Path) -> tuple[list[str], list[list[str]]]:
        """
        GPS CSV를 통째로 읽어 (header, rows) 반환.
        - 타임 파싱/정렬 없음 (LiDAR 인덱스와 1:1 정렬 가정)
        - dialect는 Sniffer, 실패 시 excel 폴백
        """
        import csv

        with gps_csv.open("r", newline="", encoding="utf-8") as f:
            sample = f.read(2048)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel

            reader = csv.reader(f, dialect)
            try:
                header = next(reader)
            except StopIteration:
                return [], []

            rows = [row for row in reader]
        return header, rows

    def _parse_sec_nsec_from_row(header: list[str], row: list[str]) -> tuple[Optional[int], Optional[int]]:
        """GPS 한 행에서 (sec, nsec) 추출 (여러 포맷 지원)"""
        import re, math
        h2i = {h: i for i, h in enumerate(header)}
        # 1) sec / nsec
        if "sec" in h2i and "nsec" in h2i:
            try:
                return int(row[h2i["sec"]]), int(row[h2i["nsec"]])
            except: pass
        # 2) header.stamp.secs / header.stamp.nsecs
        if "header.stamp.secs" in h2i and "header.stamp.nsecs" in h2i:
            try:
                return int(row[h2i["header.stamp.secs"]]), int(row[h2i["header.stamp.nsecs"]])
            except: pass
        # 3) timestamp (float sec)
        if "timestamp" in h2i:
            try:
                t = float(row[h2i["timestamp"]])
                sec = int(math.floor(t))
                nsec = int(round((t - sec) * 1e9))
                return sec, nsec
            except: pass
        # 4) 첫 컬럼이 "sec_nsec" 패턴
        if header and len(row) > 0:
            m = re.match(r"^\s*(\d+)\s*[_-]\s*(\d+)\s*$", str(row[0]))
            if m:
                try:
                    return int(m.group(1)), int(m.group(2))
                except: pass
        return None, None

    def _ts_key(sec: int, nsec: int) -> str:
        """LiDAR 키와 동일 포맷의 키 생성. (zero-pad 없이 'sec_nsec')"""
        return f"{int(sec)}_{int(nsec)}"

    def _build_gps_ts_index(header: list[str], rows: list[list[str]]) -> dict[str, int]:
        """GPS: ts_key('sec_nsec') -> row_idx 맵 생성 (중복 키는 최초만 채택)"""
        idx = {}
        for i, r in enumerate(rows):
            sec, nsec = _parse_sec_nsec_from_row(header, r)
            if sec is None or nsec is None: 
                continue
            key = _ts_key(sec, nsec)
            if key not in idx:
                idx[key] = i
        return idx

    def _inject_index_values(header: list[str], rows: list[list[str]], values: list[int], colname: str = "index"):
        """index 컬럼을 특정 값들로 주입(있으면 덮어쓰고, 없으면 맨 앞에 추가)"""
        if not header:
            return header, rows
        header_out = header[:]
        if colname in header_out:
            j = header_out.index(colname)
            out_rows = []
            for v, r in zip(values, rows):
                rr = r[:]
                if j >= len(rr):
                    rr += [""] * (j - len(rr) + 1)
                rr[j] = str(v)
                out_rows.append(rr)
            return header_out, out_rows
        else:
            header_out = [colname] + header_out
            out_rows = [[str(v)] + r for v, r in zip(values, rows)]
            return header_out, out_rows

    def _placeholder_row_like(header: list[str], sec: int, nsec: int) -> list[str]:
        """GPS 누락 시 헤더 형태를 보존하는 placeholder 생성 (sec/nsec/timestamp 채우기 시도)"""
        row = [""] * len(header)
        h2i = {h: i for i, h in enumerate(header)}
        if "sec" in h2i: row[h2i["sec"]] = str(int(sec))
        if "nsec" in h2i: row[h2i["nsec"]] = str(int(nsec))
        if "header.stamp.secs" in h2i: row[h2i["header.stamp.secs"]] = str(int(sec))
        if "header.stamp.nsecs" in h2i: row[h2i["header.stamp.nsecs"]] = str(int(nsec))
        if "timestamp" in h2i:
            row[h2i["timestamp"]] = f"{sec + nsec*1e-9:.9f}"
        # 첫 컬럼이 'sec_nsec' 관례라면 넣어줌
        if header and re.match(r"^\s*(\w+)$", header[0] or "") and "_" in header[0]:
            row[0] = _ts_key(sec, nsec)
        return row



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

    # GPS (규약 경로: base_dir/GPS/odom_data_synced.csv)
    gps_header: list[str] = []
    gps_rows: list[list[str]] = []
    gps_ts_index: dict[str, int] = {}
    gps_src = _get_gps_csv_path(base_dir)
    if gps_src:
        log(f"[info] GPS CSV     : {gps_src}")
        gps_header, gps_rows = _load_gps_csv(gps_src)
        gps_ts_index = _build_gps_ts_index(gps_header, gps_rows)
        log(f"[info] GPS rows    : {len(gps_rows)} (ts-indexed={len(gps_ts_index)})")
    else:
        log(f"[warn] GPS CSV not found at {base_dir / 'GPS' / 'odom_data_synced.csv'}")


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

        if seg_len < CFG.clip_min_frames:
            if log_cb:
                log_cb(f"[skip] scene {sid}: length={seg_len} < min_frames={CFG.clip_min_frames}")
            continue

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

                    import csv
                    
                    hdr_out, rows_out = _inject_reindex(imu_header, sel, colname="index")

                    with imu_out_csv.open("w", newline="", encoding="utf-8") as fcsv:
                        writer = csv.writer(fcsv)
                        if hdr_out:
                            writer.writerow(hdr_out)
                        writer.writerows(rows_out)

                    log(f"[ok][scene {sid}] IMU slice saved: {imu_out_csv} (rows={len(rows_out)}) [reindexed]")
                else:
                    log(f"[warn][scene {sid}] IMU slice skipped (LiDAR timestamps missing)")
            else:
                log(f"[info][scene {sid}] IMU not available, skipped")
        except Exception as e:
            log(f"[warn][scene {sid}] IMU slice failed: {e}")

        # GPS
        # ---- GPS 잘라 저장 (LiDAR TS로 정확 매칭, index = LiDAR k) ----
        try:
            if gps_rows and seg_len > 0:
                matched_rows = []
                idx_values   = []
                missing = 0

                for k in range(seg_len):
                    lidar_src_k = lidar_xyzi_files[l0 + k]
                    ts_str = ts_str_from_path(lidar_src_k) or _ts(lidar_src_k)  # "sec_nsec"
                    # sec, nsec 재구성(정규화 위해)
                    try:
                        sec_s, nsec_s = ts_str.split("_")
                        sec_i, nsec_i = int(sec_s), int(nsec_s)
                    except Exception:
                        # 비정상 파일명인 경우: 매칭 포기하고 placeholder
                        sec_i, nsec_i = 0, 0

                    key = _ts_key(sec_i, nsec_i)
                    ridx = gps_ts_index.get(key, None)
                    if ridx is not None:
                        matched_rows.append(gps_rows[ridx])
                    else:
                        # 타임스탬프가 없으면 placeholder로 alignment 유지
                        matched_rows.append(_placeholder_row_like(gps_header, sec_i, nsec_i))
                        missing += 1
                    idx_values.append(k)  # GPS index = LiDAR 로컬 k

                hdr_out, rows_out = _inject_index_values(gps_header, matched_rows, idx_values, colname="index")

                gps_out_dir = scene_dir / "GPS"
                ensure_dir(gps_out_dir)
                gps_out_csv = gps_out_dir / "odom_data_synced.csv"

                import csv
                with gps_out_csv.open("w", newline="", encoding="utf-8") as fcsv:
                    writer = csv.writer(fcsv)
                    if hdr_out:
                        writer.writerow(hdr_out)
                    writer.writerows(rows_out)

                if missing > 0:
                    log(f"[ok][scene {sid}] GPS slice saved: {gps_out_csv} (rows={len(rows_out)}, missing_ts={missing})")
                else:
                    log(f"[ok][scene {sid}] GPS slice saved: {gps_out_csv} (rows={len(rows_out)})")
            else:
                log(f"[info][scene {sid}] GPS not available, skipped")
        except Exception as e:
            log(f"[warn][scene {sid}] GPS slice failed: {e}")



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
                          current_lidar_idx, current_img_idx, control_mode, gps_allow_segs=None):
    H, W = canvas.shape[:2]
    timeline_h = CFG.timeline_h
    overview_h = CFG.overview_h

    # 1) 타임라인 생성/붙이기 (위쪽)
    timeline_img = create_timeline_matplotlib(
        current_lidar_idx, current_img_idx, control_mode,
        snap_start, snap_end, segment_id,
        W, timeline_h
    )
    canvas[H - (timeline_h + overview_h) : H - overview_h, :W] = timeline_img

    # 2) 개요 바 생성/붙이기 (아래쪽)
    segs = _extract_lidar_segments_from_marks(marks_json_path)
    merged = _intersect_segments(segs, gps_allow_segs or []) if segs else []

    overview_img = create_lidar_overview_bar(
        total_lidar=len(lidar_files),
        current_lidar_idx=current_lidar_idx,
        segs=segs,
        width=W,
        height=overview_h,
        extra_segs=(gps_allow_segs or []),
        merged_segs=merged,
    )
    canvas[H - overview_h : H, :W] = overview_img


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
        self.allowed_next: str = initial_allowed_next  # "start"/"end"
        self.segment_id: int = int(initial_segment_id)
        self.current_phase: str = None
        self.snap_start: Dict[str, Any] = copy.deepcopy(initial_snap_start) if initial_snap_start else None
        self.snap_end: Dict[str, Any] = copy.deepcopy(initial_snap_end) if initial_snap_end else None
        self.undo_stack: List[Dict[str, Any]] = []  # 최근 저장(s/e) 스냅샷 버퍼(되돌리기용 1단계 이상도 가능)
        self.cached_canvas: Optional[np.ndarray] = None

        # gnss
        self.gps_json_path: Optional[Path] = None
        self.gps_bad_segs: List[Tuple[int,int]] = []    # ← 추가
        self.gps_allow_segs: List[Tuple[int,int]] = []  # ← 추가

        # merge gps + camera
        self.merged_segs: List[Tuple[int,int]] = []   # 빨강 표시용 (옵션)

        # ---- Individual control mode ----
        self.control_mode: str = "all"  # "all", "lidar", "cam1", "cam2", ..., "cam6"
        self.individual_step: int = 1

        # 시작 인덱스 설정
        self.img_idx = [CFG.start_index_default] * self.num_cams
        self.lidar_idx = CFG.start_index_default

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
            self, "Select MERGED JSON", default_dir, "JSON Files (*.json);;All Files (*)"
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

    def _resume_from_dialog(self):
        default_dir = str(((dataset_base_dir / CFG.marks_subdir) if dataset_base_dir else Path("./marks_json")).resolve())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select marks JSON to resume",
            default_dir, "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        p = Path(path)
        if not p.exists():
            self._toast(f"File not found: {p}")
            return

        # JSON 로드
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    self._toast("Invalid marks format (not a list).")
                    return
        except Exception as e:
            self._toast(f"Failed to read JSON: {e}")
            return

        # base_dir 추정 및 데이터셋 스위치(필요 시)
        inferred = _infer_base_dir_from_marks(data)
        if inferred and (str(inferred) != str(dataset_base_dir)):
            self._toast(f"Switch dataset to: {inferred}")
            self._reload_dataset(inferred)

        # append 타겟 변경
        global marks_json_path
        marks_json_path = p

        # 상태 복구
        allowed, segid, sstart, send = _resume_state_from_marks(p)
        self.allowed_next = allowed
        self.segment_id = int(segid)
        self.snap_start = copy.deepcopy(sstart) if sstart else None
        self.snap_end = copy.deepcopy(send) if send else None

        # 보기 인덱스도 스냅샷 근처로 맞춤
        # - 다음에 저장해야 할 대상(allowed_next)에 따라 기준 스냅샷을 선택
        ref_snap = self.snap_start if (self.allowed_next == "end" and self.snap_start) else self.snap_end
        if ref_snap:
            # LiDAR
            lidx = int(ref_snap.get("lidar_idx", self.lidar_idx))
            self.lidar_idx = int(np.clip(lidx, 0, max(0, len(lidar_files)-1)))
            # 카메라들
            cam_indices = ref_snap.get("cam_indices", self.img_idx)
            if isinstance(cam_indices, list) and len(cam_indices) == 6:
                new_cam_idx = []
                for i in range(6):
                    max_i = max(0, len(camera_files[i]) - 1)
                    new_cam_idx.append(int(np.clip(int(cam_indices[i]), 0, max_i)))
                self.img_idx = new_cam_idx

        # UI 동기화
        self.sld.blockSignals(True)
        self.sld.setValue(self.lidar_idx)
        self.sld.setMaximum(max(0, len(lidar_files) - 1))
        self.sld.blockSignals(False)

        self._toast(f"Resumed: allowed_next={self.allowed_next}, segment_id={self.segment_id}")
        self._refresh_state_label()
        self._refresh()


    def _load_gnss_dialog(self):
        default_dir = str(((dataset_base_dir / CFG.marks_subdir) if dataset_base_dir else Path("./marks_json")).resolve())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select GNSS JSON", default_dir, "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self._load_gnss_json(Path(path))

    def _load_gnss_json(self, p: Path):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # GNSS JSON -> (l0,l1) 목록
            bad = []
            for sid, (l0, l1), (_c0, _c1), _st, _ed in _pair_segments_from_marks(data):
                a, b = int(l0), int(l1)
                if a > b: a, b = b, a
                bad.append((a, b))

            total_n = len(lidar_files)
            bad = _merge_and_clip_segments(bad, total_n)
            allow = _complement_segments(bad, total_n)

            self.gps_bad_ranges = bad
            self.gps_allow_ranges = allow
            self._toast(f"GNSS loaded: bad={bad} -> allow={allow}")
            self._refresh()
        except Exception as e:
            self._toast(f"GNSS load failed: {e}")

    def _load_gps_from_dialog(self):
        default_dir = str(((dataset_base_dir / CFG.marks_subdir) if dataset_base_dir else Path("./marks_json")).resolve())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select GNSS bad JSON", default_dir, "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        p = Path(path)
        try:
            bads = _read_gps_bad_ranges(p)  # [(a,b), ...] 불가구간
        except Exception as e:
            self._toast(f"GNSS JSON read failed: {e}")
            return

        self.gps_bad_segs = bads

        last = len(lidar_files) - 1
        if last >= 0:
            self.gps_allow_segs = _complement_ranges(self.gps_bad_segs, 0, last)
        else:
            self.gps_allow_segs = []

        print(f"[GNSS:comp] total_lidar={len(lidar_files)} last={last}")
        print(f"[GNSS:comp] bad_len={len(self.gps_bad_segs)} allow_len={len(self.gps_allow_segs)}")
        if self.gps_bad_segs:
            print(f"[GNSS:comp] bad_sample={self.gps_bad_segs[:8]}")
        if self.gps_allow_segs:
            print(f"[GNSS:comp] allow_sample={self.gps_allow_segs[:8]}")

        # 디버그 메시지로 실제 값 확인
        self._toast(f"GNSS loaded. bad={len(self.gps_bad_segs)} allow={len(self.gps_allow_segs)}")
        print("[GNSS] bad:", self.gps_bad_segs)
        print("[GNSS] allow:", self.gps_allow_segs)

        self._refresh()

    def on_make_merged_json(self):
        try:
            if not marks_json_path or (not Path(marks_json_path).exists()):
                self._toast("카메라 marks JSON이 로드되지 않았습니다.")
                return
            if not self.gps_allow_segs:
                self._toast("GNSS JSON(허용 구간)을 먼저 로드해주세요.")
                return

            # 1) 원본 카메라 세그먼트+카메라 시작 인덱스 로드
            with open(marks_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pairs = _pair_segments_from_marks(data)  # (sid, (l0,l1), (c0,c1), st, ed)

            N = len(lidar_files)
            if N <= 0:
                self._toast("LiDAR 파일이 없습니다.")
                return

            # 2) 교집합(빨간) 계산: (카메라 세그들의 합집합) ∩ (GNSS 허용)
            cam_segs = []
            # cam 세그먼트는 병합해서 전체 카메라 허용으로
            for sid, (l0, l1), (c0, c1), st, ed in pairs:
                a, b = int(l0), int(l1)
                if a > b: a, b = b, a
                cam_segs.append((a, b))
            cam_segs = _merge_segments(cam_segs, N)
            inter = _intersect_segments(cam_segs, self.gps_allow_segs)  # ★ 교집합

            if not inter:
                self._toast("교집합 구간이 없습니다.")
                self.merged_segs = []
                self._refresh()
                return

            # 3) 교집합을 '원본 세그먼트 쌍' 기준으로 쪼개서 start/end 스냅샷 생성
            out_list = []
            new_sid = 1

            # 원본 페어별로 교집합과 겹치는 부분만 분할
            for _sid, (l0, l1), (c0, c1), _st, _ed in pairs:
                a0, a1 = int(l0), int(l1)
                if a0 > a1: a0, a1 = a1, a0
                # 이 원본 세그와 inter의 교집합
                parts = _intersect_segments([(a0,a1)], inter)
                for (s, e) in parts:
                    if s > e: 
                        continue
                    # cam 인덱스는 시작 cam에서 offset 만큼 전진
                    off_s = s - a0
                    off_e = e - a0
                    cam_s = []
                    cam_e = []
                    for i in range(6):
                        max_i = max(0, len(camera_files[i]) - 1)
                        cs = int(np.clip(int(c0[i]) + off_s, 0, max_i))
                        ce = int(np.clip(int(c0[i]) + off_e, 0, max_i))
                        cam_s.append(cs)
                        cam_e.append(ce)

                    # 새 start/end 스냅샷
                    snapS = _build_snapshot(f"start{new_sid}", s, cam_s)
                    snapE = _build_snapshot(f"end{new_sid}",   e, cam_e)
                    out_list.append(snapS); out_list.append(snapE)
                    new_sid += 1

            # 4) 파일로 저장
            run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            worker = _sanitize_token(getattr(CFG, "worker_name", "anon"))
            base_name = dataset_base_dir.name if dataset_base_dir else "dataset"
            out_path = (marks_json_path.parent /
                        f"merged_{base_name}_{run_ts}_{worker}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_list, f, ensure_ascii=False, indent=2)

            # 5) 뷰에 빨간(교집합) ㄷ자 적용
            self.merged_segs = inter
            self._toast(f"Merged JSON saved: {out_path.name}  (segments={new_sid-1})")
            self._refresh()

        except Exception as e:
            self._toast(f"Merge failed: {e}")



    def _reload_dataset(self, new_base_dir: Path):
        """marks JSON으로부터 추정된 base_dir로 데이터셋을 갈아끼움."""
        global camera_files, lidar_files, dataset_base_dir

        dataset_base_dir = new_base_dir
        camera_files, lidar_files, _ = load_camera_and_lidar_files()

        # 슬라이더 범위/인덱스 초기화
        self.sld.blockSignals(True)
        self.sld.setMaximum(max(0, len(lidar_files) - 1))
        self.sld.blockSignals(False)

        # 기본 인덱스
        self.img_idx = [min(CFG.start_index_default, max(0, len(c)-1)) for c in camera_files]
        self.lidar_idx = min(CFG.start_index_default, max(0, len(lidar_files)-1))

    # ========== UI 구성 ==========
    def _build_right_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        # json 불러오기
        self.btn_resume = QtWidgets.QPushButton("Resume from camera JSON…")
        self.btn_resume.clicked.connect(self._resume_from_dialog)
        v.addWidget(self.btn_resume)

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

        # GPS
        v.addWidget(self._sep("Load GNSS(GPS) JSON file"))
        self.btn_load_gps = QtWidgets.QPushButton("Load GNSS(GPS) JSON…")
        self.btn_load_gps.clicked.connect(self._load_gps_from_dialog)
        v.addWidget(self.btn_load_gps)

        # Merge
        v.addWidget(self._sep("Merge GPS-Camera"))
        self.btn_make_merged = QtWidgets.QPushButton("Make merged JSON…")
        self.btn_make_merged.clicked.connect(self.on_make_merged_json)
        v.addWidget(self.btn_make_merged)

        v.addWidget(self._sep("Export Scenes"))


        # Export 버튼 + 로그창
        self.btn_export = QtWidgets.QPushButton("Export Scenes (copy)")
        self.btn_export.clicked.connect(self._start_export_dialog)
        v.addWidget(self.btn_export)

        self.txt_export = QtWidgets.QTextEdit()
        self.txt_export.setReadOnly(True)
        self.txt_export.setMinimumHeight(160)
        v.addWidget(self.txt_export)

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
                            self.lidar_idx, self.img_idx, self.control_mode, self.gps_allow_segs,)
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

# =========================
# 8) 전역 변수 및 초기화
# =========================

# 전역 변수들
camera_files: List[List[Path]] = []
lidar_files: List[Path] = []
calib_data = None
marks_json_path: Path = None
dataset_base_dir: Optional[Path] = None

initial_allowed_next: str = "start"
initial_segment_id: int = 1
initial_snap_start: Optional[Dict[str, Any]] = None
initial_snap_end: Optional[Dict[str, Any]] = None

def _resume_state_from_marks(marks_path: Path) -> Tuple[str, int, Optional[dict], Optional[dict]]:
    """
    기존 marks JSON을 읽어 다음 저장 상태(allowed_next), 다음 세그먼트 번호(segment_id),
    가장 최근 START/END 스냅샷을 유추한다.
    """
    if (not marks_path) or (not marks_path.exists()):
        return "start", 1, None, None

    try:
        with marks_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "start", 1, None, None

    if not isinstance(data, list) or not data:
        return "start", 1, None, None

    starts, ends = {}, {}
    for it in data:
        lab = str(it.get("label", ""))
        m = re.match(r"^(start|end)(\d+)$", lab)
        if not m:
            continue
        kind, sid = m.group(1), int(m.group(2))
        if kind == "start":
            starts[sid] = it
        else:
            ends[sid] = it

    if not starts and not ends:
        return "start", 1, None, None

    last_sid = max(set(starts.keys()) | set(ends.keys()))
    # case 1) 마지막 sid가 start만 있고 end가 없으면 → 다음은 end 저장
    if (last_sid in starts) and (last_sid not in ends):
        return "end", last_sid, starts.get(last_sid), ends.get(last_sid - 1)
    # case 2) 마지막 sid에 end까지 있으면 → 다음은 start 저장
    return "start", last_sid + 1, starts.get(last_sid), ends.get(last_sid)



def _find_latest_marks(marks_dir: Path, base_name: str) -> Optional[Path]:
    """marks_dir에서 base_name 접두의 *_syn_marks_*.json 중 가장 최신을 고른다."""
    if not marks_dir.exists():
        return None
    candidates = sorted(marks_dir.glob(f"{base_name}_syn_marks_*.json"))
    return candidates[-1] if candidates else None


def initialize_data():
    global camera_files, lidar_files, calib_data, marks_json_path, dataset_base_dir
    global initial_allowed_next, initial_segment_id, initial_snap_start, initial_snap_end

    print("Loading scene metadata...")
    scene_meta = load_scene_meta()

    print("Loading camera and LiDAR files...")
    camera_files, lidar_files, dataset_base_dir = load_camera_and_lidar_files()
    print(f"Base dir: {dataset_base_dir}")
    print(f"Found {len(lidar_files)} LiDAR files")
    for i, cam_files_i in enumerate(camera_files):
        print(f"Camera {i+1}: {len(cam_files_i)} files")

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

    # NEW: camera_<base>_<timestamp>_<worker>.json
    prefix = getattr(CFG, "filename_prefix", "camera")
    worker = _sanitize_token(getattr(CFG, "worker_name", "anon"))
    marks_json_path = marks_dir / f"{prefix}_{base_name}_{run_ts}_{worker}.json"

    print(f"Marks will be saved to: {marks_json_path}")

    # 새 세션 초기 상태
    initial_allowed_next, initial_segment_id = "start", 1
    initial_snap_start, initial_snap_end = None, None



# =========================
# 9) 메인 함수
# =========================

def main():
    # Initialize data (항상 새 파일 생성)
    initialize_data()

    if len(lidar_files) == 0:
        print("Error: No LiDAR files found!")
        return
    if all(len(cam_files) == 0 for cam_files in camera_files):
        print("Error: No camera files found!")
        return

    app = QtWidgets.QApplication(sys.argv)
    win = Viewer()
    win.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    main()