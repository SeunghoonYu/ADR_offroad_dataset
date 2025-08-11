#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review saved Cam6 + LiDAR snapshots from JSON

- JSON 파일(list of objects: startN/endN ...)을 읽어 각 스냅샷의 인덱스를 그대로 시각화
- 기본 조작:
  Space: LiDAR on/off
  [ / ] : 이전/다음 JSON 스냅샷으로 이동 (start1 -> end1 -> start2 -> ...)
  r     : 현재 스냅샷의 저장 인덱스로 복구
  , / . : 전체(라이다+6캠) N칸 이동 (b=1, n=10, m=50로 step 변경)
  1..6  : 단일 카메라 모드 선택 → a/d로 해당 카메라만 이동
  l     : LiDAR 모드 선택 → a/d로 라이다만 이동
  c     : 전체 카메라 모드 → a/d로 6캠 동시 이동
  a / d : 선택된 모드 대상 이동(글로벌 step 크기만큼)
  q/ESC : 종료
"""

import re, json, argparse
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import datetime as dt

# ===== 공통 유틸 (기존 파일에서 사용한 함수들 복사) =====
def rpy_matrix_ypr_zyx(yaw, pitch, roll):
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)
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

def _rectify_maps_from_ros_caminfo(caminfo):
    K = np.array(caminfo['K'], dtype=np.float64).reshape(3,3)
    D = np.array(caminfo['D'], dtype=np.float64).ravel()
    R = np.array(caminfo['R'], dtype=np.float64).reshape(3,3)
    P = np.array(caminfo['P'], dtype=np.float64).reshape(3,4)

    w_full = int(caminfo['width']); h_full = int(caminfo['height'])
    bx = int(caminfo.get('binning_x', 1)); by = int(caminfo.get('binning_y', 1))
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

    return {'map1': reduced_map1, 'map2': reduced_map2, 'P_rect': Pb, 'K_rect': Pb[:3,:3], 'out_size': out_size}

def load_calib_yaml(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    img_w = int(cfg['image']['width']); img_h = int(cfg['image']['height'])
    alpha = float(cfg.get('undistort', {}).get('alpha', 0.0))
    proj_mode = cfg.get('projection', {}).get('mode', 'undistorted')

    cams = []
    for cam in cfg['cameras']:
        Trc_cam = make_T_parent_child(cam['rotcam_extrinsic']['translation'], cam['rotcam_extrinsic']['rotation_ypr'])
        L = cam['lidar_extrinsic']
        Trc_lidar = make_T_parent_child(L['translation'], L['rotation_ypr'])

        T_cam_rotcam = np.linalg.inv(Trc_cam)
        T_cam_lidar  = T_cam_rotcam @ Trc_lidar
        R_cam_lidar  = T_cam_lidar[:3,:3].copy()
        t_cam_lidar  = T_cam_lidar[:3, 3].copy()

        cam_dict = {
            'name': cam['name'], 'frame_id': cam['frame_id'], 'rotated_frame_id': cam['rotated_frame_id'],
            'T_cam_lidar': T_cam_lidar, 'R_cam_lidar': R_cam_lidar, 't_cam_lidar': t_cam_lidar,
            'projection_mode': proj_mode,
        }

        if 'ros_caminfo' in cam and cam['ros_caminfo']:
            rc = _rectify_maps_from_ros_caminfo(cam['ros_caminfo'])
            cam_dict.update({'rect_model':'ros', 'map1':rc['map1'], 'map2':rc['map2'],
                             'P_rect':rc['P_rect'], 'K_rect':rc['K_rect'], 'out_size':rc['out_size']})
        else:
            K    = np.array(cam['intrinsics']['K'], dtype=np.float64)
            dist = np.array(cam['distortion']['coeffs'], dtype=np.float64).reshape(1,-1)
            K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (img_w,img_h), alpha, (img_w,img_h))
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (img_w,img_h), cv2.CV_16SC2)
            cam_dict.update({'rect_model':'opencv', 'K':K, 'dist':dist, 'K_new':K_new,
                             'roi': tuple(int(v) for v in roi), 'map1':map1, 'map2':map2, 'out_size':(img_w,img_h)})
        cams.append(cam_dict)

    return {'alpha': alpha, 'image_size': (img_w, img_h), 'cameras': cams}

def project_lidar_to_image(xyz_lidar, cam, image_shape=None, alpha=0.0, mode_override=None):
    R, t = cam['R_cam_lidar'], cam['t_cam_lidar']
    pts_c = (R @ xyz_lidar.T + t.reshape(3,1)).T
    Z = pts_c[:,2]
    m = (Z > 0.1) & (Z < 200.0)
    if not np.any(m):
        return np.zeros((0,2), np.int32), np.array([], int), np.array([]), np.zeros((0,3))
    pts_c = pts_c[m]; Z = Z[m]

    rect_model = cam.get('rect_model', 'opencv')
    mode = cam.get('projection_mode', 'undistorted') if mode_override is None else mode_override

    if rect_model == 'ros':
        P = cam['P_rect']
        pts_h = np.hstack([pts_c, np.ones((pts_c.shape[0],1))])
        uvw   = (P @ pts_h.T).T
        uv    = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    else:
        if mode == 'undistorted':
            K = cam['K_new']
            xs, ys = pts_c[:,0]/Z, pts_c[:,1]/Z
            uv1 = (K @ np.vstack([xs, ys, np.ones_like(xs)])).T
            uv  = uv1[:, :2] / uv1[:, 2:3]
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

def load_xyzi_bin(bin_path: Path):
    arr = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1,4)
    return arr[:, :3], arr[:, 3]

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

# ===== 메인 =====
def build_canvas(imgs):
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

def project_one_cam(cfg, camera_files, i, img_idx_i, lidar_idx, draw_lidar, alpha_yaml, color_metric, fixed_dist_range):
    cam = cfg['cameras'][i]
    files = camera_files[i]
    if not files: return np.zeros((400,640,3), np.uint8)
    img_bgr = cv2.imread(str(files[img_idx_i]), cv2.IMREAD_COLOR)
    if img_bgr is None: return np.zeros((400,640,3), np.uint8)

    # rectify
    und_bgr = cv2.remap(img_bgr, cam['map1'], cam['map2'], interpolation=cv2.INTER_LINEAR)
    img_disp = und_bgr
    if cam.get('rect_model','opencv') == 'opencv' and alpha_yaml == 0.0:
        x, y, w, h = cam['roi']
        if w > 0 and h > 0:
            img_disp = und_bgr[y:y+h, x:x+w].copy()

    if draw_lidar and 0 <= lidar_idx < len(lidar_files):
        xyz, inten = load_xyzi_bin(lidar_files[lidar_idx])
        uv_int, kept_idx, Z, pts_c_in = project_lidar_to_image(xyz, cam, image_shape=img_disp.shape, alpha=alpha_yaml)
        if fixed_dist_range is None:
            colors_bgr = distance_colors_bgr(pts_c_in, None, None, metric=color_metric)
        else:
            vmin, vmax = fixed_dist_range
            colors_bgr = distance_colors_bgr(pts_c_in, vmin, vmax, metric=color_metric)
        for (u, v), c in zip(uv_int, colors_bgr):
            cv2.circle(img_disp, (int(u), int(v)), 2, (int(c[0]), int(c[1]), int(c[2])), -1, lineType=cv2.LINE_AA)

    label = f"Cam{i+1} {img_idx_i} LiDAR {lidar_idx}" + ("" if draw_lidar else " [LIDAR OFF]")
    cv2.putText(img_disp, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120,255,0), 2)
    return cv2.resize(img_disp, (640,400))

def overlay_segment_box(canvas, segment_id, start_entry, end_entry,
                        cur_cam_idx, cur_lidar_idx, step, mode):
    """
    하단 중앙에 segment N의 start/end 저장값과 현재 커서(idx) 요약을 모두 표시
    """
    H, W = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.45, 1
    pad_x, pad_y, gap = 9, 6, 6

    # 텍스트 라인 구성
    header = f"segment={segment_id}  mode={mode}  step={step}"

    lines = [header]

    if start_entry is not None:
        sL = start_entry["lidar_idx"]; sC = start_entry["cam_idx"]
        lines.append(f"start{segment_id}:  L={sL}  C={sC}")

    if end_entry is not None:
        eL = end_entry["lidar_idx"]; eC = end_entry["cam_idx"]
        lines.append(f"end{segment_id}:    L={eL}  C={eC}")

    # 현재 커서(현재 화면 인덱스) 표시 + start와의 차이를 우선 표기(없으면 end 기준)
    base = start_entry if start_entry is not None else end_entry
    if base is not None:
        baseL = base["lidar_idx"]; baseC = base["cam_idx"]
        dL = cur_lidar_idx - baseL
        dC = [cur_cam_idx[i] - baseC[i] for i in range(6)]
        lines.append(f"current:     L={cur_lidar_idx}({dL:+})  "
                     f"C={[f'{v}({d:+})' for v,d in zip(cur_cam_idx, dC)]}")
    else:
        lines.append(f"current:     L={cur_lidar_idx}  C={cur_cam_idx}")

    # 크기 계산
    sizes = [cv2.getTextSize(t, font, fs, th)[0] for t in lines]
    text_w = max(s[0] for s in sizes)
    text_hs = [s[1] for s in sizes]
    total_h = sum(text_hs) + gap*(len(lines)-1)

    box_w = text_w + pad_x*2
    box_h = total_h + pad_y*2
    x1 = (W - box_w)//2
    y1 = H - box_h - 12
    x2 = x1 + box_w
    y2 = y1 + box_h

    # 반투명 박스
    ov = canvas.copy()
    cv2.rectangle(ov, (x1,y1), (x2,y2), (0,0,0), -1)
    cv2.addWeighted(ov, 0.45, canvas, 0.55, 0, dst=canvas)

    # 텍스트
    y = y1 + pad_y + text_hs[0]
    for i, t in enumerate(lines):
        tw,_ = sizes[i]
        tx = x1 + (box_w - tw)//2
        if i == 0:            color = (220,220,220)   # header
        elif t.startswith("start"): color = (0,255,255)
        elif t.startswith("end"):   color = (255,200,120)
        else:                  color = (200,200,200)  # current
        cv2.putText(canvas, t, (tx,y), font, fs, color, th, cv2.LINE_AA)
        y += text_hs[i] + gap


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        type=str,
        default="/media/ysh/T7/snu_mt_0807/test0807_15_11/marks_json/sync_marks_20250809_215246.json",
        help="Path to marks json file"
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default="/home/ysh/off-road/tools/calib_matrix/matrix0801.yaml",
        help="Path to calibration yaml"
    )
    parser.add_argument(
        "--color_metric",
        type=str,
        default="range",
        help="Color metric (range/intensity/etc.)"
    )
    parser.add_argument(
        "--dist_range",
        type=str,
        default="0,24",
        help="Distance range (min,max)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Undistort alpha parameter (0~1)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"[INFO] Using JSON: {args.json}")
    print(f"[INFO] Using YAML: {args.yaml}")
    print(f"[INFO] Color Metric: {args.color_metric}")
    print(f"[INFO] Distance Range: {args.dist_range}")

    marks = json.load(open(args.json, "r", encoding="utf-8"))
    
    # --- 세그먼트 맵 구성: { seg_id: {"start": {...}, "end": {...}} } ---
    import re
    segment_marks = {}
    for rec in marks:
        label = rec.get("label", "")
        m = re.search(r'(\d+)$', label)
        if not m: 
            continue
        seg_id = int(m.group(1))
        entry = {
            "lidar_idx": int(rec["indices"]["lidar_idx"]),
            "cam_idx":   [int(x) for x in rec["indices"]["cam_idx"]],
            "label":     label,
        }
        segment_marks.setdefault(seg_id, {})
        if label.startswith("start"):
            segment_marks[seg_id]["start"] = entry
        elif label.startswith("end"):
            segment_marks[seg_id]["end"] = entry

    if not segment_marks:
        raise SystemExit("세그먼트 정보가 없습니다 (startN/endN).")

    seg_ids = sorted(segment_marks.keys())
    current_segment = seg_ids[0]
    max_segment = seg_ids[-1]

    # 초기 커서는 startN으로 세팅(없으면 endN)
    start_entry = segment_marks[current_segment].get("start")
    end_entry   = segment_marks[current_segment].get("end")

    if start_entry is not None:
        saved_cam_idx   = start_entry["cam_idx"]
        saved_lidar_idx = start_entry["lidar_idx"]
        rec_label       = start_entry["label"]
    elif end_entry is not None:
        saved_cam_idx   = end_entry["cam_idx"]
        saved_lidar_idx = end_entry["lidar_idx"]
        rec_label       = end_entry["label"]
    else:
        raise SystemExit(f"segment {current_segment} 에 start/end 둘 다 없음")

    img_idx   = saved_cam_idx.copy()
    lidar_idx = int(saved_lidar_idx)

    if not isinstance(marks, list) or len(marks)==0:
        raise SystemExit("JSON에 스냅샷이 없습니다.")

    base_dir = Path(marks[0]["base_dir"])
    camera_dirs = [base_dir / "decoded_rgb" / f"camera_{i}" for i in range(1,7)]
    lidar_dir   = base_dir / "lidar_xyzi"
    camera_files = [sorted(d.glob("*.jpg")) for d in camera_dirs]
    lidar_files  = sorted(lidar_dir.glob("*.bin"))

    cfg = load_calib_yaml(Path(args.yaml))
    alpha_yaml = cfg['alpha'] if args.alpha is None else args.alpha
    color_metric = args.color_metric
    fixed_dist_range = None
    if args.dist_range:
        vmin, vmax = [float(x) for x in args.dist_range.split(",")]
        fixed_dist_range = (vmin, vmax)



    project_lidar = True
    selected_cam = None
    control_mode = None   # 'cam' | 'lidar' | 'all' | None
    global_step_size = 1  # b=1, n=10, m=50

    while True:
        imgs = [project_one_cam(cfg, camera_files, i, img_idx[i], lidar_idx, project_lidar, alpha_yaml, color_metric, fixed_dist_range) for i in range(6)]
        can  = build_canvas(imgs)
        mode_str = control_mode if control_mode else "none"
        start_entry = segment_marks.get(current_segment, {}).get("start")
        end_entry   = segment_marks.get(current_segment, {}).get("end")
        overlay_segment_box(can, current_segment, start_entry, end_entry, img_idx, lidar_idx, global_step_size, mode_str)
        cv2.imshow("Review Cam6+LiDAR (from JSON)", can)

        key = cv2.waitKey(0) & 0xFF

        def step_cam(i, delta):
            img_idx[i] = max(0, min(len(camera_files[i]) - 1, img_idx[i] + delta))
        def step_all(delta):
            for i in range(6): step_cam(i, delta)

        if key in (27, ord('q')):
            break

        elif key == 32:  # Space
            project_lidar = not project_lidar

        # 스냅샷 간 이동
        # 스냅샷(세그먼트) 간 이동: [ 이전 / ] 다음
        elif key == ord('['):
            # 이전 segment
            idx = seg_ids.index(current_segment)
            if idx > 0:
                current_segment = seg_ids[idx - 1]
                s = segment_marks[current_segment].get("start")
                e = segment_marks[current_segment].get("end")
                base = s if s is not None else e
                if base is not None:
                    saved_cam_idx   = base["cam_idx"]
                    saved_lidar_idx = base["lidar_idx"]
                    img_idx   = saved_cam_idx.copy()
                    lidar_idx = int(saved_lidar_idx)
                    rec_label = base["label"]

        elif key == ord(']'):
            # 다음 segment
            idx = seg_ids.index(current_segment)
            if idx < len(seg_ids) - 1:
                current_segment = seg_ids[idx + 1]
                s = segment_marks[current_segment].get("start")
                e = segment_marks[current_segment].get("end")
                base = s if s is not None else e
                if base is not None:
                    saved_cam_idx   = base["cam_idx"]
                    saved_lidar_idx = base["lidar_idx"]
                    img_idx   = saved_cam_idx.copy()
                    lidar_idx = int(saved_lidar_idx)
                    rec_label = base["label"]


        # 저장값으로 복구
        elif key == ord('r'):
            img_idx = saved_cam_idx.copy()
            lidar_idx = int(saved_lidar_idx)

        # step 설정
        elif key == ord('b'):
            global_step_size = 1
            print("[step] 1")
        elif key == ord('n'):
            global_step_size = 10
            print("[step] 10")
        elif key == ord('m'):
            global_step_size = 50
            print("[step] 50")

        # 전체 동기 이동
        elif key == 44:  # ','
            lidar_idx = max(0, lidar_idx - global_step_size)
            step_all(-global_step_size)
        elif key == 46:  # '.'
            lidar_idx = min(len(lidar_files) - 1, lidar_idx + global_step_size)
            step_all(+global_step_size)

        # 모드 선택
        elif 49 <= key <= 54:  # '1'~'6'
            selected_cam = key - 49
            control_mode = 'cam'
        elif key == ord('l'):
            control_mode = 'lidar'; selected_cam = None
        elif key == ord('c'):
            control_mode = 'all'; selected_cam = None

        # a/d 이동
        elif key == ord('a'):
            if control_mode == 'cam' and selected_cam is not None:
                step_cam(selected_cam, -global_step_size)
            elif control_mode == 'lidar':
                lidar_idx = max(0, lidar_idx - global_step_size)
            elif control_mode == 'all':
                step_all(-global_step_size)

        elif key == ord('d'):
            if control_mode == 'cam' and selected_cam is not None:
                step_cam(selected_cam, +global_step_size)
            elif control_mode == 'lidar':
                lidar_idx = min(len(lidar_files) - 1, lidar_idx + global_step_size)
            elif control_mode == 'all':
                step_all(+global_step_size)
