#!/usr/bin/env python3
import os
import glob
import numpy as np
import cv2

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.serialization import deserialize_message

# ====== 공통 설정 ======
INPUT_BASE = '/media/ysh/T7/sensor_setup/test0725_20_30_forward_slow'

# ====== LiDAR 설정 ======
LIDAR_INPUT_DIR = os.path.join(INPUT_BASE, 'lidar')
LIDAR_OUTPUT_DIR = os.path.join(INPUT_BASE, 'lidar_xyzi')

# ====== Camera 설정 ======
CAM_OUTPUT_BASE = os.path.join(INPUT_BASE, 'decoded_rgb')
NUM_CAMERAS = 6
BAYER_CONVERSION = cv2.COLOR_BAYER_BG2BGR  # RGGB 패턴이라면 cv2.COLOR_BAYER_RG2BGR

# ====== LiDAR 처리 ======
def decode_lidar():
    os.makedirs(LIDAR_OUTPUT_DIR, exist_ok=True)
    bin_files = sorted(glob.glob(os.path.join(LIDAR_INPUT_DIR, '*.bin')))
    print(f"[LiDAR] Found {len(bin_files)} bin files.")

    for f in bin_files:
        with open(f, 'rb') as raw:
            msg = deserialize_message(raw.read(), PointCloud2)
            points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))

        if not points:
            print(f"[LiDAR] Skip empty pointcloud: {f}")
            continue

        arr = np.array([[p[0], p[1], p[2], p[3]] for p in points], dtype=np.float32)
        fname = os.path.basename(f).replace('.bin', 'xyzi.bin')
        save_path = os.path.join(LIDAR_OUTPUT_DIR, fname)
        arr.tofile(save_path)

        print(f"[LiDAR] Saved: {save_path}  shape={arr.shape}")

# ====== Camera 처리 ======
def ensure_cam_dirs():
    for i in range(1, NUM_CAMERAS + 1):
        os.makedirs(os.path.join(CAM_OUTPUT_BASE, f'camera_{i}'), exist_ok=True)

def decode_camera():
    ensure_cam_dirs()
    for cam_id in range(1, NUM_CAMERAS + 1):
        input_dir = os.path.join(INPUT_BASE, f'camera_{cam_id}')
        output_dir = os.path.join(CAM_OUTPUT_BASE, f'camera_{cam_id}')
        if not os.path.exists(input_dir):
            print(f"[Camera {cam_id}] Skip: No folder found.")
            continue

        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith('.jpg'):
                continue

            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname.replace('.jpg', '.jpg'))

            bayer_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if bayer_img is None:
                print(f"[Camera {cam_id}] Warning: Failed to read {input_path}")
                continue

            rgb_img = cv2.cvtColor(bayer_img, BAYER_CONVERSION)
            cv2.imwrite(output_path, rgb_img)
            print(f"[Camera {cam_id}] Saved RGB image: {output_path}")

# ====== 실행 진입점 ======
if __name__ == '__main__':
    decode_lidar()
    decode_camera()
