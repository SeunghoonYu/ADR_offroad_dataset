import argparse
import os
import re
import csv
import glob
import json
import shutil
from typing import Tuple, Optional

import rclpy
from rclpy.serialization import serialize_message

from sensor_msgs.msg import Image, Imu, NavSatFix, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

from cv_bridge import CvBridge
import cv2

from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata


def ns(sec: int, nsec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nsec)


def parse_sec_nsec_from_name(fname: str) -> Optional[Tuple[int, int]]:
    """
    파일명에서 sec, nsec 추출
    지원 패턴:
      - <sec>_<nsec>.jpg / .bin
      - <sec>_<nsec>_<frame>.jpg / .bin
    nsec 자릿수는 8~9자리까지 허용
    """
    base = os.path.basename(fname)

    # sec_nsec.ext
    m = re.match(r'(\d{9,11})_(\d{8,9})\.(jpg|bin)$', base)
    if m:
        return int(m.group(1)), int(m.group(2))

    # sec_nsec_frame.ext
    m = re.match(r'(\d{9,11})_(\d{8,9})_\d+\.(jpg|bin)$', base)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None


def get_subdirs(run_dir: str):
    """run_dir 하위의 *_0, *_1, *_2 ... 서브폴더 리스트"""
    subdirs = sorted([d for d in glob.glob(os.path.join(run_dir, "*"))
                      if os.path.isdir(d) and re.search(r"_\d+$", d)])
    if not subdirs:
        return [run_dir]  # fallback
    return subdirs


class BagWriter:
    def __init__(self, out_bag_uri: str):
        self.writer = SequentialWriter()
        storage_options = StorageOptions(uri=out_bag_uri, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr',
                                             output_serialization_format='cdr')
        self.writer.open(storage_options, converter_options)
        self.created = set()

    def ensure_topic(self, name: str, typ: str):
        if name in self.created:
            return
        md = TopicMetadata(name=name, type=typ, serialization_format='cdr')
        self.writer.create_topic(md)
        self.created.add(name)

    def write(self, topic: str, typ: str, msg=None, t_ns: Optional[int] = None, serialized_bytes: bytes = None):
        self.ensure_topic(topic, typ)
        if serialized_bytes is None:
            serialized_bytes = serialize_message(msg)
            if t_ns is None and hasattr(msg, 'header'):
                t_ns = ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
        if t_ns is None:
            t_ns = 0
        self.writer.write(topic, serialized_bytes, t_ns)


def write_cameras(run_dir: str, bag: BagWriter, cams=range(1, 7)):
    for subdir in get_subdirs(run_dir):
        for cam_id in cams:
            cam_dir = os.path.join(subdir, "decoded_rgb", f"camera_{cam_id}")
            if not os.path.isdir(cam_dir):
                continue
            topic = f'/my_camera_{cam_id}/pylon_ros2_camera_node_{cam_id}/image_raw'
            files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
            print(f"[INFO] Camera {cam_id}: found {len(files)} images in {cam_dir}")
            for jf in files:
                ts = parse_sec_nsec_from_name(jf)
                if ts is None:
                    print(f"[WARN] Could not parse timestamp from {jf}")
                    continue
                sec, nsec = ts
                img = cv2.imread(jf, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[WARN] Could not read image {jf}")
                    continue
                if len(img.shape) == 2:
                    encoding = 'mono8' if img.dtype == 'uint8' else 'mono16'
                elif img.dtype == 'uint8':
                    encoding = 'bgr8'
                else:
                    encoding = 'passthrough'
                msg: Image = CvBridge().cv2_to_imgmsg(img, encoding=encoding)
                msg.header.stamp.sec = int(sec)
                msg.header.stamp.nanosec = int(nsec)
                msg.header.frame_id = f'cam{cam_id}'
                bag.write(topic, 'sensor_msgs/msg/Image', msg)


def write_lidar(run_dir: str, bag: BagWriter):
    for subdir in get_subdirs(run_dir):
        lidar_dir = os.path.join(subdir, 'lidar_xyzi')
        if not os.path.isdir(lidar_dir):
            continue
        topic = '/lidar_points'
        files = sorted(glob.glob(os.path.join(lidar_dir, '*.bin')))
        print(f"[INFO] LiDAR: found {len(files)} scans in {lidar_dir}")
        for bf in files:
            ts = parse_sec_nsec_from_name(bf)
            if ts is None:
                print(f"[WARN] Could not parse timestamp from {bf}")
                continue
            sec, nsec = ts
            with open(bf, 'rb') as f:
                raw = f.read()
            msg = PointCloud2()
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nsec
            msg.header.frame_id = "lidar_link"
            msg.data = raw
            msg.is_bigendian = False
            msg.is_dense = False
            msg.point_step = 0
            msg.row_step = 0
            msg.height = 1
            msg.width = 0
            msg.fields = []
            bag.write(topic, 'sensor_msgs/msg/PointCloud2', msg)


def write_radars(run_dir: str, bag: BagWriter, rads=range(1, 4)):
    for subdir in get_subdirs(run_dir):
        for rad_id in rads:
            rad_dir = os.path.join(subdir, f'radar{rad_id}')
            if not os.path.isdir(rad_dir):
                continue
            topic = f'/PointCloudDetectionradar{rad_id}'
            files = sorted(glob.glob(os.path.join(rad_dir, '*.bin')))
            print(f"[INFO] Radar {rad_id}: found {len(files)} scans in {rad_dir}")
            for bf in files:
                ts = parse_sec_nsec_from_name(bf)
                if ts is None:
                    print(f"[WARN] Could not parse timestamp from {bf}")
                    continue
                sec, nsec = ts
                with open(bf, 'rb') as f:
                    raw = f.read()
                msg = PointCloud2()
                msg.header.stamp.sec = sec
                msg.header.stamp.nanosec = nsec
                msg.header.frame_id = f"radar{rad_id}_link"
                msg.data = raw
                msg.is_bigendian = False
                msg.is_dense = False
                msg.point_step = 0
                msg.row_step = 0
                msg.height = 1
                msg.width = 0
                msg.fields = []
                bag.write(topic, 'sensor_msgs/msg/PointCloud2', msg)


def write_imu(run_dir: str, bag: BagWriter):
    for subdir in get_subdirs(run_dir):
        csv_path = os.path.join(subdir, 'imu', 'imu.csv')
        if not os.path.isfile(csv_path):
            continue
        topic = '/imu/data'
        print(f"[INFO] IMU: reading {csv_path}")
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg = Imu()
                msg.header.stamp.sec = int(row['sec'])
                msg.header.stamp.nanosec = int(row['nsec'])
                msg.header.frame_id = 'imu_link'
                msg.orientation.x = float(row['orient_x'])
                msg.orientation.y = float(row['orient_y'])
                msg.orientation.z = float(row['orient_z'])
                msg.orientation.w = float(row['orient_w'])
                msg.linear_acceleration.x = float(row['lin_acc_x'])
                msg.linear_acceleration.y = float(row['lin_acc_y'])
                msg.linear_acceleration.z = float(row['lin_acc_z'])
                msg.angular_velocity.x = float(row['ang_vel_x'])
                msg.angular_velocity.y = float(row['ang_vel_y'])
                msg.angular_velocity.z = float(row['ang_vel_z'])
                bag.write(topic, 'sensor_msgs/msg/Imu', msg)


def write_gps_and_odom(run_dir: str, bag: BagWriter):
    for subdir in get_subdirs(run_dir):
        odom_csv = os.path.join(subdir, 'GPS', 'odom_data_synced.csv')
        if not os.path.isfile(odom_csv):
            continue
        topic_odom = '/pva/odom'
        print(f"[INFO] GPS/Odom: reading {odom_csv}")
        with open(odom_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg = Odometry()
                msg.header.stamp.sec = int(row['sec'])
                msg.header.stamp.nanosec = int(row['nsec'])
                msg.header.frame_id = 'map'
                msg.child_frame_id = 'base_link'
                msg.pose.pose.position.x = float(row['pos_x'])
                msg.pose.pose.position.y = float(row['pos_y'])
                msg.pose.pose.position.z = float(row['pos_z'])
                msg.pose.pose.orientation.x = float(row['ori_x'])
                msg.pose.pose.orientation.y = float(row['ori_y'])
                msg.pose.pose.orientation.z = float(row['ori_z'])
                msg.pose.pose.orientation.w = float(row['ori_w'])
                msg.twist.twist.linear.x = float(row['vel_x'])
                msg.twist.twist.linear.y = float(row['vel_y'])
                msg.twist.twist.linear.z = float(row['vel_z'])
                bag.write(topic_odom, 'nav_msgs/msg/Odometry', msg)


def write_tf_static(run_dir: str, bag: BagWriter):
    for subdir in get_subdirs(run_dir):
        tf_dir = os.path.join(subdir, 'tf_static')
        json_path = os.path.join(tf_dir, 'tf_static.json')
        if not os.path.isfile(json_path):
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        transforms = data.get('transforms', [])
        if not transforms:
            continue

        tf_msg = TFMessage()
        t_ns = None
        for d in transforms:
            ts = TransformStamped()
            ts.header.stamp.sec = int(d.get('sec', 0))
            ts.header.stamp.nanosec = int(d.get('nsec', 0))
            ts.header.frame_id = d.get('frame_id', '')
            ts.child_frame_id = d.get('child_frame_id', '')
            tr = d.get('translation', {})
            q = d.get('rotation', {})
            ts.transform.translation.x = float(tr.get('x', 0.0))
            ts.transform.translation.y = float(tr.get('y', 0.0))
            ts.transform.translation.z = float(tr.get('z', 0.0))
            ts.transform.rotation.x = float(q.get('x', 0.0))
            ts.transform.rotation.y = float(q.get('y', 0.0))
            ts.transform.rotation.z = float(q.get('z', 0.0))
            ts.transform.rotation.w = float(q.get('w', 1.0))
            tf_msg.transforms.append(ts)
            if t_ns is None:
                t_ns = ns(ts.header.stamp.sec, ts.header.stamp.nanosec)

        bag.write('/tf_static', 'tf2_msgs/msg/TFMessage', tf_msg, t_ns=t_ns or 0)


def main():
    parser = argparse.ArgumentParser(description="Convert saved raw sensor files into a ROS 2 bag (sqlite3).")
    parser.add_argument('--run-dir', required=True, help="Run directory (e.g., /home/rlmodel/sensor_setup/test0819_13_53)")
    parser.add_argument('--out-bag', required=False, help="Output bag URI (directory). Default: <run-dir>")
    parser.add_argument('--overwrite', action='store_true', help="If set, delete the existing bag directory and overwrite.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    # ROS2 표준: .bag 확장자 제거, 디렉토리 이름만 사용
    out_bag = args.out_bag if args.out_bag else run_dir.rstrip('/')
    out_bag = os.path.abspath(out_bag)

    if os.path.exists(out_bag):
        if args.overwrite:
            print(f"[WARN] Removing existing bag directory: {out_bag}")
            shutil.rmtree(out_bag)
        else:
            raise RuntimeError(f"Bag directory already exists: {out_bag}\nUse --overwrite to delete it automatically.")

    parent = os.path.dirname(out_bag)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rclpy.init()
    try:
        bag = BagWriter(out_bag)
        write_cameras(run_dir, bag, cams=range(1, 7))
        write_lidar(run_dir, bag)
        write_radars(run_dir, bag, rads=range(1, 4))
        write_imu(run_dir, bag)
        write_gps_and_odom(run_dir, bag)
        write_tf_static(run_dir, bag)
    finally:
        rclpy.shutdown()

    print(f"[OK] Wrote rosbag2 at: {out_bag}")
    print("Inspect with: ros2 bag info", out_bag)
    print("Play with:    ros2 bag play", out_bag)


if __name__ == '__main__':
    main()