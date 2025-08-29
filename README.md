<h1 align="center">
  <img src="assets/off-road_logo_sample.png" alt="ADR Offroad Logo" width="120"/>
  ADR Offroad Dataset
</h1>

## ADR offroad dataset post-processing tool
<img src="assets/offroad_vis_test1.png" width="1000">
<img src="assets/offroad_vis_test2.png" width="1000">

## Directory example
```text
<data_root>/
├── test0804_11_11/                   # original capture (raw)
│   └── ...                           # (omitted)
└── <Location>_0804_11_11_scenes/     # exported scenes from the original capture
    ├── <Location>_0804_11_11_scenes_1/
    │   ├── camera_info/              # calibration matrices
    │   ├── decoded_rgb/              # RGB images
    │   │   ├── camera_1/
    │   │   ├── camera_2/
    │   │   ├── camera_3/
    │   │   ├── camera_4/
    │   │   ├── camera_5/
    │   │   └── camera_6/
    │   ├── GPS/
    │   │   └── odom_data_synced.csv               # sliced to LiDAR [start,end], index re-numbered from 0
    │   ├── imu/
    │   │   └── imu.csv               # sliced to LiDAR [start,end], index re-numbered from 0
    │   ├── lidar/                    # raw LiDAR (1:1 matched by sec_nsec)
    │   ├── lidar_xyzi/               # point clouds (x y z intensity)
    │   ├── marks_json/               # marks used for sync/export
    │   ├── radar1/                   # sliced to LiDAR [start,end]
    │   ├── radar2/
    │   ├── radar3/
    │   ├── tf_static/
    │   │   └── tf_static.json        # base_link TFs
    │   └── scene_meta.json
    ├── <Location>_0804_11_11_scenes_2/
    ├── <Location>_0804_11_11_scenes_3/
    └── <Location>_0804_11_11_scenes_4/