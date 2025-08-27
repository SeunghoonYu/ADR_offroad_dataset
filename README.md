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
└── /test0804_11_11/                    # original data
    └── ...
    └── ...    
└── <Location>_0804_11_11_scenes/       # original data to matching scenes
    └── <Location>_0804_11_11_scenes_1/
        ├── camera_info/                    # Calibration matrix info
        ├── decoded_rgb/                    # RGB images
        │   ├── camera_1/
        │   ├── camera_2/
        │   ├── camera_3/
        │   ├── camera_4/
        │   ├── camera_5/
        │   └── camera_6/
        ├── imu/
        │   └── imu.csv                     # clip by lidar timestamps
        ├── lidar/                          # raw lidar (for rosbag) 
        ├── lidar_xyzi/                     # xyzi only
        ├── marks_json/                     # sync matching index 
        ├── radar1/                         # clip by lidar timestamps
        ├── radar2/
        ├── radar3/
        ├── tf_static/                      # base_link tf
        │   └── tf_static.json
        └── scene_meta.json
    └── <Location>_0804_11_11_scenes_2/
    └── <Location>_0804_11_11_scenes_3/
    └── <Location>_0804_11_11_scenes_4/