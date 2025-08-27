<h1 align="center">
  <img src="assets/off-road_logo_sample.png" alt="ADR Offroad Logo" width="120"/>
  ADR Offroad Dataset
</h1>

## ADR offroad dataset post-processing tool
<img src="assets/offroad_vis_test1.png" width="1000">
<img src="assets/offroad_vis_test2.png" width="1000">

## Directory example
<data_root>/
└── Gwangmyeong_Hagon_0822_15_22_scenes/
    └── Gwangmyeong_Hagon_0822_15_22_scenes_<ID>/
        ├── camera_info/
        ├── decoded_rgb/
        │   ├── camera_1/
        │   ├── camera_2/
        │   ├── camera_3/
        │   ├── camera_4/
        │   ├── camera_5/
        │   └── camera_6/
        ├── imu/
        │   └── imu.csv                 # 선택 구간만, index 0부터 재부여
        ├── lidar/                      # raw lidar (sec_nsec 매칭)
        ├── lidar_xyzi/                 # xyzi 포맷 (sec_nsec_######.bin)
        ├── marks_json/                 # 사용한 마크 파일 백업
        ├── radar1/
        ├── radar2/
        ├── radar3/
        ├── tf_static/
        │   └── tf_static.json
        └── scene_meta.json
