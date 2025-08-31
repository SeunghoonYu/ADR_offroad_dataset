#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clip Viewer
- 기존 viewer.py의 시각화/보정/투영/타임라인/Export 유틸을 그대로 재사용
- merged JSON만 로드하여 오버뷰에 '교집합' 영역만 보여줌
- 교집합 내부에서 2차 clip start/end를 정의해 'final_clip' JSON으로 저장
- Export는 '내보낼 CLIP JSON'을 별도로 Load 해서 복사 수행
"""

import os
import sys
import copy
import json
import datetime as dt
from pathlib import Path
from typing import Optional, List, Tuple
import cv2

# ---- Qt ----
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence

# ---- 기존 viewer 모듈 재사용 ----
import viewer as V  # 같은 디렉토리의 viewer.py

# Wayland 경고 회피(필요 시)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


# ---------------------------------------
# 공용 경로 헬퍼 (marks 루트 및 서브폴더)
# ---------------------------------------
def compute_marks_dirs(base_dir: Path):
    """
    viewer_merge.py와 동일 규칙:
    <base_dir.name>_marks_json/ 아래에
      - merge_camera_gnss_json/
      - final_clip/
    을 사용 (camera/gnss는 clip 뷰어엔 필요 없음)
    """
    marks_root = base_dir.parent / f"{base_dir.name}_marks_json"
    merge_dir = marks_root / "merge_camera_gnss_json"
    final_clip_dir = marks_root / "final_clip"
    merge_dir.mkdir(parents=True, exist_ok=True)
    final_clip_dir.mkdir(parents=True, exist_ok=True)
    return marks_root, merge_dir, final_clip_dir


# ---------------------------------------
# Clip 전용 Viewer
# ---------------------------------------
class ClipViewer(V.QtWidgets.QMainWindow):
    """
    기존 viewer.Viewer의 구조/스타일을 최대한 유지하되,
    오른쪽 패널을 'clip' 전용 컨트롤로 구성
    """
    def __init__(self):
        super().__init__()

        # ---- 데이터 로딩 (viewer와 동일) ----
        self.camera_files, self.lidar_files, self.dataset_base_dir = V.load_camera_and_lidar_files()
        self.calib_data = None
        if V.CFG.calib_yaml and Path(V.CFG.calib_yaml).exists():
            self.calib_data = V.load_calib_yaml(Path(V.CFG.calib_yaml))

        V.camera_files = self.camera_files
        V.lidar_files = self.lidar_files
        V.calib_data = self.calib_data
        V.dataset_base_dir = self.dataset_base_dir

        # marks 디렉토리들
        self.marks_root_dir, self.merge_json_dir, self.final_clip_dir = compute_marks_dirs(self.dataset_base_dir)

        # ---- 상태 변수 ----
        self.lidar_idx = min(V.CFG.start_index_default, max(0, len(self.lidar_files) - 1))
        self.img_idx = [0] * 6
        self.draw_lidar = True
        self.point_radius = 2

        # merged 세그먼트 (오버뷰에 '교집합'만 표시)
        self.merged_segs: List[Tuple[int, int]] = []
        self.merged_json_path: Optional[Path] = None
        self._merged_pairs = []

        # clip 작업 상태(현재 편집용)
        self.allowed_next = "start"
        self.segment_id = 1
        self.snap_start = None
        self.snap_end = None
        self._clip_pairs = []

        # 현재(편집용) clip json 파일
        self.clip_json_path: Optional[Path] = None

        # Export용으로 따로 선택하는 clip json 파일
        self.export_clip_path: Optional[Path] = None
        self._export_preview: List[Tuple[int, int, int]] = []  # (sid,a,b)

        self.view_scale = getattr(V.CFG, "display_scale", 0.5)  # 표시 배율(기본 50%)

        self.num_cams = len(self.camera_files) if self.camera_files else 6
        self.step_size = 1  # ← 좌/우 이동 step

        # ---- UI 구성 ----
        self._build_ui()
        self._refresh_all()

    # ---------------- UI ----------------

    def _build_ui(self):
        self.setWindowTitle("Clip Viewer (merged→clip→export)")
        self.resize(1920, 1200)

        # 중앙: 이미지 캔버스 QLabel 하나에 합성 이미지를 넣는 방식
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.lbl_canvas = QtWidgets.QLabel()
        self.lbl_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_canvas.setStyleSheet("background-color: black;")

        self.scroll.setWidget(self.lbl_canvas)
        self.setCentralWidget(self.scroll)

        # 창 기본 크기도 살짝 줄이기
        self.resize(1400, 900)

        # 오른쪽 패널(clip 전용)
        right = self._build_right_panel()
        dock = QtWidgets.QDockWidget("Controls")
        dock.setWidget(right)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # 상태바
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # === Keyboard shortcuts ===
        QShortcut(QKeySequence(Qt.Key.Key_Space),        self, activated=self._toggle_lidar)
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self, activated=lambda: self._nudge_lidar(-self.step_size))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._nudge_lidar(+self.step_size))
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft),  self, activated=self._go_prev_merged)
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self, activated=self._go_next_merged)

    def _build_right_panel(self) -> QtWidgets.QWidget:
        """
        Clip 워크플로우 전용 패널
          1) Load merged JSON → merged 세그먼트만 오버뷰로 표시
          2) 현재 인덱스에서 Clip Start/End 찍고 final_clip JSON으로 저장
          3) Export: '내보낼 CLIP JSON'을 별도로 Load 후 복사
        """
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        # --- merged JSON 로드 ---
        self.btn_load_merged = QtWidgets.QPushButton("Load MERGED JSON…")
        self.btn_load_merged.clicked.connect(self._load_merged_json_dialog)
        v.addWidget(self.btn_load_merged)

        # --- Clip 파일(최종) 준비/선택 (편집용) ---
        h_clip = QtWidgets.QHBoxLayout()
        self.btn_new_clip = QtWidgets.QPushButton("New CLIP JSON")
        self.btn_new_clip.clicked.connect(self._make_new_clip_json)
        h_clip.addWidget(self.btn_new_clip)

        self.btn_open_clip = QtWidgets.QPushButton("Open CLIP JSON…")
        self.btn_open_clip.clicked.connect(self._open_clip_json_dialog)
        h_clip.addWidget(self.btn_open_clip)
        v.addLayout(h_clip)

        # --- Clip 마킹 ---
        g = QtWidgets.QGroupBox("Clip Marking")
        gv = QtWidgets.QVBoxLayout(g)
        self.btn_clip_start = QtWidgets.QPushButton("Set CLIP START (current)")
        self.btn_clip_start.clicked.connect(self._set_clip_start)
        gv.addWidget(self.btn_clip_start)

        self.btn_clip_end = QtWidgets.QPushButton("Set CLIP END (current)")
        self.btn_clip_end.clicked.connect(self._set_clip_end)
        gv.addWidget(self.btn_clip_end)

        self.lbl_seg = QtWidgets.QLabel("allowed_next = start | segment_id = 1")
        gv.addWidget(self.lbl_seg)
        v.addWidget(g)

        # --- Merged Navigate ---
        nav = QtWidgets.QGroupBox("Merged Navigate")
        nv = QtWidgets.QHBoxLayout(nav)
        self.btn_prev_clip = QtWidgets.QPushButton("[ Prev")
        self.btn_prev_clip.clicked.connect(self._go_prev_merged)
        nv.addWidget(self.btn_prev_clip)
        self.btn_next_clip = QtWidgets.QPushButton("Next ]")
        self.btn_next_clip.clicked.connect(self._go_next_merged)
        nv.addWidget(self.btn_next_clip)
        v.addWidget(nav)

        # --- 슬라이더(라이다 인덱스) ---
        self.sld = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld.setMinimum(0)
        self.sld.setMaximum(max(0, len(self.lidar_files) - 1))
        self.sld.setValue(self.lidar_idx)
        self.sld.valueChanged.connect(self._on_slider)
        v.addWidget(self.sld)

        # --- 원 표시 반지름 & draw toggle ---
        self.spn_radius = QtWidgets.QSpinBox()
        self.spn_radius.setRange(1, 5)
        self.spn_radius.setValue(self.point_radius)
        self.spn_radius.valueChanged.connect(self._on_radius)
        v.addWidget(QtWidgets.QLabel("LiDAR point radius"))
        v.addWidget(self.spn_radius)

        self.chk_draw = QtWidgets.QCheckBox("Draw LiDAR points")
        self.chk_draw.setChecked(self.draw_lidar)
        self.chk_draw.toggled.connect(self._on_draw_toggle)
        v.addWidget(self.chk_draw)

        # --- Export 섹션 (내보낼 CLIP JSON을 별도로 로드) ---
        v.addSpacing(8)
        exp = QtWidgets.QGroupBox("Export from Loaded CLIP JSON")
        ev = QtWidgets.QVBoxLayout(exp)

        self.btn_choose_export = QtWidgets.QPushButton("Load EXPORT CLIP JSON…")
        self.btn_choose_export.clicked.connect(self._choose_export_clip_json_dialog)
        ev.addWidget(self.btn_choose_export)

        self.lbl_export_file = QtWidgets.QLabel("No export clip selected")
        self.lbl_export_file.setWordWrap(True)
        self.lbl_export_file.setStyleSheet("color: #444;")
        ev.addWidget(self.lbl_export_file)

        self.btn_export = QtWidgets.QPushButton("Export Scenes…")
        self.btn_export.clicked.connect(self._export_from_clip_json)
        ev.addWidget(self.btn_export)

        v.addWidget(exp)

        v.addStretch(1)
        return w

    # ---------- Export용 클립 선택 ----------
    def _choose_export_clip_json_dialog(self):
        start_dir = self.final_clip_dir if self.final_clip_dir.exists() else self.marks_root_dir
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load EXPORT CLIP JSON", str(start_dir), "JSON files (*.json);;All files (*)"
        )
        if not fname:
            self._toast("Canceled.")
            return
        self.export_clip_path = Path(fname)
        self._build_export_preview_and_label()
        self._toast(f"Export clip loaded: {self.export_clip_path.name}")

    def _build_export_preview_and_label(self):
        """self.export_clip_path를 미리 읽어 (sid,a,b) 미리보기와 라벨 텍스트 구성"""
        self._export_preview = []
        if not self.export_clip_path or not self.export_clip_path.exists():
            self.lbl_export_file.setText("No export clip selected")
            return

        try:
            with self.export_clip_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            pairs = V._pair_segments_from_marks(data)  # (sid,(l0,l1),(c0,c1),st,ed)
            for sid, (l0, l1), _cc, _st, _ed in pairs:
                a, b = int(l0), int(l1)
                if a > b:
                    a, b = b, a
                self._export_preview.append((sid, a, b))
        except Exception as e:
            self.lbl_export_file.setText(f"{self.export_clip_path.name} (read failed)")
            self._toast(f"Export preview failed: {e}")
            return

        if not self._export_preview:
            self.lbl_export_file.setText(f"{self.export_clip_path.name} (no segments)")
        else:
            show = ", ".join([f"{sid}:{a}-{b}" for sid, a, b in self._export_preview[:5]])
            more = f" …(+{len(self._export_preview)-5})" if len(self._export_preview) > 5 else ""
            self.lbl_export_file.setText(f"{self.export_clip_path.name} | {len(self._export_preview)} clip(s) :: {show}{more}")

    # ---------------- 이동 ----------------

    def _go_prev_merged(self):
        self._go_merged(-1)

    def _go_next_merged(self):
        self._go_merged(+1)

    def _go_merged(self, direction: int):
        if not self._merged_pairs:
            self._toast("MERGED JSON을 먼저 로드하세요.")
            return

        C = self._merged_pairs
        cur = int(self.lidar_idx)

        # 현재 포함 세그먼트 찾기
        cur_idx = None
        for i, it in enumerate(C):
            if it["start"] <= cur <= it["end"]:
                cur_idx = i
                break

        if direction > 0:  # 다음
            if cur_idx is not None:
                target_i = min(len(C) - 1, cur_idx + 1)
            else:
                target_i = next((i for i,it in enumerate(C) if it["start"] > cur), len(C) - 1)
        else:  # 이전
            if cur_idx is not None:
                target_i = max(0, cur_idx - 1)
            else:
                prevs = [i for i,it in enumerate(C) if it["end"] < cur]
                target_i = prevs[-1] if prevs else 0

        self._jump_to_segment(C[target_i])

    def _go_prev_segment(self):
        self._go_segment(-1)

    def _go_next_segment(self):
        self._go_segment(+1)

    def _go_segment(self, direction: int):
        """로드된 merged 세그먼트(self.merged_segs)를 기준으로 이전/다음 세그먼트 시작으로 점프"""
        segs = self.merged_segs or []
        if not segs:
            self._toast("먼저 MERGED JSON을 로드하세요.")
            return

        # 정렬 보장
        segs = sorted((min(a,b), max(a,b)) for a,b in segs)
        cur = int(self.lidar_idx)

        # 현재 포함 세그먼트 인덱스 찾기
        cur_idx = None
        for i, (a, b) in enumerate(segs):
            if a <= cur <= b:
                cur_idx = i
                break

        if direction > 0:  # 다음
            if cur_idx is not None and cur_idx + 1 < len(segs):
                target = segs[cur_idx + 1][0]
            else:
                # 현재 위치 이후 첫 세그먼트
                later = [a for (a, b) in segs if a > cur]
                target = later[0] if later else segs[-1][0]
        else:  # 이전
            if cur_idx is not None and cur_idx - 1 >= 0:
                target = segs[cur_idx - 1][0]
            else:
                # 현재 위치 이전 마지막 세그먼트
                earlier = [a for (a, b) in segs if b < cur]
                target = earlier[-1] if earlier else segs[0][0]

        # 슬라이더 이동 → 공통 이동으로 모든 인덱스 동기화
        self.sld.setValue(int(target))  # valueChanged -> _on_slider -> _relative_step
        self._toast(f"Jumped to merged segment start @ {int(target)}")

    # -------------- 이벤트/핸들러 --------------

    def _on_slider(self, val: int):
        val = int(val)
        delta = val - int(self.lidar_idx)
        self.lidar_idx = val
        # 카메라 6개도 같은 delta로 이동(클램프)
        for i in range(6):
            if i < len(self.camera_files):
                max_i = max(0, len(self.camera_files[i]) - 1)
                self.img_idx[i] = max(0, min(max_i, self.img_idx[i] + delta))
        self._refresh_all()

    def _on_radius(self, val: int):
        self.point_radius = int(val)
        self._refresh_all()

    def _on_draw_toggle(self, on: bool):
        self.draw_lidar = bool(on)
        self._refresh_all()

    # -------------- 파일 로드/저장 --------------

    def _load_merged_json_dialog(self):
        start_dir = self.merge_json_dir if self.merge_json_dir.exists() else self.marks_root_dir
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open MERGED JSON", str(start_dir), "JSON files (*.json);;All files (*)"
        )
        if not fname:
            self._toast("Canceled.")
            return
        p = Path(fname)
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            pairs = V._pair_segments_from_marks(data)  # (sid,(l0,l1),(c0,c1),st,ed)
            segs = []
            self._merged_pairs = []
            for _sid, (l0, l1), _cc, _st, _ed in pairs:
                a, b = int(l0), int(l1)
                if a > b:
                    a, b = b, a
                segs.append((a, b))
                self._merged_pairs.append({
                    "sid": _sid, "start": a, "end": b, "st": _st, "ed": _ed
                })
            # 병합 및 클램프
            N = len(self.lidar_files)
            self.merged_segs = V._merge_segments(segs, N)
            self.merged_json_path = p
            self._toast(f"Merged loaded: {p.name} (segments={len(self.merged_segs)})")

            if self._merged_pairs:
                self._jump_to_segment(self._merged_pairs[0])
            self._refresh_all()
        except Exception as e:
            self._toast(f"Load merged failed: {e}")

    def _make_new_clip_json(self):
        """
        final_clip 디렉터리에 새 clip json 파일(빈 리스트)을 생성하고 핸들로 잡는다.
        """
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.dataset_base_dir.name if self.dataset_base_dir else "dataset"
        worker = V._sanitize_token(getattr(V.CFG, "worker_name", "anon"))
        out = self.final_clip_dir / f"clip_{base}_{ts}_{worker}.json"
        try:
            with out.open("w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            self.clip_json_path = out
            self.allowed_next = "start"
            self.segment_id = 1
            self.snap_start = None
            self.snap_end = None
            self._toast(f"New clip JSON: {out.name}")
            self._refresh_seg_label()
        except Exception as e:
            self._toast(f"Create clip json failed: {e}")

    def _open_clip_json_dialog(self):
        start_dir = self.final_clip_dir if self.final_clip_dir.exists() else self.marks_root_dir
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open CLIP JSON", str(start_dir), "JSON files (*.json);;All files (*)"
        )
        if not fname:
            self._toast("Canceled.")
            return
        self.clip_json_path = Path(fname)
        # segment_id를 이어서 쓰고 싶으면 현재 파일을 훑어 최대 sid+1 계산 (선택사항)
        try:
            with self.clip_json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # 기존 포맷 기반 최대 sid 추출
            max_sid = 0
            for it in data if isinstance(data, list) else []:
                label = str(it.get("label", ""))
                if label.startswith("start") or label.startswith("end"):
                    try:
                        sid = int(label.replace("start", "").replace("end", ""))
                        max_sid = max(max_sid, sid)
                    except:
                        pass
            self.segment_id = max_sid + 1 if max_sid > 0 else 1
        except:
            self.segment_id = 1
        self.allowed_next = "start"
        self._refresh_seg_label()
        self._refresh_clip_pairs()
        self._toast(f"Clip json opened: {self.clip_json_path.name}")

    # -------------- Clip 마킹 --------------

    def _check_in_merged(self, idx: int) -> bool:
        """현재 라이다 인덱스가 merged 세그먼트(교집합) 영역 안인지 검사"""
        for a, b in self.merged_segs or []:
            if a <= idx <= b:
                return True
        return False

    def _set_clip_start(self):
        if not self._ensure_clip_ready():
            return
        if not self._check_in_merged(self.lidar_idx):
            self._toast("현재 인덱스가 '교집합' 영역 밖입니다.")
            return

        cam_s = [int(self.img_idx[i] if i < len(self.img_idx) else 0) for i in range(6)]
        self.snap_start = V._build_snapshot(f"start{self.segment_id}", self.lidar_idx, cam_s)
        self.allowed_next = "end"
        self._append_to_clip(self.snap_start)
        self._refresh_seg_label()
        self._toast(f"start{self.segment_id} @ {self.lidar_idx}")

    def _set_clip_end(self):
        if not self._ensure_clip_ready():
            return
        if self.allowed_next != "end":
            self._toast("먼저 START를 찍으세요.")
            return
        if not self._check_in_merged(self.lidar_idx):
            self._toast("현재 인덱스가 '교집합' 영역 밖입니다.")
            return

        cam_e = [int(self.img_idx[i] if i < len(self.img_idx) else 0) for i in range(6)]
        self.snap_end = V._build_snapshot(f"end{self.segment_id}", self.lidar_idx, cam_e)
        self._append_to_clip(self.snap_end)
        self._toast(f"end{self.segment_id} @ {self.lidar_idx}")

        # 다음 세그 준비
        self.segment_id += 1
        self.allowed_next = "start"
        self.snap_start = None
        self.snap_end = None
        self._refresh_seg_label()

    def _append_to_clip(self, obj: dict):
        if not self.clip_json_path:
            self._toast("clip json 파일이 준비되지 않았습니다. [New/Open]을 먼저 사용하세요.")
            return
        try:
            data = []
            if self.clip_json_path.exists():
                with self.clip_json_path.open("r", encoding="utf-8") as f:
                    try:
                        d = json.load(f)
                        if isinstance(d, list):
                            data = d
                    except:
                        pass
            data.append(obj)
            with self.clip_json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._refresh_clip_pairs()
        except Exception as e:
            self._toast(f"Append failed: {e}")

    def _ensure_clip_ready(self) -> bool:
        if not self.merged_segs:
            self._toast("Merged JSON을 먼저 로드하세요.")
            return False
        if not self.clip_json_path:
            self._toast("CLIP JSON을 먼저 준비/오픈하세요.")
            return False
        return True

    def _toggle_lidar(self):
        """space: LiDAR 투영 on/off 토글 (+체크박스 동기화)"""
        self.chk_draw.setChecked(not self.chk_draw.isChecked())  # toggled -> _on_draw_toggle 호출됨

    def _nudge_lidar(self, step: int):
        self._relative_step(int(step))

    def _relative_step(self, delta: int, update_slider: bool = True):
        if delta == 0 or not self.lidar_files:
            return
        # lidar
        self.lidar_idx = max(0, min(len(self.lidar_files) - 1, self.lidar_idx + int(delta)))
        # cams
        for i in range(self.num_cams):
            max_i = len(self.camera_files[i]) - 1 if i < len(self.camera_files) and len(self.camera_files[i]) > 0 else 0
            self.img_idx[i] = max(0, min(max_i, self.img_idx[i] + int(delta)))
        # slider 동기화
        if update_slider:
            self.sld.blockSignals(True)
            self.sld.setValue(self.lidar_idx)
            self.sld.blockSignals(False)
        self._refresh_all()

    def _refresh_clip_pairs(self):
        """clip_json_path에서 (start,end,st,ed,sid) 리스트를 다시 만든다."""
        self._clip_pairs = []
        try:
            if not self.clip_json_path or not self.clip_json_path.exists():
                return
            with self.clip_json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            pairs = V._pair_segments_from_marks(data)  # (sid,(l0,l1),(c0,c1),st,ed)
            items = []
            for sid, (l0, l1), _cc, st, ed in pairs:
                a, b = int(l0), int(l1)
                if a > b:
                    a, b = b, a
                items.append({"sid": sid, "start": a, "end": b, "st": st, "ed": ed})
            # start 기준 오름차순
            self._clip_pairs = sorted(items, key=lambda x: x["start"])
        except Exception as e:
            self._toast(f"clip parse failed: {e}")

    def _go_prev_clip(self):
        self._go_clip(-1)

    def _go_next_clip(self):
        self._go_clip(+1)

    def _go_clip(self, direction: int):
        """'[' 또는 ']' : 현재 인덱스 기준 이전/다음 클립으로 점프"""
        # clip json 준비 여부
        if not self.clip_json_path or not self.clip_json_path.exists():
            self._toast("CLIP JSON을 먼저 열거나 생성하세요.")
            return

        # 최신 쌍으로 새로고침(파일이 방금 갱신되었을 수 있음)
        self._refresh_clip_pairs()
        C = self._clip_pairs
        if not C:
            self._toast("CLIP JSON에 세그먼트가 없습니다.")
            return

        cur = int(self.lidar_idx)
        # 현재 속한 클립 찾기
        cur_idx = None
        for i, it in enumerate(C):
            if it["start"] <= cur <= it["end"]:
                cur_idx = i
                break

        target_i = None
        if direction > 0:  # 다음
            if cur_idx is not None:
                target_i = min(len(C) - 1, cur_idx + 1)
            else:
                # 현재 위치 이후 첫 클립
                target_i = next((i for i, it in enumerate(C) if it["start"] > cur), len(C) - 1)
        else:  # 이전
            if cur_idx is not None:
                target_i = max(0, cur_idx - 1)
            else:
                # 현재 위치 이전 마지막 클립
                rev = [i for i, it in enumerate(C) if it["end"] < cur]
                target_i = (rev[-1] if rev else 0)

        self._jump_to_clip(C[target_i])

    def _jump_to_segment(self, it: dict):
        start = int(it["start"])
        st = it.get("st") or {}
        cams = st.get("cam_indices", None)

        # 슬라이더(→ _on_slider 통해 카메라도 delta만큼 이동)로 먼저 이동
        start = max(0, min(len(self.lidar_files)-1, start))
        self.sld.setValue(start)

        # 스냅샷에 cam_indices가 있으면 정확히 그 값으로 덮어써 동기화
        if isinstance(cams, list) and len(cams) == 6:
            new_cam_idx = []
            for i in range(6):
                max_i = max(0, len(self.camera_files[i]) - 1)
                new_cam_idx.append(min(max_i, max(0, int(cams[i]))))
            self.img_idx = new_cam_idx
            self._refresh_all()

        self._toast(f"Jumped to MERGED sid={it.get('sid')} [{it['start']}..{it['end']}]")

    def _jump_to_clip(self, it: dict):
        """지정된 클립 시작지점으로 점프 + 카메라 인덱스 동기화"""
        start = int(it["start"])
        st = it.get("st") or {}
        cams = st.get("cam_indices", None)

        # 슬라이더로 이동 (이벤트 통해 화면 갱신)
        if self.lidar_files:
            start = max(0, min(len(self.lidar_files) - 1, start))
        self.sld.setValue(start)

        # 카메라 인덱스도 스냅샷 기준으로 보정(가능할 때만)
        if isinstance(cams, list) and len(cams) == 6:
            new_cam_idx = []
            for i in range(6):
                max_i = max(0, len(self.camera_files[i]) - 1)
                new_cam_idx.append(min(max_i, max(0, int(cams[i]))))
            self.img_idx = new_cam_idx
            # 즉시 반영
            self._refresh_all()
        self._toast(f"Jumped to clip sid={it.get('sid')} [{it['start']}..{it['end']}]")

    # -------------- Export --------------

    def _export_from_clip_json(self):
        """
        Export는 'export용으로 Load한 CLIP JSON(self.export_clip_path)'을 사용해서 수행.
        필요 시 파일 선택 다이얼로그가 뜸.
        """
        # 1) 대상 export clip json 확보
        if not self.export_clip_path or not self.export_clip_path.exists():
            self._choose_export_clip_json_dialog()
            if not self.export_clip_path or not self.export_clip_path.exists():
                self._toast("Export canceled (no clip selected).")
                return

        # 2) 미리보기(이미 _build_export_preview_and_label에서 만들어둠)
        if not self._export_preview:
            # 혹시 비어 있으면 다시 시도
            self._build_export_preview_and_label()
        if not self._export_preview:
            self._toast("선택된 EXPORT CLIP JSON에 내보낼 세그먼트가 없습니다.")
            return

        show = ", ".join([f"{sid}:{a}-{b}" for sid, a, b in self._export_preview[:5]])
        more = f" ... (+{len(self._export_preview)-5} more)" if len(self._export_preview) > 5 else ""
        self._toast(f"Export {len(self._export_preview)} clip(s) from {self.export_clip_path.name} :: {show}{more}")

        # 3) 실제 Export 실행
        try:
            V.export_scenes_from_marks(self.export_clip_path, dataset_tag=V.CFG.dataset_tag)
            self._toast(f"Export done: {len(self._export_preview)} clip(s) → {self.export_clip_path.name}")
        except Exception as e:
            self._toast(f"Export failed: {e}")

    # -------------- 렌더링 --------------

    def _refresh_all(self):
        # 6캠 합성
        imgs = []
        for i in range(6):
            imgs.append(V.project_one_cam(i, self.img_idx[i], self.lidar_idx,
                                          draw_lidar=self.draw_lidar,
                                          point_radius=self.point_radius))
        canvas = V.build_canvas(imgs)

        # 오버뷰 바: '교집합' + (편집중 clip) 표시
        H = V.CFG.overview_h
        W = canvas.shape[1]

        clip_segs = [(it["start"], it["end"]) for it in self._clip_pairs] if self._clip_pairs else None
        bar = V.create_lidar_overview_bar(
            total_lidar=len(self.lidar_files),
            current_lidar_idx=self.lidar_idx,
            segs=[],                 # 내 JSON(주황) 숨김
            width=W,
            height=H,
            extra_segs=None,         # 초록 숨김
            merged_segs=self.merged_segs,  # 교집합만 표시
            clip_segs=clip_segs      # 편집중인 clip도 다른 색으로 표시
        )
        # 합성(하단 개행은 build_canvas에서 timeline+overview 공간을 이미 예약해둠)
        # 여기선 가장 아래 영역에 bar만 덮어쓰기
        h0 = canvas.shape[0] - H
        canvas[h0:h0+H, :, :] = bar

        # QLabel 갱신
        qimg = QtGui.QImage(canvas.data, canvas.shape[1], canvas.shape[0],
                            canvas.strides[0], QtGui.QImage.Format.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg)

        s = float(self.view_scale)
        if 0 < s < 1.0:
            w = int(pix.width() * s)
            h = int(pix.height() * s)
            if w > 0 and h > 0:
                pix = pix.scaled(w, h,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)

        self.lbl_canvas.setPixmap(pix)

    def _refresh_seg_label(self):
        self.lbl_seg.setText(f"allowed_next = {self.allowed_next} | segment_id = {self.segment_id}")

    def _toast(self, msg: str):
        self.status.showMessage(msg, 4000)
        print(msg)


# ---------------------------------------
# main
# ---------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ClipViewer()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
