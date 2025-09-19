#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Force Export Runner
- clip_json: 내보낼 CLIP JSON 경로
- out_root : 강제로 지정할 저장 경로
"""

from pathlib import Path
import sys
import merge_viewer as V  # 같은 디렉토리에 merge_viewer.py 필요

# ===== 여기만 수정 =====
CLIP_JSON = Path("/mnt/mydisk/offroad_dataset_marks_json/siheung_lake/test0827_14_42_marks_json/final_clip_json/final_clip_test0827_14_42.json")
OUT_ROOT  = Path("/mnt/mydisk/offroad_dataset_final_scenes/siheung_lake/test0827_14_42")   # 강제로 저장할 경로
BASE_DIR  = Path("/mnt/mydisk/offroad_dataset_origin/siheung_lake/test0827_14_42")
# ======================

def _log_cb(msg: str):
    print(msg)

def _progress_cb(done: int, total: int):
    pct = int(round(100.0 * done / total)) if total else 0
    print(f"\r[progress] {done}/{total} ({pct}%)", end="", flush=True)
    if done >= total:
        print()

def main():
    clip_path = CLIP_JSON.expanduser().resolve()
    out_root  = OUT_ROOT.expanduser().resolve()
    base_dir_override = BASE_DIR.expanduser().resolve() if isinstance(BASE_DIR, Path) else None

    if not clip_path.exists():
        print(f"[error] clip json not found: {clip_path}")
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] clip json : {clip_path}")
    print(f"[info] out_root  : {out_root}")

    try:
        V.export_scenes_from_marks(
            marks_json_path=clip_path,
            dataset_tag="manual",   # 무시됨
            log_cb=_log_cb,
            progress_cb=_progress_cb,
            base_dir_override=base_dir_override,
            out_root_override=out_root  # ⚡ merge_viewer.py 함수에 이 인자 추가 필요
        )
        print("\n[done] Export finished.")
    except Exception as e:
        print(f"\n[error] Export failed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
