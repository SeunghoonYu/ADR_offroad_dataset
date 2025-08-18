#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, shutil, sys, re
from pathlib import Path
from typing import Optional

# =========================
# User configs
# =========================
BASE_DIR     = Path("/mnt/e/off-road/test0807_15_11")       # 원본 데이터 루트
MARKS_JSON   = BASE_DIR / "marks_json" / "sync_marks_20250818_122434.json"  # 파일 또는 glob 패턴 둘 다 지원
COPY_MODE    = "symlink"   # "copy" | "symlink"
DATASET_TAG  = "SNU_mountain"  # 예: "SNU_mountain"; None이면 기본(<BASE>_scenes)

# =========================
# Helpers
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, mode: str):
    # 기존 파일/링크가 있으면 제거
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except Exception:
            pass

    if mode == "symlink":
        try:
            # 절대경로 symlink 권장 (상대경로도 됨)
            dst.symlink_to(src)
            return
        except Exception as e:
            print(f"[warn] symlink failed → fallback to copy: {src} ({e})")

    # fallback 또는 copy 모드
    shutil.copy2(src, dst)

def load_marks_json(marks: Path):
    """MARKS_JSON이 파일이면 그대로, 아니면 glob 패턴으로 최신 파일 선택."""
    if marks.exists() and marks.is_file():
        chosen = marks
    else:
        candidates = sorted(marks.parent.glob(marks.name))
        if not candidates:
            raise FileNotFoundError(f"No marks JSON found: {marks}")
        chosen = candidates[-1]
    with open(chosen, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Marks JSON must be a list of entries.")
    return data, chosen

def list_inputs(base_dir: Path):
    lidar_dir = base_dir / "lidar_xyzi"
    cam_dirs  = [base_dir / "decoded_rgb" / f"camera_{i}" for i in range(1,7)]
    lidar = sorted(lidar_dir.glob("*.bin"))
    cams  = [sorted(d.glob("*.jpg")) for d in cam_dirs]
    return lidar, cams

def pair_segments(items):
    starts, ends = {}, {}
    for it in items:
        lab = it.get("label", "")
        if lab.startswith("start"):
            sid = int(lab.replace("start", ""))
            starts[sid] = it
        elif lab.startswith("end"):
            sid = int(lab.replace("end", ""))
            ends[sid] = it
    scene_ids = sorted(set(starts) & set(ends))
    return [(sid, starts[sid], ends[sid]) for sid in scene_ids]

def derive_root_name(base_dir: Path, dataset_tag: Optional[str]) -> str:
    base_name = base_dir.name  # e.g., "test0807_15_11"
    if dataset_tag:
        # base_name 에서 처음 등장하는 숫자부터 끝까지를 suffix로 사용 (예: "0807_15_11")
        m = re.search(r"(\d.*)$", base_name)
        suffix = m.group(1) if m else base_name
        return f"{dataset_tag}_{suffix}_scenes"
    else:
        return f"{base_name}_scenes"

# =========================
# Main
# =========================
def main():
    data, chosen_json = load_marks_json(MARKS_JSON)
    lidar_files, cam_files = list_inputs(BASE_DIR)

    root_name = derive_root_name(BASE_DIR, DATASET_TAG)
    OUT_ROOT = BASE_DIR.parent / root_name
    ensure_dir(OUT_ROOT)

    print(f"[info] Base dir     : {BASE_DIR}")
    print(f"[info] Output root  : {OUT_ROOT}")
    print(f"[info] Copy mode    : {COPY_MODE}")
    print(f"[info] Marks JSON   : {chosen_json}")

    pairs = pair_segments(data)
    if not pairs:
        print("[error] No start/end pairs found.")
        sys.exit(1)

    for sid, start, end in pairs:
        scene_dir = OUT_ROOT / f"{root_name}_{sid}"
        out_lidar = scene_dir / "lidar_xyzi"
        out_cams  = [scene_dir / "decoded_rgb" / f"camera_{i}" for i in range(1,7)]
        ensure_dir(out_lidar)
        for d in out_cams:
            ensure_dir(d)

        # 인덱스 범위 (포함 구간)
        l0, l1 = start["indices"]["lidar_idx"], end["indices"]["lidar_idx"]
        c0 = start["indices"]["cam_idx"]
        c1 = end["indices"]["cam_idx"]

        # 길이 계산
        L_len = max(0, l1 - l0 + 1)
        C_len = [max(0, c1[i] - c0[i] + 1) for i in range(6)]
        seg_len = min([L_len] + C_len)

        if seg_len <= 0:
            print(f"[warn][scene {sid}] empty segment, skip.")
            continue

        if not (L_len == seg_len == C_len[0] == C_len[1] == C_len[2] == C_len[3] == C_len[4] == C_len[5]):
            print(f"[warn][scene {sid}] length mismatch → truncate to {seg_len} "
                  f"(lidar={L_len}, cams={C_len})")

        # LiDAR
        for k in range(seg_len):
            src = lidar_files[l0 + k]
            dst = out_lidar / f"{k:06d}.bin"
            link_or_copy(src, dst, COPY_MODE)

        # Cams
        for i in range(6):
            for k in range(seg_len):
                src = cam_files[i][c0[i] + k]
                dst = out_cams[i] / f"{k:06d}.jpg"
                link_or_copy(src, dst, COPY_MODE)

        # meta 저장
        meta = {
            "scene_id": sid,
            "root_name": root_name,
            "source_base_dir": str(BASE_DIR),
            "length": seg_len,
            "lidar_range": [l0, l0 + seg_len - 1],
            "cam_ranges": [[c0[i], c0[i] + seg_len - 1] for i in range(6)],
            "mode": COPY_MODE
        }
        with open(scene_dir / "scene_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[ok] {scene_dir}  (frames={seg_len})")

if __name__ == "__main__":
    main()
