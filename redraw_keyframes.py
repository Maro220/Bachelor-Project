"""
redraw_keyframes.py
-------------------
For an existing scene whose Phase 2 ran, draw fresh top-3 keyframe images
into output/<scene>/top3/keyframes/keyframe_<sample_idx>.jpg using the
current _draw_top3_boxes style (unified white box, label "ID: N" only).

Reads:
    output/<scene>/top3/keyframes/keyframe_<sec>.json   (filtered detections)
    output/<scene>/top3/keyframe_map.json               (sample_idx → sample_token)
    NuScenes raw images (via SDK) for the CAM_FRONT frame

Writes:
    output/<scene>/top3/keyframes/keyframe_<sec>.jpg

Usage:
    python redraw_keyframes.py scene-0061
"""

import glob
import json
import os
import sys

if len(sys.argv) < 2:
    sys.exit("Usage: python redraw_keyframes.py <scene-name>")

scene        = sys.argv[1]
scene_root   = f"output/{scene}"
full_dir     = f"{scene_root}/full"
kf_dir       = f"{full_dir}/keyframes"
kf_map_path  = f"{full_dir}/keyframe_map.json"

for p in (kf_dir, kf_map_path):
    if not os.path.exists(p):
        sys.exit(f"Missing: {p}")

import cv2
from nuscenes.nuscenes import NuScenes
from top3_pipeline import _draw_all_boxes_uniform

NUSCENES_DATAROOT = "data/v1.0-mini"

with open(kf_map_path) as f:
    km = {entry["sample_idx"]: entry for entry in json.load(f)}

# Source of truth for "all objects" is the FULL (unfiltered) keyframe JSONs
full_kf_dir = kf_dir

print(f"Loading NuScenes from {NUSCENES_DATAROOT}…")
nusc = NuScenes(version="v1.0-mini", dataroot=NUSCENES_DATAROOT, verbose=False)

kf_jsons = sorted(glob.glob(os.path.join(full_kf_dir, "keyframe_*.json")))
print(f"Redrawing {len(kf_jsons)} keyframes (all objects, white + track id)…")

done = 0
for full_kf_path in kf_jsons:
    base       = os.path.basename(full_kf_path)
    sample_idx = int(base.replace("keyframe_", "").replace(".json", ""))
    entry      = km.get(sample_idx)
    if not entry:
        print(f"  ⚠  no keyframe_map entry for sample_idx {sample_idx}")
        continue
    try:
        sample = nusc.get("sample", entry["sample_token"])
        sd     = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img    = cv2.imread(os.path.join(NUSCENES_DATAROOT, sd["filename"]))
    except Exception as e:
        print(f"  ⚠  load failed for sec {sample_idx}: {e}")
        continue
    if img is None:
        continue

    with open(full_kf_path) as f:
        data = json.load(f)
    objs = data.get("scene_summary", {}).get("detected_objects", []) or []

    canvas  = _draw_all_boxes_uniform(img, objs)
    out_jpg = os.path.join(kf_dir, f"keyframe_{sample_idx:04d}.jpg")
    cv2.imwrite(out_jpg, canvas)
    done += 1

print(f"\n✅ Wrote {done} jpgs into {kf_dir}/")
