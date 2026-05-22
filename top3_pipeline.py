"""
top3_pipeline.py
----------------
Phase 2 of the simple annotation pipeline.

Reads the full pipeline output from output/full/ (written by Phase 1 =
process_scene_sweeps), locks the top-3 GT-confirmed objects by combined
bounding-box area, then writes a filtered copy to output/top3/ containing
ONLY those 3 identities across every frame, keyframe, annotated image, and
cumulative summary.

Nothing in output/full/ is ever touched.

Public API
----------
    from top3_pipeline import run_top3_phase

    # Call this right after process_scene_sweeps() returns.
    run_top3_phase(
        full_out_dir = "output/full",
        top3_out_dir = "output/top3",
        nusc         = nusc_handle,          # open NuScenes instance
        stable_fields= {"environment": ...,  # from Phase 1
                        "lighting":    ...,
                        "nuscenes_description": ...},
    )

Output layout inside top3_out_dir
----------------------------------
    frames/                   per-sweep JSONs  (3 objects only)
    keyframes/                sample-aligned JSONs (3 objects only)
    annotated/                JPGs with only 3 colour-coded boxes
    scene/                    per-keyframe Qwen outputs (3-object prompt)
    summaries/                flat cumulative (3 objects only)
    keyframe_map.json         copy of full keyframe_map (unchanged)
    top3_identities.json      locked {identity → metadata} for the 3 objects
    representative_frame.jpg  best keyframe where all 3 visible + largest area
    representative_frame.json metadata about the chosen frame
"""

import base64
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── colour palette per slot (BGR for OpenCV) ────────────────────────────────
_SLOT_COLORS_BGR = [
    (255, 255, 255),   # slot 0 — white
    (255, 255, 255),   # slot 1 — white
    (255, 255, 255),   # slot 2 — white
]
_SLOT_LABELS = ["ID: 1", "ID: 2", "ID: 3"]


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _dump(path: str, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _bbox_area(bb: dict) -> int:
    if not bb:
        return 0
    return max(0, (bb.get("x2", 0) - bb.get("x1", 0)) *
                  (bb.get("y2", 0) - bb.get("y1", 0)))


def _identity_of(obj: dict) -> Optional[str]:
    """Return the stable identity key for a detection dict."""
    it = obj.get("instance_token", "")
    if it:
        return it
    tid = obj.get("track_id", -1)
    if tid != -1:
        return f"yolo_{tid}"
    return None


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — lock top-3 identities from full keyframe output
# ════════════════════════════════════════════════════════════════════════════

def lock_top3_identities(full_out_dir: str) -> List[Dict]:
    """
    Scan all keyframe JSONs in full_out_dir/keyframes/ and return the 3
    identities (instance_token or yolo_<id>) with the largest TOTAL
    bounding-box area summed across all keyframes where they appear.

    Selection priority
    ------------------
    1. GT-confirmed objects (gt_sourced=True)  — ranked by total area
    2. Fill remaining slots with highest-area YOLO-only objects

    Returns a list of up to 3 dicts:
        {
          "slot":           0 / 1 / 2,
          "slot_label":     "Object 1" / "Object 2" / "Object 3",
          "identity":       "<instance_token or yolo_tid>",
          "object_type":    "car" / "pedestrian" / …,
          "color":          "white" / … / "unknown",
          "total_area":     <int px²>,
          "seen_keyframes": [<kf_idx>, …],
          "gt_sourced":     True / False,
        }
    """
    kf_dir = os.path.join(full_out_dir, "keyframes")
    import glob
    kf_files = sorted(glob.glob(os.path.join(kf_dir, "keyframe_*.json")))
    if not kf_files:
        raise FileNotFoundError(
            f"No keyframe JSONs found in {kf_dir}. Run Phase 1 first."
        )

    # identity → aggregated stats
    agg: Dict[str, dict] = {}

    for kf_path in kf_files:
        data = _load(kf_path)
        kf_idx = data.get("sample_idx",
                          int(os.path.basename(kf_path)
                              .replace("keyframe_", "").replace(".json", "")))
        objs = (data.get("scene_summary", {})
                    .get("detected_objects", []))

        for obj in objs:
            ident = _identity_of(obj)
            if ident is None:
                continue
            area = _bbox_area(obj.get("bounding_box", {}))
            if ident not in agg:
                agg[ident] = {
                    "identity":       ident,
                    "object_type":    obj.get("object_type") or obj.get("type", "object"),
                    "color":          obj.get("color", "unknown"),
                    "gt_sourced":     obj.get("gt_sourced", False),
                    "total_area":     0,
                    "seen_keyframes": [],
                }
            agg[ident]["total_area"] += area
            agg[ident]["seen_keyframes"].append(kf_idx)
            # update color if we get a better (non-unknown) one
            if agg[ident]["color"] in ("unknown", "") and obj.get("color", "unknown") not in ("unknown", ""):
                agg[ident]["color"] = obj["color"]
            # promote gt_sourced if any frame confirms it
            if obj.get("gt_sourced"):
                agg[ident]["gt_sourced"] = True

    all_identities = list(agg.values())
    gt_objs   = [x for x in all_identities if x["gt_sourced"]]
    yolo_objs = [x for x in all_identities if not x["gt_sourced"]]

    gt_objs.sort(key=lambda x: x["total_area"], reverse=True)
    yolo_objs.sort(key=lambda x: x["total_area"], reverse=True)

    selected = (gt_objs[:3] + yolo_objs)[:3]

    result = []
    for slot, obj in enumerate(selected):
        entry = dict(obj)
        entry["slot"]       = slot
        entry["slot_label"] = _SLOT_LABELS[slot]
        result.append(entry)

    print(f"\n✅ Locked top-3 identities:")
    for e in result:
        gt_str = "GT ✓" if e["gt_sourced"] else "YOLO"
        print(f"   [{e['slot_label']}] {e['color']} {e['object_type']} "
              f"| area={e['total_area']}px² | {gt_str} "
              f"| seen in {len(e['seen_keyframes'])} keyframes")

    return result


# ── Manual override builders ────────────────────────────────────────────────

def _build_identities_from_override(override_ids: List[str],
                                    full_out_dir: str) -> List[Dict]:
    """Build the same dict shape as lock_top3_identities, but for caller-chosen
    identity strings. Aggregates area / seen_keyframes from existing JSONs."""
    import glob
    kf_files = sorted(glob.glob(
        os.path.join(full_out_dir, "keyframes", "keyframe_*.json")))
    if not kf_files:
        raise FileNotFoundError(
            f"No keyframes in {full_out_dir}/keyframes — run Phase 1 first.")

    agg = {ident: {
        "identity":       ident,
        "object_type":    "object",
        "color":          "unknown",
        "gt_sourced":     False,
        "total_area":     0,
        "seen_keyframes": [],
    } for ident in override_ids}

    for kf_path in kf_files:
        data   = _load(kf_path)
        kf_idx = data.get("sample_idx",
            int(os.path.basename(kf_path).replace("keyframe_", "").replace(".json", "")))
        for obj in data.get("scene_summary", {}).get("detected_objects", []) or []:
            ident = _identity_of(obj)
            if ident not in agg:
                continue
            agg[ident]["total_area"] += _bbox_area(obj.get("bounding_box", {}))
            agg[ident]["seen_keyframes"].append(kf_idx)
            if obj.get("object_type") or obj.get("type"):
                agg[ident]["object_type"] = obj.get("object_type") or obj.get("type")
            if agg[ident]["color"] in ("unknown", "") and obj.get("color", "unknown") not in ("unknown", ""):
                agg[ident]["color"] = obj["color"]
            if obj.get("gt_sourced"):
                agg[ident]["gt_sourced"] = True

    missing = [i for i in override_ids if agg[i]["total_area"] == 0]
    if missing:
        raise ValueError(
            f"--top3 identities not found in any keyframe: {missing}. "
            f"Pass the IDENTITY column from list_top3_candidates.py "
            f"(e.g. 'yolo_42' or a 32-char NuScenes instance_token), "
            f"not the '#' row number."
        )

    out = []
    for slot, ident in enumerate(override_ids):
        e = dict(agg[ident])
        e["slot"]       = slot
        e["slot_label"] = _SLOT_LABELS[slot]
        out.append(e)

    print(f"\n✅ Locked overridden top-3 identities:")
    for e in out:
        gt = "GT ✓" if e["gt_sourced"] else "YOLO"
        print(f"   [{e['slot_label']}] {e['color']} {e['object_type']} "
              f"| area={e['total_area']}px² | {gt} "
              f"| seen in {len(e['seen_keyframes'])} keyframes")
    return out


def _force_representative(sec: int, top3_identities: List[Dict],
                          full_out_dir: str):
    """Return (best_idx, best_kf_path, stats) for a user-pinned keyframe sec."""
    kf_path = os.path.join(full_out_dir, "keyframes", f"keyframe_{sec:04d}.json")
    if not os.path.exists(kf_path):
        raise FileNotFoundError(
            f"No keyframe at sec {sec} — looked for {kf_path}.")
    data = _load(kf_path)
    locked = {e["identity"] for e in top3_identities}
    visible, area = 0, 0
    for obj in data.get("scene_summary", {}).get("detected_objects", []) or []:
        if _identity_of(obj) in locked:
            visible += 1
            area    += _bbox_area(obj.get("bounding_box", {}))
    print(f"  sec {sec}: {visible}/3 of chosen identities visible, "
          f"combined area={area}px²")
    return sec, kf_path, {"objects_visible": visible, "combined_area": area}


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — pick best representative keyframe
# ════════════════════════════════════════════════════════════════════════════

def pick_representative_keyframe(
    top3_identities: List[Dict],
    full_out_dir: str,
) -> Tuple[int, str, dict]:
    """
    Find the keyframe where ALL 3 objects are simultaneously visible and
    their combined bounding-box area is largest.

    Falls back to best partial visibility (2 objects, then 1) if no frame
    has all 3 at once.

    Returns (sample_idx, keyframe_json_path, combined_stats_dict).
    """
    locked_ids = {e["identity"] for e in top3_identities}

    import glob
    kf_dir   = os.path.join(full_out_dir, "keyframes")
    kf_files = sorted(glob.glob(os.path.join(kf_dir, "keyframe_*.json")))

    best_count     = 0
    best_area      = 0
    best_kf_path   = kf_files[len(kf_files) // 2]  # fallback: middle frame
    best_sample_idx = 0
    best_stats     = {}

    for kf_path in kf_files:
        data    = _load(kf_path)
        sample_idx = data.get("sample_idx",
                               int(os.path.basename(kf_path)
                                   .replace("keyframe_", "").replace(".json", "")))
        objs    = (data.get("scene_summary", {})
                       .get("detected_objects", []))

        present      = {}   # identity → obj entry
        combined_area = 0
        for obj in objs:
            ident = _identity_of(obj)
            if ident in locked_ids:
                present[ident] = obj
                combined_area += _bbox_area(obj.get("bounding_box", {}))

        count = len(present)
        # Prefer frames with more objects visible; tie-break on combined area
        if (count > best_count) or (count == best_count and combined_area > best_area):
            best_count     = count
            best_area      = combined_area
            best_kf_path   = kf_path
            best_sample_idx = sample_idx
            best_stats     = {
                "sample_idx":     sample_idx,
                "objects_visible": count,
                "combined_area":   combined_area,
                "present_identities": list(present.keys()),
            }

    print(f"\n✅ Representative keyframe: sample_idx={best_sample_idx} "
          f"({best_count}/3 objects visible, combined_area={best_area}px²)")
    return best_sample_idx, best_kf_path, best_stats


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — filter ALL frame JSONs to top-3 only + redraw annotated images
# ════════════════════════════════════════════════════════════════════════════

def _filter_frame_json(data: dict, locked_ids: set) -> dict:
    """
    Return a copy of a frame JSON with detected_objects filtered to only
    the locked identities. Counts are recomputed from the filtered list.
    """
    import copy
    out = copy.deepcopy(data)
    ss  = out.setdefault("scene_summary", {})
    all_objs = ss.get("detected_objects", [])

    filtered = [o for o in all_objs if _identity_of(o) in locked_ids]
    ss["detected_objects"] = filtered

    vehicle_types = {"car", "van", "truck", "bus", "motorcycle"}
    ss["total_vehicles_detected"]       = sum(1 for o in filtered if o.get("object_type", o.get("type","")) in vehicle_types)
    ss["total_pedestrians_detected"]    = sum(1 for o in filtered if o.get("object_type", o.get("type","")) == "pedestrian")
    ss["total_cyclists_detected"]       = sum(1 for o in filtered if o.get("object_type", o.get("type","")) == "cyclist")
    ss["total_traffic_lights_detected"] = sum(1 for o in filtered if o.get("object_type", o.get("type","")) == "traffic_light")

    return out


def _draw_all_boxes_uniform(
    frame_bgr: np.ndarray,
    objs: List[dict],
) -> np.ndarray:
    """Draw every object on frame_bgr in WHITE, label = 'ID: <track_id>'.
    Used for keyframe overview images (not the rep frame)."""
    WHITE  = (255, 255, 255)
    canvas = frame_bgr.copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for obj in objs:
        bb = obj.get("bounding_box", {})
        if not bb:
            continue
        x1, y1 = int(bb.get("x1", 0)), int(bb.get("y1", 0))
        x2, y2 = int(bb.get("x2", 0)), int(bb.get("y2", 0))
        if x2 <= x1 or y2 <= y1:
            continue

        tid   = obj.get("track_id")
        label = f"ID: {tid}" if tid is not None else "ID: ?"

        # Box + corner accents
        cv2.rectangle(canvas, (x1, y1), (x2, y2), WHITE, 3)
        mark = 18
        for (px, py), (dx, dy) in [
            ((x1,y1),(1,1)),((x2,y1),(-1,1)),
            ((x1,y2),(1,-1)),((x2,y2),(-1,-1))
        ]:
            cv2.line(canvas,(px,py),(px+dx*mark,py),WHITE,4)
            cv2.line(canvas,(px,py),(px,py+dy*mark),WHITE,4)

        font_scale = 0.60
        line_h     = int(font_scale * 28)
        pad        = 6
        text_w     = cv2.getTextSize(label, font, font_scale, 1)[0][0]
        box_w      = text_w + pad*2
        box_h      = line_h + pad*2
        lx, ly     = x1, (y1-box_h-4 if y1 > box_h+8 else y2+4)

        overlay = canvas.copy()
        cv2.rectangle(overlay,(lx,ly),(lx+box_w,ly+box_h),(0,0,0),-1)
        cv2.addWeighted(overlay,0.65,canvas,0.35,0,canvas)

        ty = ly+pad+line_h-4
        cv2.putText(canvas,label,(lx+pad,ty),font,font_scale,WHITE,1,cv2.LINE_AA)

    return canvas


def _draw_top3_boxes(
    frame_bgr: np.ndarray,
    objs: List[dict],
    slot_map: Dict[str, int],   # identity → slot index
) -> np.ndarray:
    """
    Draw only the top-3 boxes on frame_bgr. Each box uses its slot colour
    and is labelled Object 1/2/3 plus motion if GT-confirmed.
    """
    canvas = frame_bgr.copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for obj in objs:
        ident = _identity_of(obj)
        if ident not in slot_map:
            continue
        slot  = slot_map[ident]
        color = _SLOT_COLORS_BGR[slot]
        label = _SLOT_LABELS[slot]

        bb = obj.get("bounding_box", {})
        if not bb:
            continue
        x1, y1 = int(bb.get("x1", 0)), int(bb.get("y1", 0))
        x2, y2 = int(bb.get("x2", 0)), int(bb.get("y2", 0))
        if x2 <= x1 or y2 <= y1:
            continue

        # Box + corner accents
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        mark = 18
        for (px, py), (dx, dy) in [
            ((x1,y1),(1,1)),((x2,y1),(-1,1)),
            ((x1,y2),(1,-1)),((x2,y2),(-1,-1))
        ]:
            cv2.line(canvas,(px,py),(px+dx*mark,py),color,4)
            cv2.line(canvas,(px,py),(px,py+dy*mark),color,4)

        # Label
        font_scale = 0.60
        line_h     = int(font_scale * 28)
        pad        = 6
        text_w     = cv2.getTextSize(label, font, font_scale, 1)[0][0]
        box_w      = text_w + pad*2
        box_h      = line_h + pad*2
        lx         = x1
        ly         = y1-box_h-4 if y1 > box_h+8 else y2+4

        overlay = canvas.copy()
        cv2.rectangle(overlay,(lx,ly),(lx+box_w,ly+box_h),(0,0,0),-1)
        cv2.addWeighted(overlay,0.65,canvas,0.35,0,canvas)

        ty = ly+pad+line_h-4
        cv2.putText(canvas,label,(lx+pad,ty),font,font_scale,color,1,cv2.LINE_AA)

    return canvas


def filter_full_to_top3(
    top3_identities: List[Dict],
    full_out_dir:    str,
    top3_out_dir:    str,
    dataroot:        str,
    nusc             = None,
) -> None:
    """
    Walk every JSON in full_out_dir/{frames,keyframes}/ and write filtered
    copies to top3_out_dir/{frames,keyframes}/. Re-draw annotated images
    showing only the 3 boxes.
    """
    import glob

    locked_ids = {e["identity"] for e in top3_identities}
    slot_map   = {e["identity"]: e["slot"] for e in top3_identities}

    for sub in ("frames", "keyframes", "annotated", "scene", "summaries"):
        os.makedirs(os.path.join(top3_out_dir, sub), exist_ok=True)

    # ── frames (all sweeps) ──────────────────────────────────────────────
    frame_files = sorted(glob.glob(
        os.path.join(full_out_dir, "frames", "frame_*.json")
    ))
    print(f"\n  Filtering {len(frame_files)} sweep frames…")
    for fp in frame_files:
        data     = _load(fp)
        filtered = _filter_frame_json(data, locked_ids)
        out_path = fp.replace(
            os.path.join(full_out_dir, "frames"),
            os.path.join(top3_out_dir,  "frames")
        )
        _dump(out_path, filtered)

    # ── keyframes (sample-aligned) ───────────────────────────────────────
    kf_files = sorted(glob.glob(
        os.path.join(full_out_dir, "keyframes", "keyframe_*.json")
    ))
    print(f"  Filtering {len(kf_files)} keyframes…")
    for kf in kf_files:
        data     = _load(kf)
        filtered = _filter_frame_json(data, locked_ids)
        out_path = kf.replace(
            os.path.join(full_out_dir, "keyframes"),
            os.path.join(top3_out_dir,  "keyframes")
        )
        _dump(out_path, filtered)

    # ── annotated images: redraw with only 3 boxes ───────────────────────
    ann_files = sorted(glob.glob(
        os.path.join(full_out_dir, "annotated", "frame_*_track.jpg")
    ))
    print(f"  Redrawing {len(ann_files)} annotated images…")

    # Build frame_idx → filtered objects lookup from top3/frames/
    frame_objs_cache: Dict[int, List[dict]] = {}
    for fp in glob.glob(os.path.join(top3_out_dir, "frames", "frame_*.json")):
        import re
        m = re.search(r"frame_(\d+)\.json", os.path.basename(fp))
        if m:
            fidx = int(m.group(1))
            d    = _load(fp)
            frame_objs_cache[fidx] = (
                d.get("scene_summary", {}).get("detected_objects", [])
            )

    for ann_path in ann_files:
        import re
        m = re.search(r"frame_(\d+)_track\.jpg", os.path.basename(ann_path))
        if not m:
            continue
        fidx   = int(m.group(1))
        frame  = cv2.imread(ann_path)
        if frame is None:
            continue
        objs   = frame_objs_cache.get(fidx, [])
        canvas = _draw_top3_boxes(frame, objs, slot_map)
        out_path = ann_path.replace(
            os.path.join(full_out_dir,  "annotated"),
            os.path.join(top3_out_dir,  "annotated")
        )
        cv2.imwrite(out_path, canvas)

    # ── copy keyframe_map unchanged ──────────────────────────────────────
    src_km = os.path.join(full_out_dir, "keyframe_map.json")
    if os.path.exists(src_km):
        shutil.copy2(src_km, os.path.join(top3_out_dir, "keyframe_map.json"))

    # ── draw one annotated jpg per keyframe (ALL objects, white + track id)
    if nusc is not None and os.path.exists(src_km):
        km_entries  = {e["sample_idx"]: e for e in _load(src_km)}
        full_kfs    = sorted(glob.glob(
            os.path.join(full_out_dir, "keyframes", "keyframe_*.json")))
        print(f"  Drawing {len(full_kfs)} keyframe jpgs (all objects)…")
        for full_kf_path in full_kfs:
            base       = os.path.basename(full_kf_path)
            try:
                sample_idx = int(base.replace("keyframe_", "").replace(".json", ""))
            except ValueError:
                continue
            entry = km_entries.get(sample_idx)
            if not entry:
                continue
            try:
                sample   = nusc.get("sample", entry["sample_token"])
                sd       = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                img      = cv2.imread(os.path.join(dataroot, sd["filename"]))
            except Exception as e:
                print(f"    ⚠  sec {sample_idx}: {e}")
                continue
            if img is None:
                continue
            objs   = (_load(full_kf_path).get("scene_summary", {})
                                          .get("detected_objects", []) or [])
            canvas = _draw_all_boxes_uniform(img, objs)
            out_jpg = os.path.join(full_out_dir, "keyframes",
                                   f"keyframe_{sample_idx:04d}.jpg")
            cv2.imwrite(out_jpg, canvas)

    print(f"  ✅ Filtered output written to {top3_out_dir}/")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — write representative frame image + metadata
# ════════════════════════════════════════════════════════════════════════════

def write_representative_frame(
    best_sample_idx:   int,
    best_kf_json_path: str,
    best_stats:        dict,
    top3_identities:   List[Dict],
    top3_out_dir:      str,
    dataroot:          str,
    nusc,
) -> str:
    """
    Load the image for the best keyframe, draw the 3 boxes, and save to
    top3_out_dir/representative_frame.jpg.
    Also saves representative_frame.json with full metadata.
    Returns the output image path.
    """
    slot_map = {e["identity"]: e["slot"] for e in top3_identities}

    # Find the source image via NuScenes SDK
    kf_map_path = os.path.join(top3_out_dir, "keyframe_map.json")
    img_path    = None
    if os.path.exists(kf_map_path):
        kf_map = _load(kf_map_path)
        entry  = next((k for k in kf_map if k["sample_idx"] == best_sample_idx), None)
        if entry:
            try:
                sample = nusc.get("sample", entry["sample_token"])
                sd     = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                img_path = os.path.join(dataroot, sd["filename"])
            except Exception as e:
                print(f"  ⚠  Could not resolve image via SDK: {e}")

    # Fallback: look in full annotated dir
    if img_path is None or not os.path.exists(img_path):
        import glob
        candidates = glob.glob(
            os.path.join(top3_out_dir, "annotated", f"frame_*_track.jpg")
        )
        if candidates:
            img_path = candidates[best_sample_idx] if best_sample_idx < len(candidates) else candidates[0]

    if img_path is None or not os.path.exists(img_path):
        print("  ⚠  Could not find source image for representative frame.")
        return ""

    frame  = cv2.imread(img_path)
    if frame is None:
        print(f"  ⚠  cv2.imread failed: {img_path}")
        return ""

    # Get filtered objects for this keyframe
    kf_json = os.path.join(
        top3_out_dir, "keyframes",
        os.path.basename(best_kf_json_path)
    )
    objs = []
    if os.path.exists(kf_json):
        objs = (_load(kf_json).get("scene_summary", {})
                               .get("detected_objects", []))

    canvas   = _draw_top3_boxes(frame, objs, slot_map)
    out_img  = os.path.join(top3_out_dir, "representative_frame.jpg")
    cv2.imwrite(out_img, canvas)

    # Metadata
    meta = {
        "sample_idx":        best_sample_idx,
        "keyframe_json":     os.path.basename(best_kf_json_path),
        "source_image":      img_path,
        "objects_visible":   best_stats.get("objects_visible", 0),
        "combined_area":     best_stats.get("combined_area",   0),
        "top3_identities":   top3_identities,
    }
    _dump(os.path.join(top3_out_dir, "representative_frame.json"), meta)

    print(f"  ✅ Representative frame saved: {out_img}")
    return out_img


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Qwen per-keyframe (3-object prompt only)
# ════════════════════════════════════════════════════════════════════════════

def run_qwen_top3(
    top3_identities: List[Dict],
    top3_out_dir:    str,
    dataroot:        str,
    nusc,
    stable_fields:   dict,
    video_fps:       float = 2.0,
):
    """
    Run Qwen analyse_frame on each keyframe in top3_out_dir/keyframes/,
    writing output to top3_out_dir/scene/output_sec_N.json.
    Uses a much shorter prompt — only 3 objects, no GT hint list.
    """
    # Import lazily to avoid circular deps
    from scene_annotator import (
        analyse_frame,
        _yolo_data_from_frame_result,
        FIELD_OPTIONS,
    )
    import glob, re

    kf_files = sorted(glob.glob(
        os.path.join(top3_out_dir, "keyframes", "keyframe_*.json")
    ))
    kf_map_path = os.path.join(top3_out_dir, "keyframe_map.json")
    kf_map      = _load(kf_map_path) if os.path.exists(kf_map_path) else []
    kf_map_by_idx = {k["sample_idx"]: k for k in kf_map}

    slot_labels = {e["identity"]: e["slot_label"] for e in top3_identities}

    print(f"\n  Running Qwen on {len(kf_files)} top3 keyframes…")
    for kf_path in kf_files:
        m = re.search(r"keyframe_(\d+)\.json", os.path.basename(kf_path))
        if not m:
            continue
        sample_idx = int(m.group(1))
        out_json   = os.path.join(top3_out_dir, "scene", f"output_sec_{sample_idx}.json")

        if os.path.exists(out_json):
            print(f"    Skipping keyframe {sample_idx} (already done)")
            continue

        # Load image
        frame = None
        kf_entry = kf_map_by_idx.get(sample_idx)
        if kf_entry:
            try:
                sample  = nusc.get("sample", kf_entry["sample_token"])
                sd      = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                img_path = os.path.join(dataroot, sd["filename"])
                frame   = cv2.imread(img_path)
            except Exception as e:
                print(f"    ⚠  Image load failed for sample {sample_idx}: {e}")

        if frame is None:
            continue

        yolo_data = _yolo_data_from_frame_result(_load(kf_path))

        # Inject slot labels into detections so Qwen references A/B/C
        for det in yolo_data.get("detections", []):
            ident = _identity_of(det)
            if ident and ident in slot_labels:
                det["slot_label"] = slot_labels[ident]

        try:
            analyse_frame(
                frame,
                str(sample_idx),
                out_json_path = out_json,
                out_img_path  = os.path.join(
                    top3_out_dir, "scene", f"frame_sec_{sample_idx}.jpg"
                ),
                yolo_data     = yolo_data,
                frame_idx     = kf_entry.get("frame_idx", 0) if kf_entry else 0,
                stable_fields = stable_fields,
                video_fps     = video_fps,
            )
        except Exception as e:
            print(f"    ⚠  analyse_frame failed (keyframe {sample_idx}): {e}")

    print("  ✅ Qwen top3 analysis complete.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — flat cumulative for top-3 only
# ════════════════════════════════════════════════════════════════════════════

def generate_top3_cumulative(
    top3_identities: List[Dict],
    top3_out_dir:    str,
    stable_fields:   dict,
) -> dict:
    """
    Wrapper around scene_annotator.generate_flat_cumulative() pointing at
    top3_out_dir/{scene,summaries,frames} for I/O.
    """
    import glob

    scene_dir   = os.path.join(top3_out_dir, "scene")
    summary_dir = os.path.join(top3_out_dir, "summaries")
    frames_dir  = os.path.join(top3_out_dir, "frames")
    os.makedirs(summary_dir, exist_ok=True)

    scene_files = sorted(glob.glob(os.path.join(scene_dir, "output_sec_*.json")))
    num_seconds = len(scene_files)
    if num_seconds == 0:
        print("  ⚠  No top3 scene files found — skipping cumulative.")
        return {}

    # generate_flat_cumulative numbers seconds 0..num_seconds-1 contiguously,
    # but top3 keyframe indices are sparse (whichever sec's the 3 objects show
    # up in). Use max-index+1 as the upper bound so every existing file is read.
    max_sec = 0
    for fp in scene_files:
        try:
            n = int(os.path.basename(fp).replace("output_sec_", "").replace(".json", ""))
            if n > max_sec:
                max_sec = n
        except ValueError:
            pass
    span = max_sec + 1

    from scene_annotator import generate_flat_cumulative
    result = generate_flat_cumulative(
        num_seconds   = span,
        stable_fields = stable_fields,
        scene_dir     = scene_dir,
        summaries_dir = summary_dir,
        frames_dir    = frames_dir,
    )

    out_path = os.path.join(summary_dir, "output_cumulative_top3.json")
    _dump(out_path, result)
    print(f"  ✅ Top3 cumulative saved: {out_path}")
    return result


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run_top3_phase(
    full_out_dir:  str,
    top3_out_dir:  str,
    nusc,
    dataroot:      str,
    stable_fields: dict,
    video_fps:     float = 2.0,
    no_qwen:       bool = False,
    override_ids:  list = None,
    override_sec:  int  = None,
) -> dict:
    """
    Run the complete Phase 2 pipeline.

    Parameters
    ----------
    full_out_dir  : path written by Phase 1 (process_scene_sweeps), e.g. "output/full"
    top3_out_dir  : output path for Phase 2, e.g. "output/top3"
    nusc          : open NuScenes instance (already loaded by Phase 1)
    dataroot      : NuScenes data root, e.g. "data/v1.0-mini"
    stable_fields : {"environment": str, "lighting": str, "nuscenes_description": str}
    video_fps     : sweep fps (default 2.0 for keyframe rate)

    Returns
    -------
    dict with keys:
        top3_identities       : list of 3 locked identity dicts
        representative_frame  : path to the best annotated frame JPG
        cumulative            : the flat cumulative summary dict
    """
    t0 = time.perf_counter()

    os.makedirs(top3_out_dir, exist_ok=True)

    # ── 1. Lock top-3 identities ──────────────────────────────────────────
    print("\n" + "="*55)
    print("  TOP3 PHASE — Step 1: locking top-3 identities")
    print("="*55)
    if override_ids:
        print(f"  Manual override: {override_ids}")
        top3_identities = _build_identities_from_override(
            override_ids, full_out_dir
        )
    else:
        top3_identities = lock_top3_identities(full_out_dir)
    _dump(os.path.join(top3_out_dir, "top3_identities.json"), top3_identities)

    # ── 2. Pick representative keyframe ───────────────────────────────────
    print("\n  Step 2: selecting representative keyframe")
    if override_sec is not None:
        print(f"  Manual override: sec {override_sec}")
        best_idx, best_kf_path, best_stats = _force_representative(
            override_sec, top3_identities, full_out_dir
        )
    else:
        best_idx, best_kf_path, best_stats = pick_representative_keyframe(
            top3_identities, full_out_dir
        )

    # ── 3. Filter all JSONs + redraw images ───────────────────────────────
    print("\n  Step 3: filtering output to top-3 objects")
    filter_full_to_top3(top3_identities, full_out_dir, top3_out_dir, dataroot, nusc=nusc)

    # ── 4. Write representative frame ─────────────────────────────────────
    print("\n  Step 4: writing representative frame")
    rep_frame_path = write_representative_frame(
        best_idx, best_kf_path, best_stats,
        top3_identities, top3_out_dir, dataroot, nusc
    )

    if no_qwen:
        print("\n  Steps 5 & 6 SKIPPED (no_qwen): YOLO + boxes only.")
        cumulative = {}
    else:
        # ── 5. Qwen per-keyframe (3-object only) ──────────────────────────────
        print("\n  Step 5: Qwen analysis (top-3 objects only)")
        run_qwen_top3(
            top3_identities, top3_out_dir,
            dataroot, nusc, stable_fields, video_fps
        )

        # ── 6. Flat cumulative ────────────────────────────────────────────────
        print("\n  Step 6: flat cumulative summary")
        cumulative = generate_top3_cumulative(
            top3_identities, top3_out_dir, stable_fields
        )

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*55}")
    print(f"  TOP3 PHASE complete in {elapsed:.1f}s")
    print(f"  Output: {top3_out_dir}/")
    print(f"  Representative frame: {rep_frame_path}")
    print(f"{'='*55}\n")

    return {
        "top3_identities":      top3_identities,
        "representative_frame": rep_frame_path,
        "cumulative":           cumulative,
    }