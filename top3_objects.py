"""
top3_objects.py
---------------
Standalone helper: given a keyframe JSON (output of run_yolo_and_track) and
the corresponding image, returns the top-3 most prominent objects and writes
an annotated image showing only those 3 boxes.

Selection priority
------------------
1. GT-confirmed objects (gt_sourced=True)  ranked by bounding-box area (largest first)
2. If fewer than 3 GT objects exist, fill remaining slots with the highest-confidence
   YOLO-only detections (also ranked by area).

"GT-confirmed" = LiDAR-validated identity → speed / direction / action are
trustworthy, so the survey question can legitimately ask about movement.

Usage (standalone)
------------------
    python top3_objects.py \
        --json  output/keyframes/keyframe_0010.json \
        --image data/v1.0-mini/samples/CAM_FRONT/xxxx.jpg \
        --out   output/top3/frame_0010_top3.jpg

Imported by scene_annotator_simple.py
--------------------------------------
    from top3_objects import pick_top3, draw_top3
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── colour palette (one per slot, colour-blind friendly) ────────────────────
_SLOT_COLORS = [
    (0,   200, 255),   # slot 1 — cyan
    (0,   230, 100),   # slot 2 — green
    (255, 160,   0),   # slot 3 — amber
]
_SLOT_LABELS = ["Object 1", "Object 2", "Object 3"]


def _bbox_area(bb: Dict) -> int:
    return max(0, (bb.get("x2", 0) - bb.get("x1", 0)) *
                  (bb.get("y2", 0) - bb.get("y1", 0)))


def pick_top3(frame_data: Dict) -> List[Dict]:
    """
    Return up to 3 objects from a run_yolo_and_track frame_data dict.

    Each returned entry is the original detection dict augmented with:
        slot        : 0 / 1 / 2  (display order)
        slot_label  : "Object 1" / "Object 2" / "Object 3"
        slot_color  : (B, G, R) tuple for drawing
    """
    detections = (frame_data.get("scene_summary", {})
                             .get("detected_objects", []))

    # Split GT-confirmed vs YOLO-only
    gt_objs   = [d for d in detections if d.get("gt_sourced", False)]
    yolo_objs = [d for d in detections if not d.get("gt_sourced", False)]

    # Sort each group by bbox area descending
    gt_objs.sort(key=lambda d: _bbox_area(d.get("bounding_box", {})),
                 reverse=True)
    yolo_objs.sort(key=lambda d: (
        # secondary sort: confidence then area
        -d.get("confidence", 0),
        -_bbox_area(d.get("bounding_box", {}))
    ))

    # Fill up to 3 slots: GT first, then YOLO fallback
    selected = (gt_objs[:3] + yolo_objs)[:3]

    # Annotate each with its slot metadata
    result = []
    for slot, obj in enumerate(selected):
        entry = dict(obj)           # shallow copy — don't mutate original
        entry["slot"]       = slot
        entry["slot_label"] = _SLOT_LABELS[slot]
        entry["slot_color"] = _SLOT_COLORS[slot]
        result.append(entry)

    return result


def draw_top3(
    frame_bgr: np.ndarray,
    top3: List[Dict],
    box_thickness: int = 3,
    font_scale: float = 0.65,
) -> np.ndarray:
    """
    Draw ONLY the top-3 bounding boxes on a copy of frame_bgr.
    Each box is drawn with its slot colour and labelled "Object 1/2/3".
    Motion info (action + direction) is shown below the label when available.
    Returns the annotated frame.
    """
    canvas = frame_bgr.copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for obj in top3:
        bb = obj.get("bounding_box", {})
        if not bb:
            continue
        x1, y1 = int(bb.get("x1", 0)), int(bb.get("y1", 0))
        x2, y2 = int(bb.get("x2", 0)), int(bb.get("y2", 0))
        if x2 <= x1 or y2 <= y1:
            continue

        color = obj["slot_color"]
        label = obj["slot_label"]

        # Optional motion line (only for GT-confirmed objects)
        motion_str = ""
        if obj.get("gt_sourced"):
            action    = obj.get("action", "")
            direction = obj.get("direction", "")
            parts = [p for p in (action, direction)
                     if p and p not in ("unknown", "stationary", "static", "")]
            if parts:
                motion_str = " · ".join(parts)

        # Draw box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

        # Draw corner accent marks (makes box look cleaner)
        mark = 18
        for (px, py), (dx, dy) in [
            ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
            ((x1, y2), (1,-1)), ((x2, y2), (-1,-1))
        ]:
            cv2.line(canvas, (px, py), (px + dx * mark, py), color, box_thickness + 1)
            cv2.line(canvas, (px, py), (px, py + dy * mark), color, box_thickness + 1)

        # Label background + text
        lines = [label]
        if motion_str:
            lines.append(motion_str)

        line_h   = int(font_scale * 28)
        pad      = 6
        max_w    = max(
            cv2.getTextSize(l, font, font_scale, 1)[0][0] for l in lines
        )
        box_w    = max_w + pad * 2
        box_h    = line_h * len(lines) + pad * 2
        lx       = x1
        ly       = y1 - box_h - 4 if y1 > box_h + 8 else y2 + 4

        # Semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (lx, ly), (lx + box_w, ly + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # Text lines
        for i, line in enumerate(lines):
            ty = ly + pad + (i + 1) * line_h - 4
            cv2.putText(canvas, line, (lx + pad, ty),
                        font, font_scale, color, 1, cv2.LINE_AA)

    return canvas


def top3_from_files(
    json_path: str,
    image_path: str,
    out_path: Optional[str] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Convenience wrapper: load JSON + image, pick top3, draw, optionally save.
    Returns (top3_list, annotated_frame).
    """
    with open(json_path) as f:
        frame_data = json.load(f)

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    top3   = pick_top3(frame_data)
    canvas = draw_top3(frame, top3)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, canvas)
        print(f"Saved: {out_path}")

    return top3, canvas


def top3_summary_for_prompt(top3: List[Dict]) -> str:
    """
    Returns a short human-readable description of the 3 objects
    to embed in the survey question text shown to participants.

    Example output:
        "Object 1 (white car, moving right), Object 2 (pedestrian, walking),
         Object 3 (blue truck, parked)"
    """
    parts = []
    for obj in top3:
        label = obj["slot_label"]
        otype = obj.get("object_type") or obj.get("type", "object")

        # Color (vehicles/cyclists only)
        color = obj.get("color", "")
        color_str = f"{color} " if color and color not in ("unknown", "") else ""

        # Motion (GT-confirmed only)
        action    = obj.get("action", "")
        direction = obj.get("direction", "")
        motion_parts = [p for p in (action, direction)
                        if p and p not in ("unknown", "stationary", "static", "")]
        motion_str = ", ".join(motion_parts) if motion_parts else "stationary"

        parts.append(f"{label} ({color_str}{otype}, {motion_str})")

    return "; ".join(parts)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pick top-3 GT-confirmed objects and draw annotated frame."
    )
    parser.add_argument("--json",  required=True, help="Path to keyframe JSON")
    parser.add_argument("--image", required=True, help="Path to frame JPG/PNG")
    parser.add_argument("--out",   default=None,  help="Output image path")
    args = parser.parse_args()

    top3, canvas = top3_from_files(args.json, args.image, args.out)

    print(f"\nTop-3 objects selected ({len(top3)}):")
    for obj in top3:
        bb    = obj.get("bounding_box", {})
        area  = _bbox_area(bb)
        gts   = "GT ✓" if obj.get("gt_sourced") else "YOLO"
        color = obj.get("color", "")
        color_str = f"{color} " if color and color != "unknown" else ""
        print(f"  [{obj['slot_label']}] {color_str}{obj.get('object_type') or obj.get('type')} "
              f"| area={area}px² | {gts} "
              f"| action={obj.get('action','?')} "
              f"| direction={obj.get('direction','?')}")

    print(f"\nSurvey question context:")
    print(f"  {top3_summary_for_prompt(top3)}")

    if args.out is None:
        out_path = args.image.replace(".jpg", "_top3.jpg").replace(".png", "_top3.png")
        cv2.imwrite(out_path, canvas)
        print(f"\nAnnotated frame saved to: {out_path}")