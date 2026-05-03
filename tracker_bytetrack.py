"""
ByteTracker Integration (Supervision Library)
==============================================
Robust multi-object tracking with ByteTrack using supervision library.
Handles occlusions, re-identification, and temporal consistency across frames.

Reads per-second per-frame JSONs, outputs deterministic tracks with ByteTracker.
"""
import json
import os
from typing import List, Dict
import math
import numpy as np
import cv2

try:
    import supervision as sv
    from supervision.tracker.byte_tracker.core import ByteTrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("⚠  ByteTracker not available. Install: pip install supervision")


class Track:
    """Track object to store temporal data."""
    def __init__(self, track_id: int, obj_type: str, first_frame: int, bbox: Dict, detection_id=None):
        self.track_id = track_id
        self.object_type = obj_type
        self.start = first_frame
        self.end = first_frame
        # frames now include optional detection_id per observation
        frame_entry = {"second": first_frame, "bbox": bbox}
        if detection_id is not None:
            frame_entry["detection_id"] = detection_id
        self.frames = [frame_entry]
        self.last_bbox = bbox
        self.detection_ids = [detection_id] if detection_id is not None else []

    def update(self, frame_idx: int, bbox: Dict, detection_id=None):
        entry = {"second": frame_idx, "bbox": bbox}
        if detection_id is not None:
            entry["detection_id"] = detection_id
            self.detection_ids.append(detection_id)
        self.frames.append(entry)
        self.last_bbox = bbox
        self.end = frame_idx


def bbox_iou(bbox1: Dict, bbox2: Dict) -> float:
    """Compute IoU between two bboxes."""
    x1 = max(bbox1["x1"], bbox2["x1"])
    y1 = max(bbox1["y1"], bbox2["y1"])
    x2 = min(bbox1["x2"], bbox2["x2"])
    y2 = min(bbox1["y2"], bbox2["y2"])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    area1 = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
    area2 = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def run_bytetrack(
    num_seconds: int = None,
    num_frames: int = None,
    frame_rate: int = 1,
    per_second_json_dir: str = "output",
    out_path: str = "output/tracks.json",
    track_thresh: float = 0.5,
    track_buffer: int = 15,
    match_thresh: float = 0.5,
    min_box_area: float = 10,
    video_path: str = None,
):
    """
    Run ByteTracker on per-second JSONs to produce deterministic tracks.

    Args:
        num_seconds: Total number of frames to process
        per_second_json_dir: Directory containing output_sec_{sec}.json files
        out_path: Path to save output tracks.json
        track_thresh: Detection confidence threshold for ByteTracker
        track_buffer: Frames to keep inactive tracks (handle temporary occlusions)
        match_thresh: Similarity threshold for matching
        min_box_area: Minimum bounding box area to consider
    """
    if not BYTETRACK_AVAILABLE:
        print("⚠  ByteTracker not available; falling back to simple IoU tracker")
        return run_simple_tracker_fallback(
            num_seconds,
            per_second_json_dir=per_second_json_dir,
            out_path=out_path
        )
    print(f"🔍 ByteTracker: track_activation_threshold={track_thresh}, lost_track_buffer={track_buffer}, minimum_matching_threshold={match_thresh}, frame_rate={frame_rate}")

    # Initialize ByteTracker (supervision API) with correct parameter names
    byte_tracker = ByteTrack(
        track_activation_threshold=track_thresh,
        lost_track_buffer=track_buffer,
        minimum_matching_threshold=match_thresh,
        frame_rate=frame_rate
    )

    tracks_dict = {}  # track_id -> Track object

    # Determine number of frames to iterate
    if num_frames is None:
        if num_seconds is None:
            raise ValueError("Either num_seconds or num_frames must be provided")
        num_frames = int(num_seconds * frame_rate)

    # If a video path is provided, open it to compute ego-motion per frame
    cap = None
    prev_gray = None
    cam_translations = {}  # frame_idx -> (dx, dy)
    if video_path:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  ! Could not open video: {video_path}")
                cap = None
        except Exception as e:
            print(f"  ! Error opening video for ego-motion: {e}")
            cap = None

    for frame_idx in range(num_frames):
        # Try frame-level JSON (high-freq detections)
        p = os.path.join(per_second_json_dir, f"frame_{frame_idx:06d}.json")
        # Fallback to per-second JSON mapped by frame_rate
        if not os.path.exists(p):
            sec = frame_idx // frame_rate
            p = os.path.join(per_second_json_dir, f"output_sec_{sec}.json")
        if not os.path.exists(p):
            print(f"  ⊘ {p} not found, skipping")
            continue

        with open(p) as f:
            d = json.load(f)

        ss = d.get("scene_summary", {})
        dets_raw = ss.get("detected_objects", []) or []

        # Build sv.Detections object from per-second JSON
        boxes = []
        confidences = []
        class_ids = []
        object_types = []

        for det in dets_raw:
            bb = det.get("bounding_box", {}) or {}
            if not all(k in bb for k in ("x1", "y1", "x2", "y2")):
                continue

            area = (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"])
            if area < min_box_area:
                continue

            # Use model confidence if available, else assume 0.8
            conf = float(det.get("confidence", 0.8)) if det.get("confidence") else 0.8
            conf = max(0.0, min(1.0, conf))  # Clamp to [0, 1]

            boxes.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]])
            confidences.append(conf)
            class_ids.append(0)  # Default class
            object_types.append(det.get("object_type", "unknown"))
            # preserve original detection id if present (support multiple possible keys)
            det_id = det.get("id", det.get("object_id", det.get("detection_id", None)))
            det_ids = locals().get('det_ids', [])
            det_ids.append(det_id)
            locals()['det_ids'] = det_ids
            # preserve original detection id if present (support multiple possible keys)
            det_id = det.get("id", det.get("object_id", det.get("detection_id", None)))
            det_ids = locals().get('det_ids', [])
            det_ids.append(det_id)
            # write back to locals so fallback code can access
            locals()['det_ids'] = det_ids

        # Fallback: if no detections in per-second JSON, try raw YOLO file
        if not boxes:
            yolo_raw_path = os.path.join(per_second_json_dir, f"yolo_raw_{sec}.json")
            if os.path.exists(yolo_raw_path):
                with open(yolo_raw_path) as yf:
                    yraw = json.load(yf)
                for det in yraw.get("detections", []) or []:
                    bb = det.get("bounding_box", {}) or {}
                    if not all(k in bb for k in ("x1", "y1", "x2", "y2")):
                        continue

                    area = (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"])
                    if area < min_box_area:
                        continue

                    conf = float(det.get("confidence", 0.8)) if det.get("confidence") else 0.8
                    conf = max(0.0, min(1.0, conf))

                    boxes.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]])
                    confidences.append(conf)
                    class_ids.append(0)
                    object_types.append(det.get("type", "unknown"))
                    det_id = det.get("id", det.get("object_id", det.get("detection_id", None)))
                    det_ids = locals().get('det_ids', [])
                    det_ids.append(det_id)
                    locals()['det_ids'] = det_ids

        # Create sv.Detections object
        if boxes:
            detections = sv.Detections(
                xyxy=np.array(boxes, dtype=np.float32),
                confidence=np.array(confidences, dtype=np.float32),
                class_id=np.array(class_ids, dtype=np.int32),
            )
            # Store object types as custom data
            detections.data["object_type"] = object_types
            # Store original detection ids (if we collected any)
            if 'det_ids' in locals():
                detections.data['detection_id'] = locals().get('det_ids', [])
        else:
            detections = sv.Detections.empty()

        # Estimate ego-motion for this frame if video capture available
        dx, dy = 0.0, 0.0
        if cap is not None:
            # read current frame from video
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, img = cap.read()
                if ok:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        # compute sparse optical flow and take median translation
                        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=7)
                        if pts is not None:
                            next_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21,21))
                            st = st.reshape(-1)
                            prev_pts = pts.reshape(-1,2)[st==1]
                            next_pts = next_pts.reshape(-1,2)[st==1]
                            if len(prev_pts) >= 5:
                                flows = next_pts - prev_pts
                                dx, dy = float(np.median(flows[:,0])), float(np.median(flows[:,1]))
                    prev_gray = gray
            except Exception:
                dx, dy = 0.0, 0.0
        cam_translations[frame_idx] = (dx, dy)

        # Update tracker with detections for this frame
        detections = byte_tracker.update_with_detections(detections)

        # Extract tracked objects
        if len(detections) > 0:
            for i, (box, track_id, conf) in enumerate(zip(
                detections.xyxy,
                detections.tracker_id,
                detections.confidence
            )):
                track_id = int(track_id)
                bbox = {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3])
                }

                # Get object type from custom data
                obj_type = "unknown"
                if hasattr(detections, 'data') and "object_type" in detections.data:
                    types_list = detections.data["object_type"]
                    if isinstance(types_list, list) and i < len(types_list):
                        obj_type = types_list[i]

                # Get original detection id (if stored)
                orig_det_id = None
                if hasattr(detections, 'data') and "detection_id" in detections.data:
                    ids_list = detections.data["detection_id"]
                    if isinstance(ids_list, list) and i < len(ids_list):
                        orig_det_id = ids_list[i]

                # Create or update track (frame index -> second mapping)
                frame_for_track = frame_idx
                sec_for_track = frame_for_track // frame_rate
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = Track(track_id, obj_type, sec_for_track, bbox, detection_id=orig_det_id)
                else:
                    tracks_dict[track_id].update(sec_for_track, bbox, detection_id=orig_det_id)
        print(f"  Frame {frame_idx}: {len(detections) if len(detections) > 0 else 0} tracked objects, {len(boxes)} detections")

    # Post-process: compute average speeds
    out_tracks = []
    for track_id, track in tracks_dict.items():
        avg_speed = None
        if len(track.frames) > 1:
            dsum = 0.0
            count = 0
            for a, b in zip(track.frames[:-1], track.frames[1:]):
                ax = (a["bbox"]["x1"] + a["bbox"]["x2"]) / 2
                ay = (a["bbox"]["y1"] + a["bbox"]["y2"]) / 2
                bx = (b["bbox"]["x1"] + b["bbox"]["x2"]) / 2
                by = (b["bbox"]["y1"] + b["bbox"]["y2"]) / 2
                # compute raw object displacement
                obj_dx = bx - ax
                obj_dy = by - ay

                # compute cumulative camera translation between the two second timestamps
                a_sec = int(a.get("second", 0))
                b_sec = int(b.get("second", a_sec))
                cam_dx = 0.0
                cam_dy = 0.0
                # sum per-frame camera translations if available
                start_idx = a_sec * frame_rate
                end_idx = b_sec * frame_rate
                for fi in range(start_idx, end_idx + 1):
                    t = cam_translations.get(fi)
                    if t:
                        cam_dx += t[0]
                        cam_dy += t[1]

                rel_dx = obj_dx - cam_dx
                rel_dy = obj_dy - cam_dy
                dist = math.hypot(rel_dx, rel_dy)
                dsum += dist
                count += 1
            avg_speed = dsum / count if count else None

        out_tracks.append({
            "track_id": track.track_id,
            "object_type": track.object_type,
            "start_second": track.start,
            "end_second": track.end,
            "duration_seconds": track.end - track.start + 1,
            "frames": track.frames,
            "first_detection_id": track.detection_ids[0] if getattr(track, 'detection_ids', None) else None,
            "avg_speed_px_per_second": avg_speed,
            "total_frames_seen": len(track.frames),
        })

    # Sort by track_id
    out_tracks.sort(key=lambda t: t["track_id"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "tracks": out_tracks,
            "num_seconds": num_seconds,
            "tracker": "bytetrack",
            "num_tracks": len(out_tracks),
        }, f, indent=2)

    print(f"✅ Saved ByteTracker results: {out_path} ({len(out_tracks)} tracks)")
    return out_path


def run_simple_tracker_fallback(
    num_seconds: int,
    per_second_json_dir: str = "output",
    out_path: str = "output/tracks.json",
    iou_threshold: float = 0.3
):
    """
    Simple IoU-based tracker (fallback if ByteTracker not available).
    """
    print(f"⚠  Using simple IoU tracker (threshold={iou_threshold})")
    try:
        from tracker_simple import run_simple_tracker
        return run_simple_tracker(num_seconds, per_second_json_dir, out_path, iou_threshold)
    except ImportError:
        print("⚠  tracker_simple not available either. Skipping tracking.")
        return None


if __name__ == "__main__":
    # Test: process output/ directory
    # default: treat as 10 seconds at 1 fps
    # when run standalone, try to use default video if present
    default_video = "assets/3.mp4"
    if os.path.exists(default_video):
        run_bytetrack(num_seconds=10, frame_rate=1, video_path=default_video)
    else:
        run_bytetrack(num_seconds=10, frame_rate=1)
