"""
yolo_bytetrack.py
=================
Integrated YOLO detection + ByteTracker with:
  - Pixel-based HSV color extraction per bounding box
  - Global Motion Compensation (GMC) via Farneback optical flow
  - Live per-frame ByteTracker (not post-hoc from JSONs)
  - Track color propagation (consistent track_id → color)
  - Motion calculations: velocity, direction, classification
  - All frame/tracking params as top-level config variables
"""

import json
import math
import os
import cv2
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# ── Configuration (change these freely) ──────────────────────────────────────
TARGET_FILE              = "assets/7.mp4"   # ← change to any video or image path
TRACKER_FRAME_RATE       = 0      # frames per second fed to tracker (0 = auto: video_fps − 1, min 5)
MIN_DETECTION_AREA       = 600    # min pixel area for non-traffic-light detections
TRACK_ACTIVATION_THRESH  = 0.25   # ByteTrack: min confidence to activate track
TRACK_BUFFER_FRAMES      = 30     # ByteTrack: frames to keep a lost track alive
MATCH_THRESH             = 0.8    # ByteTrack: min IoU/similarity to match tracks
COLOR_LOCK_FRAMES        = 5      # observations before locking a track's color
MOTION_STATIONARY_PX     = 5.0    # px/frame below which = "stationary"
MOTION_SLOW_PX           = 20.0   # px/frame below which = "slow"
MOTION_FAST_PX           = 60.0   # px/frame above which = "fast"
GMC_ENABLED              = True   # toggle Global Motion Compensation
GMC_SCALE                = 0.25   # downscale factor for GMC computation (speed)
YOLO_MODEL_PATH          = "yolo11l.pt"
YOLO_CONF_THRESH         = 0.25
YOLO_INPUT_MAX_DIM       = 1024   # resize longest edge to this before inference
VIDEO_CHUNK_SECONDS      = 5      # seconds per L1 cumulative chunk (re-exported)

# ── YOLO class → semantic type mapping ───────────────────────────────────────
VEHICLE_CLASSES    = {"car", "van", "motorcycle", "bus", "truck"}
CYCLIST_CLASSES    = {"bicycle"}
PEDESTRIAN_CLASSES = {"person"}
TRAFFIC_LIGHT_CLS  = {"traffic light"}


def _semantic_type(class_name: str) -> str:
    name = class_name.lower()
    if name in VEHICLE_CLASSES:       return name
    if name in CYCLIST_CLASSES:       return "cyclist"
    if name in PEDESTRIAN_CLASSES:    return "pedestrian"
    if name in TRAFFIC_LIGHT_CLS:     return "traffic_light"
    return name


def _size_label(bbox: Dict, frame_w: int, frame_h: int) -> str:
    """
    Classify object size relative to the frame area.
    Thresholds tuned for typical traffic / dashcam footage:
      < 1.5%  of frame  → small   (distant vehicle)
      1.5–8%  of frame  → medium  (typical mid-range vehicle)
      > 8%    of frame  → large   (close-up / bus / truck)
    """
    obj_area   = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
    frame_area = frame_w * frame_h
    if frame_area == 0:
        return "unknown"
    ratio = obj_area / frame_area
    if ratio < 0.015:
        return "small"
    if ratio < 0.08:
        return "medium"
    return "large"


# ── HSV pixel-based color extraction ─────────────────────────────────────────
_HSV_PALETTE = [
    ("red",    (0,   60,  50),  (10,  255, 255)),
    ("orange", (11,  60,  50),  (25,  255, 255)),
    ("yellow", (26,  60,  50),  (35,  255, 255)),
    ("green",  (36,  40,  40),  (85,  255, 255)),
    ("blue",   (86,  60,  50),  (130, 255, 255)),
    ("purple", (131, 40,  40),  (155, 255, 255)),
    ("red2",   (156, 60,  50),  (180, 255, 255)),  # red wraps in HSV
    ("white",  (0,   0,   200), (180, 30,  255)),
    ("black",  (0,   0,   0),   (180, 255, 50)),
    ("silver", (0,   0,   100), (180, 30,  200)),
    ("gray",   (0,   0,   51),  (180, 25,  199)),
    ("brown",  (10,  40,  40),  (20,  200, 150)),
]


def extract_dominant_color(frame_bgr: np.ndarray, bbox: Dict) -> str:
    """Extract dominant color from a bounding-box region using HSV histogram."""
    x1, y1 = max(0, bbox["x1"]), max(0, bbox["y1"])
    x2, y2 = min(frame_bgr.shape[1], bbox["x2"]), min(frame_bgr.shape[0], bbox["y2"])
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    # Use central 60% of the box to avoid background bleeding
    pw, ph = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
    roi = frame_bgr[y1 + ph:y2 - ph, x1 + pw:x2 - pw]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    counts = {}
    for name, lo, hi in _HSV_PALETTE:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        counts[name] = int(np.sum(mask > 0))
    # merge red + red2
    counts["red"] = counts.pop("red", 0) + counts.pop("red2", 0)
    
    # Unconditionally return the best matching color
    best = max(counts, key=counts.get)
    return best


# ── YOLO singleton ────────────────────────────────────────────────────────────
_yolo_model = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
            _yolo_model = YOLO(YOLO_MODEL_PATH)
        except ImportError:
            print("⚠  ultralytics not installed. YOLO unavailable.")
    return _yolo_model


# ── ByteTracker singleton (live, per-video) ───────────────────────────────────
_byte_tracker = None
_tracker_frame_rate = TRACKER_FRAME_RATE


def reset_tracker(frame_rate: int = TRACKER_FRAME_RATE):
    """Reset/init the ByteTracker — call once per video."""
    global _byte_tracker, _tracker_frame_rate
    _tracker_frame_rate = frame_rate
    try:
        from supervision.tracker.byte_tracker.core import ByteTrack
        _byte_tracker = ByteTrack(
            track_activation_threshold=TRACK_ACTIVATION_THRESH,
            lost_track_buffer=TRACK_BUFFER_FRAMES,
            minimum_matching_threshold=MATCH_THRESH,
            frame_rate=frame_rate,
        )
        print(f"✅ ByteTracker initialised (frame_rate={frame_rate})")
    except ImportError:
        _byte_tracker = None
        print("⚠  supervision not installed. ByteTracker unavailable.")


def get_tracker():
    global _byte_tracker
    if _byte_tracker is None:
        reset_tracker()
    return _byte_tracker


# ── Track state: color propagation + motion history ──────────────────────────
class TrackState:
    def __init__(self, track_id: int, obj_type: str):
        self.track_id   = track_id
        self.obj_type   = obj_type
        self.color_votes: List[str] = []
        self.locked_color: Optional[str] = None
        self.positions: List[Tuple[float, float]] = []  # (cx, cy) per frame
        self.frame_indices: List[int] = []

    def add_observation(self, frame_idx: int, bbox: Dict, color_candidate: str):
        cx = (bbox["x1"] + bbox["x2"]) / 2.0
        cy = (bbox["y1"] + bbox["y2"]) / 2.0
        self.positions.append((cx, cy))
        self.frame_indices.append(frame_idx)
        if self.locked_color is None:
            if color_candidate not in ("unknown", "mixed"):
                self.color_votes.append(color_candidate)
            if len(self.color_votes) >= COLOR_LOCK_FRAMES:
                self.locked_color = Counter(self.color_votes).most_common(1)[0][0]

    @property
    def color(self) -> str:
        if self.locked_color:
            return self.locked_color
        if self.color_votes:
            return Counter(self.color_votes).most_common(1)[0][0]
        return "unknown"

    def velocity_px_per_frame(self) -> Optional[float]:
        if len(self.positions) < 2:
            return None
        dists = [
            math.hypot(self.positions[i][0] - self.positions[i-1][0],
                       self.positions[i][1] - self.positions[i-1][1])
            for i in range(1, len(self.positions))
        ]
        return sum(dists) / len(dists)

    def direction_label(self) -> str:
        if len(self.positions) < 2:
            return "stationary"
        dx = self.positions[-1][0] - self.positions[0][0]
        dy = self.positions[-1][1] - self.positions[0][1]
        dist = math.hypot(dx, dy)
        if dist < MOTION_STATIONARY_PX:
            return "stationary"
        angle = math.degrees(math.atan2(-dy, dx))  # y-flip for screen coords
        if   -45  <= angle <  45:  return "moving right"
        elif  45  <= angle < 135:  return "moving up"
        elif -135 <= angle < -45:  return "moving down"
        else:                      return "moving left"

    def speed_label(self) -> str:
        v = self.velocity_px_per_frame()
        if v is None:                     return "unknown"
        if v < MOTION_STATIONARY_PX:     return "stationary"
        if v < MOTION_SLOW_PX:           return "slow"
        if v < MOTION_FAST_PX:           return "moving"
        return "fast"


_track_states: Dict[int, TrackState] = {}


def _get_or_create_state(track_id: int, obj_type: str) -> TrackState:
    if track_id not in _track_states:
        _track_states[track_id] = TrackState(track_id, obj_type)
    return _track_states[track_id]


def reset_track_states():
    global _track_states
    _track_states = {}


# ── GMC — Global Motion Compensation ─────────────────────────────────────────
_prev_gray_small: Optional[np.ndarray] = None
_cumulative_tx: float = 0.0
_cumulative_ty: float = 0.0


def reset_gmc():
    global _prev_gray_small, _cumulative_tx, _cumulative_ty
    _prev_gray_small = None
    _cumulative_tx = 0.0
    _cumulative_ty = 0.0


def compute_gmc(frame_bgr: np.ndarray) -> Tuple[float, float]:
    """
    Estimate camera translation (tx, ty) between this frame and the previous one.
    Returns (0, 0) on the first frame or if GMC is disabled.
    """
    global _prev_gray_small, _cumulative_tx, _cumulative_ty
    if not GMC_ENABLED:
        return 0.0, 0.0

    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (int(w * GMC_SCALE), int(h * GMC_SCALE)))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if _prev_gray_small is None:
        _prev_gray_small = gray
        return 0.0, 0.0

    flow = cv2.calcOpticalFlowFarneback(
        _prev_gray_small, gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    _prev_gray_small = gray

    # Median flow vector → robust global translation estimate
    tx_small = float(np.median(flow[..., 0]))
    ty_small = float(np.median(flow[..., 1]))

    # Scale back to original resolution
    tx = tx_small / GMC_SCALE
    ty = ty_small / GMC_SCALE
    return tx, ty


def compensate_bbox(bbox: Dict, tx: float, ty: float) -> Dict:
    """Subtract camera motion from a bounding box."""
    if tx == 0.0 and ty == 0.0:
        return bbox
    return {
        "x1": bbox["x1"] - tx,
        "y1": bbox["y1"] - ty,
        "x2": bbox["x2"] - tx,
        "y2": bbox["y2"] - ty,
    }


# ── Core per-frame function ───────────────────────────────────────────────────
def run_yolo_and_track(
    frame_bgr: np.ndarray,
    frame_idx: int,
    out_json_path: Optional[str] = None,
) -> Dict:
    """
    Run YOLO + GMC + ByteTracker on one frame.

    Returns a dict:
    {
      "frame": frame_idx,
      "gmc_tx": float, "gmc_ty": float,
      "scene_summary": {
          "total_vehicles_detected": int,
          ...
          "detected_objects": [
              {
                "id": int,           # YOLO local detection id
                "track_id": int,     # ByteTracker persistent id
                "type": str,
                "confidence": float,
                "color": str,        # HSV pixel color (locked or best-guess)
                "position": str,
                "area": int,
                "bounding_box": {"x1","y1","x2","y2"},
                "speed": str,
                "direction": str,
              }, ...
          ]
      }
    }
    """
    import supervision as sv

    model = get_yolo()
    tracker = get_tracker()

    h_orig, w_orig = frame_bgr.shape[:2]
    scale  = min(YOLO_INPUT_MAX_DIM / max(h_orig, w_orig), 1.0)
    small  = cv2.resize(frame_bgr, (int(w_orig * scale), int(h_orig * scale)))
    h, w   = small.shape[:2]

    # ── GMC ──────────────────────────────────────────────────────────────────
    tx, ty = compute_gmc(frame_bgr)

    # ── YOLO ─────────────────────────────────────────────────────────────────
    vehicle_count       = 0
    pedestrian_count    = 0
    cyclist_count       = 0
    traffic_light_count = 0

    boxes_raw, confidences_raw, class_ids_raw = [], [], []
    obj_types_raw, bboxes_raw = [], []

    if model:
        results = model(small, conf=YOLO_CONF_THRESH, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id     = int(box.cls[0])
            cls_name   = model.names[cls_id]
            confidence = float(box.conf[0])
            area       = (x2 - x1) * (y2 - y1)

            if area < MIN_DETECTION_AREA and cls_name != "traffic light":
                continue

            sem_type = _semantic_type(cls_name)
            if sem_type in VEHICLE_CLASSES:    vehicle_count += 1
            elif sem_type == "cyclist":        cyclist_count += 1
            elif sem_type == "pedestrian":     pedestrian_count += 1
            elif sem_type == "traffic_light":  traffic_light_count += 1

            bbox_orig = {
                "x1": int(x1 / scale), "y1": int(y1 / scale),
                "x2": int(x2 / scale), "y2": int(y2 / scale),
            }
            bbox_gmc  = compensate_bbox(bbox_orig, tx, ty)

            boxes_raw.append([bbox_gmc["x1"], bbox_gmc["y1"],
                               bbox_gmc["x2"], bbox_gmc["y2"]])
            confidences_raw.append(min(1.0, max(0.0, confidence)))
            class_ids_raw.append(cls_id)
            obj_types_raw.append(sem_type)
            bboxes_raw.append(bbox_orig)  # keep original (not compensated) for display

    # ── ByteTracker update ────────────────────────────────────────────────────
    tracked_objects = []

    if tracker and boxes_raw:
        detections = sv.Detections(
            xyxy       = np.array(boxes_raw, dtype=np.float32),
            confidence = np.array(confidences_raw, dtype=np.float32),
            class_id   = np.array(class_ids_raw, dtype=np.int32),
        )
        detections.data["object_type"] = obj_types_raw

        detections = tracker.update_with_detections(detections)

        for i in range(len(detections)):
            box      = detections.xyxy[i]
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            conf     = float(detections.confidence[i])
            obj_type = (detections.data["object_type"][i]
                        if "object_type" in detections.data else "unknown")

            # Re-map tracked box back to original (uncompensated) coords for display
            # Use bboxes_raw if index aligns; otherwise use tracked box
            if i < len(bboxes_raw):
                bbox = bboxes_raw[i]
            else:
                bbox = {"x1": int(box[0]), "y1": int(box[1]),
                        "x2": int(box[2]), "y2": int(box[3])}

            # ── Pixel color (skip for pedestrians) ───────────────────────
            is_pedestrian = obj_type == "pedestrian"
            pixel_color = "unknown" if is_pedestrian else extract_dominant_color(frame_bgr, bbox)

            # ── Track state update ────────────────────────────────────────
            state = _get_or_create_state(track_id, obj_type)
            state.add_observation(frame_idx, bbox, pixel_color)

            # ── Position label ────────────────────────────────────────────
            cx = (bbox["x1"] + bbox["x2"]) / 2
            horizontal = ("Left"   if cx < w_orig * 0.40 else
                          "Right"  if cx > w_orig * 0.60 else "Center")
            cy_px = (bbox["y1"] + bbox["y2"]) / 2
            depth  = "Foreground" if cy_px > h_orig * 0.50 else "Background"
            position = f"{depth} {horizontal}"

            obj_entry = {
                "id":           i + 1,
                "track_id":     track_id,
                "type":         obj_type,
                "confidence":   round(conf, 2),
                "position":     position,
                "area":         (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"]),
                "bounding_box": bbox,
                "speed":        state.speed_label(),
                "direction":    state.direction_label(),
            }
            if not is_pedestrian:
                obj_entry["color"] = state.color
            # Size for vehicles and cyclists only
            if obj_type in VEHICLE_CLASSES or obj_type == "cyclist":
                obj_entry["size"] = _size_label(bbox, w_orig, h_orig)
            tracked_objects.append(obj_entry)

    elif not tracker and boxes_raw:
        # Fallback: no tracker, just enumerate detections
        for i, (bbox, conf, obj_type) in enumerate(
                zip(bboxes_raw, confidences_raw, obj_types_raw)):
            is_pedestrian = obj_type == "pedestrian"
            pixel_color = "unknown" if is_pedestrian else extract_dominant_color(frame_bgr, bbox)
            cx = (bbox["x1"] + bbox["x2"]) / 2
            horizontal = ("Left"   if cx < w_orig * 0.40 else
                          "Right"  if cx > w_orig * 0.60 else "Center")
            cy_px = (bbox["y1"] + bbox["y2"]) / 2
            depth  = "Foreground" if cy_px > h_orig * 0.50 else "Background"
            obj_entry = {
                "id":           i + 1,
                "track_id":     -1,
                "type":         obj_type,
                "confidence":   round(conf, 2),
                "position":     f"{depth} {horizontal}",
                "area":         (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"]),
                "bounding_box": bbox,
                "speed":        "unknown",
                "direction":    "unknown",
            }
            if not is_pedestrian:
                obj_entry["color"] = pixel_color
            if obj_type in VEHICLE_CLASSES or obj_type == "cyclist":
                obj_entry["size"] = _size_label(bbox, w_orig, h_orig)
            tracked_objects.append(obj_entry)

    output = {
        "frame": frame_idx,
        "gmc_tx": round(tx, 2),
        "gmc_ty": round(ty, 2),
        "scene_summary": {
            "total_vehicles_detected":       vehicle_count,
            "total_pedestrians_detected":    pedestrian_count,
            "total_cyclists_detected":       cyclist_count,
            "total_traffic_lights_detected": traffic_light_count,
            "detected_objects":              tracked_objects,
        }
    }

    if out_json_path:
        os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(output, f, indent=2)

    return output


# ── Annotated canvas (for saving debug images) ────────────────────────────────
def draw_tracks(frame_bgr: np.ndarray, frame_data: Dict) -> np.ndarray:
    """Draw bounding boxes + track IDs only (clean version for user-facing images)."""
    canvas = frame_bgr.copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX
    for obj in frame_data.get("scene_summary", {}).get("detected_objects", []):
        bb  = obj["bounding_box"]
        tid = obj.get("track_id", obj["id"])
        x1, y1, x2, y2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]

        # Draw bounding box (white outline)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw track ID label only
        label = f"#{tid}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
        ly = y1 - 8 if y1 > 20 else y1 + th + 8
        cv2.rectangle(canvas, (x1, ly - th - 4), (x1 + tw + 6, ly + 4), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x1 + 3, ly), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ── Full-video pipeline ───────────────────────────────────────────────────────
def process_video_frames(
    video_path: str,
    out_dir: str = "output",
    frame_rate: int = TRACKER_FRAME_RATE,
) -> Tuple[int, int]:
    """
    Run YOLO + GMC + ByteTracker on every Nth frame of a video.

    Writes:
      output/frames/frame_{idx:06d}.json        — per-frame JSON (track_id, color, size, motion)
      output/annotated/frame_{idx:06d}_track.jpg — user-facing image (boxes + IDs only)

    Returns:
      (total_frames_written, video_duration_seconds)
    """
    import shutil
    frames_dir    = os.path.join(out_dir, "frames")
    annotated_dir = os.path.join(out_dir, "annotated")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)
    if os.path.exists(annotated_dir):
        shutil.rmtree(annotated_dir, ignore_errors=True)
    os.makedirs(frames_dir,    exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps           = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_secs = max(1, int(total_frames / fps))

    # Auto frame-rate: if 0, sample at 0.5x video_fps, but never drop below 15 FPS (unless video < 15 FPS)
    actual_rate = frame_rate if frame_rate > 0 else min(int(fps), max(15, int(math.floor(fps * 0.5))))
    step_float  = fps / actual_rate

    print(f"\n{'='*55}")
    print(f"  Video original FPS : {fps:.2f} fps")
    print(f"  Sampling at        : {actual_rate:.2f} fps  (step size: {step_float:.2f} frames)")
    print(f"  Duration           : {duration_secs}s  |  ~{int(duration_secs * actual_rate)} frames to process")
    print(f"{'='*55}\n")

    # Fresh state for new video
    reset_tracker(int(actual_rate))
    reset_track_states()
    reset_gmc()

    frames_written = 0
    current_float_idx = 0.0

    while current_float_idx < total_frames:
        frame_idx = int(current_float_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        json_path = os.path.join(frames_dir,    f"frame_{frame_idx:06d}.json")
        jpg_path  = os.path.join(annotated_dir, f"frame_{frame_idx:06d}_track.jpg")

        data = run_yolo_and_track(frame, frame_idx, out_json_path=json_path)

        # Save annotated user-facing image
        canvas = draw_tracks(frame, data)
        cv2.imwrite(jpg_path, canvas)

        n = len(data["scene_summary"]["detected_objects"])
        print(f"  Frame {frame_idx:06d}: {n} tracked objects | GMC ({data['gmc_tx']:+.1f}, {data['gmc_ty']:+.1f})")

        frames_written += 1
        current_float_idx += step_float

    cap.release()
    print(f"\u2705 Processed {frames_written} frames \u2192 {frames_dir}/ | {annotated_dir}/")

    # Generate and save a single global tracker summary JSON for the whole video
    global_summary = get_track_motion_summary(0, int(duration_secs) + 1, fps=actual_rate)
    summary_path = os.path.join(out_dir, "tracks_summary.json")
    with open(summary_path, "w") as f:
        json.dump(global_summary, f, indent=2)
    print(f"  \u2705 Saved full track summary to: {summary_path}")

    return frames_written, duration_secs


# ── Cumulative motion summary (used by scene_annotator) ──────────────────────
def get_track_motion_summary(
    span_start_sec: int,
    span_end_sec: int,
    fps: Optional[float] = None,
) -> List[Dict]:
    """
    Return a list of motion summaries for tracks that were active in the span.
    Uses the in-memory _track_states built during process_video_frames().
    Falls back to reading tracks.json if called post-hoc.
    """
    actual_fps = fps if fps is not None else float(_tracker_frame_rate)
    if actual_fps <= 0:
        actual_fps = 25.0  # Safe fallback if never initialized

    summaries = []
    start_frame = int(span_start_sec * actual_fps)
    end_frame   = int(span_end_sec   * actual_fps)

    for tid, state in _track_states.items():
        active_frames = [
            (fi, pos)
            for fi, pos in zip(state.frame_indices, state.positions)
            if start_frame <= fi <= end_frame
        ]
        if not active_frames:
            continue

        indices = [af[0] for af in active_frames]
        positions = [af[1] for af in active_frames]

        if len(positions) >= 2:
            dx = positions[-1][0] - positions[0][0]
            dy = positions[-1][1] - positions[0][1]
            dist = math.hypot(dx, dy)
        else:
            dist = 0.0

        summaries.append({
            "track_id":        tid,
            "object_type":     state.obj_type,
            "color":           state.color,
            "speed":           state.speed_label(),
            "direction":       state.direction_label(),
            "distance_px":     round(dist, 1),
            "frame_span":      [indices[0], indices[-1]],
            "evidence_seconds": sorted(set(int(fi / actual_fps) for fi in indices)),
        })
    return summaries


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    target = TARGET_FILE
    process_video_frames(target, out_dir="output", frame_rate=TRACKER_FRAME_RATE)
    summary = get_track_motion_summary(0, 9999)
    print(f"\nMotion summary ({len(summary)} tracks):")
    for s in summary[:10]:
        print(f"  Track {s['track_id']} ({s['color']} {s['object_type']}): "
              f"{s['direction']} @ {s['speed']} | {s['distance_px']:.0f}px")
