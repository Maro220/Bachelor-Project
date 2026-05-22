import json
import math
import os
import time
import cv2
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("⚠   nuscenes SDK not available.")

try:
    import open_clip
    import torch
    from PIL import Image as _PIL_Image
    CLIP_AVAILABLE = True  # if false, use HSV instead
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠   open_clip not installed. Falling back to HSV color extraction.")
    print("   Install with: pip install open_clip_torch")

_clip_model       = None
_clip_preprocess  = None
_clip_tokenizer   = None

_COLOR_NAMES = [
    "white", "black", "silver", "gray",
    "red", "blue", "green", "yellow",
    "orange", "brown", "dark",
]

# Type-specific noun used in the CLIP prompt. Lets CLIP match e.g.
# "orange traffic cone" instead of "orange vehicle" so non-vehicle
# objects get sensible color predictions too.
_TYPE_NOUN = {
    "car":           "vehicle",
    "van":           "vehicle",
    "truck":         "vehicle",
    "bus":           "vehicle",
    "motorcycle":    "vehicle",
    "cyclist":       "bicycle",
    "cone":          "traffic cone",
    "barrier":       "barrier",
    "traffic_light": "traffic light",
}
_DEFAULT_NOUN = "object"

# Per-noun cache of normalized text embeddings, lazy-built on first use.
_clip_text_emb_by_noun: Dict[str, object] = {}


def _init_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None or not CLIP_AVAILABLE:
        return
    try:
        import open_clip, warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*QuickGELU mismatch.*')
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai', quick_gelu=True
            )
        _clip_model.eval()
        _clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print("✓ CLIP color model loaded (ViT-B/32)")
    except Exception as e:
        print(f"⚠   CLIP init failed: {e}. Falling back to HSV.")
        _clip_model = None


def _get_clip_text_emb_for_noun(noun: str):
    """Lazy-build a normalized text embedding for one '{color} {noun}' set."""
    if noun in _clip_text_emb_by_noun:
        return _clip_text_emb_by_noun[noun]
    import torch
    labels = [f"{c} {noun}" for c in _COLOR_NAMES]
    with torch.no_grad():
        tokens = _clip_tokenizer(labels)
        emb    = _clip_model.encode_text(tokens)
        emb    = emb / emb.norm(dim=-1, keepdim=True)
    _clip_text_emb_by_noun[noun] = emb
    return emb

TARGET_SCENE             = "scene-0757"   # NuScenes scene NAME (not a path) — SDK resolves frames from data/v1.0-mini/sweeps/CAM_FRONT/
TRACKER_FRAME_RATE       = 0      # frames per second fed to tracker (0 = auto: all frames at video fps)
MIN_DETECTION_AREA       = 150    # min pixel area for non-traffic-light detections
TRACK_ACTIVATION_THRESH  = 0.25   # ByteTrack: min confidence to activate track (raised from 0.15)
TRACK_BUFFER_FRAMES      = 120    # ByteTrack: frames to keep a lost track alive (~10s at 12 Hz — survives occlusions)
MATCH_THRESH             = 0.85   # ByteTrack: IoU-DISTANCE threshold for matching (higher = more permissive; default 0.8). Was 0.55 → fragmented.
COLOR_LOCK_FRAMES        = 5      # observations before locking a track's color
MOTION_STATIONARY_PX     = 1.5    # px/frame below which = "stationary" (tuned for 25fps)
MOTION_SLOW_PX           = 8.0    # px/frame below which = "slow"
MOTION_FAST_PX           = 25.0   # px/frame above which = "fast"
YOLO_MODEL_PATH          = "yolo11l.pt"
YOLO_CONF_THRESH         = 0.03   # Lowered from 0.20 — same car was flickering in/out at ~0.18-0.22
YOLO_INPUT_MAX_DIM       = 1600   # resize longest edge to this before inference
YOLO_IMGSZ               = 1280   # inference resolution passed to ultralytics (was using default 640)

NUSCENES_DATAROOT        = "data/v1.0-mini"
NUSCENES_VERSION         = "v1.0-mini"
NUSCENES_SCENE_TOKEN     = None  
VALIDATE_WITH_GROUNDTRUTH= True   # Compare YOLO detections against NuScenes annotations

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
# HSV pixel-based color extraction 
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


def extract_dominant_color(frame_bgr: np.ndarray, bbox: Dict, obj_type: str = "",
                            other_bboxes: Optional[List[Dict]] = None) -> str:
    x1 = max(0, bbox["x1"])
    y1 = max(0, bbox["y1"])
    x2 = min(frame_bgr.shape[1], bbox["x2"])
    y2 = min(frame_bgr.shape[0], bbox["y2"])

    if x2 <= x1 or y2 <= y1:
        return "unknown"

    if obj_type in VEHICLE_CLASSES:
        y2 = y1 + max(4, int((y2 - y1) * 0.70))

    pw = max(1, int((x2 - x1) * 0.15))
    ph = max(1, int((y2 - y1) * 0.15))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_w = max(2, (x2 - x1) // 2 - pw)
    half_h = max(2, (y2 - y1) // 2 - ph)

    roi_x1 = cx - half_w
    roi_y1 = cy - half_h

    roi = frame_bgr[roi_y1: cy + half_h, roi_x1: cx + half_w]
    if roi.size == 0:
        return "unknown"

    valid_mask = np.full(roi.shape[:2], 255, dtype=np.uint8)
    if other_bboxes:
        for ob in other_bboxes:
            ox1 = max(0, int(ob["x1"]) - roi_x1)
            oy1 = max(0, int(ob["y1"]) - roi_y1)
            ox2 = min(roi.shape[1], int(ob["x2"]) - roi_x1)
            oy2 = min(roi.shape[0], int(ob["y2"]) - roi_y1)
            if ox2 > ox1 and oy2 > oy1:
                valid_mask[oy1:oy2, ox1:ox2] = 0
        if int(np.sum(valid_mask > 0)) < 20:
            valid_mask[:] = 255

    hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    counts = {}
    for name, lo, hi in _HSV_PALETTE:
        color_mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        color_mask = cv2.bitwise_and(color_mask, valid_mask)
        counts[name] = int(np.sum(color_mask > 0))

    # Merge the two red ranges
    counts["red"] = counts.pop("red", 0) + counts.pop("red2", 0)
    best = max(counts, key=counts.get)
    return best
def _normalize_illumination(crop_bgr: np.ndarray) -> np.ndarray:
    #CLAHE on the L channel only — fixes shadows/overexposure without changing hue.
    if crop_bgr.size == 0:
        return crop_bgr
    lab     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq    = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def _is_low_light(crop_bgr: np.ndarray) -> bool:
    if crop_bgr.size == 0:
        return True
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < 45


def extract_dominant_color_clip(frame_bgr: np.ndarray, bbox: Dict,
                                 obj_type: str = "",
                                 other_bboxes: Optional[List[Dict]] = None
                                 ) -> Tuple[str, float]:
    """
    Extract vehicle color using CLIP zero-shot classification.
    Returns (color_name, confidence). Falls back to HSV if CLIP unavailable.
    """
    _init_clip()

    x1 = max(0, bbox["x1"])
    y1 = max(0, bbox["y1"])
    x2 = min(frame_bgr.shape[1], bbox["x2"])
    y2 = min(frame_bgr.shape[0], bbox["y2"])

    if x2 <= x1 or y2 <= y1:
        return "unknown", 0.0

    if obj_type in VEHICLE_CLASSES:
        y2 = y1 + max(8, int((y2 - y1) * 0.70))

    crop_bgr = frame_bgr[y1:y2, x1:x2].copy()
    if crop_bgr.size == 0:
        return "unknown", 0.0

    crop_bgr  = _normalize_illumination(crop_bgr)
    low_light = _is_low_light(crop_bgr)

    if _clip_model is not None and CLIP_AVAILABLE:
        try:
            import torch
            from PIL import Image as _PIL_Image
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img  = _PIL_Image.fromarray(crop_rgb)
            noun     = _TYPE_NOUN.get(obj_type, _DEFAULT_NOUN)
            text_emb = _get_clip_text_emb_for_noun(noun)
            with torch.no_grad():
                img_t = _clip_preprocess(pil_img).unsqueeze(0)
                img_e = _clip_model.encode_image(img_t)
                img_e = img_e / img_e.norm(dim=-1, keepdim=True)
                probs = (img_e @ text_emb.T * 100).softmax(dim=-1)[0]
            best_idx   = int(probs.argmax())
            confidence = float(probs[best_idx])
            color_name = _COLOR_NAMES[best_idx]

            if low_light and confidence < 0.55:
                return "dark", 0.3
            if confidence < 0.35:
                return "unknown", confidence
            return color_name, confidence
        except Exception as e:
            print(f"  ⚠   CLIP color extraction error: {e}. Falling back to HSV.")


    norm_bbox = {"x1": 0, "y1": 0,
                 "x2": crop_bgr.shape[1], "y2": crop_bgr.shape[0]}
    color_hsv = extract_dominant_color(crop_bgr, norm_bbox, obj_type, other_bboxes)
    return color_hsv, 0.6


#  NuScenes SDK loader 
_nusc = None  
_nusc_frame_data: Dict[int, Dict] = {}   # Maps frame_idx → ground truth annotations


def _init_nuscenes_sdk(dataroot: str, version: str = "v1.0-mini"):
    global _nusc
    if _nusc is None:
        _nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        print(f"✅ Loaded NuScenes {version}")
    return _nusc


def load_nuscenes_scene(scene_name: str, dataroot: str = NUSCENES_DATAROOT) -> bool:

    global _nusc_frame_data
    
    if not NUSCENES_AVAILABLE:
        return False
    
    nusc = _init_nuscenes_sdk(dataroot)
    
    # Find scene by name
    scene_record = None
    for scene in nusc.scene:
        if scene_name in scene["name"]:
            scene_record = scene
            break
    
    if not scene_record:
        print(f"⚠   Scene '{scene_name}' not found. Available scenes:")
        for scene in nusc.scene[:10]:
            print(f"     - {scene['name']}")
        return False
    
    print(f"✅ Found scene: {scene_name} (token: {scene_record['token'][:8]}...)")
    _extract_frame_annotations_sdk(nusc, scene_record["token"])
    print(f"✅ Extracted {len(_nusc_frame_data)} frames with annotations")
    return True


def _extract_frame_annotations_sdk(nusc: 'NuScenes', scene_token: str):
    global _nusc_frame_data
    
    _nusc_frame_data = {}
    
    # Get scene and iterate through samples
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]
    
    frame_idx = 0
    while sample_token:
        sample = nusc.get("sample", sample_token)
        
        # Get ego-pose from CAM_FRONT sample_data
        cam_data_token = sample["data"]["CAM_FRONT"]
        sample_data = nusc.get("sample_data", cam_data_token)
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        
        # Extract annotations for this frame
        annotations = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            # Get instance to find category
            instance = nusc.get("instance", ann["instance_token"])
            category = nusc.get("category", instance["category_token"])
            category_name = category["name"]
            
            # Only include vehicles and pedestrians
            if any(x in category_name for x in ["car", "truck", "bus", "motorcycle", "pedestrian", "bicycle"]):
                annotations.append({
                    "category": category_name,
                    "translation": ann["translation"],
                    "size": ann["size"],
                    "rotation": ann["rotation"],
                    "instance_token": ann["instance_token"],
                })
        
        _nusc_frame_data[frame_idx] = {
            "ego_pose": ego_pose,
            "annotations": annotations,
        }
        
        frame_idx += 1
        sample_token = sample["next"]


def get_frame_ground_truth(frame_idx: int) -> Optional[Dict]:
    return _nusc_frame_data.get(frame_idx)


# ── GT merge state (per-sweep injection into ByteTracker) ────────────────────
_NUSCENES_TO_YOLO_TYPE = {
    "human.pedestrian.adult":              "pedestrian",
    "human.pedestrian.child":              "pedestrian",
    "human.pedestrian.wheelchair":         "pedestrian",
    "human.pedestrian.stroller":           "pedestrian",
    "human.pedestrian.personal_mobility":  "pedestrian",
    "vehicle.car":                         "car",
    "vehicle.truck":                       "truck",
    "vehicle.bus.rigid":                   "bus",
    "vehicle.bus.bendy":                   "bus",
    "vehicle.motorcycle":                  "motorcycle",
    "vehicle.bicycle":                     "cyclist",
    "vehicle.trailer":                     "truck",
    "vehicle.construction":                "truck",
    "movable_object.trafficcone":          "cone",
    "movable_object.barrier":              "barrier",
    "movable_object.pushable_pullable":    "other",
    "static_object.bicycle_rack":          "other",
}


def _nuscenes_to_yolo_type(semantic_type: str) -> str:
    return _NUSCENES_TO_YOLO_TYPE.get(semantic_type, "other")


_nuscenes_gt_by_frame: Dict[int, Dict] = {}


def set_nuscenes_gt(gt_data: dict):
    global _nuscenes_gt_by_frame
    _nuscenes_gt_by_frame = {
        f["frame_video_idx"]: f
        for f in gt_data.get("frames", [])
    }
    total_anns = sum(len(f.get("annotations", [])) for f in gt_data.get("frames", []))
    print(f"✅ NuScenes GT loaded: {len(_nuscenes_gt_by_frame)} frames, "
          f"{total_anns} total annotations")


def reset_gt_state():
    global _nuscenes_gt_by_frame
    _nuscenes_gt_by_frame = {}


def _derive_action(obj_type: str, speed: str, direction: str) -> str:
    """Map tracker speed/direction (GMC-corrected) to a human-readable action label.
    Called instead of asking Qwen, which only sees a single static frame and cannot
    judge motion reliably."""
    t = obj_type.lower()
    moving = speed not in ("stationary", "unknown")

    if "traffic light" in t or (t == "light"):
        return "static"
    if any(x in t for x in ["person", "pedestrian"]):
        if not moving:          return "standing"
        if speed == "slow":     return "walking"
        return "running"
    if any(x in t for x in ["cyclist", "bicycle"]):
        if not moving:          return "stopped"
        if "left" in direction: return "turning"
        if "right" in direction:return "turning"
        return "moving"
    # vehicles get "parked" when stationary; other non-vehicle objects
    # (cones, barriers, debris, "other") get "static" instead — "parked"
    # only makes sense for vehicles.
    is_vehicle = t in VEHICLE_CLASSES or "vehicle" in t
    if not is_vehicle:
        return "moving" if moving else "static"
    if not moving:              return "parked"
    if "left" in direction:     return "turning left"
    if "right" in direction:    return "turning right"
    return "moving"


def build_nuscenes_gt_2d(nusc, sd_records: list,
                          image_size=(1600, 900)) -> dict:
    """
    Project NuScenes 3D sample annotations into 2D bboxes for every sweep
    frame (~12 Hz), not just keyframes. 3D boxes are world-frame and valid at
    any timestamp, so each sweep's own ego_pose gives an accurate 2D bbox.
    Motion attributes (speed_label) come from nusc.box_velocity()
    on the keyframe annotation; direction is computed per-instance from
    sweep-to-sweep pixel displacement.

    Output shape matches what set_nuscenes_gt() expects:
      { "frames": [ { "frame_video_idx": <frame_idx>,
                      "annotations": [ {instance_token, semantic_type,
                                        visibility, bbox_2d, speed_label,
                                        direction, action }, ... ]
                    } ] }
    """
    from nuscenes.utils.geometry_utils import view_points
    from nuscenes.utils.data_classes import Box as _NuscBox
    from pyquaternion import Quaternion
    import numpy as _np
    import math as _math

    W, H = image_size

    calib = nusc.get("calibrated_sensor", sd_records[0]["calibrated_sensor_token"])
    K     = _np.array(calib["camera_intrinsic"])

    def _speed_label(v_mps: float) -> str:
        if v_mps < 0.5: return "stationary"
        if v_mps < 2.0: return "slow"
        if v_mps < 8.0: return "moving"
        return "fast"

    def _direction(dx: float, dy: float, speed_label: str) -> str:
        if speed_label == "stationary" or (abs(dx) < 1.0 and abs(dy) < 1.0):
            return "stationary"
        # Image y grows downward; flip so +y = up for human-readable labels
        angle = _math.degrees(_math.atan2(-dy, dx))
        if   -45 <= angle <  45: return "moving right"
        elif  45 <= angle < 135: return "moving up"
        elif -135 <= angle < -45:return "moving down"
        else:                    return "moving left"

    # Build keyframe annotation cache (raw 3D world-frame, indexed by sample_token).
    # Sweeps inherit annotations from the most recent keyframe in the chain.
    sample_annotations: dict = {}
    for sd in sd_records:
        if not sd["is_key_frame"]:
            continue
        sample = nusc.get("sample", sd["sample_token"])
        anns = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cat = nusc.get("category",
                           nusc.get("instance", ann["instance_token"])["category_token"])
            if cat["name"] not in _NUSCENES_TO_YOLO_TYPE:
                continue
            try:
                vis_lvl = int(nusc.get("visibility", ann.get("visibility_token", ""))["token"])
            except (KeyError, ValueError):
                vis_lvl = 0
            try:
                v_world   = nusc.box_velocity(ann_token)
                speed_mps = float(_np.linalg.norm(v_world[:2]))
                if _np.isnan(speed_mps):
                    speed_mps = 0.0
            except Exception:
                speed_mps = 0.0
            anns.append({
                "instance_token": ann["instance_token"],
                "semantic_type":  cat["name"],
                "translation":    ann["translation"],
                "size":           ann["size"],
                "rotation":       ann["rotation"],
                "visibility":     vis_lvl,
                "speed_label":    _speed_label(speed_mps),
            })
        sample_annotations[sd["sample_token"]] = anns

    def _project_box(ann_3d, ego_pose):
        box = _NuscBox(ann_3d["translation"], ann_3d["size"],
                       Quaternion(ann_3d["rotation"]))
        box.translate(-_np.array(ego_pose["translation"]))
        box.rotate(Quaternion(ego_pose["rotation"]).inverse)
        box.translate(-_np.array(calib["translation"]))
        box.rotate(Quaternion(calib["rotation"]).inverse)
        corners_3d = box.corners()
        # Reject if ANY corner is at or behind the camera plane. Partial
        # behind-camera boxes blow up under perspective divide and clamp
        # to a full-frame bbox; the object reappears cleanly on the next
        # sweep once all 8 corners are in front.
        if (corners_3d[2, :] <= 0.1).any():
            return None
        corners_2d = view_points(corners_3d, K, normalize=True)[:2]
        x1_raw, y1_raw = float(corners_2d[0].min()), float(corners_2d[1].min())
        x2_raw, y2_raw = float(corners_2d[0].max()), float(corners_2d[1].max())
        # If the raw projection is fully outside the image, skip — clamping
        # would otherwise glue it to the frame edge as a thin strip.
        if x2_raw < 0 or y2_raw < 0 or x1_raw >= W or y1_raw >= H:
            return None
        x1 = max(0,   int(x1_raw))
        y1 = max(0,   int(y1_raw))
        x2 = min(W-1, int(x2_raw))
        y2 = min(H-1, int(y2_raw))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0}

    frames_out      = []
    last_kf_anns    = []
    prev_centers: dict = {}  # instance_token -> (cx, cy) from previous sweep

    for frame_idx, sd in enumerate(sd_records):
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
        if sd["is_key_frame"]:
            last_kf_anns = sample_annotations.get(sd["sample_token"], [])

        anns_out = []
        for ann_3d in last_kf_anns:
            proj = _project_box(ann_3d, ego_pose)
            if proj is None:
                continue
            it     = ann_3d["instance_token"]
            cx, cy = proj["cx"], proj["cy"]

            direction = "stationary"
            if ann_3d["speed_label"] != "stationary" and it in prev_centers:
                pcx, pcy = prev_centers[it]
                direction = _direction(cx - pcx, cy - pcy, ann_3d["speed_label"])
            prev_centers[it] = (cx, cy)

            action = _derive_action(ann_3d["semantic_type"],
                                    ann_3d["speed_label"], direction)

            anns_out.append({
                "instance_token": it,
                "semantic_type":  ann_3d["semantic_type"],
                "visibility":     ann_3d["visibility"],
                "bbox_2d":        {"x1": proj["x1"], "y1": proj["y1"],
                                   "x2": proj["x2"], "y2": proj["y2"]},
                "speed_label":    ann_3d["speed_label"],
                "direction":      direction,
                "action":         action,
            })

        frames_out.append({
            "frame_video_idx": frame_idx,
            "annotations":     anns_out,
        })

    n_kf = sum(1 for sd in sd_records if sd["is_key_frame"])
    print(f"✓ GT projected at all {len(frames_out)} sweep frames ({n_kf} keyframes)")
    return {"frames": frames_out}


def _bbox_iou(bb1: dict, bb2: dict) -> float:
    required = ("x1", "y1", "x2", "y2")
    if not bb1 or not bb2:
        return 0.0
    if not all(k in bb1 for k in required) or not all(k in bb2 for k in required):
        return 0.0
    ix1 = max(bb1["x1"], bb2["x1"])
    iy1 = max(bb1["y1"], bb2["y1"])
    ix2 = min(bb1["x2"], bb2["x2"])
    iy2 = min(bb1["y2"], bb2["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = max(1, (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"]))
    a2 = max(1, (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"]))
    return inter / (a1 + a2 - inter)


def merge_gt_into_detections(
    yolo_data:      dict,
    gt_annotations: list,
    frame_w:        int,
    frame_h:        int,
    iou_threshold:  float = 0.2,
) -> dict:
    """
    Merge NuScenes GT annotations into yolo_data["detections"].

    Pass 1 — IoU-match existing YOLO detections against GT:
        • Stamp gt_sourced=True, motion_source="nuscenes_gt", instance_token
        • Overwrite type with GT type if different  → gt_type_override=True
        • Overwrite speed, action, direction with GT LiDAR values

    Pass 2 — Inject GT objects YOLO missed entirely:
        • source="gt_injected", instance_token, all GT motion fields set

    Counts recomputed from full merged list.
    """
    # For IoU-match enrichment (pass 1) accept any annotated visibility — YOLO
    # already detected the box; GT only needs to confirm identity so the
    # instance_token bridges across ByteTrack fragments.
    matchable_gt = [a for a in gt_annotations if a.get("visibility", 0) >= 1]
    # For injection (pass 2) keep the stricter ≥2 threshold so heavily-
    # occluded boxes don't pollute the tracker.
    injectable_gt = [a for a in matchable_gt if a.get("visibility", 0) >= 2]
    if not matchable_gt:
        return yolo_data
    visible_gt = matchable_gt   # used for stats / iteration where both apply

    detections             = yolo_data["detections"]
    matched_gt_tokens: set = set()
    type_corrections       = 0
    gt_enriched            = 0

    # PASS 1: enrich existing YOLO detections
    for det in detections:
        yolo_bb = det.get("bounding_box", {})
        if not yolo_bb:
            continue

        best_iou, best_gt = 0.0, None
        for ann in visible_gt:
            iou = _bbox_iou(yolo_bb, ann.get("bbox_2d", {}))
            if iou > best_iou:
                best_iou, best_gt = iou, ann

        if best_gt is None or best_iou < iou_threshold:
            continue

        it           = best_gt["instance_token"]
        gt_type      = _nuscenes_to_yolo_type(best_gt["semantic_type"])
        yolo_type    = det.get("type", "")
        type_changed = (gt_type != yolo_type and gt_type != "other")

        det["gt_sourced"]     = True
        det["motion_source"]  = "nuscenes_gt"
        det["instance_token"] = it
        det["gt_iou"]         = round(best_iou, 3)

        if type_changed:
            det["yolo_type_original"] = yolo_type
            det["type"]               = gt_type
            det["gt_type_override"]   = True
            type_corrections += 1
        else:
            det["gt_type_override"] = False

        det["speed"]     = best_gt.get("speed_label", det.get("speed",     "unknown"))
        det["action"]    = best_gt.get("action",      det.get("action",    "unknown"))
        det["direction"] = best_gt.get("direction",   det.get("direction", "unknown"))

        matched_gt_tokens.add(it)
        gt_enriched += 1

    # PASS 2: inject GT objects YOLO missed entirely (stricter ≥2 visibility)
    injected = 0
    for ann in injectable_gt:
        it = ann["instance_token"]
        if it in matched_gt_tokens:
            continue

        gt_bbox = ann.get("bbox_2d", {})
        if not gt_bbox:
            continue

        already_covered = any(
            _bbox_iou(gt_bbox, d.get("bounding_box", {})) > iou_threshold
            for d in detections
        )
        if already_covered:
            continue

        cx         = (gt_bbox["x1"] + gt_bbox["x2"]) / 2
        cy         = (gt_bbox["y1"] + gt_bbox["y2"]) / 2
        horizontal = ("Left"   if cx < frame_w * 0.4 else
                      "Right"  if cx > frame_w * 0.6 else "Center")
        depth      = "Foreground" if cy > frame_h * 0.5 else "Background"
        gt_type    = _nuscenes_to_yolo_type(ann["semantic_type"])

        detections.append({
            "id":               len(detections) + injected + 1,
            "type":             gt_type,
            "confidence":       1.0,
            "position":         f"{depth} {horizontal}",
            "area":             max(0, (gt_bbox["x2"] - gt_bbox["x1"]) *
                                       (gt_bbox["y2"] - gt_bbox["y1"])),
            "bounding_box":     gt_bbox,
            "track_id":         _canonical_id_for(it),
            "instance_token":   it,
            "speed":            ann.get("speed_label",   "unknown"),
            "direction":        ann.get("direction",     "stationary"),
            "action":           ann.get("action",        "unknown"),
            "color":            "unknown",
            "size":             ann.get("size_category", "unknown"),
            "source":           "gt_injected",
            "gt_sourced":       True,
            "motion_source":    "nuscenes_gt",
            "gt_type_override": False,
        })
        injected += 1

    vehicle_types = {"car", "van", "truck", "bus", "motorcycle"}
    type_counts: dict = {}
    for d in detections:
        t = d.get("type", "other")
        type_counts[t] = type_counts.get(t, 0) + 1

    yolo_data["vehicle_count"]       = sum(type_counts.get(t, 0) for t in vehicle_types)
    yolo_data["pedestrian_count"]    = type_counts.get("pedestrian", 0)
    yolo_data["cyclist_count"]       = type_counts.get("cyclist", 0)
    yolo_data["traffic_light_count"] = type_counts.get("traffic_light", 0)
    yolo_data["total_objects"]       = len(detections)

    yolo_only = sum(1 for d in detections if not d.get("gt_sourced", False))
    yolo_data["gt_merge_stats"] = {
        "total_gt_visible":      len(visible_gt),
        "yolo_gt_matched":       gt_enriched,
        "yolo_type_corrections": type_corrections,
        "gt_injected":           injected,
        "yolo_only":             yolo_only,
        "pct_gt_enriched":       round(
            (gt_enriched + injected) / max(len(detections), 1) * 100, 1),
    }

    return yolo_data


#  YOLO
_yolo_model = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
            _yolo_model = YOLO(YOLO_MODEL_PATH)
        except ImportError:
            print(" ultralytics not installed. YOLO unavailable.")
    return _yolo_model


#  ByteTracker  
_byte_tracker        = None
_tracker_frame_rate  = TRACKER_FRAME_RATE
_tracker_actual_fps: float = 0.0  

def reset_tracker(frame_rate: int):

    global _byte_tracker, _tracker_frame_rate, _tracker_actual_fps
    _tracker_frame_rate = frame_rate
    _tracker_actual_fps = float(frame_rate) if frame_rate > 0 else 0.0
 
    kalman_rate = max(10, frame_rate)   # ← THE FIX
 
    try:
        from supervision.tracker.byte_tracker.core import ByteTrack
        _byte_tracker = ByteTrack(
            track_activation_threshold=TRACK_ACTIVATION_THRESH,
            lost_track_buffer=TRACK_BUFFER_FRAMES,
            minimum_matching_threshold=MATCH_THRESH,
            frame_rate=kalman_rate,          # use clamped value here
        )
        print(f"✅ ByteTracker initialised (video fps={frame_rate}, kalman_rate={kalman_rate})")
    except ImportError:
        _byte_tracker = None
        print("⚠  supervision not installed. ByteTracker unavailable.")
 
 
def get_tracker():
    global _byte_tracker
    if _byte_tracker is None:
        reset_tracker()
    return _byte_tracker

#  Track state: color propagation + motion history 
class TrackState:
    _EMA_ALPHA = 0.30   # smoothing factor: higher = faster response to changes

    def __init__(self, identity: str, obj_type: str):
        # identity is a string: instance_token for GT-confirmed objects,
        # f"yolo_{track_id}" for YOLO-only. Promoting identity above
        # ByteTracker's track_id lets the same physical car keep one
        # TrackState across fragmented ByteTrack IDs.
        self.identity    = identity
        self.obj_type    = obj_type
        self.color_votes: List[Tuple[str, float]] = []
        self.locked_color: Optional[str]          = None
        self.positions: List[Tuple[float, float]] = []
        self.frame_indices: List[int]             = []

        # Anchor: cumulative camera offset at first observation
        self._anchor_tx: float = 0.0
        self._anchor_ty: float = 0.0
        self._anchor_set: bool = False

        # EMA state
        self._smoothed_v: Optional[float] = None
        # Raw (un-compensated) centre from the previous frame — used with the
        # per-frame affine inverse to compute camera-cancelled velocity.
        self._prev_raw: Optional[Tuple[float, float]] = None

    def add_observation(
        self,
        frame_idx: int,
        bbox: Dict,
        color_candidate: str,
        color_confidence: float = 1.0,
        cum_tx: float = 0.0,
        cum_ty: float = 0.0,
        frame_M_inv: Optional[np.ndarray] = None,
    ):

        #cum_tx / cum_ty : cumulative camera translation since video start.
        #frame_M_inv     : 2×3 inverse affine for this frame (curr→prev, full-res).
         #                 raw centre into the previous frame's coordinate system and
          #                comparing it to the actual previous raw centre.  This cancels
           #               translation + rotation + scale in one step, so parked objects
            #              near the frame edge no longer get flagged as moving when the
             #             ego-vehicle turns.  Falls back to translation-only when None.
        raw_cx = (bbox["x1"] + bbox["x2"]) / 2.0
        raw_cy = (bbox["y1"] + bbox["y2"]) / 2.0

        if not self._anchor_set:
            self._anchor_tx = cum_tx
            self._anchor_ty = cum_ty
            self._anchor_set = True

        stable_cx = raw_cx - (cum_tx - self._anchor_tx)
        stable_cy = raw_cy - (cum_ty - self._anchor_ty)

        if self._prev_raw is not None:
            if frame_M_inv is not None:
                pt = np.array([raw_cx, raw_cy, 1.0])
                corrected_cx = float(frame_M_inv[0] @ pt)
                corrected_cy = float(frame_M_inv[1] @ pt)
                inst_v = math.hypot(corrected_cx - self._prev_raw[0],
                                    corrected_cy - self._prev_raw[1])
            else:
                prev_cx, prev_cy = self.positions[-1]
                inst_v = math.hypot(stable_cx - prev_cx, stable_cy - prev_cy)

            if self._smoothed_v is None:
                self._smoothed_v = inst_v
            else:
                self._smoothed_v = (
                    self._EMA_ALPHA * inst_v
                    + (1.0 - self._EMA_ALPHA) * self._smoothed_v
                )

        self._prev_raw = (raw_cx, raw_cy)
        self.positions.append((stable_cx, stable_cy))
        self.frame_indices.append(frame_idx)

        # Confidence-weighted color voting
        if self.locked_color is None:
            if color_candidate not in ("unknown", "dark", "mixed"):
                self.color_votes.append((color_candidate, color_confidence))
            high_conf = [(c, w) for c, w in self.color_votes if w > 0.5]
            if len(high_conf) >= COLOR_LOCK_FRAMES:
                from collections import defaultdict
                scores = defaultdict(float)
                for c, w in high_conf:
                    scores[c] += w
                self.locked_color = max(scores, key=scores.get)

    @property
    def color(self) -> str:
        if self.locked_color:
            return self.locked_color
        if self.color_votes:
            from collections import defaultdict
            scores = defaultdict(float)
            for c, w in self.color_votes:
                scores[c] += w
            return max(scores, key=scores.get) if scores else "unknown"
        return "unknown"
    def velocity_px_per_frame(self) -> Optional[float]:
        return self._smoothed_v  

    def speed_label(self) -> str:
        # Traffic lights are stationary by domain truth — the rotation-only
        # GMC homography (compute_gmc) doesn't cancel forward-motion parallax,
        # and GT doesn't enrich traffic_light, so pixel motion would otherwise
        # leak through as "moving".
        if self.obj_type == "traffic_light":
            return "stationary"

        ema_v = self.velocity_px_per_frame()
        avg_v = None
        if len(self.positions) >= 2:
            span = self.frame_indices[-1] - self.frame_indices[0]
            if span > 0:
                dx = self.positions[-1][0] - self.positions[0][0]
                dy = self.positions[-1][1] - self.positions[0][1]
                avg_v = math.hypot(dx, dy) / span

        v = max(x for x in (ema_v, avg_v) if x is not None) if (ema_v is not None or avg_v is not None) else None
        if v is None:                    return "unknown"
        if v < MOTION_STATIONARY_PX:    return "stationary"
        if v < MOTION_SLOW_PX:          return "slow"
        if v < MOTION_FAST_PX:          return "moving"
        return "fast"

    def direction_label(self) -> str:
        if self.obj_type == "traffic_light":
            return "stationary"
        speed = self.speed_label()
        if speed in ("stationary", "unknown"):
            return "stationary"

        if len(self.positions) < 2:
            return "stationary"

        dx = self.positions[-1][0] - self.positions[0][0]
        dy = self.positions[-1][1] - self.positions[0][1]
        dist = math.hypot(dx, dy)

        if dist < MOTION_STATIONARY_PX:
            return "stationary"
        angle = math.degrees(math.atan2(-dy, dx))
        if   -45  <= angle <  45:  return "moving right"
        elif  45  <= angle < 135:  return "moving up"
        elif -135 <= angle < -45:  return "moving down"
        else:                      return "moving left"

_track_states: Dict[str, TrackState] = {}   # keyed by identity string
_identity_to_canonical: Dict[str, int] = {} # identity → small int for display
_next_canonical_id: int = 1


def _get_or_create_state(identity: str, obj_type: str) -> TrackState:
    if identity not in _track_states:
        _track_states[identity] = TrackState(identity, obj_type)
    return _track_states[identity]


def _canonical_id_for(identity: str) -> int:
    """Allocate a stable small-integer ID per identity by first sight.
    Same identity (instance_token or yolo_{tid}) → same canonical_id forever."""
    global _next_canonical_id
    cid = _identity_to_canonical.get(identity)
    if cid is None:
        cid = _next_canonical_id
        _identity_to_canonical[identity] = cid
        _next_canonical_id += 1
    return cid


def reset_track_states():
    global _track_states, _identity_to_canonical, _next_canonical_id
    _track_states          = {}
    _identity_to_canonical = {}
    _next_canonical_id     = 1

#  GMC  Global Motion Compensation (ego-pose only)
_cumulative_tx: float = 0.0
_cumulative_ty: float = 0.0
_frame_gmc_M_inv: Optional[np.ndarray] = None   # 3×3 homography inverse (curr→prev, full-res)

#  Ego-pose state (NuScenes only) 
_ego_poses: Optional[List[Dict]] = None
_ego_pose_cam_K: Optional[np.ndarray] = None
_ego_pose_cam_R: Optional[np.ndarray] = None   # R: ego frame → camera frame (3×3)
_ego_pose_cam_t: Optional[np.ndarray] = None   # t: sensor position in ego frame (3,)
 
def _quat_to_rotmat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)],
    ], dtype=np.float64)

def set_ego_poses(
    poses: List[Dict],
    cam_intrinsic: List[List[float]],
    cam_rotation_quat,
    cam_translation,
):
    #Load NuScenes ground-truth ego-pose data.
    global _ego_poses, _ego_pose_cam_K, _ego_pose_cam_R, _ego_pose_cam_t
    _ego_poses      = poses
    _ego_pose_cam_K = np.array(cam_intrinsic, dtype=np.float64)
    _ego_pose_cam_R = _quat_to_rotmat(cam_rotation_quat)
    _ego_pose_cam_t = np.array(cam_translation, dtype=np.float64)
    print(f"✅ Ego-pose GMC loaded ({len(poses)} keyframes)")
def reset_gmc():
    global _cumulative_tx, _cumulative_ty, _frame_gmc_M_inv
    _cumulative_tx    = 0.0
    _cumulative_ty    = 0.0
    _frame_gmc_M_inv  = None


def compute_gmc(frame_idx: int = 0) -> Tuple[float, float, float, float]:
    """
    Ego-pose-based Global Motion Compensation for NuScenes scenes.
    Builds a rotation-only homography from the ego-pose quaternions and stores
    the inverse in _frame_gmc_M_inv for compensate_bbox / add_observation.
    """
    global _cumulative_tx, _cumulative_ty, _frame_gmc_M_inv

    if _ego_poses is None or frame_idx == 0 or frame_idx >= len(_ego_poses):
        _frame_gmc_M_inv = None
        return 0.0, 0.0, _cumulative_tx, _cumulative_ty

    pose1 = _ego_poses[frame_idx - 1]
    pose2 = _ego_poses[frame_idx]

    R_w_e1 = _quat_to_rotmat(pose1["rotation"])
    R_w_e2 = _quat_to_rotmat(pose2["rotation"])

    # Camera orientation in world frame
    R_w_c1 = R_w_e1 @ _ego_pose_cam_R
    R_w_c2 = R_w_e2 @ _ego_pose_cam_R

    # Relative rotation: cam2 → cam1  (exact for background at optical infinity)
    R_c2_to_c1 = R_w_c1.T @ R_w_c2

    K     = _ego_pose_cam_K
    K_inv = np.linalg.inv(K)

    # 3×3 rotation homography mapping current pixels → previous pixels
    H_bwd = K @ R_c2_to_c1 @ K_inv

    denom = H_bwd[2, 2]
    if abs(denom) < 1e-9:
        _frame_gmc_M_inv = None
        return 0.0, 0.0, _cumulative_tx, _cumulative_ty

    _frame_gmc_M_inv = H_bwd / denom          # shape (3, 3)

    # Apparent pixel shift of the principal point (for cumulative offset display)
    cx, cy = K[0, 2], K[1, 2]
    pt     = H_bwd @ np.array([cx, cy, 1.0])
    pt    /= pt[2]
    tx = cx - float(pt[0])
    ty = cy - float(pt[1])

    _cumulative_tx += tx
    _cumulative_ty += ty

    return tx, ty, _cumulative_tx, _cumulative_ty


def compensate_bbox(bbox: Dict, M_inv: Optional[np.ndarray],
                    frame_w: int = 99999, frame_h: int = 99999) -> Dict:
    """
    Apply the inverse camera-motion homography to a bbox so ByteTracker's IoU
    matching works in a stable (previous-frame) coordinate system.

    M_inv is a 3×3 homography → must perspective-divide by the w-component.
    Output is clamped to [0, frame_w/h] so ByteTracker doesn't drop edge tracks.
    """
    if M_inv is None:
        return bbox

    corners = np.array([
        [bbox["x1"], bbox["y1"], 1.0],
        [bbox["x2"], bbox["y1"], 1.0],
        [bbox["x2"], bbox["y2"], 1.0],
        [bbox["x1"], bbox["y2"], 1.0],
    ])  # (4, 3)

    warped_h = (M_inv @ corners.T).T            # (4, 3) homogeneous
    w_col    = warped_h[:, 2:3]
    w_col    = np.where(np.abs(w_col) < 1e-9, 1e-9, w_col)
    warped   = warped_h[:, :2] / w_col          # (4, 2) Euclidean

    return {
        "x1": max(0,       int(warped[:, 0].min())),
        "y1": max(0,       int(warped[:, 1].min())),
        "x2": min(frame_w, int(warped[:, 0].max())),
        "y2": min(frame_h, int(warped[:, 1].max())),
    }
 
# Core per-frame function 
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
      "gmc_cum_tx": float, "gmc_cum_ty": float,
      "scene_summary": { ... }
    }
    """
    import supervision as sv

    model   = get_yolo()
    tracker = get_tracker()

    h_orig, w_orig = frame_bgr.shape[:2]
    scale  = min(YOLO_INPUT_MAX_DIM / max(h_orig, w_orig), 1.0)
    small  = cv2.resize(frame_bgr, (int(w_orig * scale), int(h_orig * scale)))

    #  GMC: returns per-frame delta AND cumulative offset 
    tx, ty, cum_tx, cum_ty = compute_gmc(frame_idx)

    #  YOLO detection 
    vehicle_count       = 0
    pedestrian_count    = 0
    cyclist_count       = 0
    traffic_light_count = 0

    boxes_raw: List[List[int]]  = []
    confidences_raw: List[float] = []
    class_ids_raw: List[int]    = []
    obj_types_raw: List[str]    = []
    bboxes_raw: List[Dict]      = []  

    if model:
        results = model(small, conf=YOLO_CONF_THRESH, imgsz=YOLO_IMGSZ, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id     = int(box.cls[0])
            cls_name   = model.names[cls_id]
            confidence = float(box.conf[0])
            area       = (x2 - x1) * (y2 - y1)

         #   if area < MIN_DETECTION_AREA and cls_name != "traffic light":
          #      continue

            sem_type = _semantic_type(cls_name)
            if sem_type in VEHICLE_CLASSES:    vehicle_count      += 1
            elif sem_type == "cyclist":        cyclist_count      += 1
            elif sem_type == "pedestrian":     pedestrian_count   += 1
            elif sem_type == "traffic_light":  traffic_light_count += 1
            bbox_orig = {
                "x1": int(x1 / scale), "y1": int(y1 / scale),
                "x2": int(x2 / scale), "y2": int(y2 / scale),
            }
            # GMC-compensated bbox for stable IoU matching in ByteTracker
            bbox_gmc  = compensate_bbox(bbox_orig, _frame_gmc_M_inv, w_orig, h_orig)

            boxes_raw.append([bbox_gmc["x1"], bbox_gmc["y1"],
                               bbox_gmc["x2"], bbox_gmc["y2"]])
            confidences_raw.append(min(1.0, max(0.0, confidence)))
            class_ids_raw.append(cls_id)
            obj_types_raw.append(sem_type)
            bboxes_raw.append(bbox_orig)

    # Per-detection provenance: GT-merge flags survive the tracker via this
    # parallel list (re-attached to the tracked output further down).
    provenance_raw: List[Dict] = [
        {"source": "yolo", "gt_sourced": False} for _ in bboxes_raw
    ]

    # GT injection (Bug 8): at sweep rate, IoU-match YOLO ↔ GT and append
    # any GT objects YOLO missed entirely so ByteTracker sees them too.
    frame_gt = _nuscenes_gt_by_frame.get(frame_idx, {})
    gt_anns  = frame_gt.get("annotations", []) if frame_gt else []
    if gt_anns:
        _tmp_dets = [
            {"id": i, "type": ot, "confidence": cf,
             "bounding_box": bb, "track_id": -1}
            for i, (bb, cf, ot) in enumerate(
                zip(bboxes_raw, confidences_raw, obj_types_raw))
        ]
        _tmp_yolo = {
            "detections":          _tmp_dets,
            "vehicle_count":       vehicle_count,
            "pedestrian_count":    pedestrian_count,
            "cyclist_count":       cyclist_count,
            "traffic_light_count": traffic_light_count,
            "total_objects":       len(_tmp_dets),
        }
        n_before = len(_tmp_dets)
        _tmp_yolo = merge_gt_into_detections(
            _tmp_yolo, gt_anns, w_orig, h_orig
        )

        # Pass 1: copy enrichment flags onto provenance for existing detections.
        # If GT corrected the type, also update obj_types_raw so ByteTracker
        # downstream uses the correct class.
        for i, det in enumerate(_tmp_yolo["detections"][:n_before]):
            for flag in ("gt_sourced", "gt_type_override", "motion_source",
                          "instance_token", "gt_iou", "yolo_type_original",
                          "speed", "direction", "action"):
                if flag in det:
                    provenance_raw[i][flag] = det[flag]
            if det.get("gt_type_override"):
                obj_types_raw[i] = det["type"]

        # Pass 2: append GT-injected entries to tracker input so ByteTracker
        # assigns them a stable track_id across sweeps.
        for det in _tmp_yolo["detections"][n_before:]:
            bb     = det["bounding_box"]
            bb_gmc = compensate_bbox(bb, _frame_gmc_M_inv, w_orig, h_orig)
            bboxes_raw.append(bb)
            boxes_raw.append([bb_gmc["x1"], bb_gmc["y1"],
                              bb_gmc["x2"], bb_gmc["y2"]])
            confidences_raw.append(0.99)
            class_ids_raw.append(0)
            obj_types_raw.append(det["type"])
            provenance_raw.append({
                "source":           "gt_injected",
                "gt_sourced":       True,
                "gt_type_override": False,
                "motion_source":    "nuscenes_gt",
                "instance_token":   det.get("instance_token", ""),
                "speed":            det.get("speed",     "unknown"),
                "direction":        det.get("direction", "stationary"),
                "action":           det.get("action",    "unknown"),
            })

    # ByteTracker update
    tracked_objects: List[Dict] = []

    if tracker and boxes_raw:
        detections = sv.Detections(
            xyxy       = np.array(boxes_raw,       dtype=np.float32),
            confidence = np.array(confidences_raw, dtype=np.float32),
            class_id   = np.array(class_ids_raw,   dtype=np.int32),
        )
        detections.data["object_type"] = obj_types_raw
        gmc_cx_to_orig: Dict[Tuple[int, int], List[Tuple[Dict, Dict]]] = {}
        for gmc_box, orig_bbox, prov in zip(boxes_raw, bboxes_raw, provenance_raw):
            key = (
                int((gmc_box[0] + gmc_box[2]) / 2),
                int((gmc_box[1] + gmc_box[3]) / 2),
            )
            gmc_cx_to_orig.setdefault(key, []).append((orig_bbox, prov))

        detections = tracker.update_with_detections(detections)

        for i in range(len(detections)):
            box      = detections.xyxy[i]
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            conf     = float(detections.confidence[i])
            obj_type = (detections.data["object_type"][i]
                        if "object_type" in detections.data else "unknown")

            tracked_cx = int((box[0] + box[2]) / 2)
            tracked_cy = int((box[1] + box[3]) / 2)
            best_key   = min(
                gmc_cx_to_orig.keys(),
                key=lambda k: math.hypot(k[0] - tracked_cx, k[1] - tracked_cy),
                default=None,
            )
            prov: Dict = {"source": "yolo", "gt_sourced": False}
            if best_key is not None:
                euclidean = math.hypot(
                    best_key[0] - tracked_cx, best_key[1] - tracked_cy
                )
                if euclidean < 50.0:
                    candidates  = gmc_cx_to_orig[best_key]
                    bbox, prov  = candidates.pop(0)
                    if not candidates:
                        del gmc_cx_to_orig[best_key]
                else:
                    bbox = {
                        "x1": int(box[0] + tx), "y1": int(box[1] + ty),
                        "x2": int(box[2] + tx), "y2": int(box[3] + ty),
                    }
            else:
                bbox = {
                    "x1": int(box[0] + tx), "y1": int(box[1] + ty),
                    "x2": int(box[2] + tx), "y2": int(box[3] + ty),
                }

            is_pedestrian = obj_type == "pedestrian"
            pixel_color, color_conf = (
                ("unknown", 0.0) if is_pedestrian
                else extract_dominant_color_clip(
                    frame_bgr, bbox, obj_type,
                    other_bboxes=[b for b in bboxes_raw if b is not bbox],
                )
            )

            # Identity-first state: instance_token bridges across ByteTrack
            # fragments; YOLO-only objects fall back to per-track-id identity.
            it       = prov.get("instance_token", "") if prov else ""
            identity = it if it else f"yolo_{track_id}"
            cid      = _canonical_id_for(identity)

            state = _get_or_create_state(identity, obj_type)
            state.add_observation(
                frame_idx, bbox, pixel_color,
                color_confidence=color_conf,
                cum_tx=cum_tx, cum_ty=cum_ty,
                frame_M_inv=_frame_gmc_M_inv,
            )
            cx_px = (bbox["x1"] + bbox["x2"]) / 2
            horizontal = (
                "Left"   if cx_px < w_orig * 0.40 else
                "Right"  if cx_px > w_orig * 0.60 else
                "Center"
            )
            cy_px  = (bbox["y1"] + bbox["y2"]) / 2
            depth  = "Foreground" if cy_px > h_orig * 0.50 else "Background"
            position = f"{depth} {horizontal}"

            obj_entry: Dict = {
                "id":           i + 1,
                "track_id":     cid,            # primary ID surfaced to JSON / draw_tracks
                "canonical_id": cid,
                "bytetrack_id": track_id,       # raw tracker ID kept for debugging
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
            if obj_type in VEHICLE_CLASSES or obj_type == "cyclist":
                obj_entry["size"] = _size_label(bbox, w_orig, h_orig)

            # Attach GT provenance and override tracker motion with LiDAR-GT
            # values where available (mirrors merge_gt_into_detections semantics).
            if prov.get("gt_sourced"):
                obj_entry["gt_sourced"]    = True
                obj_entry["motion_source"] = prov.get("motion_source", "nuscenes_gt")
                if prov.get("instance_token"):
                    obj_entry["instance_token"] = prov["instance_token"]
                if "gt_iou" in prov:
                    obj_entry["gt_iou"] = prov["gt_iou"]
                obj_entry["gt_type_override"] = bool(prov.get("gt_type_override", False))
                if "yolo_type_original" in prov:
                    obj_entry["yolo_type_original"] = prov["yolo_type_original"]
                for f in ("speed", "direction", "action"):
                    if f in prov and prov[f] is not None:
                        obj_entry[f] = prov[f]
            if prov.get("source") == "gt_injected":
                obj_entry["source"] = "gt_injected"

            tracked_objects.append(obj_entry)

    elif not tracker and boxes_raw:
        # Fallback: no ByteTracker — basic detection only
        for i, (bbox, conf, obj_type) in enumerate(
                zip(bboxes_raw, confidences_raw, obj_types_raw)):
            is_pedestrian = obj_type == "pedestrian"
            pixel_color, _color_conf = (
                ("unknown", 0.0) if is_pedestrian
                else extract_dominant_color_clip(
                    frame_bgr, bbox, obj_type,
                    other_bboxes=[b for b in bboxes_raw if b is not bbox],
                )
            )
            cx_px = (bbox["x1"] + bbox["x2"]) / 2
            horizontal = (
                "Left"   if cx_px < w_orig * 0.40 else
                "Right"  if cx_px > w_orig * 0.60 else
                "Center"
            )
            cy_px  = (bbox["y1"] + bbox["y2"]) / 2
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
        "frame":      frame_idx,
        "gmc_tx":     round(tx,     2),
        "gmc_ty":     round(ty,     2),
        "gmc_cum_tx": round(cum_tx, 2),
        "gmc_cum_ty": round(cum_ty, 2),
        "scene_summary": {
            "total_vehicles_detected":       vehicle_count,
            "total_pedestrians_detected":    pedestrian_count,
            "total_cyclists_detected":       cyclist_count,
            "total_traffic_lights_detected": traffic_light_count,
            "detected_objects":              tracked_objects,
        },
    }

    if out_json_path:
        os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(output, f, indent=2)

    return output


#  Annotated imgs
def draw_tracks(frame_bgr: np.ndarray, frame_data: Dict) -> np.ndarray:
    """
    Draw bounding boxes with colour-coded provenance:
      Orange  = GT injected        (YOLO missed this object entirely)
      Yellow  = GT type-corrected  (YOLO found it but had wrong class)
      Green   = GT motion-enriched (YOLO found it, GT fixed speed/action)
      White   = YOLO only          (no GT match)
    """
    canvas = frame_bgr.copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for obj in frame_data.get("scene_summary", {}).get("detected_objects", []):
        bb = obj.get("bounding_box", {})
        if not bb:
            continue
        x1 = int(bb.get("x1", 0)); y1 = int(bb.get("y1", 0))
        x2 = int(bb.get("x2", 0)); y2 = int(bb.get("y2", 0))
        # skip zero-size or inverted boxes — prevents phantom label at (0,0)
        if x2 <= x1 or y2 <= y1:
            continue

        source = obj.get("source", "yolo")

        # Prefer canonical_id (allocated inline from instance_token in
        # run_yolo_and_track, stable across ByteTrack fragments). track_id
        # is now also the canonical id; fall back to id for legacy rows.
        cid = obj.get("canonical_id")
        if cid is None or cid == -1:
            cid = obj.get("track_id", obj.get("id", "?"))

        if source == "gt_injected":
            color = (0, 165, 255)           # orange
            label = f"#{cid} GT-inject"
        elif obj.get("gt_type_override"):
            color = (0, 255, 255)           # yellow
            label = f"#{cid} GT-type"
        elif obj.get("gt_sourced"):
            color = (0, 255, 128)           # green
            label = f"#{cid}"
        else:
            color = (255, 255, 255)         # white
            label = f"#{cid}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        ly = y1 - 6 if y1 > 16 else y1 + th + 6
        cv2.rectangle(canvas, (x1, ly - th - 3), (x1 + tw + 5, ly + 3),
                      (0, 0, 0), -1)
        cv2.putText(canvas, label, (x1 + 2, ly), font, 0.45,
                    color, 1, cv2.LINE_AA)
    return canvas


def process_scene_sweeps(
    scene_name: str,
    dataroot: str = NUSCENES_DATAROOT,
    out_dir:  str = "output",
    camera:   str = "CAM_FRONT",
) -> Tuple[int, int]:
    """
    Process a NuScenes scene at native sweep rate (~12 Hz) through YOLO +
    ByteTracker, reading JPGs directly from the SDK (no mp4, no cv2 seek).

    Per-frame detections are written for every sweep so tracking has dense
    motion data; a separate keyframes/ directory holds only sample-aligned
    frames (2 Hz) for the downstream Qwen annotation pipeline. A
    keyframe_map.json links sweep frame_idx ↔ sample_idx ↔ sample_token.
    """
    global _tracker_actual_fps

    _t_start = time.perf_counter()

    import shutil
    if not NUSCENES_AVAILABLE:
        raise RuntimeError("nuScenes SDK not available")

    frames_dir    = os.path.join(out_dir, "frames")
    annotated_dir = os.path.join(out_dir, "annotated")
    keyframes_dir = os.path.join(out_dir, "keyframes")
    for d in (frames_dir, annotated_dir, keyframes_dir):
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    nusc  = _init_nuscenes_sdk(dataroot)
    scene = next((s for s in nusc.scene if scene_name in s["name"]), None)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' not found in {dataroot}")

    # Walk CAM sample_data chain: 12 Hz mix of samples (is_key_frame=True) + sweeps
    first_sample = nusc.get("sample", scene["first_sample_token"])
    sd_token     = first_sample["data"][camera]
    sd_records   = []
    while sd_token:
        sd = nusc.get("sample_data", sd_token)
        sd_records.append(sd)
        sd_token = sd["next"]

    n_total     = len(sd_records)
    n_keyframes = sum(1 for sd in sd_records if sd["is_key_frame"])

    # Native fps from timestamps (microseconds)
    t_first       = sd_records[0]["timestamp"]
    t_last        = sd_records[-1]["timestamp"]
    duration_secs = max(1e-6, (t_last - t_first) / 1e6)
    actual_fps    = n_total / duration_secs

    print(f"\n{'='*55}")
    print(f"  Scene             : {scene['name']}")
    print(f"  Camera            : {camera}")
    print(f"  Total frames      : {n_total}  ({n_keyframes} keyframes)")
    print(f"  Duration          : {duration_secs:.1f}s @ {actual_fps:.2f} fps")
    print(f"{'='*55}\n")

    # Camera calibration & full 12 Hz ego-pose list (one per sample_data record)
    calib     = nusc.get("calibrated_sensor", sd_records[0]["calibrated_sensor_token"])
    ego_poses = [nusc.get("ego_pose", sd["ego_pose_token"]) for sd in sd_records]

    _tracker_actual_fps = actual_fps
    reset_tracker(int(round(actual_fps)))
    reset_track_states()
    reset_gmc()
    set_ego_poses(
        ego_poses,
        cam_intrinsic     = calib["camera_intrinsic"],
        cam_rotation_quat = calib["rotation"],
        cam_translation   = calib["translation"],
    )

    keyframe_map: List[Dict] = []
    sample_idx               = 0

    for frame_idx, sd in enumerate(sd_records):
        img_path = os.path.join(dataroot, sd["filename"])
        frame    = cv2.imread(img_path)
        if frame is None:
            print(f"  ⚠ could not read {img_path}")
            continue

        json_path = os.path.join(frames_dir,    f"frame_{frame_idx:06d}.json")
        jpg_path  = os.path.join(annotated_dir, f"frame_{frame_idx:06d}_track.jpg")

        data = run_yolo_and_track(frame, frame_idx, out_json_path=None)
        data["is_keyframe"] = bool(sd["is_key_frame"])
        data["timestamp"]   = sd["timestamp"]
        if sd["is_key_frame"]:
            data["sample_token"] = sd["sample_token"]
            data["sample_idx"]   = sample_idx

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        canvas = draw_tracks(frame, data)
        cv2.imwrite(jpg_path, canvas)

        if sd["is_key_frame"]:
            # Sample-aligned copy for Qwen / annotation pipeline
            kf_path = os.path.join(keyframes_dir, f"keyframe_{sample_idx:04d}.json")
            with open(kf_path, "w") as f:
                json.dump(data, f, indent=2)
            keyframe_map.append({
                "sample_idx":   sample_idx,
                "frame_idx":    frame_idx,
                "sample_token": sd["sample_token"],
                "timestamp":    sd["timestamp"],
            })
            sample_idx += 1

        n      = len(data["scene_summary"]["detected_objects"])
        marker = " [KEY]" if sd["is_key_frame"] else ""
        print(
            f"  Frame {frame_idx:06d}: {n} tracked | "
            f"GMC ({data['gmc_tx']:+.1f}, {data['gmc_ty']:+.1f}){marker}"
        )

    with open(os.path.join(out_dir, "keyframe_map.json"), "w") as f:
        json.dump(keyframe_map, f, indent=2)

    global_summary = get_track_motion_summary(0, int(duration_secs) + 1)
    summary_path   = os.path.join(out_dir, "tracks_summary.json")
    with open(summary_path, "w") as f:
        json.dump(global_summary, f, indent=2)

    _elapsed = time.perf_counter() - _t_start
    print(f"\n✅ Processed {n_total} sweeps ({n_keyframes} keyframes) "
          f"in {_elapsed:.1f}s ({_elapsed / max(n_total, 1):.2f}s/sweep)")
    print(f"   Per-sweep:  {frames_dir}/")
    print(f"   Annotated:  {annotated_dir}/")
    print(f"   Keyframes:  {keyframes_dir}/  (sample-aligned for Qwen)")
    print(f"   Summary:    {summary_path}")

    return n_total, int(duration_secs)


#  Cumulative motion summary
def get_track_motion_summary(
    span_start_sec: int,
    span_end_sec: int,
    fps: Optional[float] = None,
) -> List[Dict]:

    if fps is not None:
        actual_fps = fps
    elif _tracker_actual_fps > 0:
        actual_fps = _tracker_actual_fps
    elif _tracker_frame_rate > 0:
        actual_fps = float(_tracker_frame_rate)
    else:
        actual_fps = 25.0 

    summaries   = []
    start_frame = int(span_start_sec * actual_fps)
    end_frame   = int(span_end_sec   * actual_fps)

    for identity, state in _track_states.items():
        active = [
            (fi, pos)
            for fi, pos in zip(state.frame_indices, state.positions)
            if start_frame <= fi <= end_frame
        ]
        if not active:
            continue

        indices   = [a[0] for a in active]
        positions = [a[1] for a in active]

        dist = 0.0
        if len(positions) >= 2:
            dx   = positions[-1][0] - positions[0][0]
            dy   = positions[-1][1] - positions[0][1]
            dist = math.hypot(dx, dy)

        summary: Dict = {
            "identity":         identity,
            "track_id":         _identity_to_canonical.get(identity, -1),
            "object_type":      state.obj_type,
            "speed":            state.speed_label(),
            "direction":        state.direction_label(),
            "distance_px":      round(dist, 1),
            "frame_span":       [indices[0], indices[-1]],
            "evidence_seconds": sorted(set(int(fi / actual_fps) for fi in indices)),
        }
        if state.obj_type != "pedestrian":
            summary["color"] = state.color
        summaries.append(summary)
    return summaries
if __name__ == "__main__":
    # Sweep-rate (~12 Hz) tracking on NuScenes; downstream Qwen consumes the
    # sample-aligned keyframes/ output (2 Hz).
    process_scene_sweeps(TARGET_SCENE, dataroot=NUSCENES_DATAROOT, out_dir="output")
    summary = get_track_motion_summary(0, 9999)
    print(f"\nMotion summary ({len(summary)} tracks):")
    for s in summary[:10]:
        color_str = s.get("color", "n/a")
        print(
            f"  Track {s['track_id']} ({color_str} {s['object_type']}): "
            f"{s['direction']} @ {s['speed']} | {s['distance_px']:.0f}px"
        )