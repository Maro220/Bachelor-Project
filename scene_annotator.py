import base64
import math
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import whisper
from datetime import datetime
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()
import cv2
import requests
from flask import Flask, jsonify, render_template, request, send_file
try:
    from yolo_bytetrack import (
        run_yolo_and_track,
        process_video_frames,
        process_scene_sweeps,
        get_track_motion_summary,
        reset_tracker,
        draw_tracks,
        set_ego_poses,
        TRACKER_FRAME_RATE,
        NUSCENES_DATAROOT,
        VIDEO_CHUNK_SECONDS as _YBT_CHUNK,
    )
    TRACKER_AVAILABLE = True
except Exception as e:
    print(f"yolo_bytetrack import failed: {e}")
    TRACKER_AVAILABLE = False
    run_yolo_and_track = None
    process_video_frames = None
    process_scene_sweeps = None
    get_track_motion_summary = None
    NUSCENES_DATAROOT = "data/v1.0-mini"

try:
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False
    print("google-api-python-client not installed. Responses saved locally only.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ultralytics not installed. Skipping YOLO detection.")

OLLAMA_URL       = "http://localhost:11434/api/chat"
MODEL     = "qwen2.5vl:7b"
CREDENTIALS_FILE = "credentials.json"
OAUTH_FILE       = "oauth_client.json"
TOKEN_FILE       = "token.json"
SHEET_ID_FILE    = "spreadsheet_id.txt"
SCOPES           = ["https://www.googleapis.com/auth/spreadsheets"]
SERVER_PORT      = 7860
DEFAULT_SCENE    = "scene-0757"   # NuScenes scene name (sweep-based pipeline; no path)
DEFAULT_TARGET   = f"output_nuscenes/{DEFAULT_SCENE}/{DEFAULT_SCENE}_CAM_FRONT.mp4"  # mp4 path used ONLY by /video/preview in the form
VIDEO_CHUNK_SECONDS = _YBT_CHUNK if TRACKER_AVAILABLE else 5

FIELD_OPTIONS = {
    "environment":    ["urban street", "highway", "parking lot", "intersection", "residential area", "school zone", "construction zone"],
    "lighting":       ["bright daylight", "low-light", "night with street lights", "night without lighting"],
    "traffic_density":["empty", "light", "moderate", "heavy", "gridlock"],
    "traffic_flow":   ["free-flowing", "slow-moving", "stopped", "mixed"],
}
OBJECT_TYPE_OPTIONS   = ["car", "van", "truck", "bus", "motorcycle", "cyclist", "pedestrian", "traffic_light", "road_sign", "other"]
VEHICLE_CLASSES       = {"car", "van", "motorcycle", "bus", "truck"}
SHEET_HEADERS = [
    "Annotator Type",      
    "Participant Name",
    "Country",
    "Age Category",
    "Gender",
    "Profession",
    "Driving Skill",
    "Scene ID",
    "Environment",
    "Lighting",
    "Traffic Density",
    "Traffic Flow",
    "Total Vehicles",
    "Total Pedestrians",
    "Total Cyclists",
    "Total Traffic Lights",
    "Scene Narrative",
    "Hazards and Events",
    "Video Review",
    "Object Count Summary",
]
_nuscenes_gt_by_frame: dict = {}  
def set_nuscenes_gt(gt_data: dict):
    global _nuscenes_gt_by_frame
    _nuscenes_gt_by_frame = {
        f["frame_video_idx"]: f
        for f in gt_data.get("frames", [])
    }
    total_anns = sum(len(f.get("annotations", [])) for f in gt_data.get("frames", []))
    print(f"✅ NuScenes GT loaded: {len(_nuscenes_gt_by_frame)} frames, "
          f"{total_anns} total annotations")
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
    "movable_object.trafficcone":          "other",
    "movable_object.barrier":              "other",
    "movable_object.pushable_pullable":    "other",
    "static_object.bicycle_rack":          "other",
}
def _nuscenes_to_yolo_type(semantic_type: str) -> str:
    return _NUSCENES_TO_YOLO_TYPE.get(semantic_type, "other")
def inject_missed_gt_objects(yolo_data: dict, gt_annotations: list,
                              frame_w: int, frame_h: int) -> dict:
    """
    Add GT-annotated objects that YOLO missed into the detection list.
    Flagged as source='gt_injected' so downstream code knows provenance.
    Only injects objects with visibility >= 2 (≥40% visible).
    Uses IoU > 0.3 to avoid double-counting.
    """
    existing_bboxes = [
        d["bounding_box"] for d in yolo_data["detections"]
        if d.get("bounding_box")
    ]

    injected = 0
    for ann in gt_annotations:
        if ann.get("visibility", 0) < 2:
            continue

        gt_bbox = ann.get("bbox_2d", {})
        if not gt_bbox:
            continue

        already_detected = any(
            _bbox_iou(gt_bbox, eb) > 0.3 for eb in existing_bboxes
        )
        if already_detected:
            continue

        cx         = (gt_bbox["x1"] + gt_bbox["x2"]) / 2
        cy         = (gt_bbox["y1"] + gt_bbox["y2"]) / 2
        horizontal = "Left" if cx < frame_w * 0.4 else "Right" if cx > frame_w * 0.6 else "Center"
        depth      = "Foreground" if cy > frame_h * 0.5 else "Background"
        yolo_type  = _nuscenes_to_yolo_type(ann["semantic_type"])

        injected_obj = {
            "id":           len(yolo_data["detections"]) + injected + 1,
            "type":         yolo_type,
            "confidence":   1.0,
            "position":     f"{depth} {horizontal}",
            "area":         max(0, (gt_bbox["x2"] - gt_bbox["x1"]) *
                                   (gt_bbox["y2"] - gt_bbox["y1"])),
            "bounding_box": gt_bbox,
            "track_id":     -1,
            "speed":        ann.get("speed_label", "unknown"),
            "direction":    ann.get("direction", "unknown"),
            "action":       ann.get("action", "unknown"),
            "color":        "unknown",
            "size":         ann.get("size_category", "unknown"),
            "source":       "gt_injected",
        }

        yolo_data["detections"].append(injected_obj)
        existing_bboxes.append(gt_bbox)
        injected += 1

    if injected > 0:
        type_counts = {}
        for d in yolo_data["detections"]:
            t = d.get("type", "other")
            type_counts[t] = type_counts.get(t, 0) + 1

        vehicle_types = {"car", "van", "truck", "bus", "motorcycle"}
        yolo_data["vehicle_count"]       = sum(type_counts.get(t, 0) for t in vehicle_types)
        yolo_data["pedestrian_count"]    = type_counts.get("pedestrian", 0)
        yolo_data["cyclist_count"]       = type_counts.get("cyclist", 0)
        yolo_data["traffic_light_count"] = type_counts.get("traffic_light", 0)
        yolo_data["total_objects"]       = len(yolo_data["detections"])
        print(f"  ✓ GT injection: +{injected} missed objects → "
              f"total now {yolo_data['total_objects']}")

    return yolo_data


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


def _yolo_data_from_frame_result(frame_data: dict) -> dict:
    """Convert yolo_bytetrack frame output to the legacy yolo_data format expected by analyse_frame."""
    ss = frame_data.get("scene_summary", {})
    dets = ss.get("detected_objects", [])
    detections = []
    for d in dets:
        detections.append({
            "id":           d.get("id"),
            "type":         d.get("type"),
            "confidence":   d.get("confidence"),
            "position":     d.get("position"),
            "area":         d.get("area"),
            "bounding_box": d.get("bounding_box") or {},
            "track_id":     d.get("track_id", -1),
            "color":        d.get("color", "unknown"),
            "size":         d.get("size", "unknown"),
            "speed":        d.get("speed", "unknown"),
            "direction":    d.get("direction", "unknown"),
        })
    return {
        "frame_info":          {"width": 0, "height": 0},
        "total_objects":       len(dets),
        "vehicle_count":       ss.get("total_vehicles_detected", 0),
        "pedestrian_count":    ss.get("total_pedestrians_detected", 0),
        "cyclist_count":       ss.get("total_cyclists_detected", 0),
        "traffic_light_count": ss.get("total_traffic_lights_detected", 0),
        "other_count":         0,
        "detections":          detections,
    }
def _safe_parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    s = raw.find("{")
    e = raw.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(raw[s:e + 1])
        except json.JSONDecodeError:
            pass
    snippet = raw[s:] if s != -1 else raw
    depth_b, depth_sq = 0, 0
    in_str, escape = False, False
    for ch in snippet:
        if escape:       escape = False; continue
        if ch == "\\": escape = True;  continue
        if ch == '"':    in_str = not in_str; continue
        if in_str:       continue
        if ch == "{":    depth_b  += 1
        elif ch == "}":  depth_b  -= 1
        elif ch == "[":  depth_sq += 1
        elif ch == "]":  depth_sq -= 1
    closing = "]" * max(depth_sq, 0) + "}" * max(depth_b, 0)
    try:
        return json.loads(snippet + closing)
    except json.JSONDecodeError:
        pass
    print("  ! Could not parse Qwen JSON — returning empty dict")
    print(f"    raw length: {len(raw)} chars")
    print(f"    raw[:400]:  {raw[:400]!r}")
    print(f"    raw[-200:]: {raw[-200:]!r}")
    return {}


def call_qwen_vision(frame_bgr, prompt):
    h, w   = frame_bgr.shape[:2]
    scale  = min(512 / max(h, w), 1.0)
    small  = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    _, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    b64    = base64.b64encode(encoded.tobytes()).decode()
    payload = {
        "model":   MODEL,
        "format":  "json",
        "stream":  False,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "options": {
            "temperature": 0.1,
            "num_ctx":     8192,
            "num_predict": 2048,
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        return _safe_parse_json(r.json()["message"]["content"])
    except requests.Timeout:
        print(f" Qwen vision timeout (180s) - Ollama server slow or unreachable")
        return {}
    except requests.ConnectionError as e:
        print(f" Cannot connect to Ollama at {OLLAMA_URL}: {e}")
        return {}
    except requests.HTTPError as e:
        print(f" Ollama returned HTTP error: {e.response.status_code} {e}")
        return {}
    except (KeyError, ValueError) as e:
        print(f" Invalid response format from Ollama (missing message/content): {e}")
        return {}


def call_qwen_text(prompt):
    payload = {
        "model":   MODEL,
        "format":  "json",
        "stream":  False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.0, "num_ctx": 16384, "num_predict": 3072}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        return _safe_parse_json(r.json()["message"]["content"])
    except requests.Timeout:
        print(f"  ✗ Qwen text timeout (300s) - Ollama server slow or unreachable")
        return {}
    except requests.ConnectionError as e:
        print(f"  ✗ Cannot connect to Ollama at {OLLAMA_URL}: {e}")
        return {}
    except requests.HTTPError as e:
        print(f"  ✗ Ollama returned HTTP error: {e.response.status_code} {e}")
        return {}
    except (KeyError, ValueError) as e:
        print(f"  ✗ Invalid response format from Ollama (missing message/content): {e}")
        return {}

def _derive_traffic_density(n_vehicles: int, n_cyclists: int) -> str:
    total = n_vehicles + n_cyclists
    if total == 0:  return "empty"
    if total <= 3:  return "light"
    if total <= 8:  return "moderate"
    if total <= 15: return "heavy"
    return "gridlock"


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
    # vehicles
    if not moving:              return "parked"
    if "left" in direction:     return "turning left"
    if "right" in direction:    return "turning right"
    return "moving"


def validate_qwen_against_gt(scene_summary: dict, gt_annotations: list) -> dict:
    """Score Qwen's scene_summary against NuScenes ground truth."""
    scores = {}
    vis_anns = [a for a in gt_annotations if a.get("visibility", 0) >= 2]

    gt_has_moving = any(a.get("action") in ("moving", "walking") for a in vis_anns)
    qwen_flow = scene_summary.get("traffic_flow", "")
    scores["traffic_flow_correct"] = bool(
        (gt_has_moving  and qwen_flow not in ("stopped",)) or
        (not gt_has_moving and qwen_flow in ("stopped", "slow-moving"))
    )

    vehicle_types = {"car", "van", "truck", "bus", "motorcycle"}
    gt_vehicles   = sum(1 for a in vis_anns
                        if any(vt in a.get("semantic_type", "") for vt in vehicle_types))
    qwen_vehicles = scene_summary.get("total_vehicles_detected", 0)
    scores["vehicle_count_close"] = abs(gt_vehicles - qwen_vehicles) <= 1

    gt_peds   = sum(1 for a in vis_anns if "pedestrian" in a.get("semantic_type", ""))
    qwen_peds = scene_summary.get("total_pedestrians_detected", 0)
    scores["pedestrian_count_close"] = abs(gt_peds - qwen_peds) <= 1

    qwen_hazards = scene_summary.get("hazards_and_events", "none").lower()

    HAZARD_TYPES = {
        "movable_object.trafficcone",
        "movable_object.barrier",
        "movable_object.debris",
        "movable_object.pushable_pullable",
    }
    gt_has_hazard = any(
        a.get("semantic_type", "") in HAZARD_TYPES or
        a.get("semantic_type", "").startswith("movable_object")
        for a in vis_anns
    )

    if gt_has_hazard:
        scores["hazard_detected"] = qwen_hazards != "none"
        scores["no_false_hazard"] = True
    else:
        scores["hazard_detected"] = True
        scores["no_false_hazard"] = qwen_hazards == "none"

    scores["overall_accuracy"] = round(
        sum(1 for v in scores.values() if v is True) / max(len(scores), 1), 2
    )

    return scores


def _build_ego_motion_attrs(frame_idx: int, fps: float) -> dict:
    #Returns ego-vehicle motion as a structured attribute dict (NuScenes only).
    try:
        from yolo_bytetrack import _ego_poses
        if _ego_poses is None or frame_idx == 0 or frame_idx >= len(_ego_poses):
            return {}
        p1    = _ego_poses[frame_idx - 1]["translation"]
        p2    = _ego_poses[frame_idx]["translation"]
        speed = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) * max(fps, 1.0)
        return {
            "ego_vehicle_speed_mps": round(speed, 1),
            "ego_vehicle_moving":    speed > 0.3,
        }
    except Exception:
        return {}
def analyse_frame(frame, scene_id, out_json_path, out_img_path, yolo_data=None, frame_idx=None, stable_fields=None, video_fps: float = 2.0):
    is_static = (scene_id == "static")

    if yolo_data is None:
        if TRACKER_AVAILABLE and run_yolo_and_track is not None:
            _fidx = frame_idx if frame_idx is not None else 0
            frame_data = run_yolo_and_track(frame, _fidx)
            yolo_data  = _yolo_data_from_frame_result(frame_data)
            if draw_tracks is not None:
                canvas = draw_tracks(frame, frame_data)
                yolo_img_path = out_img_path.replace(".jpg", "_yolo.jpg")
                cv2.imwrite(yolo_img_path, canvas)
                print(f"  Track-annotated image: {yolo_img_path}")
        else:
            print("  ⚠  No detector available — empty detections.")
            yolo_data = {
                "total_objects": 0, "vehicle_count": 0,
                "pedestrian_count": 0, "cyclist_count": 0,
                "traffic_light_count": 0, "other_count": 0,
                "detections": [],
            }

    # Inject GT objects that YOLO missed (NuScenes mode only)
    if _nuscenes_gt_by_frame and frame_idx is not None and frame is not None:
        frame_gt_all = _nuscenes_gt_by_frame.get(frame_idx, {})
        gt_anns      = frame_gt_all.get("annotations", [])
        if gt_anns:
            h_frame, w_frame = frame.shape[:2]
            yolo_data = inject_missed_gt_objects(
                yolo_data, gt_anns, w_frame, h_frame
            )

    os.makedirs("output/yolo_raw", exist_ok=True)
    with open(f"output/yolo_raw/yolo_raw_{scene_id}.json", "w") as f:
        json.dump(yolo_data, f, indent=2)

    cv2.imwrite(out_img_path, frame)

    n_det  = yolo_data["total_objects"]
    n_veh  = yolo_data["vehicle_count"]
    n_ped  = yolo_data["pedestrian_count"]
    n_cyc  = yolo_data["cyclist_count"]
    n_tl   = yolo_data["traffic_light_count"]

    track_motions = {}
    if not is_static:
        try:
            current_sec = int(scene_id)
            if get_track_motion_summary is not None:
                motions = get_track_motion_summary(0, current_sec + 1)
                for m in motions:
                    track_motions[m["track_id"]] = f"{m['speed']}, {m['direction']}"
        except ValueError as e:
            print(f"  ✗ Invalid scene_id format (expected integer): {e}")

    # Build structured detection data — pure attributes only.
    # Qwen synthesizes descriptions from these; it does not add facts.
    structured_dets = []
    for d in yolo_data["detections"]:
        tid   = d.get("track_id", -1)
        # gt_injected objects are 100% from NuScenes GT even before the
        # final_objs loop sets gt_motion. motion_source="nuscenes_gt"
        # is set by the earlier override for IoU-matched YOLO detections.
        _is_gt = (
            d.get("source") == "gt_injected" or
            d.get("motion_source") == "nuscenes_gt" or
            d.get("gt_motion", False)
        )
        entry = {
            "type":       d["type"],
            "position":   d["position"],
            "color":      d.get("color",     "unknown"),
            "size":       d.get("size",      "unknown"),
            "action":     d.get("action",    "unknown"),
            "speed":      d.get("speed",     "unknown"),
            "direction":  d.get("direction", "unknown"),
            "gt_sourced": _is_gt,
            "source":     d.get("source",    "yolo"),
        }
        if d.get("speed_mps") is not None:
            entry["speed_mps"] = d["speed_mps"]
        if tid in track_motions:
            entry["track_history"] = track_motions[tid]
        structured_dets.append(entry)

    # Structured GT confirmed objects list (replaces old gt_hint string)
    gt_confirmed = []
    if _nuscenes_gt_by_frame and frame_idx is not None:
        frame_gt     = _nuscenes_gt_by_frame.get(frame_idx, {})
        visible_anns = [a for a in frame_gt.get("annotations", [])
                        if a.get("visibility", 0) >= 2]
        if visible_anns:
            w_frame = frame.shape[1] if frame is not None else 1600
            for a in visible_anns:
                bb   = a.get("bbox_2d", {})
                cx   = (bb.get("x1", 0) + bb.get("x2", w_frame)) / 2 / max(w_frame, 1)
                zone = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
                gt_confirmed.append({
                    "type":       a["semantic_type"].split(".")[-1],
                    "full_type":  a["semantic_type"],
                    "zone":       zone,
                    "action":     a.get("action", "unknown"),
                    "speed_mps":  a.get("speed_mps", 0.0),
                    "speed":      a.get("speed_label", "unknown"),
                    "direction":  a.get("direction", "stationary"),
                    "dist_m":     a.get("dist_ego_m", None),
                    "visibility": a.get("visibility", 0),
                    "size":       a.get("size_category", "unknown"),
                })

    # Ego motion context (structured, not a sentence)
    ego_attrs = _build_ego_motion_attrs(frame_idx or 0, video_fps)

    env_options      = " | ".join(FIELD_OPTIONS["environment"])
    lighting_options = " | ".join(FIELD_OPTIONS["lighting"])
    flow_options     = " | ".join(FIELD_OPTIONS["traffic_flow"])

    stable_context = ""
    if stable_fields:
        stable_context = (
            f"Previously confirmed: environment={stable_fields.get('environment','')}, "
            f"lighting={stable_fields.get('lighting','')}. "
            f"Keep these values unless the image clearly contradicts them.\n"
        )

    prompt = f"""You are annotating a traffic scene image, taking in considertion that it involves the ego vehicle.
Your task: fill the JSON schema below using ONLY the structured data provided.
Do not invent any fact. Do not infer motion from a single frame — use the
speed/direction/action attributes already provided in the data.

{stable_context}
## EGO VEHICLE
{json.dumps(ego_attrs) if ego_attrs else "unknown (non-NuScenes video)"}

## DETECTED OBJECTS
Each object has pre-computed attributes from LiDAR/GPS (gt_sourced=true)
or from the visual tracker (gt_sourced=false). Trust gt_sourced=true values
exactly. For gt_sourced=false values, use your visual judgement to verify.
{json.dumps(structured_dets, indent=2)}

## GROUND TRUTH CONFIRMED OBJECTS (expert-labeled — use to fill detection gaps)
Objects confirmed present by NuScenes LiDAR annotations.
If an object appears in this list but NOT in DETECTED OBJECTS, include it
in your description using these attributes.
{json.dumps(gt_confirmed, indent=2) if gt_confirmed else "[]"}

## OUTPUT SCHEMA
Return ONLY valid JSON matching this exact structure.
No markdown, no explanation, no extra keys.
{{
  "scene_summary": {{
    "scene_description": "<3-4 sentences synthesized strictly from the object attributes above. Sentence 1: environment type and overall layout. Sentence 2: list each vehicle with its color, size, position, and action. Sentence 3: pedestrians and cyclists with their action and position. Sentence 4: traffic flow summary and any interactions between objects.>",
    "environment": "<{env_options}>",
    "lighting": "<{lighting_options}>",
    "traffic_flow": "<{flow_options}>",
    "spatial_description": "<1 sentence: what is in the foreground vs background>",
    "hazards_and_events": "<describe any hazard clearly visible in image, or exactly: none>"
  }}
}}

RULES:
1. scene_description must reflect the object attributes above — not image impressions.
2. Every vehicle mentioned must include: color, size, position, action.
3. speed/direction/action values come from the data above — do not guess from pixels.
4. If color=unknown, write 'unknown-colored' — do not guess the color.
5. If gt_confirmed contains objects not in detected_objects, include them.
6. environment and lighting: if stable_context is set, keep those values unless image clearly contradicts.
"""

    print(f"  > Qwen analysing frame '{scene_id}'")
    try:
        result = call_qwen_vision(frame, prompt)
    except Exception as e:
        print(f"  ! Qwen vision error: {e}")
        result = {}

    result["frame"] = scene_id
    ss = result.setdefault("scene_summary", {})
    ss["total_vehicles_detected"]       = n_veh
    ss["total_pedestrians_detected"]    = n_ped
    ss["total_cyclists_detected"]       = n_cyc
    ss["total_traffic_lights_detected"] = n_tl
    ss["traffic_density"]               = _derive_traffic_density(n_veh, n_cyc)
    # Inject stable fields so they appear in every per-second output
    if stable_fields:
        ss.setdefault("environment", stable_fields.get("environment", ""))
        ss.setdefault("lighting",    stable_fields.get("lighting", ""))

    if _nuscenes_gt_by_frame and frame_idx is not None:
        frame_gt_all  = _nuscenes_gt_by_frame.get(frame_idx, {})
        gt_anns_valid = frame_gt_all.get("annotations", [])
        if gt_anns_valid:
            result["qwen_gt_scores"] = validate_qwen_against_gt(ss, gt_anns_valid)

    final_objs = []
    for y in yolo_data["detections"]:
        obj_type = y["type"].lower()
        is_vehicle_or_cyclist = any(x in obj_type for x in
                                    ["car", "van", "truck", "bus", "motorcycle", "cyclist", "bicycle"])

        obj = {
            "track_id":         y.get("track_id", -1),
            "object_type":      y["type"],
            "confidence":       y["confidence"],
            "position":         y["position"],
            "bounding_box":     y["bounding_box"],
            "bounding_box_area":y["area"],
            "speed":            y.get("speed", "unknown"),
            "direction":        y.get("direction", "unknown"),
        }
        if y.get("source"):
            obj["source"] = y["source"]
        if is_vehicle_or_cyclist:
            obj["color"] = y.get("color", "unknown")
            obj["size"]  = y.get("size",  "unknown")

        if not is_static:
            obj["action"] = y.get("action") or _derive_action(
                y["type"], y.get("speed", "unknown"), y.get("direction", "unknown")
            )
        final_objs.append(obj)

    # GT motion override — LiDAR/GPS values replace pixel-based tracker estimates.
    # Tracker is kept as fallback for non-NuScenes videos / unmatched detections.
    if _nuscenes_gt_by_frame and frame_idx is not None:
        frame_gt_all = _nuscenes_gt_by_frame.get(frame_idx, {})
        gt_anns      = frame_gt_all.get("annotations", [])

        for obj in final_objs:
            if obj.get("source") == "gt_injected":
                obj["gt_motion"] = True
                continue

            bb = obj.get("bounding_box", {})
            if not bb:
                obj["gt_motion"] = False
                continue

            best_iou, best_gt = 0.25, None
            for gt_ann in gt_anns:
                iou = _bbox_iou(bb, gt_ann.get("bbox_2d", {}))
                if iou > best_iou:
                    best_iou, best_gt = iou, gt_ann

            if best_gt:
                obj["speed"]     = best_gt.get("speed_label", obj.get("speed", "unknown"))
                obj["direction"] = best_gt.get("direction",   obj.get("direction", "unknown"))
                obj["action"]    = best_gt.get("action",      obj.get("action", "unknown"))
                obj["speed_mps"] = best_gt.get("speed_mps",   None)
                obj["gt_motion"] = True
            else:
                obj["gt_motion"] = False

    ss["detected_objects"] = final_objs
    result.pop("detected_objects", None)

    # Save annotated frame (ID-only boxes) matching exactly what Qwen analysed
    _font = cv2.FONT_HERSHEY_SIMPLEX
    if frame is not None:
        canvas = frame.copy()
        for obj in final_objs:
            bb = obj.get("bounding_box", {})
            if not bb:
                continue
            x1 = int(bb.get("x1", 0)); y1 = int(bb.get("y1", 0))
            x2 = int(bb.get("x2", 0)); y2 = int(bb.get("y2", 0))
            label = f"#{obj.get('track_id', '?')}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
            (tw, th), _ = cv2.getTextSize(label, _font, 0.50, 1)
            ly = y1 - 8 if y1 > 20 else y1 + th + 8
            cv2.rectangle(canvas, (x1, ly - th - 4), (x1 + tw + 6, ly + 4), (0, 0, 0), -1)
            cv2.putText(canvas, label, (x1 + 3, ly), _font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(out_img_path.replace(".jpg", "_yolo.jpg"), canvas)

    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_json_path}")
    return result
#  AI Cumulative Summary 
def _safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


#  Generate AI cumulative for static image 
def generate_static_cumulative(result):
    ss = result.get("scene_summary", {})
    detections = ss.get("detected_objects", [])

    from collections import defaultdict
    groups_raw = defaultdict(list)
    for obj in detections:
        groups_raw[obj.get("object_type", "other")].append(obj)

    object_groups = []
    for otype, objs in groups_raw.items():
        is_ped = otype.lower() == "pedestrian"
        positions = [o.get("position", "") for o in objs]
        zones = []
        for p in positions:
            if "Foreground" in p: zones.append("foreground")
            elif "Background" in p: zones.append("background")
        zone     = max(set(zones), key=zones.count) if zones else "foreground"
        actions  = [o.get("action", "static") for o in objs if "action" in o]
        behavior = max(set(actions), key=actions.count) if actions else "static"
        group = {
            "group_label": f"{len(objs)} {otype}(s)",
            "object_type": otype,
            "count":       len(objs),
            "zone":        zone,
            "behavior":    behavior,
        }
        if not is_ped:
            sizes = [o.get("size", "medium") for o in objs]
            group["typical_size"] = max(set(sizes), key=sizes.count) if sizes else "medium"
        object_groups.append(group)

    cumulative = {
        "annotator_type":       "ai",
        "scene_id":             "static",
        "environment":          ss.get("environment", ""),
        "lighting":             ss.get("lighting", ""),
        "traffic_density":      ss.get("traffic_density", ""),
        "traffic_flow":         ss.get("traffic_flow", ""),
        "total_vehicles":       ss.get("total_vehicles_detected", 0),
        "total_pedestrians":    ss.get("total_pedestrians_detected", 0),
        "total_cyclists":       ss.get("total_cyclists_detected", 0),
        "total_traffic_lights": ss.get("total_traffic_lights_detected", 0),
        "object_groups":        object_groups,
        "scene_narrative":      ss.get("scene_description", ""),
        "spatial_description":  ss.get("spatial_description", ""),
        "hazards_and_events":   ss.get("hazards_and_events", "none"),
        "annotation_confidence": 0.85
    }
    os.makedirs("output/summaries", exist_ok=True)
    with open("output/summaries/output_ai_cumulative.json", "w") as f:
        json.dump(cumulative, f, indent=2)
    print("  Saved: output/summaries/output_ai_cumulative.json")
    return cumulative


#  Google Sheets 
_sheets_service = None
_spreadsheet_id = None


def _get_sheets():
    global _sheets_service
    if not SHEETS_AVAILABLE:
        return None
    if _sheets_service:
        return _sheets_service

    creds = None
    if os.path.exists(CREDENTIALS_FILE):
        try:
            creds = service_account.Credentials.from_service_account_file(
                CREDENTIALS_FILE, scopes=SCOPES)
            print("✓ Using service account credentials")
        except Exception as e:
            print(f"⚠  Service account load failed: {e}")
            creds = None

    if not creds and os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            if not creds.valid:
                creds = None
        except (IOError, ValueError) as e:
            print(f"⚠  Token file invalid: {e}")
            creds = None

    if not creds and os.path.exists(OAUTH_FILE):
        try:
            flow  = InstalledAppFlow.from_client_secrets_file(OAUTH_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
        except Exception as e:
            print(f"⚠  OAuth failed: {e}")
            return None

    if not creds:
        print("⚠  No Google credentials found. Responses saved locally only.")
        return None

    try:
        _sheets_service = build("sheets", "v4", credentials=creds)
        return _sheets_service
    except Exception as e:
        print(f"⚠  Sheets build failed: {e}")
        return None
def _ensure_spreadsheet():
    global _spreadsheet_id
    if _spreadsheet_id:
        return _spreadsheet_id

    if os.path.exists(SHEET_ID_FILE):
        with open(SHEET_ID_FILE) as f:
            _spreadsheet_id = f.read().strip()
        print(f"✓ Using spreadsheet: {_spreadsheet_id}")
        return _spreadsheet_id

    svc = _get_sheets()
    if not svc:
        return None
    try:
        body   = {"properties": {"title": f"Scene Annotations {datetime.now():%Y-%m-%d}"}}
        result = svc.spreadsheets().create(body=body).execute()
        _spreadsheet_id = result["spreadsheetId"]
        with open(SHEET_ID_FILE, "w") as f:
            f.write(_spreadsheet_id)
        print(f"✓ Created spreadsheet: {_spreadsheet_id}")
        svc.spreadsheets().values().update(
            spreadsheetId=_spreadsheet_id,
            range="Sheet1!A1",
            valueInputOption="RAW",
            body={"values": [SHEET_HEADERS]}
        ).execute()
        return _spreadsheet_id
    except Exception as e:
        print(f"⚠  Could not create spreadsheet: {e}")
        return None
def append_row(summary: dict) -> bool:
    sid = _ensure_spreadsheet()
    svc = _get_sheets()
    if not sid or not svc:
        return False
    try:
        object_groups  = summary.get("object_groups", [])
        object_summary = "; ".join([f"{g['count']} {g['object_type']}" for g in object_groups])
        p_info = summary.get("participant_info", {})
        row = [
            summary.get("annotator_type", "human"),
            p_info.get("name",          summary.get("participant_id", "")),
            p_info.get("country",       ""),
            p_info.get("age_category",  ""),
            p_info.get("gender",        ""),
            p_info.get("profession",    ""),
            p_info.get("driving_skill", ""),
            summary.get("scene_id", ""),
            summary.get("environment",     ""),
            summary.get("lighting",        ""),
            summary.get("traffic_density", ""),
            summary.get("traffic_flow",    ""),
            summary.get("total_vehicles",       0),
            summary.get("total_pedestrians",    0),
            summary.get("total_cyclists",       0),
            summary.get("total_traffic_lights", 0),
            summary.get("scene_narrative",    ""),
            summary.get("hazards_and_events", ""),
            summary.get("video_review",       ""),
            object_summary,
        ]
        svc.spreadsheets().values().append(
            spreadsheetId=sid,
            range="Sheet1!A:T",
            valueInputOption="RAW",
            body={"values": [row]}
        ).execute()
        print("✓ Row appended to Google Sheet")
        return True
    except Exception as e:
        print(f"⚠  Sheet append failed: {e}")
        return False
#  Flask 
_scene_b64_cache = {}

def _img_to_b64(path):
    if path in _scene_b64_cache:
        return _scene_b64_cache[path]
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    _scene_b64_cache[path] = data
    return data


_whisper_model = whisper.load_model("base") 

flask_app = Flask(__name__)
def _find_image(scene_id):
    for c in [f"output/scene/frame_{scene_id}.jpg",
              f"output/scene/frame_sec_{scene_id}.jpg",
              "output/scene/frame_static.jpg"]:
        if os.path.exists(c):
            return c
    return None
@flask_app.route("/")
def serve_form():
    return render_template("form.html")
@flask_app.route("/api/scene-image")
def api_scene_image():
    scene_id = request.args.get("scene_id", "static")
    path = _find_image(scene_id)
    return jsonify({"b64": _img_to_b64(path) if path else None})
@flask_app.route("/api/scene-video")
def api_scene_video():
    target = DEFAULT_TARGET
    if not os.path.exists(target):
        return jsonify({"url": None})
    ext = os.path.splitext(target)[1].lower()
    if ext in {".mp4", ".mov", ".avi", ".mkv"}:
        return jsonify({"url": "/video/preview"})
    return jsonify({"url": None})

@flask_app.route("/api/annotated-frames")
def api_annotated_frames():
    import glob, re
    files = glob.glob("output/scene/frame_sec_*_yolo.jpg")
    if not files:
        files = [f for f in glob.glob("output/scene/frame_sec_*.jpg") if "_yolo" not in f]
    frames = []
    for f in files:
        m = re.search(r'frame_sec_(\d+)', os.path.basename(f))
        if m:
            sec = int(m.group(1))
            frames.append({"sec": sec, "url": f"/frames/annotated/{sec}"})
    frames.sort(key=lambda x: x["sec"]) 
    return jsonify({"frames": frames})

@flask_app.route("/frames/annotated/<int:sec>")
def frame_annotated(sec):
    base = os.path.abspath("output/scene")
    yolo_path  = os.path.join(base, f"frame_sec_{sec}_yolo.jpg")
    plain_path = os.path.join(base, f"frame_sec_{sec}.jpg")
    if os.path.exists(yolo_path):
        return send_file(yolo_path, mimetype="image/jpeg")
    if os.path.exists(plain_path):
        return send_file(plain_path, mimetype="image/jpeg")
    return ("Not found", 404)
@flask_app.route('/video/preview')
def video_preview():
    target = DEFAULT_TARGET
    if not os.path.exists(target):
        return ("Not found", 404)
    return send_file(target, mimetype='video/mp4', conditional=True)
@flask_app.route("/api/submit", methods=["POST"])
def api_submit():
    data             = request.get_json(force=True)
    scene_id         = data.get("scene_id", "static")
    participant_id   = data.get("participant_id", "").strip()
    narrative        = data.get("narrative", "").strip()
    scene_fields     = data.get("scene_fields", {})
    hazards_and_events = data.get("hazards_and_events", "").strip()
    objects          = data.get("objects", [])
    video_review     = data.get("video_review", "").strip()

    if not participant_id or not narrative:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        from collections import defaultdict
        object_groups_dict = defaultdict(lambda: {
            "count": 0, "colors": [], "sizes": [], "positions": [], "actions": [], "raw_instances": []
        })

        for obj in objects:
            obj_type = obj.get("object_type", "other")
            color    = obj.get("color")
            size     = obj.get("size")
            position = obj.get("position", "")
            action   = obj.get("action")
            object_groups_dict[obj_type]["count"] += 1
            if color:    object_groups_dict[obj_type]["colors"].append(color)
            if size:     object_groups_dict[obj_type]["sizes"].append(size)
            if position: object_groups_dict[obj_type]["positions"].append(position)
            if action:   object_groups_dict[obj_type]["actions"].append(action)
            object_groups_dict[obj_type]["raw_instances"].append(obj)

        object_groups = []
        for obj_type, data_dict in object_groups_dict.items():
            zones = set()
            for pos in data_dict["positions"]:
                if "Foreground" in pos: zones.add("foreground")
                elif "Midground" in pos: zones.add("midground")
                elif "Background" in pos: zones.add("background")
            zone = list(zones)[0] if zones else "foreground"
            colors        = data_dict["colors"]
            typical_color = max(set(colors), key=colors.count) if colors else None
            sizes         = data_dict["sizes"]
            typical_size  = max(set(sizes), key=sizes.count) if sizes else "medium"
            actions       = data_dict["actions"]
            behavior      = max(set(actions), key=actions.count) if actions else "static"
            object_groups.append({
                "group_label":   f"{data_dict['count']} {obj_type}(s)",
                "object_type":   obj_type,
                "count":         data_dict["count"],
                "typical_size":  typical_size,
                "zone":          zone,
                "behavior":      behavior,
                "typical_color": typical_color,
                "raw_instances": data_dict["raw_instances"]
            })

        total_vehicles      = sum(g["count"] for g in object_groups if g["object_type"] in ["car", "van", "truck", "bus", "motorcycle"])
        total_pedestrians   = sum(g["count"] for g in object_groups if g["object_type"] == "pedestrian")
        total_cyclists      = sum(g["count"] for g in object_groups if g["object_type"] == "cyclist")
        total_traffic_lights = sum(g["count"] for g in object_groups if g["object_type"] == "traffic_light")

        human_summary = {
            "annotator_type":       "human",
            "participant_id":       participant_id,
            "participant_info":     data.get("participant_info", {}),
            "scene_id":             scene_id,
            "environment":          scene_fields.get("environment", ""),
            "lighting":             scene_fields.get("lighting", ""),
            "traffic_density":      scene_fields.get("traffic_density", ""),
            "traffic_flow":         scene_fields.get("traffic_flow", ""),
            "total_vehicles":       total_vehicles,
            "total_pedestrians":    total_pedestrians,
            "total_cyclists":       total_cyclists,
            "total_traffic_lights": total_traffic_lights,
            "object_groups":        object_groups,
            "scene_narrative":      narrative,
            "spatial_description":  "",
            "hazards_and_events":   hazards_and_events if hazards_and_events else "none",
            "raw_narrative":        narrative,
            "raw_objects":          objects,
            "annotation_confidence": None
        }

        if video_review:
            human_summary["video_review"] = video_review

        os.makedirs("annotations", exist_ok=True)
        fname = f"annotations/{participant_id}_annotation.json"
        with open(fname, "w") as f:
            json.dump(human_summary, f, indent=2)
        print(f"✓ Saved: {fname}")

        saved_to_sheets = append_row(human_summary)
        return jsonify({"success": True, "sheets": saved_to_sheets})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@flask_app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "no audio"}), 400
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        audio.save(f.name)
        try:
            result = _whisper_model.transcribe(f.name)
        finally:
            os.unlink(f.name)
    return jsonify({"text": result["text"].strip()})


# ngrok Server 
_ngrok_proc = None

def _cleanup_ngrok():
    print("Shutting down ngrok...")
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception as e:
        print(f"⚠  ngrok kill failed: {e}")
    global _ngrok_proc
    if _ngrok_proc:
        try:
            _ngrok_proc.terminate()
            _ngrok_proc.wait(timeout=2)
        except (OSError, ProcessLookupError) as e:
            print(f"⚠  ngrok process cleanup failed: {e}")

def _try_ngrok(port):
    global _ngrok_proc
    try:
        from pyngrok import ngrok as _ngrok, conf as _ngrok_conf
        _ngrok_conf.get_default().auth_token = os.getenv("NGROK_AUTH_TOKEN", "")
        tunnel = _ngrok.connect(port, "http")
        url = tunnel.public_url.replace("http://", "https://")
        print(f"\n  PUBLIC URL (share this): {url}/?scene_id=static")
        return url
    except ImportError:
        pass
    except Exception as e:
        print(f"  pyngrok error: {e}")
        import traceback; traceback.print_exc()

    import subprocess, re
    try:
        _ngrok_proc = subprocess.Popen(
            ["ngrok", "http", str(port), "--log=stdout"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        deadline = time.time() + 8
        while time.time() < deadline:
            line = _ngrok_proc.stdout.readline()
            m = re.search(r"https://[\w\-]+\.ngrok[\.\-]\S+", line)
            if m:
                url = m.group(0).rstrip("/")
                print(f"\n  PUBLIC URL: {url}/?scene_id=static")
                return url
    except (FileNotFoundError, Exception):
        pass
    print("  ngrok not found — only accessible on local network.")
    return None


def start_server(port=SERVER_PORT):
    def _run():
        flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
    t = Thread(target=_run, daemon=True)
    t.start()
    time.sleep(1.5)
    print(f"\n  Local: http://localhost:{port}/?scene_id=static")
    _try_ngrok(port)
    print(f"  Press Ctrl+C to stop.\n")


# ── Flat cumulative (single Qwen text call over all tracker + per-second data) ─
def generate_flat_cumulative(num_seconds: int, stable_fields: dict = None) -> dict:
    """Replace the hierarchical tree with one Qwen text call.
    All object-level facts come from the tracker; Qwen only writes the narrative."""
    second_paths = [
        f"output/scene/output_sec_{s}.json"
        for s in range(num_seconds)
        if os.path.exists(f"output/scene/output_sec_{s}.json")
    ]
    if not second_paths:
        print("  No per-second files found. Skipping cumulative.")
        return {}

    # Authoritative counts from tracker
    movements = []
    if get_track_motion_summary is not None:
        movements = get_track_motion_summary(0, num_seconds + 1) or []

    # Count unique track_ids from per-second JSONs only (excludes micro-tracks
    # that appear between 1fps samples and inflate the total).
    vehicle_types = {"car", "vehicle", "truck", "bus", "motorcycle", "van", "taxi"}
    uv, up, uc, utl = set(), set(), set(), set()

    # Aggregate per-track data for objects_seen list
    track_agg = {}   # track_id -> {type, colors[], sizes[], actions[], seconds[]}

    # Per-second summary for prompt
    frame_summaries = []
    for p in second_paths:
        d = _load_json(p)
        ss = d.get("scene_summary", {})
        sec = _safe_int(d.get("frame"), -1)
        for obj in ss.get("detected_objects", []) or []:
            tid    = obj.get("track_id", -1)
            source = obj.get("source", "yolo")

            # GT-injected objects have no tracker ID.
            # Synthesize a stable pseudo-ID so they reach objects_seen
            # and the Qwen cumulative prompt.
            if tid == -1 and source == "gt_injected":
                tid = f"gt_{obj.get('object_type','?')}_{obj.get('position','?')}_{sec}"

            if tid == -1:
                continue
            t = obj.get("object_type", "").lower()
            if t in vehicle_types:     uv.add(tid)
            elif t == "pedestrian":    up.add(tid)
            elif t == "cyclist":       uc.add(tid)
            elif t == "traffic_light": utl.add(tid)

            if tid not in track_agg:
                track_agg[tid] = {"object_type": obj.get("object_type", t),
                                  "colors": [], "sizes": [], "actions": [], "seconds": []}
            agg = track_agg[tid]
            if obj.get("color") and obj["color"] not in ("unknown", ""):
                agg["colors"].append(obj["color"])
            if obj.get("size") and obj["size"] not in ("unknown", ""):
                agg["sizes"].append(obj["size"])
            if obj.get("action") and obj["action"] not in ("unknown", ""):
                agg["actions"].append(obj["action"])
            if sec not in agg["seconds"]:
                agg["seconds"].append(sec)

        frame_summaries.append({
            "second":       sec,
            "vehicles":     ss.get("total_vehicles_detected", 0),
            "pedestrians":  ss.get("total_pedestrians_detected", 0),
            "cyclists":     ss.get("total_cyclists_detected", 0),
            "traffic_flow": ss.get("traffic_flow", ""),
            "hazards":      ss.get("hazards_and_events", "none"),
            "description":  (ss.get("scene_description", "") or "")[:120],
        })

    # Build objects_seen: one entry per unique track_id, most-common color/size/action
    def _most_common(lst):
        return max(set(lst), key=lst.count) if lst else "unknown"

    objects_seen = []
    for tid, agg in sorted(track_agg.items()):
        entry = {
            "track_id":    tid,
            "object_type": agg["object_type"],
            "action":      _most_common(agg["actions"]),
            "seen_seconds": sorted(agg["seconds"]),
        }
        if agg["object_type"].lower() != "pedestrian":
            entry["color"] = _most_common(agg["colors"])
            entry["size"]  = _most_common(agg["sizes"])
        objects_seen.append(entry)

    env   = (stable_fields or {}).get("environment", "")
    light = (stable_fields or {}).get("lighting", "")
    scene_id = f"video_0_{num_seconds - 1}"

    # Split moving vs static so Qwen focuses narrative on motion
    moving_objects = [o for o in objects_seen
                      if o.get("action") not in
                      ("parked", "static", "standing", "stopped", "unknown", "")]
    static_objects = [o for o in objects_seen if o not in moving_objects]

    env_options      = " | ".join(FIELD_OPTIONS["environment"])
    lighting_options = " | ".join(FIELD_OPTIONS["lighting"])
    density_options  = " | ".join(FIELD_OPTIONS["traffic_density"])
    flow_options     = " | ".join(FIELD_OPTIONS["traffic_flow"])

    prompt = f"""You are writing the final annotation summary for a {num_seconds}-second traffic video.
Your task: fill the JSON schema below using ONLY the structured data provided.
Do not invent any object, motion, or event not present in the data.

## SCENE CONTEXT (confirmed)
environment: {env}
lighting: {light}
unique_object_counts:
  vehicles: {len(uv)}
  pedestrians: {len(up)}
  cyclists: {len(uc)}
  traffic_lights: {len(utl)}

## PER-SECOND OBJECT DATA (sampled — do not extrapolate between samples)
Each entry contains counts and traffic_flow label for that second.
description field = Qwen's own description from that frame (truncated).
{json.dumps(frame_summaries, indent=2)}

## MOVING OBJECTS — focus your narrative on these
Each entry has: type, color, size, action, speed, direction, seen_seconds.
All speed/direction values are pre-computed from LiDAR/GPS or tracker.
Do not reinterpret them.
{json.dumps(moving_objects, indent=2)}

## STATIC / PARKED OBJECTS — mention briefly
{json.dumps(static_objects, indent=2)}

## OUTPUT SCHEMA
Return ONLY valid JSON matching this exact structure.
No markdown, no explanation, no extra keys.
{{
  "scene_id": "{scene_id}",
  "environment": "<{env_options}>",
  "lighting": "<{lighting_options}>",
  "traffic_density": "<{density_options}>",
  "traffic_flow": "<{flow_options}>",
  "scene_narrative": "<4 sentences synthesized from the data above. Sentence 1: environment type and overall layout. Sentence 2: each vehicle with its color, size, and action — use exact color/size from moving_objects and static_objects data. Sentence 3: pedestrians and cyclists with their actions. Sentence 4: describe movements using the direction and speed attributes from moving_objects only.>",
  "spatial_description": "<1 sentence on foreground vs background layout>",
  "hazards_and_events": "<describe any hazard present in the per-second data, or exactly: none>",
  "temporal_movements": [
    {{
      "object_type": "<type from moving_objects>",
      "color":       "<color from moving_objects>",
      "action":      "<action from moving_objects>",
      "speed":       "<speed label from moving_objects>",
      "speed_mps":   <speed_mps value or null>,
      "direction":   "<direction from moving_objects>",
      "evidence_seconds": [<seen_seconds list from moving_objects>]
    }}
  ],
  "annotation_confidence": <0.9 if most objects have gt_sourced=true, else 0.7>
}}

RULES:
1. unique_object_counts are authoritative — do not change them.
2. temporal_movements: one entry per object in moving_objects. Empty list [] if moving_objects is empty.
3. Every vehicle in scene_narrative must include its color and size from the data.
4. Do not describe motion for objects in static_objects.
5. annotation_confidence = 0.9 if NuScenes GT data was available (check if gt_sourced fields exist), else 0.7.
"""

    result = call_qwen_text(prompt)
    cumulative = result if isinstance(result, dict) else {}
    cumulative["annotator_type"]      = "ai"
    cumulative["scene_id"]            = scene_id
    cumulative["time_span"]           = {"start_second": 0, "end_second": num_seconds - 1}
    cumulative["total_vehicles"]      = len(uv)
    cumulative["total_pedestrians"]   = len(up)
    cumulative["total_cyclists"]      = len(uc)
    cumulative["total_traffic_lights"]= len(utl)
    cumulative["objects_seen"]        = objects_seen
    if not cumulative.get("temporal_movements") and movements:
        cumulative["temporal_movements"] = [
            {
                "object_type":      m.get("object_type", "unknown"),
                "color":            m.get("color", "unknown"),
                "movement":         f"{m.get('direction','unknown')} at {m.get('speed','unknown')}",
                "distance_px":      m.get("distance_px", 0.0),
                "evidence_seconds": m.get("evidence_seconds", []),
            }
            for m in movements
        ]

    #  NuScenes GT validation
    if _nuscenes_gt_by_frame:
        import glob as _glob, re as _re
        from collections import Counter as _Counter

        # Unique GT instances across the whole scene (by visibility tier)
        gt_all:   dict = {}   # instance_token → semantic_type (all visibility)
        gt_vis40: dict = {}   # instance_token → semantic_type (≥40% visible only)
        for fd in _nuscenes_gt_by_frame.values():
            for ann in fd.get("annotations", []):
                it, st = ann["instance_token"], ann["semantic_type"]
                gt_all[it] = st
                if ann.get("visibility", 0) >= 2:
                    gt_vis40[it] = st

        # IoU-based track-instance matching across all YOLO frame JSONs
        track_instances: dict = {}   
        instance_tracks: dict = {}  
        for ff in sorted(_glob.glob("output/frames/frame_*.json")):
            m = _re.match(r".*frame_(\d+)\.json", ff)
            if not m:
                continue
            fidx     = int(m.group(1))
            frame_gt = _nuscenes_gt_by_frame.get(fidx, {})
            gt_anns  = [a for a in frame_gt.get("annotations", []) if a.get("visibility", 0) >= 2]
            if not gt_anns:
                continue
            try:
                with open(ff) as fh:
                    fd = json.load(fh)
            except (IOError, json.JSONDecodeError):
                continue
            for yo in fd.get("scene_summary", {}).get("detected_objects", []):
                tid = yo.get("track_id", -1)
                if tid < 0:
                    continue
                ybb = yo.get("bounding_box", {})
                best_iou, best_it = 0.25, None
                for ga in gt_anns:
                    iou = _bbox_iou(ybb, ga.get("bbox_2d", {}))
                    if iou > best_iou:
                        best_iou, best_it = iou, ga["instance_token"]
                if best_it:
                    track_instances.setdefault(tid, {})
                    track_instances[tid][best_it] = track_instances[tid].get(best_it, 0) + 1
                    instance_tracks.setdefault(best_it, {})
                    instance_tracks[best_it][tid]  = instance_tracks[best_it].get(tid, 0) + 1

        matched_vis40 = set(instance_tracks) & set(gt_vis40)
        recall        = round(len(matched_vis40) / max(1, len(gt_vis40)), 3)
        split_tracks  = [
            {"track_id": t, "matched_instances": list(its)}
            for t, its in track_instances.items() if len(its) > 1
        ]
        fragmented_gt = [
            {"instance_token": it[:8] + "…", "track_ids": list(tids)}
            for it, tids in instance_tracks.items() if len(tids) > 1
        ]

        cumulative["gt_validation"] = {
            "gt_unique_all":        dict(_Counter(gt_all.values())),
            "gt_unique_40pct_vis":  dict(_Counter(gt_vis40.values())),
            "pipeline_recall_40pct": recall,
            "split_tracks":         split_tracks,
            "fragmented_gt_objects": fragmented_gt,
            "note": "Only GT objects with ≥40% visibility used for recall and IoU matching.",
        }
        print(f"  GT validation: recall={recall:.1%}, "
              f"split={len(split_tracks)}, fragmented={len(fragmented_gt)}")

    os.makedirs("output/summaries", exist_ok=True)
    _dump_json("output/summaries/output_cumulative_mega.json", cumulative)
    print("  Saved: output/summaries/output_cumulative_mega.json")
    return cumulative


def process_scene(scene_name: str):
    """
    NuScenes scene pipeline (sweep-based, no mp4 required):
      1. process_scene_sweeps  → YOLO + ByteTracker on every CAM_FRONT sweep
                                 (~12 Hz); writes output/keyframes/*.json
                                 (one per sample, 2 Hz) + output/keyframe_map.json
      2. Qwen analyse_frame    → one call per keyframe (sample), reading the
                                 sample JPG directly via the SDK
      3. generate_flat_cumulative across all keyframes
    """
    if not TRACKER_AVAILABLE or process_scene_sweeps is None:
        sys.exit("Tracker / nuScenes SDK not available — cannot run scene pipeline.")

    from nuscenes.nuscenes import NuScenes

    if os.path.exists("output"):
        print("Clearing existing output folder...")
        shutil.rmtree("output", ignore_errors=True)
    for _d in ["output", "output/frames", "output/annotated", "output/keyframes",
               "output/scene", "output/yolo_raw",
               "output/summaries", "output/cumulative_tree"]:
        os.makedirs(_d, exist_ok=True)

    print(f"\n🔍 Step 1: YOLO + ByteTracker on scene '{scene_name}' (sweeps)...")
    process_scene_sweeps(scene_name, dataroot=NUSCENES_DATAROOT, out_dir="output")

    print("\n Step 2: Qwen per-keyframe scene analysis...")
    with open("output/keyframe_map.json") as f:
        keyframe_map = json.load(f)
    if not keyframe_map:
        sys.exit("No keyframes found in output/keyframe_map.json")

    nusc = NuScenes(version="v1.0-mini", dataroot=NUSCENES_DATAROOT, verbose=False)

    frames_data = []   # list[(sec_label, frame_bgr, yolo_data, frame_idx)]
    for kf in keyframe_map:
        sample   = nusc.get("sample", kf["sample_token"])
        sd       = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img_path = os.path.join(NUSCENES_DATAROOT, sd["filename"])
        frame    = cv2.imread(img_path)

        kf_json_path = f"output/keyframes/keyframe_{kf['sample_idx']:04d}.json"
        yolo_data    = None
        if os.path.exists(kf_json_path):
            try:
                with open(kf_json_path) as pf:
                    yolo_data = _yolo_data_from_frame_result(json.load(pf))
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"  ⚠  Keyframe load failed (sample {kf['sample_idx']}): {e}")

        # sample_idx is the per-keyframe label (replaces per-second "sec")
        frames_data.append((kf["sample_idx"], frame, yolo_data, kf["frame_idx"]))

    # Keyframe 0 synchronously — lock stable fields
    stable_fields = {}
    sec0, frame0, yolo0, fidx0 = frames_data[0]
    if frame0 is not None:
        r0 = analyse_frame(
            frame0, str(sec0),
            out_json_path = f"output/scene/output_sec_{sec0}.json",
            out_img_path  = f"output/scene/frame_sec_{sec0}.jpg",
            yolo_data     = yolo0,
            frame_idx     = fidx0,
        )
        ss0 = r0.get("scene_summary", {})
        stable_fields = {
            "environment": ss0.get("environment", ""),
            "lighting":    ss0.get("lighting", ""),
        }
        print(f"  Stable fields locked — {stable_fields['environment']}, {stable_fields['lighting']}")

    def _analyse_kf(item):
        sec, frame, yolo_data, frame_idx = item
        if frame is None:
            return
        try:
            analyse_frame(
                frame, str(sec),
                out_json_path = f"output/scene/output_sec_{sec}.json",
                out_img_path  = f"output/scene/frame_sec_{sec}.jpg",
                yolo_data     = yolo_data,
                frame_idx     = frame_idx,
                stable_fields = stable_fields,
            )
        except Exception as e:
            print(f"  ⚠  analyse_frame failed (keyframe {sec}): {e}")

    if len(frames_data) > 1:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(_analyse_kf, frames_data[1:]))

    print("\n Step 3: Building flat cumulative summary...")
    generate_flat_cumulative(len(frames_data), stable_fields=stable_fields)

    _ensure_spreadsheet()
    start_server()
    print(f"\nProcessed {len(frames_data)} keyframes. Open the form in browser.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping.")
        _cleanup_ngrok()


def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit("Error: cannot open video file.")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = max(1, int(total_frames / fps))
    cap.release()
    print(f"Video: {duration}s @ {fps:.1f} FPS")

    if os.path.exists("output"):
        print("Clearing existing output folder...")
        shutil.rmtree("output", ignore_errors=True)

    for _d in ["output", "output/frames", "output/annotated",
               "output/scene", "output/yolo_raw",
               "output/summaries", "output/cumulative_tree"]:
        os.makedirs(_d, exist_ok=True)

    if TRACKER_AVAILABLE and process_video_frames is not None:
        try:
            print("\n🔍 Step 1: Running YOLO + GMC + ByteTracker...")
            process_video_frames(
                video_path=path,
                out_dir="output",
                frame_rate=TRACKER_FRAME_RATE,
            )
        except Exception as e:
            print(f" yolo_bytetrack error: {e}")
            traceback.print_exc()
    else:
        print(" yolo_bytetrack unavailable; skipping tracking step.")

    print("\n Step 2: Qwen per-second scene analysis...")
    actual_rate = TRACKER_FRAME_RATE if TRACKER_FRAME_RATE > 0 else min(int(fps), max(15, int(math.floor(fps * 0.5))))
    step        = max(1, int(round(fps / actual_rate)))

    frames_data = []  
    cap2 = cv2.VideoCapture(path)
    for sec in range(duration):
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap2.read()
        frame_idx_for_sec = int(sec * fps)
        tracked_idx  = (frame_idx_for_sec // step) * step
        precomp_path = f"output/frames/frame_{tracked_idx:06d}.json"
        yolo_data = None
        if os.path.exists(precomp_path):
            try:
                with open(precomp_path) as pf:
                    yolo_data = _yolo_data_from_frame_result(json.load(pf))
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"  ⚠  Precomputed frame load failed (sec {sec}): {e}")
        frames_data.append((sec, frame if ok else None, yolo_data, frame_idx_for_sec))
    cap2.release()

    # process second 0 synchronously - lock stable fields 
    stable_fields = {}
    sec0, frame0, yolo0, fidx0 = frames_data[0]
    if frame0 is not None:
        r0 = analyse_frame(
            frame0, str(sec0),
            out_json_path = f"output/scene/output_sec_{sec0}.json",
            out_img_path  = f"output/scene/frame_sec_{sec0}.jpg",
            yolo_data     = yolo0,
            frame_idx     = fidx0,
        )
        ss0 = r0.get("scene_summary", {})
        stable_fields = {
            "environment": ss0.get("environment", ""),
            "lighting":    ss0.get("lighting", ""),
        }
        print(f"  Stable fields locked — {stable_fields['environment']}, {stable_fields['lighting']}")

    #remaining seconds in parallel (no prev_context chain) 
    def _analyse_sec(item):
        sec, frame, yolo_data, frame_idx = item
        if frame is None:
            return
        try:
            analyse_frame(
                frame, str(sec),
                out_json_path = f"output/scene/output_sec_{sec}.json",
                out_img_path  = f"output/scene/frame_sec_{sec}.jpg",
                yolo_data     = yolo_data,
                frame_idx     = frame_idx,
                stable_fields = stable_fields,
            )
        except Exception as e:
            print(f"  ⚠  analyse_frame failed (sec {sec}): {e}")

    if len(frames_data) > 1:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(_analyse_sec, frames_data[1:]))

    #single flat cumulative (one Qwen text call) 
    print("\n Step 3: Building flat cumulative summary...")
    generate_flat_cumulative(duration, stable_fields=stable_fields)

    _ensure_spreadsheet()
    start_server()
    print("\nProcessed all frames. Open the form in browser.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping.")
        _cleanup_ngrok()
#  Image Pipeline 
def process_image(path):
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Error: cannot read image: {path}")

    if os.path.exists("output"):
        print("🧹 Clearing existing output folder...")
        shutil.rmtree("output", ignore_errors=True)
    for _d in ["output", "output/scene", "output/summaries"]:
        os.makedirs(_d, exist_ok=True)

    print(f"Processing image: {path}")
    result = analyse_frame(frame,
                           scene_id      = "static",
                           out_json_path = "output/scene/output_static.json",
                           out_img_path  = "output/scene/frame_static.jpg")
    generate_static_cumulative(result)
    _ensure_spreadsheet()
    start_server()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping.")
        _cleanup_ngrok()

# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── NuScenes mode: python scene_annotator.py --nuscenes [scene_idx] ──
    if len(sys.argv) > 1 and sys.argv[1] == "--nuscenes":
        try:
            from nuscenes.nuscenes import NuScenes as _NuScenes
            from nuscenes_to_pipeline import list_scenes, scene_to_video
        except ImportError:
            sys.exit("nuscenes-devkit not installed. Run: pip install nuscenes-devkit")

        _DATAROOT = "data/v1.0-mini"
        _VERSION  = "v1.0-mini"
        _nusc = _NuScenes(version=_VERSION, dataroot=_DATAROOT, verbose=False)

        if len(sys.argv) > 2 and sys.argv[2].lstrip("-").isdigit():
            _scene_idx = int(sys.argv[2])
        else:
            list_scenes(_nusc)
            try:
                _scene_idx = int(input(f"\nEnter scene number (0–{len(_nusc.scene)-1}): ").strip())
            except (ValueError, KeyboardInterrupt):
                sys.exit("Cancelled.")

        _scene_name = _nusc.scene[_scene_idx]["name"]
        _out_dir    = os.path.join("output_nuscenes", _scene_name)
        os.makedirs(_out_dir, exist_ok=True)
        target = os.path.join(_out_dir, f"{_scene_name}_CAM_FRONT.mp4")
        print(f"\nConverting NuScenes scene {_scene_idx} ({_scene_name}) → {target}")
        scene_to_video(_nusc, _scene_idx, "CAM_FRONT", target, fps=2)

        # Load ego-pose sidecar → ground-truth motion compensation
        _ego_path = target.replace(".mp4", "_ego_poses.json")
        if os.path.exists(_ego_path) and TRACKER_AVAILABLE:
            try:
                with open(_ego_path) as _ef:
                    _ep = json.load(_ef)
                set_ego_poses(
                    _ep["poses"],
                    _ep["camera_intrinsic"],
                    _ep["camera_rotation"],
                    _ep["camera_translation"],
                )
                print("  ✅ NuScenes ego-pose motion compensation enabled.")
            except Exception as _e:
                print(f"  ⚠  Could not load ego-poses: {_e} — falling back to optical-flow GMC.")

        # Load GT annotation sidecar → Qwen hints + cumulative validation
        _gt_path = target.replace(".mp4", "_gt_annotations.json")
        if os.path.exists(_gt_path):
            try:
                with open(_gt_path) as _gf:
                    set_nuscenes_gt(json.load(_gf))
                print("  ✅ NuScenes GT annotations loaded.")
            except Exception as _e:
                print(f"  ⚠  Could not load GT annotations: {_e}")

    else:
        target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE

    ext = os.path.splitext(target)[1].lower()

    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        process_image(target)
    elif ext in {".mp4", ".mov", ".avi", ".mkv"}:
        DEFAULT_TARGET = target
        process_video(target)
    else:
        # Treat as NuScenes scene name (e.g. "scene-0757"); update DEFAULT_TARGET
        # so the form's /video/preview can still serve the pre-rendered mp4 if one
        # exists at the expected location.
        scene = target
        DEFAULT_TARGET = f"output_nuscenes/{scene}/{scene}_CAM_FRONT.mp4"
        process_scene(scene)