
import base64
import math
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from threading import Thread

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, send_file
try:
    from yolo_bytetrack import (
        run_yolo_and_track,
        process_video_frames,
        get_track_motion_summary,
        reset_tracker,
        draw_tracks,
        TRACKER_FRAME_RATE,
        VIDEO_CHUNK_SECONDS as _YBT_CHUNK,
    )
    TRACKER_AVAILABLE = True
except Exception as e:
    print(f"⚠  yolo_bytetrack import failed: {e}")
    TRACKER_AVAILABLE = False
    run_yolo_and_track = None
    process_video_frames = None
    get_track_motion_summary = None

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
VISION_MODEL     = "qwen2.5vl:7b"
TEXT_MODEL       = "qwen2.5:3b"
CREDENTIALS_FILE = "credentials.json"
OAUTH_FILE       = "oauth_client.json"
TOKEN_FILE       = "token.json"
SHEET_ID_FILE    = "spreadsheet_id.txt"
SCOPES           = ["https://www.googleapis.com/auth/spreadsheets"]
SERVER_PORT      = 7860
DEFAULT_TARGET   = "assets/7.mp4"
TOP_N_OBJECTS_FOR_GAP_FILL = 3
# Frame/tracking constants live in yolo_bytetrack.py — imported here for use
VIDEO_CHUNK_SECONDS = _YBT_CHUNK if TRACKER_AVAILABLE else 5


CUMULATIVE_SUMMARY_SCHEMA = {
    "annotator_type": "ai | human",
    "scene_id": "",

    "environment":    "urban street | highway | parking lot | intersection | residential area | school zone | construction zone",
    "lighting":       "bright daylight | low-light | night with street lights | night without lighting",

    "total_vehicles":         0,
    "total_pedestrians":      0,
    "total_cyclists":         0,
    "total_traffic_lights":   0,

    "traffic_density":  "empty | light | moderate | heavy | gridlock",
    "traffic_flow":     "free-flowing | slow-moving | stopped | mixed | one-directional | bidirectional",

    # ── Object groups
    
    "scene_narrative": "3–4 sentences describing the overall scene holistically: what kind of place, what is happening, what stands out.",
    "hazards_and_events": "any safety concerns, unusual events, obstructions, or noteworthy observations. 'none' if absent.",
    "spatial_description": "brief description of depth layers: what occupies foreground vs background, lane structure, sidewalks, etc.",
    "annotation_confidence": None   # 0.0–1.0 for AI, null for human
}

#  Internal per-frame AI schema (for AI processing only) 
# not used in the survey , it feeds the cumulative summary above.
STATIC_SCHEMA = {
    "frame": "static",
    "scene_summary": {
        "scene_description":             "",
        "environment":                   "urban street / highway / parking lot / intersection / residential area / school zone / construction zone",
        "lighting":                      "bright daylight / low-light / night with street lights / night without lighting",
        "traffic_density":               "empty / light / moderate / heavy / gridlock",
        "traffic_flow":                  "free-flowing / slow-moving / stopped / mixed / one-directional / bidirectional",
        "total_vehicles_detected":       0,
        "total_pedestrians_detected":    0,
        "total_cyclists_detected":       0,
        "total_traffic_lights_detected": 0,
        "spatial_description":           "",
        "hazards_and_events":            "",
        "detected_objects": [
            {
                "object_id":   0,
                "object_type": "type",
                "color":       "color",
                "size":        "small / medium / large",
                "position":    "Foreground Left / Foreground Center / Foreground Right / Background Left / Background Center / Background Right",
                "bounding_box_area": 0
            }
        ]
    }
}

VIDEO_SCHEMA = {
    "frame": 0,
    "scene_summary": {
        "scene_description":             "",
        "environment":                   "urban street / highway / parking lot / intersection / residential area / school zone / construction zone",
        "lighting":                      "bright daylight / low-light / night with street lights / night without lighting",
        "traffic_density":               "empty / light / moderate / heavy / gridlock",
        "traffic_flow":                  "free-flowing / slow-moving / stopped / mixed / one-directional / bidirectional",
        "total_vehicles_detected":       0,
        "total_pedestrians_detected":    0,
        "total_cyclists_detected":       0,
        "total_traffic_lights_detected": 0,
        "spatial_description":           "",
        "hazards_and_events":            "",
        "detected_objects": [
            {
                "object_id":   0,
                "object_type": "type",
                "color":       "color",
                "size":        "small / medium / large",
                "position":    "Foreground Left / Foreground Center / Foreground Right / Background Left / Background Center / Background Right",
                "action":      "parked / moving / turning / crossing / stopped / static",
                "bounding_box_area": 0
            }
        ]
    }
}

# AI cumulative (video only) — also maps to CUMULATIVE_SUMMARY_SCHEMA
CUMULATIVE_AI_SCHEMA = {
    "video_summary": {
        "total_frames_analyzed":          0,
        "environment":                    "",
        "lighting":                       "",
        "traffic_density":                "",
        "traffic_flow":                   "",
        "peak_vehicle_count":             0,
        "peak_pedestrian_count":          0,
        "avg_vehicle_count":              0,
        "avg_pedestrian_count":           0,
        "unique_object_types":            [],
        "recurring_object_types":         [],
        "spatial_description":            "",
        "scene_narrative":                "",
        "hazards_and_events":             "",
        "annotation_confidence":          0.0
    }
}

FIELD_OPTIONS = {
    "environment":    ["urban street", "highway", "parking lot", "intersection", "residential area", "school zone", "construction zone"],
    "lighting":       ["bright daylight", "low-light", "night with street lights", "night without lighting"],
    "traffic_density":["empty", "light", "moderate", "heavy", "gridlock"],
    "traffic_flow":   ["free-flowing", "slow-moving", "stopped", "mixed", "one-directional", "bidirectional"],
}

OBJECT_TYPE_OPTIONS   = ["car", "van", "truck", "bus", "motorcycle", "cyclist", "pedestrian", "traffic_light", "road_sign", "other"]
COLOR_OPTIONS         = ["white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "brown", "beige", "mixed"]
SIZE_OPTIONS          = ["small (e.g. hatchback)", "medium (e.g. sedan/SUV)", "large (e.g. truck/bus)"]
POSITION_OPTIONS      = ["Foreground Left", "Foreground Center", "Foreground Right",
                         "Midground Left", "Midground Center", "Midground Right",
                         "Background Left",  "Background Center",  "Background Right"]
ACTION_OPTIONS        = ["moving", "parked", "stopped", "turning left", "turning right", "crossing", "static", "unknown"]
VEHICLE_CLASSES       = {"car", "van", "motorcycle", "bus", "truck"}

SHEET_HEADERS = [
    "Annotator Type", "Participant ID", "Scene ID",
    "Environment", "Lighting",
    "Traffic Density", "Traffic Flow",
    "Total Vehicles", "Total Pedestrians", "Total Cyclists",
    "Total Traffic Lights",
    "Scene Narrative", "Hazards and Events",
    "Object Count Summary"
]

# YOLO detection and tracking are handled by yolo_bytetrack.py.
# Use run_yolo_and_track(frame, frame_idx) for per-frame detection+tracking.

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
            "bounding_box": d.get("bounding_box"),
            "track_id":     d.get("track_id", -1),
            "color":        d.get("color", "unknown"),
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


#  Qwen helpers 
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
    return {}


# GMC, get_bbox_center, calc_movement_distance are now in yolo_bytetrack.py.
# Thin helpers kept here for any legacy callers:

def get_bbox_center(bbox):
    return ((bbox["x1"] + bbox["x2"]) / 2, (bbox["y1"] + bbox["y2"]) / 2)

def calc_movement_distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)


def build_detection_color_map(seconds_data):
    color_map = {}
    for frame in seconds_data:
        detected_objs = frame.get("scene_summary", {}).get("detected_objects", []) or []
        for obj in detected_objs:
            obj_id = str(obj.get("object_id"))
            color = obj.get("color", "unknown")
            if obj_id not in color_map:
                color_map[obj_id] = []
            if color and color != "unknown":
                color_map[obj_id].append(color)
    
    color_summary = {}
    for obj_id, colors in color_map.items():
        if colors:
            from collections import Counter
            color_summary[obj_id] = Counter(colors).most_common(1)[0][0]
        else:
            color_summary[obj_id] = "unknown"
    return color_summary


def enrich_movement_with_colors(movements, color_map):
    """Add color info to movements based on detected_objects."""
    for mov in movements:
        hint = mov.get("object_hint", "")  #  "track_5:car"
        if ":" in hint:
            parts = hint.split(":")
            if len(parts) == 2:
                # Try to match with detection color (if available)
                obj_type = parts[1]
                # Color already encoded in DETECTED_MOVEMENTS via YOLO tracking
                if "color" not in mov:
                    mov["color"] = "unknown"
    return movements


def call_qwen_vision(frame_bgr, prompt):
    h, w   = frame_bgr.shape[:2]
    scale  = min(512 / max(h, w), 1.0)
    small  = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    _, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    b64    = base64.b64encode(encoded.tobytes()).decode()
    payload = {
        "model":   VISION_MODEL,
        "format":  "json",
        "stream":  False,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "options": {
            "temperature": 0.1,
            "num_ctx":     4096,  
            "num_predict": 2048,   
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return _safe_parse_json(r.json()["message"]["content"])


def call_qwen_text(prompt):
    payload = {
        "model":   TEXT_MODEL,
        "format":  "json",
        "stream":  False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.0, "num_ctx": 2048, "num_predict": 1024}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)  
    r.raise_for_status()
    return _safe_parse_json(r.json()["message"]["content"])


#  Per-Frame Analysis 
def analyse_frame(frame, scene_id, out_json_path, out_img_path, yolo_data=None, frame_idx=None, annotated_frame=None):
    """
    Analyse one frame with Qwen vision.

    yolo_data:       pre-computed dict from _yolo_data_from_frame_result().
                     If None, falls back to calling run_yolo_and_track() directly.
    frame_idx:       raw frame index (for tracker calls when yolo_data is None).
    annotated_frame: optional frame with YOLO bounding boxes + IDs already drawn.
                     When provided, this is sent to Qwen instead of the raw frame.
    """
    is_static = (scene_id == "static")

    # ── Obtain YOLO + tracker data ─────────────────────────────────────────
    if yolo_data is None:
        if TRACKER_AVAILABLE and run_yolo_and_track is not None:
            _fidx = frame_idx if frame_idx is not None else 0
            frame_data = run_yolo_and_track(frame, _fidx)
            yolo_data  = _yolo_data_from_frame_result(frame_data)
            # Save annotated debug image from yolo_bytetrack
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

    os.makedirs("output/yolo_raw", exist_ok=True)
    with open(f"output/yolo_raw/yolo_raw_{scene_id}.json", "w") as f:
        json.dump(yolo_data, f, indent=2)

    cv2.imwrite(out_img_path, frame)

    n_det  = yolo_data["total_objects"]
    n_veh  = yolo_data["vehicle_count"]
    n_ped  = yolo_data["pedestrian_count"]
    n_cyc  = yolo_data["cyclist_count"]
    n_tl   = yolo_data["traffic_light_count"]

    # Grab track motion summary up to the current second if available
    track_motions = {}
    prev_context = ""
    if not is_static:
        try:
            current_sec = int(scene_id)
            if get_track_motion_summary is not None:
                motions = get_track_motion_summary(0, current_sec)
                for m in motions:
                    track_motions[m["track_id"]] = f"{m['speed']}, {m['direction']}"
            
            if current_sec > 0:
                prev_path = f"output/scene/output_sec_{current_sec - 1}.json"
                if os.path.exists(prev_path):
                    with open(prev_path) as f:
                        prev_data = json.load(f)
                        narrative = prev_data.get("scene_summary", {}).get("scene_description", "")
                        if narrative:
                            prev_context = (
                                f"PREVIOUS FRAME CONTEXT (1 second ago):\\n"
                                f"\\\"{narrative}\\\"\\n\\n"
                                "Based on the PREVIOUS FRAME and the CURRENT DETECTIONS, describe what has changed or progressed.\\n\\n"
                            )
        except Exception:
            pass

    # Compact detection list for Qwen — include pixel color and motion history
    compact_dets = []
    for d in yolo_data["detections"]:
        tid = d.get("track_id", -1)
        det = {
            "id":    d["id"],
            "type":  d["type"],
            "pos":   d["position"],
            "bbox":  d["bounding_box"],
            "area":  d["area"],
            "track_id": tid,
            "pixel_color": d.get("color", "unknown"),  # HSV pixel color hint
        }
        if tid in track_motions:
            det["motion_history"] = track_motions[tid]
        compact_dets.append(det)

    action_note = ""
    if not is_static:
        action_note = (
            "For each object also add \"action\" based on type:\n"
            "  - For VEHICLES (car, truck, bus, motorcycle): parked | moving | turning left | turning right | stopped\n"
            "  - For PEOPLE (person, pedestrian): standing | walking | running | crossing\n"
            "  - For CYCLISTS (cyclist, bicycle): moving | stopped | turning\n"
        )

    # Tell Qwen whether the image has pre-drawn boxes
    has_boxes = annotated_frame is not None
    box_context = (
        "IMPORTANT: This image has YOLO bounding boxes already drawn on it. "
        "Each box is labelled with #ID (e.g. #1, #2, …). "
        "The #ID corresponds to the track_id field in the detections list below. "
        "Use the boxes and IDs to spatially ground your descriptions.\n\n"
    ) if has_boxes else ""

    # Use annotated frame for Qwen when available
    qwen_frame = annotated_frame if has_boxes else frame

    prompt = (
        f"{prev_context}"
        f"{box_context}"
        f"You are annotating a traffic scene image. {n_det} objects were detected by YOLO+ByteTrack at 1 fps.\n\n"
        f"DETECTIONS (id, type, position, bounding_box, track_id, pixel_color already computed):\n"
        f"{json.dumps(compact_dets)}\n\n"
        f"YOUR JOB — return JSON with these keys only:\n"
        f"1. scene_summary with:\n"
        f"   - scene_description: 3-5 sentences describing the scene. MUST INCLUDE COLORS and SIZES of prominent vehicles.\n"
        f"   - environment: urban street/highway/parking lot/intersection\n"
        f"   - lighting: bright daylight/low-light/night\n"
        f"   - traffic_density: empty/light/moderate/heavy\n"
        f"   - traffic_flow: free-flowing/slow-moving/stopped/mixed\n"
        f"   - spatial_description: 1 sentence on foreground vs background.\n"
        f"   - hazards_and_events: 1 sentence or \"none\".\n"
        f"2. detected_objects: array of {n_det} items, one per YOLO detection:\n"
        f"   {{object_id, color, size}}\n"
        f"   color (REQUIRED for vehicles/cyclists ONLY): use pixel_color hint unless you can clearly see a different color.\n"
        f"   color must be: white/black/silver/gray/red/blue/green/yellow/orange/brown/mixed\n"
        f"   size (REQUIRED for vehicles/cyclists ONLY): small(<5% frame)/medium(5-20%)/large(>20%)\n"
        f"   {action_note}"
        f"CRITICAL:\n"
        f"  - Color and Size are REQUIRED for: cars, vans, trucks, buses, motorcycles, cyclists, bicycles.\n"
        f"  - Color and Size are NOT included for: pedestrians, traffic lights, road signs.\n"
        f"Return ONLY the JSON object, no explanation."
    )

    print(f"  > Qwen analysing frame '{scene_id}' ({'annotated+boxes' if annotated_frame is not None else 'raw frame'})...")
    try:
        result = call_qwen_vision(qwen_frame, prompt)
    except Exception as e:
        print(f"  ! Qwen vision error: {e}")
        result = {}

    result["frame"] = scene_id
    ss = result.setdefault("scene_summary", {})
    ss["total_vehicles_detected"]       = n_veh
    ss["total_pedestrians_detected"]    = n_ped
    ss["total_cyclists_detected"]       = n_cyc
    ss["total_traffic_lights_detected"] = n_tl

    qwen_objs  = {str(o.get("object_id")): o for o in (ss.get("detected_objects") or [])}
    final_objs = []
    for y in yolo_data["detections"]:
        yid = str(y["id"])
        ai  = qwen_objs.get(yid, {})

        obj_type = y["type"].lower()
        is_vehicle = any(x in obj_type for x in ["car", "van", "truck", "bus", "motorcycle"])
        is_vehicle_or_cyclist = is_vehicle or any(x in obj_type for x in ["cyclist", "bicycle"])

        obj = {
            "object_id":        y["id"],
            "track_id":         y.get("track_id", -1),
            "object_type":      y["type"],
            "confidence":       y["confidence"],
            "position":         y["position"],
            "bounding_box":     y["bounding_box"],
            "bounding_box_area":y["area"],
            "speed":            y.get("speed", "unknown"),
            "direction":        y.get("direction", "unknown"),
        }

        is_pedestrian = any(x in obj_type for x in ["person", "pedestrian"])
        if not is_pedestrian:
            # Colour priority: STRICTLY pixel-based, no fallbacks or 'unknown'
            pixel_color = y.get("color")
            if not pixel_color or pixel_color == "unknown":
                pixel_color = "black"
            
            obj["color"] = pixel_color
            obj["pixel_color"] = pixel_color

        if is_vehicle_or_cyclist:
            obj["size"] = ai.get("size") or "unknown"
        elif ai.get("size"):
            obj["size"] = ai.get("size")

        if not is_static:
            obj_type_l = obj_type
            if "traffic light" in obj_type_l or "light" in obj_type_l:
                default_action = "static"
            elif any(x in obj_type_l for x in ["person", "pedestrian"]):
                default_action = "standing"
            elif any(x in obj_type_l for x in ["cyclist", "bicycle"]):
                default_action = "stopped"
            else:
                default_action = "parked"
            obj["action"] = ai.get("action", default_action)
        final_objs.append(obj)

    ss["detected_objects"] = final_objs
    result.pop("detected_objects", None)

    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_json_path}")
    return result


# AI Cumulative Summary
def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)





def _normalize_second_frame(path):
    """Normalize per-second frame JSON into a stable payload for cumulative prompts."""
    d = _load_json(path)
    ss = d.get("scene_summary", {})
    objs = []
    for obj in ss.get("detected_objects", []) or []:
        bbox = obj.get("bounding_box", {}) if isinstance(obj.get("bounding_box"), dict) else {}
        objs.append({
            "object_id": obj.get("object_id"),
            "object_type": obj.get("object_type"),
            "color": obj.get("color"),
            "size": obj.get("size"),
            "position": obj.get("position"),
            "action": obj.get("action"),
            "bounding_box": bbox,  # Include bbox for explicit coordinate tracking
            "bounding_box_area": obj.get("bounding_box_area", 0)
        })
    return {
        "second": _safe_int(d.get("frame"), default=-1),
        "scene_summary": {
            "environment": ss.get("environment", ""),
            "lighting": ss.get("lighting", ""),
            "traffic_density": ss.get("traffic_density", ""),
            "traffic_flow": ss.get("traffic_flow", ""),
            "total_vehicles_detected": ss.get("total_vehicles_detected", 0),
            "total_pedestrians_detected": ss.get("total_pedestrians_detected", 0),
            "total_cyclists_detected": ss.get("total_cyclists_detected", 0),
            "total_traffic_lights_detected": ss.get("total_traffic_lights_detected", 0),
            "spatial_description": ss.get("spatial_description", ""),
            "hazards_and_events": ss.get("hazards_and_events", ""),
            "detected_objects": objs,
        }
    }


def _build_cumulative_from_seconds(frames_payload, span_start, span_end, scene_id):
    # Pre-aggregate object counts and detect movements
    all_vehicles = []
    all_pedestrians = []
    all_cyclists = []
    all_traffic_lights = []
    movement_map = {}  # Track same object across frames
    
    for frame in frames_payload:
        ss = frame.get("scene_summary", {})
        all_vehicles.append(ss.get("total_vehicles_detected", 0))
        all_pedestrians.append(ss.get("total_pedestrians_detected", 0))
        all_cyclists.append(ss.get("total_cyclists_detected", 0))
        all_traffic_lights.append(ss.get("total_traffic_lights_detected", 0))
        
        for obj in ss.get("detected_objects", []) or []:
            obj_id = obj.get("object_id")
            obj_type = obj.get("object_type")
            position = obj.get("position", "")
            second = frame.get("second", -1)
            key = f"{obj_type}_{obj_id}"
            if key not in movement_map:
                movement_map[key] = {"type": obj_type, "positions": []}
            movement_map[key]["positions"].append({"second": second, "position": position})
    
    # Detect actual movements (position changes across time)
    movements_detected = []
    for key, data in movement_map.items():
        positions = data["positions"]
        if len(positions) > 1:
            start_pos = positions[0]["position"]
            end_pos = positions[-1]["position"]
            if start_pos != end_pos:
                movements_detected.append({
                    "object_type": data["type"],
                    "from_position": start_pos,
                    "to_position": end_pos,
                    "frames": [p["second"] for p in positions]
                })

    # Fetch motion summaries directly from yolo_bytetrack.py in-memory state
    if get_track_motion_summary is not None:
        tracker_movements = get_track_motion_summary(span_start, span_end)
        if tracker_movements:
            movements_detected = tracker_movements + movements_detected
    
    avg_vehicles = int(sum(all_vehicles) / len(all_vehicles) + 0.5) if all_vehicles else 0
    avg_pedestrians = int(sum(all_pedestrians) / len(all_pedestrians) + 0.5) if all_pedestrians else 0
    avg_cyclists = int(sum(all_cyclists) / len(all_cyclists) + 0.5) if all_cyclists else 0
    avg_traffic_lights = int(sum(all_traffic_lights) / len(all_traffic_lights) + 0.5) if all_traffic_lights else 0
    
    movements_json = json.dumps(movements_detected, indent=2) if movements_detected else "[]"
    
    # Extract concise summaries from per-second data instead of passing full JSON
    frame_summaries = []
    for frame in frames_payload:
        ss = frame.get("scene_summary", {})
        # Include color and size information for each object type
        detected_objs = ss.get("detected_objects", []) or []
        
        # Group objects by type with colors and sizes
        objs_by_type = {}
        for obj in detected_objs:
            otype = obj.get("object_type", "other")
            color = obj.get("color", "unknown")
            size = obj.get("size", "")
            if otype not in objs_by_type:
                objs_by_type[otype] = []
            objs_by_type[otype].append({"color": color, "size": size})
        
        # Build color and size descriptions like "white large cars, 2x; red medium SUV"
        detail_desc = []
        for otype, items in objs_by_type.items():
            detail_summary = {}
            for item in items:
                color = item.get("color", "unknown")
                size = item.get("size", "")
                key = f"{color} {size}" if size else color
                detail_summary[key] = detail_summary.get(key, 0) + 1
            
            for detail, cnt in detail_summary.items():
                if cnt == 1:
                    detail_desc.append(f"{detail} {otype}")
                else:
                    detail_desc.append(f"{cnt}x {detail} {otype}")
        
        frame_summaries.append({
            "second": frame.get("second", -1),
            "vehicles": ss.get("total_vehicles_detected", 0),
            "pedestrians": ss.get("total_pedestrians_detected", 0),
            "cyclists": ss.get("total_cyclists_detected", 0),
            "traffic_lights": ss.get("total_traffic_lights_detected", 0),
            "hazards": ss.get("hazards_and_events", "none"),
            "object_details": detail_desc,  # Add color+size details
        })
    frame_summaries_json = json.dumps(frame_summaries, indent=2)
    
    prompt = (
        "You are merging consecutive per-second traffic scene annotations into ONE cumulative JSON summary.\n"
        "For AI-human comparison: environment, lighting, traffic_density, traffic_flow must match form options.\n\n"
        "CRITICAL RULES:\n"
        "1. Use ONLY the data provided below—do NOT invent observations or movements.\n"
        "2. Counts are pre-aggregated. MUST match them exactly. Do NOT change.\n"
        "3. ONLY describe movements in DETECTED MOVEMENTS or TRACKER DATA.\n"
        "4. Be FACTUAL: your narrative is grounded in provided data, not inference.\n"
        "5. ALWAYS INCLUDE COLORS in descriptions. Example: 'white sedan moved left' not 'car moved left'.\n"
        "6. Narrative MUST mention BOTH static and moving objects. Include parked vehicles and pedestrian presence.\n"
        "7. Do NOT claim movements or object presence unsupported by data.\n\n"
        f"## PER-SECOND SUMMARY (seconds {span_start} to {span_end}):\n{frame_summaries_json}\n\n"
        f"## AGGREGATED COUNTS (MUST use these, do NOT change):\n"
        f"- Average vehicles: {avg_vehicles}\n"
        f"- Average pedestrians: {avg_pedestrians}\n"
        f"- Average cyclists: {avg_cyclists}\n"
        f"- Average traffic lights: {avg_traffic_lights}\n\n"
        f"## DETECTED MOVEMENTS (ONLY these in narrative):\n"
        f"{movements_json}\n\n"
        "Return ONLY this JSON (no markdown, no explanation):\n"
        "{\n"
        "  \"annotator_type\": \"ai\",\n"
        f"  \"scene_id\": \"{scene_id}\",\n"
        "  \"time_span\": {\"start_second\": " + str(span_start) + ", \"end_second\": " + str(span_end) + "},\n"
        "  \"environment\": \"urban street\",\n"
        "  \"lighting\": \"bright daylight\",\n"
        "  \"traffic_density\": \"moderate\",\n"
        "  \"traffic_flow\": \"free-flowing\",\n"
        f"  \"total_vehicles\": {avg_vehicles},\n"
        f"  \"total_pedestrians\": {avg_pedestrians},\n"
        f"  \"total_cyclists\": {avg_cyclists},\n"
        f"  \"total_traffic_lights\": {avg_traffic_lights},\n"
        "  \"scene_narrative\": \"Specific 3-4 sentence description with colors, types, movements, and counts.\",\n"
        "  \"spatial_description\": \"Describe foreground vs background layout.\",\n"
        "  \"hazards_and_events\": \"none\",\n"
        "  \"temporal_movements\": [\n"
        "    {\"object_type\": \"car\", \"color\": \"white\", \"movement\": \"moving right\", \"distance_px\": 45.2, \"evidence_seconds\": [0,1,2]}\n"
        "  ],\n"
        "  \"annotation_confidence\": 0.8\n"
        "}\n\n"
        "INSTRUCTIONS FOR FILLING JSON:\n"
        "- environment: Choose ONE: urban street, highway, parking lot, intersection, residential area, school zone, or construction zone\n"
        "- lighting: Choose ONE: bright daylight, low-light, night with street lights, or night without lighting\n"
        "- traffic_density: Choose ONE: empty, light, moderate, heavy, or gridlock\n"
        "- traffic_flow: Choose ONE: free-flowing, slow-moving, stopped, mixed, one-directional, or bidirectional\n"
        "- total_vehicles, total_pedestrians, total_cyclists, total_traffic_lights: MUST be exactly: " + str(avg_vehicles) + ", " + str(avg_pedestrians) + ", " + str(avg_cyclists) + ", " + str(avg_traffic_lights) + "\n"
        "- scene_narrative: Write 3-4 sentences describing: (1) Overall scene and environment, (2) Vehicle types/colors and their state (parked/moving), (3) Pedestrian presence and activity, (4) Any movements with colors and directions. Use ONLY data from PER-SECOND SUMMARY.\n"
        "- temporal_movements: For each item in DETECTED MOVEMENTS, create object with: object_type (car/person/truck/etc), color, movement description, distance_px, evidence_seconds. Must identify WHAT object is moving (e.g., 'car', 'person', 'truck').\n"
        "- If no movements detected, say: 'Scene with stationary parked vehicles and standing pedestrians, no movement detected.'\n"
        "- If no hazards, always write: 'none'"
    )
    result = call_qwen_text(prompt)
    cumulative = result if isinstance(result, dict) else {}
    cumulative["annotator_type"] = "ai"
    cumulative["scene_id"] = scene_id
    cumulative["time_span"] = {
        "start_second": span_start,
        "end_second": span_end
    }
    
    # Ensure aggregated counts are used (force override if LLM gave wrong numbers)
    if cumulative.get("total_vehicles") == 0 and avg_vehicles > 0:
        cumulative["total_vehicles"] = avg_vehicles
    if cumulative.get("total_pedestrians") == 0 and avg_pedestrians > 0:
        cumulative["total_pedestrians"] = avg_pedestrians
    if cumulative.get("total_cyclists") == 0 and avg_cyclists > 0:
        cumulative["total_cyclists"] = avg_cyclists
    if cumulative.get("total_traffic_lights") == 0 and avg_traffic_lights > 0:
        cumulative["total_traffic_lights"] = avg_traffic_lights
    
    if not cumulative.get("temporal_movements") and movements_detected:
        cumulative["temporal_movements"] = movements_detected
    else:
        cumulative.setdefault("temporal_movements", movements_detected)
    cumulative.setdefault("annotation_confidence", None)
    return cumulative


def _build_cumulative_from_cumulatives(left_summary, right_summary, scene_id):
    # Consolidate movement data from both sides
    left_movements = left_summary.get("temporal_movements", []) or []
    right_movements = right_summary.get("temporal_movements", []) or []
    
    # Aggregate counts
    left_vehicles = left_summary.get("total_vehicles", 0)
    right_vehicles = right_summary.get("total_vehicles", 0)
    left_peds = left_summary.get("total_pedestrians", 0)
    right_peds = right_summary.get("total_pedestrians", 0)
    left_cyclists = left_summary.get("total_cyclists", 0)
    right_cyclists = right_summary.get("total_cyclists", 0)
    left_tls = left_summary.get("total_traffic_lights", 0)
    right_tls = right_summary.get("total_traffic_lights", 0)
    
    avg_vehicles = int((left_vehicles + right_vehicles) / 2 + 0.5)
    avg_peds = int((left_peds + right_peds) / 2 + 0.5)
    avg_cyclists = int((left_cyclists + right_cyclists) / 2 + 0.5)
    avg_tls = int((left_tls + right_tls) / 2 + 0.5)
    
    combined_movements = left_movements + right_movements
    left_ts = left_summary.get("time_span", {}) or {}
    right_ts = right_summary.get("time_span", {}) or {}
    span_start = _safe_int(left_ts.get("start_second"), default=0)
    span_end = _safe_int(right_ts.get("end_second"), default=span_start)
    track_summaries = get_track_motion_summary(span_start, span_end) if get_track_motion_summary is not None else []
    track_summaries_json = json.dumps(track_summaries, indent=2) if track_summaries else "[]"
    
    prompt = (
        "You are merging TWO consecutive cumulative traffic summaries into one larger cumulative summary.\n"
        "CRITICAL: Do NOT lose or omit movement data. Preserve all detected movements and describe transitions.\n"
        "CRITICAL: Use ONLY movements present in the combined summaries. Do NOT invent new movements.\n"
        "Temporal progression: LEFT (earlier) → RIGHT (later).\n\n"
        f"## LEFT SUMMARY (earlier):\n{json.dumps(left_summary, indent=2)}\n\n"
        f"## RIGHT SUMMARY (later):\n{json.dumps(right_summary, indent=2)}\n\n"
        f"## TRACKER SUMMARY (tracks overlapping this combined span):\n{track_summaries_json}\n\n"
        f"## COMBINED MOVEMENT EVIDENCE (Use colors and distances from this section):\n{json.dumps(combined_movements, indent=2)}\n\n"
        f"## AGGREGATED COUNTS (MUST match these exactly):\n"
        f"- Average vehicles: {avg_vehicles}\n"
        f"- Average pedestrians: {avg_peds}\n"
        f"- Average cyclists: {avg_cyclists}\n"
        f"- Average traffic lights: {avg_tls}\n\n"
        "Return ONLY this JSON (no markdown, no explanation):\n"
        "{\n"
        "  \"annotator_type\": \"ai\",\n"
        f"  \"scene_id\": \"{scene_id}\",\n"
        "  \"time_span\": {\"start_second\": " + str(span_start) + ", \"end_second\": " + str(span_end) + "},\n"
        "  \"environment\": \"urban street\",\n"
        "  \"lighting\": \"bright daylight\",\n"
        "  \"traffic_density\": \"moderate\",\n"
        "  \"traffic_flow\": \"free-flowing\",\n"
        f"  \"total_vehicles\": {avg_vehicles},\n"
        f"  \"total_pedestrians\": {avg_peds},\n"
        f"  \"total_cyclists\": {avg_cyclists},\n"
        f"  \"total_traffic_lights\": {avg_tls},\n"
        "  \"scene_narrative\": \"Comprehensive 3-4 sentence description of combined scene.\",\n"
        "  \"spatial_description\": \"Describe foreground vs background.\",\n"
        "  \"hazards_and_events\": \"none\",\n"
        "  \"temporal_movements\": [\n"
        "    {\"object_type\": \"car\", \"color\": \"white\", \"movement\": \"moving right\", \"distance_px\": 50.0, \"evidence_seconds\": [2,3,4,5]}\n"
        "  ],\n"
        "  \"annotation_confidence\": 0.8\n"
        "}\n\n"
        "INSTRUCTIONS:\n"
        "- environment: Choose ONE: urban street, highway, parking lot, intersection, residential area, school zone, or construction zone\n"
        "- lighting: Choose ONE: bright daylight, low-light, night with street lights, or night without lighting\n"
        "- traffic_density: Choose ONE: empty, light, moderate, heavy, or gridlock\n"
        "- traffic_flow: Choose ONE: free-flowing, slow-moving, stopped, mixed, one-directional, or bidirectional\n"
        f"- total_vehicles, total_pedestrians, total_cyclists, total_traffic_lights: MUST be exactly {avg_vehicles}, {avg_peds}, {avg_cyclists}, {avg_tls}\n"
        "- scene_narrative: Write 3-4 sentences describing: (1) Overall combined scene, (2) Vehicle types/colors and activity, (3) Pedestrian presence across both periods, (4) Key movements with colors and directions. Use LEFT and RIGHT summaries.\n"
        "- temporal_movements: For each movement in COMBINED MOVEMENT EVIDENCE, include: object_type (car/person/truck/cyclist/etc), color, movement description, distance_px, evidence_seconds. Each movement MUST identify the object type being described.\n"
        "- If no movements, write: 'static scene - parked/stationary objects only'\n"
        "- If no hazards, always write: 'none'"
    )
    result = call_qwen_text(prompt)
    merged = result if isinstance(result, dict) else {}
    merged["annotator_type"] = "ai"
    merged["scene_id"] = scene_id
    
    # Ensure aggregated counts override if LLM gives 0
    if merged.get("total_vehicles") == 0 and avg_vehicles > 0:
        merged["total_vehicles"] = avg_vehicles
    if merged.get("total_pedestrians") == 0 and avg_peds > 0:
        merged["total_pedestrians"] = avg_peds
    if merged.get("total_cyclists") == 0 and avg_cyclists > 0:
        merged["total_cyclists"] = avg_cyclists
    if merged.get("total_traffic_lights") == 0 and avg_tls > 0:
        merged["total_traffic_lights"] = avg_tls
    
    if not merged.get("temporal_movements") and combined_movements:
        merged["temporal_movements"] = combined_movements
    else:
        merged.setdefault("temporal_movements", combined_movements)
    merged.setdefault("annotation_confidence", None)
    return merged


def _copy_with_updated_scene_id(src_path, dst_path, scene_id):
    data = _load_json(src_path)
    data["scene_id"] = scene_id
    _dump_json(dst_path, data)


def generate_hierarchical_cumulative(num_seconds, chunk_seconds=VIDEO_CHUNK_SECONDS):
    print("\nGenerating hierarchical cumulative summaries...")

    second_paths = []
    for sec in range(num_seconds):
        p = f"output/scene/output_sec_{sec}.json"
        if os.path.exists(p):
            second_paths.append(p)

    if not second_paths:
        print("  No per-second files found. Skipping cumulative generation.")
        return None

    tree_root = "output/cumulative_tree"
    os.makedirs(tree_root, exist_ok=True)

    # Level 1: merge every 5 consecutive seconds.
    level = 1
    level_dir = os.path.join(tree_root, f"level_{level}")
    os.makedirs(level_dir, exist_ok=True)
    current_nodes = []

    total_chunks = math.ceil(len(second_paths) / chunk_seconds)
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_seconds
        chunk = second_paths[start_idx:start_idx + chunk_seconds]
        if not chunk:
            continue

        frames_payload = [_normalize_second_frame(p) for p in chunk]
        span_start = _safe_int(frames_payload[0].get("second"), default=start_idx)
        span_end = _safe_int(frames_payload[-1].get("second"), default=start_idx + len(chunk) - 1)
        scene_id = f"video_l1_{span_start:05d}_{span_end:05d}"
        out_path = os.path.join(level_dir, f"cumulative_{span_start:05d}_{span_end:05d}.json")

        try:
            cumulative = _build_cumulative_from_seconds(frames_payload, span_start, span_end, scene_id)
        except Exception as e:
            print(f"  Level 1 merge error ({scene_id}): {e}")
            cumulative = {
                "annotator_type": "ai",
                "scene_id": scene_id,
                "time_span": {"start_second": span_start, "end_second": span_end},
                "environment": "",
                "lighting": "",
                "traffic_density": "",
                "traffic_flow": "",
                "total_vehicles": 0,
                "total_pedestrians": 0,
                "total_cyclists": 0,
                "total_traffic_lights": 0,
                "scene_narrative": "",
                "spatial_description": "",
                "hazards_and_events": "none",
                "temporal_movements": [],
                "annotation_confidence": None,
            }

        _dump_json(out_path, cumulative)
        print(f"  Saved L1 cumulative: {out_path}")
        current_nodes.append({"path": out_path, "start": span_start, "end": span_end})

    # Higher levels: pairwise merge neighboring cumulative files until one remains.
    level = 2
    while len(current_nodes) > 1:
        prev_nodes = current_nodes
        current_nodes = []
        level_dir = os.path.join(tree_root, f"level_{level}")
        os.makedirs(level_dir, exist_ok=True)

        i = 0
        while i < len(prev_nodes):
            left = prev_nodes[i]
            if i + 1 >= len(prev_nodes):
                # Carry forward odd tail node unchanged.
                scene_id = f"video_l{level}_{left['start']:05d}_{left['end']:05d}"
                out_path = os.path.join(level_dir, f"cumulative_{left['start']:05d}_{left['end']:05d}.json")
                _copy_with_updated_scene_id(left["path"], out_path, scene_id)
                print(f"  Carried forward: {out_path}")
                current_nodes.append({"path": out_path, "start": left["start"], "end": left["end"]})
                i += 1
                continue

            right = prev_nodes[i + 1]
            span_start = left["start"]
            span_end = right["end"]
            scene_id = f"video_l{level}_{span_start:05d}_{span_end:05d}"
            out_path = os.path.join(level_dir, f"cumulative_{span_start:05d}_{span_end:05d}.json")

            try:
                merged = _build_cumulative_from_cumulatives(
                    _load_json(left["path"]),
                    _load_json(right["path"]),
                    scene_id,
                )
                merged["time_span"] = {
                    "start_second": span_start,
                    "end_second": span_end
                }
            except Exception as e:
                print(f"  Level {level} merge error ({scene_id}): {e}")
                merged = {
                    "annotator_type": "ai",
                    "scene_id": scene_id,
                    "time_span": {"start_second": span_start, "end_second": span_end},
                    "environment": "",
                    "lighting": "",
                    "traffic_density": "",
                    "traffic_flow": "",
                    "total_vehicles": 0,
                    "total_pedestrians": 0,
                    "total_cyclists": 0,
                    "total_traffic_lights": 0,
                    "scene_narrative": "",
                    "spatial_description": "",
                    "hazards_and_events": "none",
                    "temporal_movements": [],
                    "annotation_confidence": None,
                }

            _dump_json(out_path, merged)
            print(f"  Saved L{level} cumulative: {out_path}")
            current_nodes.append({"path": out_path, "start": span_start, "end": span_end})
            i += 2

        level += 1

    final_path = current_nodes[0]["path"]
    final_summary = _load_json(final_path)

    os.makedirs("output/summaries", exist_ok=True)
    _dump_json("output/summaries/output_cumulative_mega.json", final_summary)


    manifest = {
        "chunk_seconds": chunk_seconds,
        "total_seconds_processed": len(second_paths),
        "final_summary_path": "output/summaries/output_cumulative_mega.json",
        "tree_root": tree_root
    }
    _dump_json("output/cumulative_tree/manifest.json", manifest)

    print("  Saved: output/summaries/output_cumulative_mega.json")

    print("  Saved: output/cumulative_tree/manifest.json")
    return final_summary


def generate_cumulative(num_seconds):
    """Backward-compatible wrapper."""
    return generate_hierarchical_cumulative(num_seconds, chunk_seconds=VIDEO_CHUNK_SECONDS)


# ── Generate AI cumulative for static image ───────────────────────────────────
def generate_static_cumulative(result):
    """Convert a single-frame result into CUMULATIVE_SUMMARY_SCHEMA format for AI."""
    ss = result.get("scene_summary", {})
    detections = ss.get("detected_objects", [])

    # Build object_groups from detections — group by type
    from collections import defaultdict
    groups_raw = defaultdict(list)
    for obj in detections:
        groups_raw[obj.get("object_type", "other")].append(obj)

    object_groups = []
    for otype, objs in groups_raw.items():
        sizes  = [o.get("size", "medium") for o in objs]
        typical_size = max(set(sizes), key=sizes.count) if sizes else "medium"
        positions = [o.get("position", "") for o in objs]
        zones = []
        for p in positions:
            if "Foreground" in p: zones.append("foreground")
            elif "Background" in p: zones.append("background")
        zone = max(set(zones), key=zones.count) if zones else "foreground"
        actions = [o.get("action", "static") for o in objs if "action" in o]
        behavior = max(set(actions), key=actions.count) if actions else "static"
        object_groups.append({
            "group_label":       f"{len(objs)} {otype}(s)",
            "object_type":       otype,
            "count":             len(objs),
            "typical_size":      typical_size,
            "zone":              zone,
            "behavior":          behavior
        })

    cumulative = {
        "annotator_type":         "ai",
        "scene_id":               "static",
        "environment":            ss.get("environment", ""),
        "lighting":               ss.get("lighting", ""),
        "traffic_density":        ss.get("traffic_density", ""),
        "traffic_flow":           ss.get("traffic_flow", ""),
        "total_vehicles":         ss.get("total_vehicles_detected", 0),
        "total_pedestrians":      ss.get("total_pedestrians_detected", 0),
        "total_cyclists":         ss.get("total_cyclists_detected", 0),
        "total_traffic_lights":   ss.get("total_traffic_lights_detected", 0),
        "object_groups":          object_groups,
        "scene_narrative":        ss.get("scene_description", ""),
        "spatial_description":    ss.get("spatial_description", ""),
        "hazards_and_events":     ss.get("hazards_and_events", "none"),
        "annotation_confidence":  0.85
    }
    os.makedirs("output/summaries", exist_ok=True)
    with open("output/summaries/output_ai_cumulative.json", "w") as f:
        json.dump(cumulative, f, indent=2)
    print("  Saved: output/summaries/output_ai_cumulative.json")
    return cumulative


# ── Google Sheets ─────────────────────────────────────────────────────────────
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
        except Exception:
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
    """
    Append a CUMULATIVE_SUMMARY_SCHEMA dict as a flat Sheets row.
    Column order matches SHEET_HEADERS exactly.
    """
    sid = _ensure_spreadsheet()
    svc = _get_sheets()
    if not sid or not svc:
        return False
    try:
        # Build object count summary
        object_groups = summary.get("object_groups", [])
        object_summary = "; ".join([f"{g['count']} {g['object_type']}" for g in object_groups])
        
        row = [
            summary.get("annotator_type", "human"),
            summary.get("participant_id", ""),
            summary.get("scene_id", ""),
            summary.get("environment", ""),
            summary.get("lighting", ""),
            summary.get("traffic_density", ""),
            summary.get("traffic_flow", ""),
            summary.get("total_vehicles", 0),
            summary.get("total_pedestrians", 0),
            summary.get("total_cyclists", 0),
            summary.get("total_traffic_lights", 0),
            summary.get("scene_narrative", ""),
            summary.get("hazards_and_events", ""),
            object_summary
        ]
        svc.spreadsheets().values().append(
            spreadsheetId=sid,
            range="Sheet1!A:N",
            valueInputOption="RAW",
            body={"values": [row]}
        ).execute()
        print("✓ Row appended to Google Sheet")
        return True
    except Exception as e:
        print(f"⚠  Sheet append failed: {e}")
        return False


# ── Gap Detection (top-N objects only) ───────────────────────────────────────
def detect_gaps(description, scene_fields, scene_id):
    """
    Check if the narrative covers required scene-level fields.
    Returns: { missing_scene_fields: [...] }
    """

    # Scene-level fields not yet covered by MCQ (MCQ already captured categorical ones)
    scene_text_fields = ["scene_narrative", "spatial_description", "hazards_and_events",
                         "traffic_density", "traffic_flow"]

    # Check which scene text fields are missing from the narrative
    filled_scene_fields = list(scene_fields.keys())

    prompt = f"""Analyse this traffic scene description written by a human.
"{description}"

Check if any of these specific details are COMPLETELY ABSENT:
1. "spatial_description": Did they mention depth layout, foreground/background, lanes, or sidewalks? (If they mention ANY spatial relationships, do NOT flag this).
2. "hazards_and_events": Did they mention safety concerns or lack thereof? (If they mention anything about the safety or events, do NOT flag this).

Strict rule: DO NOT flag "scene_narrative". The user has already provided the main narrative.

Already answered via dropdowns (do NOT flag these): {json.dumps(filled_scene_fields)}

Return ONLY valid JSON (no markdown):
{{
  "missing_scene_fields": ["spatial_description", "hazards_and_events"], // ONLY include those completely absent! If in doubt, assume they answered it.
  "objects_described_count": 0 // Integer count of how many specific distinct objects the user clearly described (e.g. type + color + position). Example: "white sedan and black SUV" = 2.
}}
"""

    try:
        result = call_qwen_text(prompt)
        return {
            "missing_scene_fields":  result.get("missing_scene_fields", []),
        }
    except Exception as e:
        print(f"  Gap detection error: {e}")
        return {
            "missing_scene_fields":  scene_text_fields,
        }


# ── Flask ─────────────────────────────────────────────────────────────────────
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


# (Form moved to templates/form.html)


flask_app = Flask(__name__)


def _find_image(scene_id):
    for c in [f"output/scene/frame_{scene_id}.jpg", f"output/scene/frame_sec_{scene_id}.jpg", "output/scene/frame_static.jpg"]:
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
    """Return a JSON with a preview URL when a video target exists."""
    # If DEFAULT_TARGET is a video file and exists, provide preview URL
    target = DEFAULT_TARGET
    if not os.path.exists(target):
        return jsonify({"url": None})
    ext = os.path.splitext(target)[1].lower()
    if ext in {".mp4", ".mov", ".avi", ".mkv"}:
        return jsonify({"url": "/video/preview"})
    return jsonify({"url": None})


@flask_app.route('/video/preview')
def video_preview():
    """Stream the target video file for preview in the form."""
    target = DEFAULT_TARGET
    if not os.path.exists(target):
        return ("Not found", 404)
    # Use send_file to stream the file directly
    return send_file(target, mimetype='video/mp4', conditional=True)


@flask_app.route("/api/detect-gaps", methods=["POST"])
def api_detect_gaps():
    data        = request.get_json(force=True)
    scene_id    = data.get("scene_id", "static")
    description = data.get("description", "")
    scene_fields= data.get("scene_fields", {})
    if not description:
        return jsonify({"error": "No description provided"}), 400
    try:
        result = detect_gaps(description, scene_fields, scene_id)
        described_count = result.get("objects_described_count", 0)

        # Add up to 3 major objects for the user to describe
        top_n = 3
        try:
            if os.path.exists("output/scene/output_static.json"):
                with open("output/scene/output_static.json") as f:
                    d = json.load(f)
                    scene_sum = d.get("scene_summary", {})
                    
                    # Number of objects actually mapped by AI
                    detected = scene_sum.get("detected_objects", [])
                    num = len(detected)
                    
                    # Alternatively use total vehicles if array is somehow truncated
                    if num == 0:
                        num = scene_sum.get("total_vehicles_detected", 0) + scene_sum.get("total_pedestrians_detected", 0) + scene_sum.get("total_cyclists_detected", 0)

                    top_n = min(3, num)
        except Exception as e:
            print(f"Failed to load static JSON for object gap count: {e}")
            top_n = 3
            
        remaining_to_ask = max(0, top_n - described_count)
            
        if remaining_to_ask > 0:
            result["missing_objects"] = [{"id": f"obj_{i}", "label": f"Prominent Object {i}"} for i in range(1, remaining_to_ask + 1)]

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@flask_app.route("/api/submit", methods=["POST"])
def api_submit():
    data            = request.get_json(force=True)
    scene_id        = data.get("scene_id", "static")
    participant_id  = data.get("participant_id", "").strip()
    narrative       = data.get("narrative", "").strip()
    scene_fields    = data.get("scene_fields", {})
    hazards_and_events = data.get("hazards_and_events", "").strip()
    objects         = data.get("objects", [])  # Array of individual object instances
    traffic_behavior = data.get("traffic_behavior", "").strip()
    events_over_time = data.get("events_over_time", "").strip()

    if not participant_id or not narrative:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Build object_groups from the detailed objects array
        # Objects come as individual instances from the form
        from collections import defaultdict
        object_groups_dict = defaultdict(lambda: {"count": 0, "colors": [], "sizes": [], "positions": [], "actions": [], "raw_instances": []})
        
        for obj in objects:
            obj_type = obj.get("object_type", "other")
            color = obj.get("color")
            size = obj.get("size")
            position = obj.get("position", "")
            action = obj.get("action")
            
            object_groups_dict[obj_type]["count"] += 1
            if color: object_groups_dict[obj_type]["colors"].append(color)
            if size: object_groups_dict[obj_type]["sizes"].append(size)
            if position: object_groups_dict[obj_type]["positions"].append(position)
            if action: object_groups_dict[obj_type]["actions"].append(action)
            object_groups_dict[obj_type]["raw_instances"].append(obj)
        
        # Convert to CUMULATIVE_SUMMARY_SCHEMA format
        object_groups = []
        for obj_type, data_dict in object_groups_dict.items():
            # Determine zone from positions
            zones = set()
            for pos in data_dict["positions"]:
                if "Foreground" in pos: zones.add("foreground")
                elif "Midground" in pos: zones.add("midground")
                elif "Background" in pos: zones.add("background")
            zone = list(zones)[0] if zones else "foreground"
            
            # Get most common values
            colors = data_dict["colors"]
            typical_color = max(set(colors), key=colors.count) if colors else None
            sizes = data_dict["sizes"]
            typical_size = max(set(sizes), key=sizes.count) if sizes else "medium"
            actions = data_dict["actions"]
            behavior = max(set(actions), key=actions.count) if actions else "static"
            
            object_groups.append({
                "group_label":       f"{data_dict['count']} {obj_type}(s)",
                "object_type":       obj_type,
                "count":             data_dict["count"],
                "typical_size":      typical_size,
                "zone":              zone,
                "behavior":          behavior,
                "typical_color":     typical_color,
                "raw_instances":     data_dict["raw_instances"]  # Include raw instance data
            })
        
        # Count objects by type from the groups
        total_vehicles = sum(g["count"] for g in object_groups if g["object_type"] in ["car", "van", "truck", "bus", "motorcycle"])
        total_pedestrians = sum(g["count"] for g in object_groups if g["object_type"] == "pedestrian")
        total_cyclists = sum(g["count"] for g in object_groups if g["object_type"] == "cyclist")
        total_traffic_lights = sum(g["count"] for g in object_groups if g["object_type"] == "traffic_light")

        # Build CUMULATIVE_SUMMARY_SCHEMA-compatible human summary
        human_summary = {
            "annotator_type":         "human",
            "participant_id":         participant_id,
            "scene_id":               scene_id,
            "environment":            scene_fields.get("environment", ""),
            "lighting":               scene_fields.get("lighting", ""),
            "traffic_density":        scene_fields.get("traffic_density", ""),
            "traffic_flow":           scene_fields.get("traffic_flow", ""),
            "total_vehicles":         total_vehicles,
            "total_pedestrians":      total_pedestrians,
            "total_cyclists":         total_cyclists,
            "total_traffic_lights":   total_traffic_lights,
            "object_groups":          object_groups,
            "scene_narrative":        narrative,
            "spatial_description":    "",  # Can be collected via follow-up if needed
            "hazards_and_events":     hazards_and_events if hazards_and_events else "none",
            "raw_narrative":          narrative,
            "raw_objects":            objects,  # Store raw object data for detailed analysis
            "annotation_confidence":  None
        }
        
        # Add video-specific fields if present
        if traffic_behavior:
            human_summary["traffic_behavior"] = traffic_behavior
        if events_over_time:
            human_summary["events_over_time"] = events_over_time

        # Local backup
        fname = f"annotations/{participant_id}_annotation.json"
        with open(fname, "w") as f:
            json.dump(human_summary, f, indent=2)
        print(f"✓ Saved: {fname}")

        saved_to_sheets = append_row(human_summary)

        return jsonify({"success": True, "sheets": saved_to_sheets})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Server ────────────────────────────────────────────────────────────────────
_ngrok_proc = None

def _cleanup_ngrok():
    print("Shutting down ngrok...")
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception:
        pass
    global _ngrok_proc
    if _ngrok_proc:
        try:
            _ngrok_proc.terminate()
            _ngrok_proc.wait(timeout=2)
        except Exception:
            pass

def _try_ngrok(port):
    global _ngrok_proc
    try:
        from pyngrok import ngrok as _ngrok, conf as _ngrok_conf
        _ngrok_conf.get_default().auth_token = "3CgoRdxOoU7NIRyNklz6KitDJGT_ucohrp9iQkCz44Sjakkw"
        tunnel = _ngrok.connect(port, "http")
        url = tunnel.public_url.replace("http://", "https://")
        print(f"\n  PUBLIC URL (share this): {url}/?scene_id=static")
        return url
    except ImportError:
        pass
    except Exception as e:
        print(f"  pyngrok error: {e}")

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


# ── Video Pipeline ────────────────────────────────────────────────────────────
def process_video(path):
    """
    Full video pipeline:
      1. yolo_bytetrack.process_video_frames() — YOLO + GMC + ByteTracker on
         every Nth frame. Writes output/frame_{idx:06d}.json with track_id + color.
      2. Per-second Qwen vision analysis — reads each second's frame, enriches
         with scene description + confirmed colors.
      3. Hierarchical cumulative summary via Qwen text.
      4. Flask annotation server.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit("Error: cannot open video file.")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = max(1, int(total_frames / fps))
    cap.release()
    print(f"Video: {duration}s @ {fps:.1f} FPS")

    # ── Clear existing output to ensure files are newly created for every run
    if os.path.exists("output"):
        print("Clearing existing output folder...")
        shutil.rmtree("output", ignore_errors=True)

    for _d in ["output", "output/frames", "output/annotated",
               "output/scene", "output/yolo_raw",
               "output/summaries", "output/cumulative_tree"]:
        os.makedirs(_d, exist_ok=True)

    # ── Step 1: YOLO + GMC + ByteTracker (yolo_bytetrack.py) ─────────────────
    if TRACKER_AVAILABLE and process_video_frames is not None:
        try:
            print("\n🔍 Step 1: Running YOLO + GMC + ByteTracker...")
            process_video_frames(
                video_path=path,
                out_dir="output",
                frame_rate=TRACKER_FRAME_RATE,
            )
        except Exception as e:
            print(f"  ❌ yolo_bytetrack error: {e}")
            traceback.print_exc()
    else:
        print("⚠  yolo_bytetrack unavailable; skipping tracking step.")

    # ── Step 2: Per-second Qwen analysis ─────────────────────────────────────
    print("\n🧠 Step 2: Qwen per-second scene analysis...")
    cap2 = cv2.VideoCapture(path)
    for sec in range(duration):
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap2.read()
        if not ok:
            break

        # Load pre-computed frame-level detections from yolo_bytetrack if available
        # Find the closest frame_{idx:06d}.json for this second
        actual_rate = TRACKER_FRAME_RATE if TRACKER_FRAME_RATE > 0 else min(int(fps), max(15, int(math.floor(fps * 0.5))))
        step = max(1, int(round(fps / actual_rate)))
        frame_idx_for_sec = int(sec * fps)
        # Round to nearest tracked frame
        tracked_idx = (frame_idx_for_sec // step) * step
        precomp_path = f"output/frames/frame_{tracked_idx:06d}.json"
        yolo_data = None
        if os.path.exists(precomp_path):
            try:
                with open(precomp_path) as pf:
                    precomp = json.load(pf)
                yolo_data = _yolo_data_from_frame_result(precomp)
            except Exception:
                yolo_data = None

        # Load pre-computed annotated image (YOLO boxes + IDs) for Qwen
        annotated_img_path = f"output/annotated/frame_{tracked_idx:06d}_track.jpg"
        annotated_frame = None
        if os.path.exists(annotated_img_path):
            annotated_frame = cv2.imread(annotated_img_path)

        analyse_frame(
            frame,
            scene_id        = str(sec),
            out_json_path   = f"output/scene/output_sec_{sec}.json",
            out_img_path    = f"output/scene/frame_sec_{sec}.jpg",
            yolo_data       = yolo_data,
            frame_idx       = frame_idx_for_sec,
            annotated_frame = annotated_frame,
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap2.release()

    # ── Step 3: Hierarchical cumulative summary ───────────────────────────────
    print("\n📊 Step 3: Building hierarchical cumulative summary...")
    generate_hierarchical_cumulative(duration, chunk_seconds=VIDEO_CHUNK_SECONDS)

    _ensure_spreadsheet()
    start_server()
    print("\nProcessed all frames. Open the form in your browser.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping.")
        _cleanup_ngrok()


# ── Image Pipeline ────────────────────────────────────────────────────────────
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

    # Also write AI cumulative in the shared schema format
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
# Edit DEFAULT_TARGET at the top of this file, then run:  python scene_annotator.py
if __name__ == "__main__":
    target = DEFAULT_TARGET
    ext    = os.path.splitext(target)[1].lower()

    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        process_image(target)
    elif ext in {".mp4", ".mov", ".avi", ".mkv"}:
        process_video(target)
    else:
        sys.exit(f"Unsupported file type: {ext}")
