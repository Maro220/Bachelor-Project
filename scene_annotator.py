
import base64
import math
import json
import os
import sys
import time
import traceback
from datetime import datetime
from threading import Thread

import cv2
import requests
from flask import Flask, jsonify, render_template, request, send_file
# tracker integration — ByteTracker (robust multi-object tracking)
try:
    from tracker_bytetrack import run_bytetrack
    TRACKER_AVAILABLE = True
except Exception as e:
    print(f"⚠  ByteTracker import failed: {e}. Install: pip install ultralytics")
    TRACKER_AVAILABLE = False
    run_bytetrack = None

# ── Optional Google Sheets ───────────────────────────────────────────────────
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

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_URL       = "http://localhost:11434/api/chat"
VISION_MODEL     = "qwen2.5vl:7b"
TEXT_MODEL       = "qwen2.5:3b"
CREDENTIALS_FILE = "credentials.json"
OAUTH_FILE       = "oauth_client.json"
TOKEN_FILE       = "token.json"
SHEET_ID_FILE    = "spreadsheet_id.txt"
SCOPES           = ["https://www.googleapis.com/auth/spreadsheets"]
SERVER_PORT      = 7860
DEFAULT_TARGET   = "assets/3.mp4"  # can be image or video path
TOP_N_OBJECTS_FOR_GAP_FILL = 3  # only ask about largest N objects in Stage 2
VIDEO_CHUNK_SECONDS = 5
TRACKER_FRAME_RATE = 5  # how many detections per second to generate for the tracker (higher than Qwen sampling)

# ── Shared Cumulative Summary Schema ─────────────────────────────────────────
# Both AI and Human annotations produce this exact structure.
# This is the unit of comparison in evaluation.
CUMULATIVE_SUMMARY_SCHEMA = {
    "annotator_type": "ai | human",
    "scene_id": "",

    # ── Scene context ────────────────────────────────────────────────────────
    "environment":    "urban street | highway | parking lot | intersection | residential area | school zone | construction zone",
    "lighting":       "bright daylight | low-light | night with street lights | night without lighting",

    # ── Object counts (ground-truth-anchored for AI, estimated for human) ───
    "total_vehicles":         0,
    "total_pedestrians":      0,
    "total_cyclists":         0,
    "total_traffic_lights":   0,

    # ── Traffic dynamics ─────────────────────────────────────────────────────
    "traffic_density":  "empty | light | moderate | heavy | gridlock",
    "traffic_flow":     "free-flowing | slow-moving | stopped | mixed | one-directional | bidirectional",

    # ── Object groups (NOT per-object — summarised by category) ─────────────
    # Each entry describes a group of similar objects observed
    "object_groups": [
        {
            "group_label":   "e.g. 'parked cars' / 'crossing pedestrians' / 'delivery trucks'",
            "object_type":   "car | truck | bus | person | cyclist | motorcycle | traffic_light",
            "count":         0,
            "typical_size":  "small | medium | large",
            "zone":          "foreground | background",
            "behavior":      "parked | moving | stopped | crossing | turning | static | mixed"
        }
    ],
    "scene_narrative": "3–4 sentences describing the overall scene holistically: what kind of place, what is happening, what stands out.",
    "hazards_and_events": "any safety concerns, unusual events, obstructions, or noteworthy observations. 'none' if absent.",
    "spatial_description": "brief description of depth layers: what occupies foreground vs background, lane structure, sidewalks, etc.",
    "annotation_confidence": None   # 0.0–1.0 for AI, null for human
}

# ── Internal per-frame AI schema (more granular, for AI processing only) ─────
# This is NOT used in the survey — it feeds the cumulative summary above.
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

# ── Google Sheets columns (in order) ─────────────────────────────────────────
SHEET_HEADERS = [
    "Annotator Type", "Participant ID", "Scene ID",
    "Environment", "Lighting",
    "Traffic Density", "Traffic Flow",
    "Total Vehicles", "Total Pedestrians", "Total Cyclists",
    "Total Traffic Lights",
    "Scene Narrative", "Hazards and Events",
    "Object Count Summary"
]

# ── YOLO ─────────────────────────────────────────────────────────────────────
_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None and YOLO_AVAILABLE:
        print("Loading YOLO11 model...")
        _yolo_model = YOLO("yolo11l.pt")
    return _yolo_model


def run_yolo(frame):
    model = get_yolo()
    h_orig, w_orig = frame.shape[:2]
    scale  = min(1024 / max(h_orig, w_orig), 1.0)
    small  = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)))
    h, w   = small.shape[:2]
    canvas = small.copy()

    vehicle_count       = 0
    pedestrian_count    = 0
    cyclist_count       = 0
    traffic_light_count = 0
    other_count         = 0
    detections          = []
    drawn_labels        = []
    valid_id            = 0

    if model:
        results = model(small, conf=0.25)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id          = int(box.cls[0])
            class_name      = model.names[cls_id]
            confidence      = float(box.conf[0])
            area            = (x2 - x1) * (y2 - y1)

            if area < 600 and class_name != "traffic light":
                continue

            valid_id += 1

            if class_name in VEHICLE_CLASSES:       vehicle_count    += 1
            elif class_name == "person":             pedestrian_count += 1
            elif class_name == "bicycle":            cyclist_count    += 1
            elif class_name == "traffic light":      traffic_light_count += 1
            else:                                    other_count      += 1

            cx = (x1 + x2) / 2
            horizontal = "Left" if cx < w * 0.40 else ("Right" if cx > w * 0.60 else "Center")
            depth      = "Foreground" if y2 > h * 0.50 else "Background"
            position   = f"{depth} {horizontal}"

            detections.append({
                "id":           valid_id,
                "type":         class_name,
                "confidence":   round(confidence, 2),
                "position":     position,
                "area":         area,
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

            color = ((cls_id * 85) % 255, (cls_id * 150) % 255, (255 - (cls_id * 45) % 255))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label           = f"ID:{valid_id} {confidence:.2f}"
            font            = cv2.FONT_HERSHEY_SIMPLEX
            fs, ft          = 0.45, 1
            (tw, th), _     = cv2.getTextSize(label, font, fs, ft)
            ly              = y1 - 10 if y1 > 40 else y1 + th + 10
            for _ in range(20):
                if not any(abs(x1 - px) < tw + 30 and abs(ly - py) < th + 15
                           for (px, py, _, _) in drawn_labels):
                    break
                ly -= th + 15
            drawn_labels.append((x1, ly, x1 + tw, ly + th))
            cv2.rectangle(canvas, (x1, ly - th - 5), (x1 + tw + 5, ly + 5), (0, 0, 0), -1)
            cv2.putText(canvas, label, (x1 + 2, ly), font, fs, (255, 255, 255), ft, cv2.LINE_AA)

    yolo_data = {
        "frame_info":         {"width": w, "height": h},
        "total_objects":      valid_id,
        "vehicle_count":      vehicle_count,
        "pedestrian_count":   pedestrian_count,
        "cyclist_count":      cyclist_count,
        "traffic_light_count":traffic_light_count,
        "other_count":        other_count,
        "detections":         detections
    }
    return yolo_data, canvas


# ── Qwen helpers ──────────────────────────────────────────────────────────────
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


def call_qwen_vision(frame_bgr, prompt):
    # Downscale to 512px long-side — enough for color/scene, much faster to encode
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
            "num_ctx":     4096,   # was 8192 — smaller context = faster KV cache
            "num_predict": 2048,   # was 6000 — caps output, avoids runaway generation
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
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return _safe_parse_json(r.json()["message"]["content"])


# ── Per-Frame Analysis ────────────────────────────────────────────────────────
def analyse_frame(frame, scene_id, out_json_path, out_img_path):
    is_static         = (scene_id == "static")
    yolo_data, canvas = run_yolo(frame)

    with open(f"output/yolo_raw_{scene_id}.json", "w") as f:
        json.dump(yolo_data, f, indent=2)

    cv2.imwrite(out_img_path, frame)
    yolo_img_path = out_img_path.replace(".jpg", "_yolo.jpg")
    cv2.imwrite(yolo_img_path, canvas)
    print(f"  YOLO bounded image: {yolo_img_path}")

    n_det  = yolo_data["total_objects"]
    n_veh  = yolo_data["vehicle_count"]
    n_ped  = yolo_data["pedestrian_count"]
    n_cyc  = yolo_data["cyclist_count"]
    n_tl   = yolo_data["traffic_light_count"]
    schema = STATIC_SCHEMA if is_static else VIDEO_SCHEMA

    # Compact detection list — only what Qwen needs to look up (id + bbox for color sampling)
    compact_dets = [
        {"id": d["id"], "type": d["type"], "pos": d["position"],
         "bbox": d["bounding_box"], "area": d["area"]}
        for d in yolo_data["detections"]
    ]

    action_note = ""
    if not is_static:
        action_note = (
            "For each object also add \"action\" based on type:\n"
            "  - For VEHICLES (car, truck, bus, motorcycle): parked | moving | turning left | turning right | stopped\n"
            "  - For PEOPLE (person, pedestrian): standing | walking | running | crossing\n"
            "  - For CYCLISTS (cyclist, bicycle): moving | stopped | turning\n"
        )

    prompt = (
        f"You are annotating a traffic scene image recorded from a moving car dashboard. "
        f"Distinguish parked cars and static objects from active traffic and moving objects. "
        f"{n_det} objects were detected by YOLO.\n\n"
        f"DETECTIONS (id, type, position, bounding_box already confirmed):\n"
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
        f"   color (REQUIRED): white/black/silver/gray/red/blue/green/yellow/orange/brown/mixed/unknown\n"
        f"   size (REQUIRED for vehicles/cyclists ONLY): small(<5% frame)/medium(5-20%)/large(>20%)\n"
        f"   {action_note}"
        f"CRITICAL:\n"
        f"  - EVERY object MUST have a color. Do NOT skip or leave null.\n"
        f"  - Size is REQUIRED for: cars, vans, trucks, buses, motorcycles, cyclists, bicycles.\n"
        f"  - Size is NOT included for: pedestrians, traffic lights, road signs.\n"
        f"Return ONLY the JSON object, no explanation."
    )

    print(f"  > Qwen analysing frame '{scene_id}'...")
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

    qwen_objs  = {str(o.get("object_id")): o for o in (ss.get("detected_objects") or [])}
    final_objs = []
    for y in yolo_data["detections"]:
        yid = str(y["id"])
        ai  = qwen_objs.get(yid, {})
        
        # Determine if this object type should have size
        obj_type = y["type"].lower()
        is_vehicle_or_cyclist = any(x in obj_type for x in ["car", "van", "truck", "bus", "motorcycle", "cyclist", "bicycle"])
        
        obj = {
            "object_id":        y["id"],
            "object_type":      y["type"],
            "confidence":       y["confidence"],
            "color":            ai.get("color"),  # Fallback to unknown if missing
            "position":         y["position"],
            "bounding_box":     y["bounding_box"],
            "bounding_box_area":y["area"],
        }
        
        # Only add size for vehicles/cyclists, not for pedestrians or traffic lights
        if is_vehicle_or_cyclist:
            obj["size"] = ai.get("size") or "unknown"
        elif ai.get("size"):  # Only include if explicitly provided (e.g., for other object types)
            obj["size"] = ai.get("size")
        
        if not is_static:
            # Object-type-aware action defaults
            if "traffic light" in obj_type or "light" in obj_type:
                default_action = "static"
            elif any(x in obj_type for x in ["person", "pedestrian"]):
                default_action = "standing"
            elif any(x in obj_type for x in ["cyclist", "bicycle"]):
                default_action = "stopped"
            else:  # vehicles (car, truck, bus, etc.)
                default_action = "parked"
            obj["action"] = ai.get("action", default_action)
        final_objs.append(obj)

    ss["detected_objects"] = final_objs
    result.pop("detected_objects", None)

    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_json_path}")
    return result


# ── AI Cumulative Summary → CUMULATIVE_SUMMARY_SCHEMA ────────────────────────
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


def _collect_track_movements_for_span(span_start, span_end, tracks_path="output/tracks.json"):
    if not os.path.exists(tracks_path):
        return []
    try:
        data = _load_json(tracks_path)
    except Exception:
        return []

    movements = []
    for track in data.get("tracks", []) or []:
        frames = track.get("frames", []) or []
        span_frames = [f for f in frames if span_start <= _safe_int(f.get("second"), -1) <= span_end]
        if len(span_frames) < 2:
            continue

        start_bbox = span_frames[0].get("bbox", {})
        end_bbox = span_frames[-1].get("bbox", {})
        start_cx = (start_bbox.get("x1", 0) + start_bbox.get("x2", 0)) / 2
        start_cy = (start_bbox.get("y1", 0) + start_bbox.get("y2", 0)) / 2
        end_cx = (end_bbox.get("x1", 0) + end_bbox.get("x2", 0)) / 2
        end_cy = (end_bbox.get("y1", 0) + end_bbox.get("y2", 0)) / 2

        delta_x = end_cx - start_cx
        delta_y = end_cy - start_cy
        if abs(delta_x) < 1 and abs(delta_y) < 1:
            continue

        movements.append({
            "object_hint": f"track_{track.get('track_id')}:{track.get('object_type', 'object')}",
            "movement": f"dx={delta_x:.1f}px, dy={delta_y:.1f}px",
            "evidence_seconds": [_safe_int(f.get("second"), -1) for f in span_frames],
            "source": "tracker"
        })
    return movements


def _load_track_summaries(span_start, span_end, tracks_path="output/tracks.json"):
    """Return compact summaries of tracks overlapping the given span."""
    if not os.path.exists(tracks_path):
        return []
    try:
        data = _load_json(tracks_path)
    except Exception:
        return []

    summaries = []
    for track in data.get("tracks", []) or []:
        s = _safe_int(track.get("start_second"), -1)
        e = _safe_int(track.get("end_second"), -1)
        # overlap test
        if e < span_start or s > span_end:
            continue
        summaries.append({
            "track_id": track.get("track_id"),
            "object_type": track.get("object_type"),
            "start_second": s,
            "end_second": e,
            "frames_count": len(track.get("frames", []) or []),
            "avg_speed_px_per_second": track.get("avg_speed_px_per_second")
        })
    return summaries


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

    tracker_movements = _collect_track_movements_for_span(span_start, span_end)
    if tracker_movements:
        movements_detected = tracker_movements + movements_detected
    
    avg_vehicles = int(sum(all_vehicles) / len(all_vehicles) + 0.5) if all_vehicles else 0
    avg_pedestrians = int(sum(all_pedestrians) / len(all_pedestrians) + 0.5) if all_pedestrians else 0
    avg_cyclists = int(sum(all_cyclists) / len(all_cyclists) + 0.5) if all_cyclists else 0
    avg_traffic_lights = int(sum(all_traffic_lights) / len(all_traffic_lights) + 0.5) if all_traffic_lights else 0
    
    movements_json = json.dumps(movements_detected, indent=2) if movements_detected else "[]"
    track_summaries = _load_track_summaries(span_start, span_end)
    track_summaries_json = json.dumps(track_summaries, indent=2) if track_summaries else "[]"
    
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
        "CONTEXT: This video is recorded from a moving car dashboard — when describing movements, distinguish parked/static objects from active traffic and moving objects.\n"
        "CRITICAL RULES:\n"
        "1. Use ONLY the data provided below—do NOT invent observations.\n"
        "2. If object counts are in the data, USE those counts. Do NOT change them.\n"
        "3. If movements are listed, DESCRIBE them in the narrative with COLORS and specifics.\n"
        "4. Be FACTUAL: only describe what you see in the movement and count data.\n"
        "5. ALWAYS INCLUDE COLORS in your narrative. Example: 'white sedan moved left' not just 'car moved left'.\n\n"
        f"## PER-SECOND SUMMARY (seconds {span_start} to {span_end}):\n{frame_summaries_json}\n\n"
        f"## AGGREGATED COUNTS (from per-second data above):\n"
        f"- Average vehicles: {avg_vehicles}\n"
        f"- Average pedestrians: {avg_pedestrians}\n"
        f"- Average cyclists: {avg_cyclists}\n"
        f"- Average traffic lights: {avg_traffic_lights}\n\n"
        f"## DETECTED MOVEMENTS:\n{movements_json}\n\n"
        f"## TRACKER DATA (if available):\n{track_summaries_json}\n\n"
        "Return JSON with these exact keys:\n"
        "{\n"
        "  \"annotator_type\": \"ai\",\n"
        f"  \"scene_id\": \"{scene_id}\",\n"
        "  \"time_span\": {\"start_second\": 0, \"end_second\": 0},\n"
        "  \"environment\": \"\",\n"
        "  \"lighting\": \"\",\n"
        "  \"traffic_density\": \"\",\n"
        "  \"traffic_flow\": \"\",\n"
        "  \"total_vehicles\": 0,\n"
        "  \"total_pedestrians\": 0,\n"
        "  \"total_cyclists\": 0,\n"
        "  \"total_traffic_lights\": 0,\n"
        "  \"object_groups\": [],\n"
        "  \"scene_narrative\": \"\",\n"
        "  \"spatial_description\": \"\",\n"
        "  \"hazards_and_events\": \"\",\n"
        "  \"temporal_movements\": [],\n"
        "  \"annotation_confidence\": 0.0\n"
        "}\n\n"
        "REQUIRED:\n"
        "- total_vehicles/pedestrians/etc: MUST match the aggregated counts (do NOT invent).\n"
        "- scene_narrative: If count data shows changes or movements exist, describe them factually WITH COLORS AND SIZES.\n"
        "  Example: 'Average 5 vehicles present: white sedans, red large SUV. Pedestrians crossing observed from frame 0-3.'\n"
        "- object_groups: Based on detected types and movements. INCLUDE COLORS and SIZES in group labels like '3 white large cars, 1x red medium SUV'.\n"
        "- temporal_movements: Copy from DETECTED MOVEMENTS section above. Include color and size descriptors.\n"
        "- If no movements detected, narrative should reflect static/stable scene.\n"
        "- If hazards absent, write 'none'."
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
    
    cumulative.setdefault("object_groups", [])
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
    track_summaries = _load_track_summaries(span_start, span_end)
    track_summaries_json = json.dumps(track_summaries, indent=2) if track_summaries else "[]"
    
    prompt = (
        "You are merging TWO consecutive cumulative traffic summaries into one larger cumulative summary.\n"
        "CRITICAL: Do NOT lose or omit movement data. Preserve all detected movements and describe transitions across the boundary.\n"
        "Temporal progression: LEFT (earlier) → RIGHT (later).\n\n"
        f"## LEFT SUMMARY (earlier):\n{json.dumps(left_summary, indent=2)}\n\n"
        f"## RIGHT SUMMARY (later):\n{json.dumps(right_summary, indent=2)}\n\n"

        f"## TRACKER SUMMARY (tracks overlapping this combined span):\n{track_summaries_json}\n\n"
        f"## COMBINED MOVEMENT EVIDENCE:\n{json.dumps(combined_movements, indent=2)}\n\n"
        f"## AGGREGATED COUNTS:\n"
        f"- Average vehicles: {avg_vehicles}\n"
        f"- Average pedestrians: {avg_peds}\n"
        f"- Average cyclists: {avg_cyclists}\n"
        f"- Average traffic lights: {avg_tls}\n\n"
        "Return JSON only with these exact keys:\n"
        "{\n"
        "  \"annotator_type\": \"ai\",\n"
        f"  \"scene_id\": \"{scene_id}\",\n"
        "  \"time_span\": {\"start_second\": 0, \"end_second\": 0},\n"
        "  \"environment\": \"\",\n"
        "  \"lighting\": \"\",\n"
        "  \"traffic_density\": \"\",\n"
        "  \"traffic_flow\": \"\",\n"
        "  \"total_vehicles\": 0,\n"
        "  \"total_pedestrians\": 0,\n"
        "  \"total_cyclists\": 0,\n"
        "  \"total_traffic_lights\": 0,\n"
        "  \"object_groups\": [],\n"
        "  \"scene_narrative\": \"\",\n"
        "  \"spatial_description\": \"\",\n"
        "  \"hazards_and_events\": \"\",\n"
        "  \"temporal_movements\": [],\n"
        "  \"annotation_confidence\": 0.0\n"
        "}\n\n"
        "STRICT Rules:\n"
        "- total_vehicles, total_pedestrians, etc.: USE the aggregated counts above.\n"
        "- temporal_movements: INCLUDE all movements from LEFT + RIGHT summaries. Do not drop movement data.\n"
        "- scene_narrative: MUST include all described movements from both sides. Describe how activity progresses and transitions.\n"
        "- CRITICAL: If LEFT or RIGHT had 'pedestrians crossing' or 'cars moving', this MUST appear in your narrative.\n"
        "- Do NOT compress away motion details. Keep the narrative detailed.\n"
        "- If hazards are absent, write 'none'."
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
    
    merged.setdefault("object_groups", [])
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
        p = f"output/output_sec_{sec}.json"
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
                "object_groups": [],
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
                    "object_groups": [],
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

    # Backward compatibility with existing consumer path.
    _dump_json("output/output_cumulative_mega.json", final_summary)
    _dump_json("output/output_cumulative.json", final_summary)

    manifest = {
        "chunk_seconds": chunk_seconds,
        "total_seconds_processed": len(second_paths),
        "final_summary_path": "output/output_cumulative_mega.json",
        "tree_root": tree_root
    }
    _dump_json("output/cumulative_tree/manifest.json", manifest)

    print("  Saved: output/output_cumulative_mega.json")
    print("  Saved: output/output_cumulative.json")
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
    with open("output/output_ai_cumulative.json", "w") as f:
        json.dump(cumulative, f, indent=2)
    print("  Saved: output/output_ai_cumulative.json")
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
    for c in [f"output/frame_{scene_id}.jpg", f"output/frame_sec_{scene_id}.jpg", "output/frame_static.jpg"]:
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
            if os.path.exists("output/output_static.json"):
                with open("output/output_static.json") as f:
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
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit("Error: cannot open video file.")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = max(1, int(total_frames / fps))
    print(f"Video: {duration}s @ {fps:.1f} FPS")

    # --- High-frequency lightweight YOLO detections for tracker ---
    try:
        os.makedirs("output", exist_ok=True)
        step = max(1, int(round(fps / TRACKER_FRAME_RATE)))
        print(f"Writing frame-level detections every {step} frames (approx {TRACKER_FRAME_RATE} fps) into output/frame_*.json")
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            ydata, _ = run_yolo(frame)
            # minimal per-frame JSON compatible with tracker expectations
            per_frame = {
                "frame": frame_idx,
                "scene_summary": {
                    "total_vehicles_detected": ydata.get("vehicle_count", 0),
                    "total_pedestrians_detected": ydata.get("pedestrian_count", 0),
                    "total_cyclists_detected": ydata.get("cyclist_count", 0),
                    "total_traffic_lights_detected": ydata.get("traffic_light_count", 0),
                    "detected_objects": ydata.get("detections", [])
                }
            }
            with open(f"output/frame_{frame_idx:06d}.json", "w") as f:
                json.dump(per_frame, f, indent=2)
            frame_idx += step
    except Exception as e:
        print(f"  ! Error writing frame-level detections: {e}")


    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap.read()
        if not ok:
            break
        analyse_frame(frame,
                      scene_id      = str(sec),
                      out_json_path = f"output/output_sec_{sec}.json",
                      out_img_path  = f"output/frame_sec_{sec}.jpg")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    # --- Run ByteTracker on per-second detections ---
    try:
        if TRACKER_AVAILABLE and run_bytetrack:
            print("🔍 Running ByteTracker on per-second outputs...")
            run_bytetrack(
                num_seconds=duration,
                frame_rate=TRACKER_FRAME_RATE,
                per_second_json_dir="output",
                out_path="output/tracks.json",
                track_thresh=0.25,      # Confidence threshold
                track_buffer=30,       # Frames to keep inactive tracks
                match_thresh=0.8,      # Similarity threshold
                video_path=path
            )
        else:
            print("⚠  ByteTracker unavailable; skipping tracking step.")
            print("   Install: pip install ultralytics")
    except Exception as e:
        print(f"  ❌ Tracker error: {e}")
        traceback.print_exc()

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

    print(f"Processing image: {path}")
    result = analyse_frame(frame,
                           scene_id      = "static",
                           out_json_path = "output/output_static.json",
                           out_img_path  = "output/frame_static.jpg")

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
