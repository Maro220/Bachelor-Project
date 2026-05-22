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
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()

# ── Simple pipeline imports ──────────────────────────────────────────────────
try:
    from top3_pipeline import run_top3_phase
    TOP3_PIPELINE_AVAILABLE = True
except ImportError as _e:
    print(f"top3_pipeline not found: {_e}")
    TOP3_PIPELINE_AVAILABLE = False
    run_top3_phase = None
import cv2
import requests
from flask import Flask, jsonify, render_template, request, send_file
try:
    import yolo_bytetrack as _ybt  # module reference for mutable globals (e.g. _nuscenes_gt_by_frame)
    from yolo_bytetrack import (
        run_yolo_and_track,
        process_scene_sweeps,
        get_track_motion_summary,
        draw_tracks,
        NUSCENES_DATAROOT,
        TARGET_SCENE,
        set_nuscenes_gt,
        reset_gt_state,
        _bbox_iou,
        _derive_action,
        build_nuscenes_gt_2d,
    )
    TRACKER_AVAILABLE = True
except Exception as e:
    print(f"yolo_bytetrack import failed: {e}")
    TRACKER_AVAILABLE = False
    _ybt = None
    run_yolo_and_track = None
    process_scene_sweeps = None
    get_track_motion_summary = None
    draw_tracks = None
    NUSCENES_DATAROOT = "data/v1.0-mini"
    TARGET_SCENE = ""
    merge_gt_into_detections = None
    set_nuscenes_gt = None
    reset_gt_state = None
    _bbox_iou = lambda a, b: 0.0
    _derive_action = None
    build_nuscenes_gt_2d = None


def _gt_by_frame() -> dict:
    """Current GT-by-frame map. Goes through the module so reassignments
    inside yolo_bytetrack.set_nuscenes_gt() are visible here."""
    return _ybt._nuscenes_gt_by_frame if _ybt is not None else {}
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
DEFAULT_SCENE    = "scene-0103" 
DEFAULT_TARGET   = f"output_nuscenes/{DEFAULT_SCENE}/{DEFAULT_SCENE}_CAM_FRONT.mp4"

# Active scene output root — set at runtime by process_scene() or serve_form.py.
# All Flask routes read from this so the server always points at the right scene.
_SCENE_OUT_DIR    = "output/scene-0757"   # safe default, overwritten at runtime
_ACTIVE_SCENE_NAME = "scene-0757"         # safe default, overwritten at runtime

FIELD_OPTIONS = {
    "environment":    ["urban street", "highway", "parking lot", "intersection", "residential area", "school zone", "construction zone"],
    "lighting":       ["day", "night"],
    "traffic_density":["empty", "light", "moderate", "heavy", "gridlock"],
    "traffic_flow":   ["free-flowing", "slow-moving", "stopped", "mixed"],
}
OBJECT_TYPE_OPTIONS   = ["car", "van", "truck", "bus", "motorcycle", "cyclist", "pedestrian", "traffic_light", "cone", "barrier", "other"]
VEHICLE_CLASSES       = {"car", "van", "motorcycle", "bus", "truck"}


def _infer_scene_fields_from_nuscenes(description: str) -> dict:
    
    #Map a NuScenes scene description (e.g. 'Parked truck, construction,
   # intersection, turn left') to environment + lighting. Both are locked from
   # the NuScenes description; Qwen no longer chooses either.
   
    desc = (description or "").lower()

    # Environment keyword priority order matters: 'construction' beats 'intersection'
    # because construction zones often happen AT intersections.
    if   "construction" in desc: env = "construction zone"
    elif "school"       in desc: env = "school zone"
    elif "parking"      in desc: env = "parking lot"
    elif "highway"      in desc: env = "highway"
    elif "residential"  in desc: env = "residential area"
    elif "intersection" in desc: env = "intersection"
    else:                        env = "urban street"

    lighting = "night" if any(kw in desc for kw in ("night", "dusk", "dawn")) else "day"

    return {
        "environment": env,
        "lighting":    lighting,
    }
SHEET_HEADERS = [
    "Timestamp",
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
    "Hazards Y/N",
    "Hazards and Events",
    "Video Review",
    "Object Count Summary",
    "Object IDs",
    "Object Evidence Seconds",
    "Objects Detail (JSON)",
]
def _yolo_data_from_frame_result(frame_data: dict) -> dict:
   # Convert yolo_bytetrack frame output to the yolo_data format expected by analyse_frame.
   # Preserves all provenance flags (gt_sourced, instance_token, source, gt_type_override)
    #so merge_gt_into_detections and the strict motion policy work correctly downstream
    ss   = frame_data.get("scene_summary", {})
    dets = ss.get("detected_objects", [])
    detections = []
    for d in dets:
        detections.append({
            "id":               d.get("id"),
            "type":             d.get("type"),
            "confidence":       d.get("confidence"),
            "position":         d.get("position"),
            "area":             d.get("area"),
            "bounding_box":     d.get("bounding_box") or {},
            "track_id":         d.get("track_id", -1),
            "canonical_id":     d.get("canonical_id", -1),
            "color":            d.get("color",     "unknown"),
            "size":             d.get("size",      "unknown"),
            "speed":            d.get("speed",     "unknown"),
            "direction":        d.get("direction", "unknown"),
            "action":           d.get("action",    "unknown"),
            # provenance flags — must be preserved so GT merge works correctly
            "source":             d.get("source",             "yolo"),
            "instance_token":     d.get("instance_token",     ""),
            "gt_sourced":         d.get("gt_sourced",         False),
            "gt_type_override":   d.get("gt_type_override",   False),
            "yolo_type_original": d.get("yolo_type_original", None),
            "gt_iou":             d.get("gt_iou",             None),
            "motion_source":      d.get("motion_source",      ""),
        })
    _vt = {"car", "van", "motorcycle", "bus", "truck"}
    return {
            "frame_info":          {"width": 0, "height": 0},
            "total_objects":       len(detections),
            "vehicle_count":       sum(1 for d in detections if d["type"].lower() in _vt),
            "pedestrian_count":    sum(1 for d in detections if d["type"].lower() == "pedestrian"),
            "cyclist_count":       sum(1 for d in detections if d["type"].lower() == "cyclist"),
            "traffic_light_count": sum(1 for d in detections if d["type"].lower() == "traffic_light"),
            "other_count":         sum(1 for d in detections if d["type"].lower() not in
                                    (_vt | {"pedestrian", "cyclist", "traffic_light"})),
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
        "options": {"temperature": 0.0, "num_ctx": 12288, "num_predict": 4096}
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


def _call_with_retry(fn, *args, retries: int = 2, wait: int = 10, **kwargs):
    """Retry an Ollama caller when it returns an empty/falsy result. Useful for
    transient empty-JSON parses and reload windows after a runner restart. Does
    not help with sustained OOM — those keep failing for the same reason."""
    for attempt in range(retries + 1):
        result = fn(*args, **kwargs)
        if result:
            return result
        if attempt < retries:
            print(f"  ⚠  Empty/failed Qwen response, retrying in {wait}s "
                  f"({attempt + 1}/{retries})")
            time.sleep(wait)
    return {}


def _derive_traffic_density(n_vehicles: int, n_cyclists: int) -> str:
    total = n_vehicles + n_cyclists
    if total == 0:  return "empty"
    if total <= 3:  return "light"
    if total <= 8:  return "moderate"
    if total <= 15: return "heavy"
    return "gridlock"


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

    yolo_raw_dir = os.path.join(_SCENE_OUT_DIR, "full", "yolo_raw")
    os.makedirs(yolo_raw_dir, exist_ok=True)
    with open(os.path.join(yolo_raw_dir, f"yolo_raw_{scene_id}.json"), "w") as f:
        json.dump(yolo_data, f, indent=2)

    cv2.imwrite(out_img_path, frame)

    # Counts recomputed from yolo_data["detections"] so GT-injected objects are
    # included. The cached *_count fields in yolo_data come from
    # run_yolo_and_track's local counters which only see YOLO output, not the
    # post-merge list.
    _known_types = (VEHICLE_CLASSES
                    | {"pedestrian", "cyclist", "traffic_light",
                       "cone", "barrier"})
    _dets = yolo_data["detections"]
    n_veh = sum(1 for d in _dets if d["type"].lower() in VEHICLE_CLASSES)
    n_ped = sum(1 for d in _dets if d["type"].lower() == "pedestrian")
    n_cyc = sum(1 for d in _dets if d["type"].lower() == "cyclist")
    n_tl  = sum(1 for d in _dets if d["type"].lower() == "traffic_light")
    n_cn  = sum(1 for d in _dets if d["type"].lower() == "cone")
    n_br  = sum(1 for d in _dets if d["type"].lower() == "barrier")
    n_oth = sum(1 for d in _dets if d["type"].lower() not in _known_types)
    n_det = len(_dets)

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
        # gt_sourced is stamped by merge_gt_into_detections (both injected and
        # IoU-matched detections). motion_source="nuscenes_gt" is a redundant
        # signal kept for legacy callers.
        _is_gt = (
            d.get("gt_sourced", False) or
            d.get("source") == "gt_injected" or
            d.get("motion_source") == "nuscenes_gt"
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
        if tid in track_motions:
            entry["track_history"] = track_motions[tid]
        structured_dets.append(entry)

    # Structured GT confirmed objects list (replaces old gt_hint string)
    gt_confirmed = []
    if _gt_by_frame() and frame_idx is not None:
        frame_gt     = _gt_by_frame().get(frame_idx, {})
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
                    "zone":       zone,
                    "action":     a.get("action", "unknown"),
                    "speed":      a.get("speed_label", "unknown"),
                    "direction":  a.get("direction", "stationary"),
                    "size":       a.get("size_category", "unknown"),
                })

    # Ego motion context (structured, not a sentence)
    ego_attrs = _build_ego_motion_attrs(frame_idx or 0, video_fps)

    env_options      = " | ".join(FIELD_OPTIONS["environment"])
    flow_options     = " | ".join(FIELD_OPTIONS["traffic_flow"])

    stable_context = ""
    if stable_fields:
        env_locked = stable_fields.get("environment", "")
        nusc_desc  = stable_fields.get("nuscenes_description", "")
        stable_context = (
            f"## SCENE METADATA (from NuScenes — authoritative)\n"
            f"nuscenes_description: {nusc_desc!r}\n"
            f"environment: {env_locked}  (FIXED — output this value verbatim)\n\n"
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
6. environment: if stable_context is set, output that value verbatim.
"""
    print(f"  > Qwen analysing frame '{scene_id}'")
    try:
        result = _call_with_retry(call_qwen_vision, frame, prompt)
    except Exception as e:
        print(f"  ! Qwen vision error: {e}")
        result = {}

    result["frame"] = scene_id
    ss = result.setdefault("scene_summary", {})
    ss["total_objects_detected"]        = n_det
    ss["total_vehicles_detected"]       = n_veh
    ss["total_pedestrians_detected"]    = n_ped
    ss["total_cyclists_detected"]       = n_cyc
    ss["total_traffic_lights_detected"] = n_tl
    ss["total_cones_detected"]          = n_cn
    ss["total_barriers_detected"]       = n_br
    ss["total_other_detected"]          = n_oth
    ss["traffic_density"]               = _derive_traffic_density(n_veh, n_cyc)
    # Lock environment and lighting from NuScenes — Qwen no longer chooses either.
    if stable_fields:
        ss["environment"] = stable_fields.get("environment", ss.get("environment", ""))
        ss["lighting"]    = stable_fields.get("lighting",    ss.get("lighting",    ""))

    if _gt_by_frame() and frame_idx is not None:
        frame_gt_all  = _gt_by_frame().get(frame_idx, {})
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
            "canonical_id":     y.get("canonical_id", -1),
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
        # Propagate provenance flags from merge_gt_into_detections so the
        # strict motion policy and downstream dedup can use them.
        for flag in ("gt_sourced", "gt_type_override", "motion_source",
                     "instance_token", "gt_iou",
                     "yolo_type_original"):
            if flag in y:
                obj[flag] = y[flag]
        if is_vehicle_or_cyclist:
            obj["color"] = y.get("color", "unknown")
            obj["size"]  = y.get("size",  "unknown")

        if not is_static:
            obj["action"] = y.get("action") or _derive_action(
                y["type"], y.get("speed", "unknown"), y.get("direction", "unknown")
            )
        final_objs.append(obj)

    # Strict motion policy: motion/direction/action only trusted when LiDAR GT
    # confirms it (gt_sourced=True set by merge_gt_into_detections).
    # Anything without GT confirmation is forced to stationary.
    for obj in final_objs:
        if not obj.get("gt_sourced", False):
            obj["speed"]     = "stationary"
            obj["direction"] = "stationary"
            obj["action"]    = "static"

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
            cid = obj.get("canonical_id", -1)
            if cid == -1 or cid is None:
                cid = obj.get("track_id", "?")
            label = f"#{cid}"
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
        # Preserve the full per-instance answers (color, size, action, direction,
        # position, signal_state, age_group, …) as a JSON dump in one column.
        objects_detail_json = json.dumps(summary.get("raw_objects", []), ensure_ascii=False)
        object_ids = ", ".join(
            f"{o.get('object_type','?')}:{(o.get('tracker_id') or '').strip() or '—'}"
            for o in summary.get("raw_objects", [])
        )
        object_evidence_seconds = "; ".join(
            f"{o.get('object_type','?')}:{(o.get('tracker_id') or '').strip() or '—'}=["
            f"{(o.get('evidence_seconds') or '').strip()}]"
            for o in summary.get("raw_objects", [])
        )
        p_info     = summary.get("participant_info", {})
        timestamp  = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        row = [
            timestamp,
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
            summary.get("hazards_yn",         ""),
            summary.get("hazards_and_events", ""),
            summary.get("video_review",       ""),
            object_summary,
            object_ids,
            object_evidence_seconds,
            objects_detail_json,
        ]
        last_col = chr(ord("A") + len(SHEET_HEADERS) - 1)
        svc.spreadsheets().values().append(
            spreadsheetId=sid,
            range=f"Sheet1!A:{last_col}",
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
    base = _SCENE_OUT_DIR
    for c in [
        f"{base}/full/scene/frame_{scene_id}.jpg",
        f"{base}/full/scene/frame_sec_{scene_id}.jpg",
        f"{base}/full/scene/frame_static.jpg",
    ]:
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
    target = f"output_nuscenes/{_ACTIVE_SCENE_NAME}/{_ACTIVE_SCENE_NAME}_CAM_FRONT.mp4"
    if not os.path.exists(target):
        return jsonify({"url": None})
    return jsonify({"url": "/video/preview"})

@flask_app.route('/video/preview')
def video_preview():
    target = f"output_nuscenes/{_ACTIVE_SCENE_NAME}/{_ACTIVE_SCENE_NAME}_CAM_FRONT.mp4"
    if not os.path.exists(target):
        return ("Not found", 404)
    return send_file(target, mimetype='video/mp4', conditional=True)

@flask_app.route("/api/annotated-frames")
def api_annotated_frames():
    import glob, re
    search_dir = os.path.join(_SCENE_OUT_DIR, "full", "scene")
    files = glob.glob(f"{search_dir}/frame_sec_*_yolo.jpg")
    if not files:
        files = [f for f in glob.glob(f"{search_dir}/frame_sec_*.jpg") if "_yolo" not in f]
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
    base      = os.path.abspath(os.path.join(_SCENE_OUT_DIR, "full", "scene"))
    yolo_path = os.path.join(base, f"frame_sec_{sec}_yolo.jpg")
    plain_path = os.path.join(base, f"frame_sec_{sec}.jpg")
    if os.path.exists(yolo_path):
        return send_file(yolo_path, mimetype="image/jpeg")
    if os.path.exists(plain_path):
        return send_file(plain_path, mimetype="image/jpeg")
    return ("Not found", 404)
@flask_app.route("/api/submit", methods=["POST"])
def api_submit():
    data             = request.get_json(force=True)
    scene_id         = data.get("scene_id", "static")
    participant_id   = data.get("participant_id", "").strip()
    narrative        = data.get("narrative", "").strip()
    scene_fields     = data.get("scene_fields", {})
    hazards_yn         = data.get("hazards_yn", "").strip()
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

        # The form's scene_id (URL query param) is just a request handle —
        # the canonical Scene ID for the spreadsheet is the real NuScenes
        # scene name (e.g. "scene-0916"). Fall back to the form's value when
        # TARGET_SCENE isn't configured (non-NuScenes runs).
        canonical_scene_id = TARGET_SCENE or scene_id

        human_summary = {
            "annotator_type":       "human",
            "participant_id":       participant_id,
            "participant_info":     data.get("participant_info", {}),
            "scene_id":             canonical_scene_id,
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
            "hazards_yn":           hazards_yn,
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

def _print_public_urls(base_url: str):
    scene = _ACTIVE_SCENE_NAME or "static"
    print(f"\n  PUBLIC — full form  : {base_url}/?scene_id={scene}")
    print(f"  PUBLIC — simple form: {base_url}/simple?scene_id={scene}")


def _try_ngrok(port):
    global _ngrok_proc
    try:
        from pyngrok import ngrok as _ngrok, conf as _ngrok_conf
        _ngrok_conf.get_default().auth_token = os.getenv("NGROK_AUTH_TOKEN", "")
        tunnel = _ngrok.connect(port, "http")
        url = tunnel.public_url.replace("http://", "https://")
        _print_public_urls(url)
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
                _print_public_urls(url)
                return url
    except (FileNotFoundError, Exception):
        pass

    # Final fallback: query the local ngrok inspector (works if a tunnel is
    # already up from a previous run / external launch).
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
            for t in _json.load(r).get("tunnels", []):
                if t.get("proto") == "https" and str(port) in t.get("config", {}).get("addr", ""):
                    _print_public_urls(t["public_url"].rstrip("/"))
                    return t["public_url"]
    except Exception:
        pass

    print("  ngrok not found — only accessible on local network.")
    return None


def start_server(port=SERVER_PORT):
    def _run():
        flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
    t = Thread(target=_run, daemon=True)
    t.start()
    time.sleep(1.5)
    scene = _ACTIVE_SCENE_NAME or "static"
    print(f"\n  LOCAL  — full form  : http://localhost:{port}/?scene_id={scene}")
    print(f"  LOCAL  — simple form: http://localhost:{port}/simple?scene_id={scene}")
    _try_ngrok(port)
    print(f"  Press Ctrl+C to stop.\n")


# ── Flat cumulative (single Qwen text call over all tracker + per-second data) ─
def generate_flat_cumulative(num_seconds: int, stable_fields: dict = None, scene_dir: str = "output/scene", summaries_dir: str = "output/summaries", frames_dir: str = "output/frames") -> dict:
    """Replace the hierarchical tree with one Qwen text call.
    All object-level facts come from the tracker; Qwen only writes the narrative."""
    second_paths = [
        f"{scene_dir}/output_sec_{s}.json"
        for s in range(num_seconds)
        if os.path.exists(f"{scene_dir}/output_sec_{s}.json")
    ]
    if not second_paths:
        print("  No per-second files found. Skipping cumulative.")
        return {}

    # Authoritative counts from tracker
    movements = []
    if get_track_motion_summary is not None:
        movements = get_track_motion_summary(0, num_seconds + 1) or []

    # Deduplicate by instance_token (stable NuScenes GT identity) when available.
    # Falls back to track_id for YOLO-only objects (no GT match).
    # Prevents track fragmentation from inflating unique object counts:
    # the same bus getting IDs t11, t15, t21 counts as ONE vehicle, not three.
    vehicle_types = {"car", "vehicle", "truck", "bus", "motorcycle", "van", "taxi"}
    known_types   = vehicle_types | {"pedestrian", "cyclist", "traffic_light",
                                     "cone", "barrier"}
    uv, up, uc, utl, ucn, ubr, uoth = (set(), set(), set(), set(),
                                       set(), set(), set())

    # Aggregate per-identity data for objects_seen list
    track_agg = {}   # identity -> {object_type, instance_token, track_ids, colors, sizes, actions, seconds, gt_sourced}

    # Per-second summary for prompt
    frame_summaries = []
    for p in second_paths:
        d   = _load_json(p)
        ss  = d.get("scene_summary", {})
        sec = _safe_int(d.get("frame"), -1)

        for obj in ss.get("detected_objects", []) or []:
            tid    = obj.get("track_id", -1)
            it     = obj.get("instance_token", "")
            source = obj.get("source", "yolo")

            # Deduplication key priority:
            #   1. instance_token  – stable GT identity, collapses fragmented tracks
            #   2. track_id        – YOLO-only objects with no GT match
            #   3. position-based  – gt_injected fallback (should have token from Fix 1)
            if it:
                identity = it
            elif tid != -1:
                identity = f"track_{tid}"
            elif source == "gt_injected":
                identity = f"gt_{obj.get('object_type','?')}_{obj.get('position','?')}"
            else:
                continue

            t = obj.get("object_type", "").lower()
            if t in vehicle_types:        uv.add(identity)
            elif t == "pedestrian":       up.add(identity)
            elif t == "cyclist":          uc.add(identity)
            elif t == "traffic_light":    utl.add(identity)
            elif t == "cone":             ucn.add(identity)
            elif t == "barrier":          ubr.add(identity)
            elif t not in known_types:    uoth.add(identity)

            if identity not in track_agg:
                track_agg[identity] = {
                    "object_type":    obj.get("object_type", t),
                    "instance_token": it or None,
                    "track_ids":      set(),
                    "colors":         [],
                    "sizes":          [],
                    "actions":        [],
                    "seconds":        [],
                    "gt_sourced":     obj.get("gt_sourced", False),
                }
            agg = track_agg[identity]
            if tid != -1:
                agg["track_ids"].add(tid)
            if obj.get("gt_sourced"):
                agg["gt_sourced"] = True
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

    # Build objects_seen: one entry per unique identity (instance_token or track_id).
    def _most_common(lst):
        return max(set(lst), key=lst.count) if lst else "unknown"

    objects_seen = []
    for identity, agg in sorted(track_agg.items(), key=lambda kv: str(kv[0])):
        entry = {
            "identity":     identity,
            "track_ids":    sorted(agg["track_ids"]),
            "object_type":  agg["object_type"],
            "action":       _most_common(agg["actions"]),
            "seen_seconds": sorted(agg["seconds"]),
            "gt_sourced":   agg["gt_sourced"],
        }
        if agg["object_type"].lower() != "pedestrian":
            entry["color"] = _most_common(agg["colors"])
            entry["size"]  = _most_common(agg["sizes"])
        # thesis metric: fragmented = same real object got multiple track_ids
        if len(agg["track_ids"]) > 1:
            entry["fragmented"]     = True
            entry["fragment_count"] = len(agg["track_ids"])
        objects_seen.append(entry)

    env   = (stable_fields or {}).get("environment", "")
    light = (stable_fields or {}).get("lighting",    "")
    scene_id = f"video_0_{num_seconds - 1}"

    # Split moving vs static so Qwen focuses narrative on motion
    moving_objects = [o for o in objects_seen
                      if o.get("action") not in
                      ("parked", "static", "standing", "stopped", "unknown", "")]
    static_objects = [o for o in objects_seen if o not in moving_objects]

    # Summarize static objects to a type→count dict to avoid dumping 100+
    # full JSON entries into the prompt. Qwen only needs counts for the
    # narrative; full detail is only needed for moving objects.
    static_summary: dict = {}
    for o in static_objects:
        t = o.get("object_type", "other")
        static_summary[t] = static_summary.get(t, 0) + 1

    # Ego motion timeline across keyframes (from NuScenes ego_pose). Used so
    # Qwen can mention whether the ego car stopped mid-video or moved
    # throughout. Computed directly from _ego_poses + keyframe_map so Step 3
    # can recompute without re-running per-keyframe analysis.
    ego_summary = "unknown"
    try:
        from yolo_bytetrack import _ego_poses as _eps
        SWEEP_FPS = 12.0
        if _eps and os.path.exists(os.path.join(_SCENE_OUT_DIR, "full", "keyframe_map.json")):
            with open(os.path.join(_SCENE_OUT_DIR, "full", "keyframe_map.json")) as _f:
                _km = json.load(_f)
            timeline = []  # (sec, moving_bool)
            for kf in _km:
                sec  = kf.get("sample_idx")
                fidx = kf.get("frame_idx", 0)
                if 0 < fidx < len(_eps):
                    p1 = _eps[fidx - 1]["translation"]
                    p2 = _eps[fidx]["translation"]
                    spd = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) * SWEEP_FPS
                    timeline.append((sec, spd > 0.3))
            timeline.sort()
            if timeline:
                states = [m for _, m in timeline]
                if all(states):
                    ego_summary = "ego vehicle was moving throughout the video"
                elif not any(states):
                    ego_summary = "ego vehicle remained stationary throughout the video"
                else:
                    segments = []
                    cur_state = states[0]
                    cur_start = timeline[0][0]
                    for i in range(1, len(timeline)):
                        if states[i] != cur_state:
                            segments.append(
                                (cur_start, timeline[i - 1][0], cur_state)
                            )
                            cur_state = states[i]
                            cur_start = timeline[i][0]
                    segments.append((cur_start, timeline[-1][0], cur_state))
                    parts = [
                        f"{'moving' if mov else 'stopped'} from s{s} to s{e}"
                        for s, e, mov in segments
                    ]
                    ego_summary = "ego vehicle: " + ", then ".join(parts)
    except Exception as _e:
        print(f"  ⚠  ego timeline computation failed: {_e}")

    env_options      = " | ".join(FIELD_OPTIONS["environment"])
    density_options  = " | ".join(FIELD_OPTIONS["traffic_density"])
    flow_options     = " | ".join(FIELD_OPTIONS["traffic_flow"])

    prompt = f"""You are writing the final annotation summary for a {num_seconds}-second traffic video.
Your task: fill the JSON schema below using ONLY the structured data provided.
Do not invent any object, motion, or event not present in the data.

## SCENE CONTEXT (confirmed)
environment: {env}
lighting: {light}
ego_motion: {ego_summary}
unique_object_counts:
  vehicles: {len(uv)}
  pedestrians: {len(up)}
  cyclists: {len(uc)}
  traffic_lights: {len(utl)}
  cones: {len(ucn)}
  barriers: {len(ubr)}
  other: {len(uoth)}

## PER-SECOND OBJECT DATA (sampled — do not extrapolate between samples)
Each entry contains counts and traffic_flow label for that second.
description field = Qwen's own description from that frame (truncated).
{json.dumps(frame_summaries, indent=2)}

## MOVING OBJECTS — focus your narrative on these
Each entry has: type, color, size, action, direction, seen_seconds.
All speed/direction values are pre-computed from LiDAR/GPS or tracker.
Do not reinterpret them.
{json.dumps(moving_objects, indent=2)}

## STATIC / PARKED OBJECTS — mention briefly (counts only)
{json.dumps(static_summary)}

## OUTPUT SCHEMA
Return ONLY valid JSON matching this exact structure.
No markdown, no explanation, no extra keys.
{{
  "scene_id": "{scene_id}",
  "environment": "<{env_options}>",
  "traffic_density": "<{density_options}>",
  "traffic_flow": "<{flow_options}>",
  "scene_narrative": "<5 sentences synthesized from the data above. Sentence 1: environment type and overall layout. Sentence 2: each vehicle (if exists) with its color, size, and action — use exact color/size from moving_objects and static_objects data. Sentence 3:(if exists) pedestrians and cyclists with their actions. Sentence 4: (if exists) describe movements using the direction from moving_objects only. Sentence 5: ego-vehicle motion using ONLY the ego_motion line above (e.g. mention if the ego car stopped mid-video or moved throughout — use the seconds given verbatim, do not invent any).>",
  "spatial_description": "<1 sentence on foreground vs background layout>",
  "hazards_and_events": "<describe any hazard present in the per-second data, or exactly: none>"
}}

RULES:
1. unique_object_counts are authoritative — do not change them.
2. Every vehicle in scene_narrative must include its color and size from the data.
3. Do not describe motion for objects in static_objects.
(temporal_movements is built deterministically by the pipeline — do not emit it.)
"""

    print(f"  moving_objects count : {len(moving_objects)}")
    print(f"  static_objects count : {len(static_objects)}")
    print(f"  frame_summaries count: {len(frame_summaries)}")
    print(f"  Prompt chars         : {len(prompt)}")
    print(f"  Prompt est. tokens   : ~{int(len(prompt.split()) * 1.3)}")
    _t_cumulative_qwen = time.perf_counter()
    result = _call_with_retry(call_qwen_text, prompt)
    print(f"  Cumulative Qwen call : {time.perf_counter() - _t_cumulative_qwen:.1f}s")
    cumulative = result if isinstance(result, dict) else {}
    cumulative["annotator_type"] = "ai"
    cumulative["scene_id"]       = scene_id
    cumulative["environment"]    = env
    cumulative["lighting"]       = light
    cumulative["time_span"]      = {"start_second": 0, "end_second": num_seconds - 1}
    cumulative["total_cones"]    = len(ucn)
    cumulative["total_barriers"] = len(ubr)
    cumulative["total_other"]    = len(uoth)
    # Traffic-lights are not annotated in NuScenes — keep YOLO-derived count.
    cumulative["total_traffic_lights"] = len(utl)

    # Vehicles/pedestrians/cyclists: prefer NuScenes GT counts when available
    # (hand-labeled, no YOLO false positives). Hybrid fallback per type:
    # when GT projects zero of a type but the tracker saw some, use the
    # tracker count — handles cases like pedestrians whose 3D boxes failed
    # the camera-frustum projection filter in build_nuscenes_gt_2d.
    _all_gt: dict = {}
    counts_source_per_type: dict = {}
    if _gt_by_frame():
        _gt_vt = {"vehicle.car", "vehicle.truck",
                  "vehicle.bus.rigid", "vehicle.bus.bendy",
                  "vehicle.motorcycle", "vehicle.construction",
                  "vehicle.trailer",
                  "vehicle.emergency.ambulance", "vehicle.emergency.police"}
        for fd in _gt_by_frame().values():
            for ann in fd.get("annotations", []):
                if ann.get("visibility", 0) >= 2:
                    _all_gt[ann["instance_token"]] = ann["semantic_type"]
        gt_veh = sum(1 for st in _all_gt.values() if st in _gt_vt)
        gt_ped = sum(1 for st in _all_gt.values() if "pedestrian" in st)
        gt_cyc = sum(1 for st in _all_gt.values() if st == "vehicle.bicycle")

        def _pick(gt_n: int, tracker_n: int, label: str) -> int:
            if gt_n > 0:
                counts_source_per_type[label] = "nuscenes_gt"
                return gt_n
            counts_source_per_type[label] = "tracker_dedup"
            return tracker_n

        cumulative["total_vehicles"]    = _pick(gt_veh, len(uv), "vehicles")
        cumulative["total_pedestrians"] = _pick(gt_ped, len(up), "pedestrians")
        cumulative["total_cyclists"]    = _pick(gt_cyc, len(uc), "cyclists")
        cumulative["counts_source"]     = (
            "nuscenes_gt" if all(v == "nuscenes_gt"
                                 for v in counts_source_per_type.values())
            else "hybrid"
        )
        cumulative["counts_source_per_type"] = counts_source_per_type
    else:
        cumulative["total_vehicles"]    = len(uv)
        cumulative["total_pedestrians"] = len(up)
        cumulative["total_cyclists"]    = len(uc)
        cumulative["counts_source"]     = "tracker_dedup"

    # Fix 2 — Fragment merge: flag each objects_seen entry as gt_confirmed,
    # then collapse non-GT fragments into their GT-confirmed counterpart when
    # they share object_type and at least one observed second. Handles the
    # occlusion case where ByteTrack loses a GT track and spawns a parallel
    # non-GT track for the same physical object.
    gt_tokens = set(_all_gt.keys())
    for obj in objects_seen:
        it = obj.get("instance_token") or obj.get("identity", "")
        obj["gt_confirmed"] = bool(gt_tokens and it in gt_tokens)

    if _all_gt:
        confirmed   = [o for o in objects_seen if o.get("gt_sourced")]
        unconfirmed = [o for o in objects_seen if not o.get("gt_sourced")]
        merged_ids: set = set()

        for uc_obj in unconfirmed:
            uc_type    = uc_obj.get("object_type")
            uc_seconds = set(uc_obj.get("seen_seconds", []))
            best_match   = None
            best_overlap = 0
            for gt_obj in confirmed:
                if gt_obj.get("object_type") != uc_type:
                    continue
                overlap = len(uc_seconds & set(gt_obj.get("seen_seconds", [])))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match   = gt_obj
            if best_match is not None and best_overlap > 0:
                best_match["seen_seconds"] = sorted(
                    set(best_match["seen_seconds"]) | uc_seconds
                )
                best_match["fragment_merged"]    = True
                best_match["fragment_track_ids"] = (
                    best_match.get("fragment_track_ids", [])
                    + uc_obj.get("track_ids", [])
                )
                merged_ids.add(uc_obj["identity"])

        objects_seen = [o for o in objects_seen
                        if o["identity"] not in merged_ids]
        if merged_ids:
            print(f"  Fragment merge: {len(merged_ids)} unconfirmed tracks "
                  f"merged into GT entries")

    cumulative["objects_seen"]        = objects_seen

    # temporal_movements is built deterministically from moving_objects (real
    # tracker identity + aggregates) and the tracker's movements list (real
    # direction + speed labels). Qwen is no longer asked for this — every
    # field is grounded in tracker data, so IDs and motion are guaranteed
    # correct.
    movements_by_identity = {m.get("identity"): m for m in (movements or [])}
    movements_by_track_id = {m.get("track_id"): m for m in (movements or [])}
    cumulative["temporal_movements"] = []
    for mo in moving_objects:
        ident = mo.get("identity")
        tids  = mo.get("track_ids", []) or []
        # Look up direction/speed by identity first, then any matching track_id.
        mv = movements_by_identity.get(ident)
        if mv is None:
            for t in tids:
                if t in movements_by_track_id:
                    mv = movements_by_track_id[t]
                    break
        cumulative["temporal_movements"].append({
            "identity":         ident,
            "track_ids":        tids,
            "object_type":      mo.get("object_type", "unknown"),
            "color":            mo.get("color", "unknown"),
            "size":             mo.get("size",  "unknown"),
            "action":           mo.get("action", "unknown"),
            "direction":        (mv or {}).get("direction", "unknown"),
            "speed":            (mv or {}).get("speed",     "unknown"),
            "evidence_seconds": mo.get("seen_seconds", []),
            "gt_sourced":       mo.get("gt_sourced", False),
        })

    #  NuScenes GT validation
    if _gt_by_frame():
        import glob as _glob, re as _re
        from collections import Counter as _Counter

        gt_all:   dict = {}
        gt_vis40: dict = {}
        for fd in _gt_by_frame().values():
            for ann in fd.get("annotations", []):
                it, st = ann["instance_token"], ann["semantic_type"]
                gt_all[it] = st
                if ann.get("visibility", 0) >= 2:
                    gt_vis40[it] = st
        track_instances: dict = {}   
        instance_tracks: dict = {}  
        for ff in sorted(_glob.glob(f"{frames_dir}/frame_*.json")):
            m = _re.match(r".*frame_(\d+)\.json", ff)
            if not m:
                continue
            fidx     = int(m.group(1))
            frame_gt = _gt_by_frame().get(fidx, {})
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
                best_iou, best_it = 0.20, None
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

    os.makedirs(summaries_dir, exist_ok=True)
    _dump_json(f"{summaries_dir}/output_cumulative_mega.json", cumulative)
    print(f"  Saved: {summaries_dir}/output_cumulative_mega.json")
    return cumulative

def process_scene(scene_name: str):
    """
    Run the full pipeline for one scene. Saves all output to:
        output/<scene_name>/full/   — all objects
        output/<scene_name>/top3/   — top-3 objects only

    Does NOT start the Flask server — use serve_form.py for that.
    """
    global _SCENE_OUT_DIR, _ACTIVE_SCENE_NAME

    if not TRACKER_AVAILABLE or process_scene_sweeps is None:
        sys.exit("Tracker / nuScenes SDK not available — cannot run scene pipeline.")
    from nuscenes.nuscenes import NuScenes

    if reset_gt_state is not None:
        reset_gt_state()

    _resume   = "--resume" in sys.argv
    _no_qwen  = "--no-qwen" in sys.argv

    # Optional manual override: --top3 id1,id2,id3   and   --sec N
    _override_ids = None
    _override_sec = None
    for _i, _a in enumerate(sys.argv):
        if _a == "--top3" and _i + 1 < len(sys.argv):
            _override_ids = [x.strip() for x in sys.argv[_i + 1].split(",") if x.strip()][:3]
        elif _a == "--sec" and _i + 1 < len(sys.argv):
            try:
                _override_sec = int(sys.argv[_i + 1])
            except ValueError:
                pass
    if _override_ids and len(_override_ids) != 3:
        sys.exit(f"--top3 needs exactly 3 comma-separated identities (got {len(_override_ids)})")

    SCENE_ROOT = f"output/{scene_name}"
    FULL_OUT   = f"{SCENE_ROOT}/full"
    TOP3_OUT   = f"{SCENE_ROOT}/top3"

    # Set globals so any route called during pipeline also resolves correctly
    _SCENE_OUT_DIR    = SCENE_ROOT
    _ACTIVE_SCENE_NAME = scene_name

    if _resume:
        print(f"  --resume: keeping existing output for {scene_name}.")
    else:
        if os.path.exists(SCENE_ROOT):
            print(f"Clearing existing output for {scene_name}…")
            shutil.rmtree(SCENE_ROOT, ignore_errors=True)

    for _d in [
        FULL_OUT,
        f"{FULL_OUT}/frames", f"{FULL_OUT}/annotated", f"{FULL_OUT}/keyframes",
        f"{FULL_OUT}/scene",  f"{FULL_OUT}/yolo_raw",  f"{FULL_OUT}/summaries",
    ]:
        os.makedirs(_d, exist_ok=True)

    nusc         = NuScenes(version="v1.0-mini", dataroot=NUSCENES_DATAROOT, verbose=False)
    scene_record = next((s for s in nusc.scene if scene_name in s["name"]), None)
    if scene_record is None:
        sys.exit(f"Scene '{scene_name}' not found in {NUSCENES_DATAROOT}")

    print("\n Step 0: Projecting GT at all sweep frames...")
    try:
        first_sample = nusc.get("sample", scene_record["first_sample_token"])
        sd_token     = first_sample["data"]["CAM_FRONT"]
        sd_records   = []
        while sd_token:
            sd = nusc.get("sample_data", sd_token)
            sd_records.append(sd)
            sd_token = sd["next"]
        gt_data = build_nuscenes_gt_2d(nusc, sd_records)
        set_nuscenes_gt(gt_data)
    except Exception as _e:
        print(f"  ⚠  GT 2D projection failed: {_e}")

    print(f"\n Step 1: YOLO + ByteTracker → {FULL_OUT}/")
    process_scene_sweeps(scene_name, dataroot=NUSCENES_DATAROOT, out_dir=FULL_OUT)

    with open(f"{FULL_OUT}/keyframe_map.json") as f:
        keyframe_map = json.load(f)
    if not keyframe_map:
        sys.exit(f"No keyframes found in {FULL_OUT}/keyframe_map.json")

    nusc_desc     = (scene_record or {}).get("description", "")
    inferred      = _infer_scene_fields_from_nuscenes(nusc_desc)
    stable_fields = {
        "environment":          inferred["environment"],
        "lighting":             inferred["lighting"],
        "nuscenes_description": nusc_desc,
    }
    print(f"  env={inferred['environment']!r}, lighting={inferred['lighting']!r}")

    frames_data = []
    for kf in keyframe_map:
        sample   = nusc.get("sample", kf["sample_token"])
        sd       = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img_path = os.path.join(NUSCENES_DATAROOT, sd["filename"])
        frame    = cv2.imread(img_path)
        kf_json_path = f"{FULL_OUT}/keyframes/keyframe_{kf['sample_idx']:04d}.json"
        yolo_data    = None
        if os.path.exists(kf_json_path):
            try:
                with open(kf_json_path) as pf:
                    yolo_data = _yolo_data_from_frame_result(json.load(pf))
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"  ⚠  Keyframe load failed (sample {kf['sample_idx']}): {e}")
        frames_data.append((kf["sample_idx"], frame, yolo_data, kf["frame_idx"]))

    if _no_qwen:
        print("\n Step 2 & 3 SKIPPED (--no-qwen): YOLO output only.")
    else:
        print("\n Step 2: Qwen per-keyframe analysis...")

        def _analyse_kf(item):
            sec, frame, yolo_data, frame_idx = item
            if frame is None:
                return
            out_json = f"{FULL_OUT}/scene/output_sec_{sec}.json"
            if os.path.exists(out_json):
                print(f"   Skipping keyframe {sec} (already analysed)")
                return
            try:
                analyse_frame(
                    frame, str(sec),
                    out_json_path = out_json,
                    out_img_path  = f"{FULL_OUT}/scene/frame_sec_{sec}.jpg",
                    yolo_data     = yolo_data,
                    frame_idx     = frame_idx,
                    stable_fields = stable_fields,
                )
            except Exception as e:
                print(f"  ⚠  analyse_frame failed (keyframe {sec}): {e}")

        _t2 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as pool:
            list(pool.map(_analyse_kf, frames_data))
        print(f"\n✅ Step 2 done in {time.perf_counter()-_t2:.1f}s")

        print("\n Step 3: Flat cumulative (all objects)...")
        _t3 = time.perf_counter()
        generate_flat_cumulative(
            len(frames_data),
            stable_fields = stable_fields,
            scene_dir     = f"{FULL_OUT}/scene",
            summaries_dir = f"{FULL_OUT}/summaries",
            frames_dir    = f"{FULL_OUT}/frames",
        )
        print(f"✅ Step 3 done in {time.perf_counter()-_t3:.1f}s")

    # ── Phase 2: top-3 ───────────────────────────────────────────────────
    if TOP3_PIPELINE_AVAILABLE and run_top3_phase is not None:
        print("\n" + "═"*55)
        print("  PHASE 2: top-3 filtered pipeline")
        print("═"*55)
        sweep_fps = 2.0
        try:
            ts_list = [k["timestamp"] for k in keyframe_map if "timestamp" in k]
            if len(ts_list) >= 2:
                duration  = (ts_list[-1] - ts_list[0]) / 1e6
                sweep_fps = max(1.0, len(ts_list) / duration)
        except Exception:
            pass
        run_top3_phase(
            full_out_dir    = FULL_OUT,
            top3_out_dir    = TOP3_OUT,
            nusc            = nusc,
            dataroot        = NUSCENES_DATAROOT,
            stable_fields   = stable_fields,
            video_fps       = sweep_fps,
            no_qwen         = _no_qwen,
            override_ids    = _override_ids,
            override_sec    = _override_sec,
        )
    else:
        print("\n⚠  top3_pipeline not available — skipping Phase 2.")

    print(f"\n{'='*55}")
    print(f"  Pipeline complete for {scene_name}")
    print(f"  Full output : {FULL_OUT}/")
    print(f"  Top-3 output: {TOP3_OUT}/")
    print(f"\n  To serve the form run:")
    print(f"    python serve_form.py {scene_name}")
    print(f"{'='*55}\n")
# ── Simple survey routes ─────────────────────────────────────────────────────

SIMPLE_SHEET_HEADERS = [
    "Timestamp", "Participant Name", "Country", "Age Category", "Gender",
    "Profession", "Driving Skill", "Scene ID",
    "Free Text Response",
]
_SIMPLE_TAB_NAME = "Simple Annotations"


def _ensure_simple_tab():
    sid = _ensure_spreadsheet()
    svc = _get_sheets()
    if not sid or not svc:
        return False
    try:
        meta     = svc.spreadsheets().get(spreadsheetId=sid).execute()
        existing = [s["properties"]["title"] for s in meta.get("sheets", [])]
        if _SIMPLE_TAB_NAME not in existing:
            svc.spreadsheets().batchUpdate(
                spreadsheetId=sid,
                body={"requests": [{"addSheet": {"properties": {"title": _SIMPLE_TAB_NAME}}}]}
            ).execute()
            last_col = chr(ord("A") + len(SIMPLE_SHEET_HEADERS) - 1)
            svc.spreadsheets().values().update(
                spreadsheetId=sid,
                range=f"{_SIMPLE_TAB_NAME}!A1",
                valueInputOption="RAW",
                body={"values": [SIMPLE_SHEET_HEADERS]}
            ).execute()
            print(f"✓ Created tab: {_SIMPLE_TAB_NAME}")
        return True
    except Exception as e:
        print(f"⚠  Could not ensure simple tab: {e}")
        return False


def _append_simple_row(summary: dict) -> bool:
    _ensure_simple_tab()
    sid = _ensure_spreadsheet()
    svc = _get_sheets()
    if not sid or not svc:
        return False
    try:
        p_info = summary.get("participant_info", {})
        row = [
            datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            p_info.get("name",          summary.get("participant_name", "")),
            p_info.get("country",       ""),
            p_info.get("age_category",  ""),
            p_info.get("gender",        ""),
            p_info.get("profession",    ""),
            p_info.get("driving_skill", ""),
            summary.get("scene_id", ""),
            summary.get("free_text", ""),
        ]
        last_col = chr(ord("A") + len(SIMPLE_SHEET_HEADERS) - 1)
        svc.spreadsheets().values().append(
            spreadsheetId=sid,
            range=f"{_SIMPLE_TAB_NAME}!A:{last_col}",
            valueInputOption="RAW",
            body={"values": [row]}
        ).execute()
        print("✓ Simple row appended to Google Sheet")
        return True
    except Exception as e:
        print(f"⚠  Simple sheet append failed: {e}")
        return False


def _get_simple_top3():
    """Read Phase 2 output directly — no re-detection."""
    top3_dir = os.path.join(_SCENE_OUT_DIR, "top3")
    rep_json = os.path.join(top3_dir, "representative_frame.json")

    if not os.path.exists(rep_json):
        return None, None, None, f"Phase 2 not run yet for {_ACTIVE_SCENE_NAME}. Run the pipeline first."

    meta       = json.load(open(rep_json))
    top3       = meta.get("top3_identities", [])
    sample_idx = meta.get("sample_idx", None)

    # Prefer the Qwen-annotated keyframe (top3/scene/frame_sec_<idx>.jpg);
    # fall back to representative_frame.jpg if absent.
    rep_img = os.path.join(top3_dir, "scene", f"frame_sec_{sample_idx}.jpg")
    if not os.path.exists(rep_img):
        rep_img = os.path.join(top3_dir, "representative_frame.jpg")

    canvas_b64 = None
    if os.path.exists(rep_img):
        with open(rep_img, "rb") as f:
            canvas_b64 = base64.b64encode(f.read()).decode()

    parts = []
    for obj in top3:
        label     = obj.get("slot_label", "Object 1")
        color     = obj.get("color", "")
        color_str = f"{color} " if color and color not in ("unknown", "") else ""
        otype     = obj.get("object_type", "object")
        parts.append(f"{label} ({color_str}{otype})")
    question_context = "; ".join(parts)

    return top3, canvas_b64, sample_idx, question_context


@flask_app.route("/simple")
def serve_simple_form():
    return render_template("form_simple.html")


@flask_app.route("/api/simple/top3")
def api_simple_top3():
    try:
        top3, canvas_b64, sample_idx, question_context = _get_simple_top3()
        if top3 is None:
            return jsonify({"error": question_context}), 404
        safe_top3 = [{k: v for k, v in obj.items() if k != "slot_color"} for obj in top3]
        return jsonify({
            "top3":             safe_top3,
            "question_context": question_context,
            "frame_b64":        canvas_b64,
            "sample_idx":       sample_idx,
            "scene_id":         TARGET_SCENE or "unknown",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@flask_app.route("/api/submit_simple", methods=["POST"])
def api_submit_simple():
    data             = request.get_json(force=True)
    participant_name = data.get("participant_name", "").strip()
    free_text        = data.get("free_text", "").strip()
    top3_objects     = data.get("top3_objects", [])
    scene_id         = data.get("scene_id", TARGET_SCENE or "unknown")

    if not participant_name or not free_text:
        return jsonify({"error": "participant_name and free_text are required"}), 400

    summary = {
        "annotator_type":   "human_simple",
        "participant_name": participant_name,
        "participant_info": data.get("participant_info", {}),
        "scene_id":         scene_id,
        "free_text":        free_text,
        "top3_objects":     top3_objects,
    }
    os.makedirs(f"annotations/{_ACTIVE_SCENE_NAME}/simple", exist_ok=True)
    fname = f"annotations/{_ACTIVE_SCENE_NAME}/simple/{participant_name}_{scene_id}_simple.json"
    with open(fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Simple annotation saved: {fname}")

    saved_to_sheets = _append_simple_row(summary)
    return jsonify({"success": True, "sheets": saved_to_sheets})


@flask_app.route("/simple/frames/latest")
def simple_frame_latest():
    path = os.path.abspath(os.path.join(_SCENE_OUT_DIR, "top3", "representative_frame.jpg"))
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return ("Not found", 404)


# Entry
if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE
    process_scene(scene)