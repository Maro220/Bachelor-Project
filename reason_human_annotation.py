"""
reason_human_annotation.py
──────────────────────────
Reasons over a human's free-text annotation by comparing it against:
  - GT facts for the 3 tracked objects  (from full/summaries/output_cumulative.json,
                                          matched by top3_identities.json hashes)
  - Scene context (environment, lighting, traffic only — no object narratives)
  - Representative frame image (visual grounding for type/color mistakes)

Edit ANNOTATION_PATH below and run:
    python reason_human_annotation.py

Output:
    annotations/<scene>/reasoned/<name>_reasoned.json
"""

import base64
import json
import os
import re
import sys
import time

import requests


# ── EDIT THIS ─────────────────────────────────────────────────────────────────
ANNOTATION_PATH = "annotations/scene-0103/simple/Faris Mohamed Izzeldin_scene-0103_simple.json"
SCENE_DIR_OVERRIDE = ""
# ─────────────────────────────────────────────────────────────────────────────


# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://127.0.0.1:11434/api/chat"
QWEN_MODEL  = "qwen2.5vl:7b"
NUM_CTX     = 8192
NUM_PREDICT = 600
TEMPERATURE = 0.0
RETRY_COUNT = 2
RETRY_WAIT  = 10


# ── Helpers ───────────────────────────────────────────────────────────────────
def derive_scene_dir(annotation_path: str) -> str:
    parts = os.path.normpath(annotation_path).split(os.sep)
    if "annotations" in parts:
        i = parts.index("annotations")
        if i + 1 < len(parts):
            return os.path.join("output", parts[i + 1])
    raise ValueError(f"Cannot derive scene dir from {annotation_path!r}")


def find_cumulative_summary(scene_dir: str) -> str | None:
    """Return path to full/summaries/output_cumulative*.json."""
    candidates = [
        os.path.join(scene_dir, "full", "summaries", "output_cumulative.json"),
        os.path.join(scene_dir, "full", "summaries", "output_cumulative_full.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fallback: any output_cumulative*.json in full/summaries
    sumdir = os.path.join(scene_dir, "full", "summaries")
    if os.path.isdir(sumdir):
        for fn in sorted(os.listdir(sumdir)):
            if fn.startswith("output_cumulative") and fn.endswith(".json"):
                return os.path.join(sumdir, fn)
    return None


def find_representative_image(scene_dir: str) -> str | None:
    """Return path to the representative frame image for the top3."""
    candidates = [
        os.path.join(scene_dir, "top3", "representative_frame.jpg"),
        os.path.join(scene_dir, "top3", "representative_frame.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for subdir in ["annotated", "frames", "keyframes", ""]:
        d = os.path.join(scene_dir, "top3", subdir) if subdir else os.path.join(scene_dir, "top3")
        if os.path.isdir(d):
            imgs = sorted([f for f in os.listdir(d) if f.endswith((".jpg", ".png"))])
            if imgs:
                return os.path.join(d, imgs[0])
    return None


def extract_gt_for_top3(summary: dict, top3_identities: list) -> list[dict]:
    """
    Filter objects_seen in the cumulative summary to the 3 tracked identities.
    Returns list of dicts with: slot_label (Object 1/2/3), object_type, color, action.
    """
    identity_hashes = {obj["identity"]: obj for obj in top3_identities}
    # Use "Object N" to match exactly what the human writes
    slot_map = {
        obj["identity"]: f"Object {obj.get('slot', 0) + 1}"
        for obj in top3_identities
    }

    gt_objects = []
    for obj in summary.get("objects_seen", []):
        ident = obj.get("identity")
        if ident in identity_hashes:
            gt_objects.append({
                "slot_label":  slot_map[ident],
                "object_type": obj.get("object_type", "unknown"),
                "color":       obj.get("color", "unknown"),
                "action":      obj.get("action", "unknown"),
            })

    # sort by slot label: Object 1, Object 2, Object 3
    gt_objects.sort(key=lambda x: x["slot_label"])
    return gt_objects


def build_scene_context(summary: dict) -> str:
    """Build a compact scene context — environment and conditions only, no object narratives."""
    parts = []
    if summary.get("environment"):
        parts.append(f"Environment: {summary['environment']}")
    if summary.get("lighting"):
        parts.append(f"Lighting: {summary['lighting']}")
    if summary.get("traffic_density"):
        parts.append(f"Traffic density: {summary['traffic_density']}")
    if summary.get("traffic_flow"):
        parts.append(f"Traffic flow: {summary['traffic_flow']}")
    if summary.get("hazards_and_events") and summary["hazards_and_events"] != "none":
        parts.append(f"Hazards: {summary['hazards_and_events']}")
    return "\n".join(parts)


# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(free_text: str, gt_objects: list[dict], scene_context: str) -> str:
    gt_lines = "\n".join(
        f"{o['slot_label']} | type={o['object_type']} | color={o['color']} | motion={o['action']}"
        for o in gt_objects
    ) or "No GT available."

    return f"""You are analysing how a human perceived a traffic scene from a video.

You have four inputs:
1. The attached image — a representative frame from the video
2. Scene context (environment and lighting only — do not use this to reference other objects)
3. Ground truth for the 3 specific labeled objects the human was asked about
4. What the human wrote after watching the video

──────────────────────────────────────
SCENE CONTEXT (environment and lighting only):
{scene_context}
──────────────────────────────────────
GROUND TRUTH — the 3 objects the human was asked about:
{gt_lines}
──────────────────────────────────────
HUMAN ANNOTATION:
"{free_text}"
──────────────────────────────────────

Instructions:
- The human was only asked to describe these 3 specific objects. Do NOT mention anything they missed.
- Do NOT reference any other objects from the scene context when reasoning about the 3 labeled objects.
- Do NOT say "this is not relevant to Object X".
- For each object the human described, compare ONLY type, color, and motion to the ground truth.
- Where they got something wrong, look at the image and explain WHY they likely perceived it that way — be specific about what you see (lighting, angle, shape, shadow, etc.).
- Where they got it right, briefly confirm what in the image supports their description.

Write 3-5 sentences total. Be specific — cite what you actually see in the image."""


# ── Qwen vision caller ────────────────────────────────────────────────────────
def call_qwen_vision(image_path: str, prompt: str) -> str | None:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx":     NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }

    for attempt in range(RETRY_COUNT + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.RequestException as e:
            print(f"  ⚠  Request error on attempt {attempt+1}: {e}")
        if attempt < RETRY_COUNT:
            print(f"     Retrying in {RETRY_WAIT}s …")
            time.sleep(RETRY_WAIT)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    annotation_path = ANNOTATION_PATH
    scene_dir = SCENE_DIR_OVERRIDE or derive_scene_dir(annotation_path)

    if not os.path.exists(annotation_path):
        print(f"ERROR: annotation not found: {annotation_path}")
        sys.exit(1)

    with open(annotation_path) as f:
        ann = json.load(f)

    scene_id    = ann.get("scene_id", "unknown")
    free_text   = ann.get("free_text", "").strip()
    participant = ann.get("participant_info", {})
    p_name      = participant.get("name", ann.get("participant_name", "unknown"))
    safe_name   = re.sub(r"\s+", "_", p_name.lower())

    if not free_text:
        print("ERROR: free_text is empty.")
        sys.exit(1)

    # ── Load top3_identities ──────────────────────────────────────────────────
    identities_path = os.path.join(scene_dir, "top3", "top3_identities.json")
    if not os.path.exists(identities_path):
        print(f"ERROR: top3_identities.json not found: {identities_path}")
        sys.exit(1)
    with open(identities_path) as f:
        top3_identities = json.load(f)

    # ── Load cumulative summary ───────────────────────────────────────────────
    summary_path = find_cumulative_summary(scene_dir)
    if not summary_path:
        print(f"ERROR: output_cumulative.json not found in {scene_dir}/full/summaries/")
        sys.exit(1)
    with open(summary_path) as f:
        summary = json.load(f)

    # ── Extract GT for the 3 objects ──────────────────────────────────────────
    gt_objects = extract_gt_for_top3(summary, top3_identities)
    if not gt_objects:
        print("WARNING: no GT objects matched — check identity hashes in top3_identities.json")

    # ── Scene context (no narratives) ─────────────────────────────────────────
    scene_context = build_scene_context(summary)

    # ── Find image ────────────────────────────────────────────────────────────
    img_path = find_representative_image(scene_dir)
    if not img_path:
        print(f"ERROR: no representative image found in {scene_dir}/top3/")
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Scene      : {scene_id}")
    print(f"Annotator  : {p_name}")
    print(f"Image      : {img_path}")
    print(f"Summary    : {summary_path}")
    print(f"GT objects : {len(gt_objects)} matched")
    for o in gt_objects:
        print(f"  {o['slot_label']} | {o['object_type']} | {o['color']} | {o['action']}")
    print(f"Free text  : {free_text[:120]}{'...' if len(free_text) > 120 else ''}")
    print(f"\nCalling Qwen …")

    prompt    = build_prompt(free_text, gt_objects, scene_context)
    reasoning = call_qwen_vision(img_path, prompt)

    if not reasoning:
        print("ERROR: Qwen returned nothing after retries.")
        sys.exit(1)

    print(f"\nReasoning:\n{reasoning}\n")

    # ── Save output ───────────────────────────────────────────────────────────
    output = {
        "annotator_type":    "human_reasoned",
        "participant_name":  p_name,
        "participant_info":  participant,
        "scene_id":          scene_id,
        "source_annotation": annotation_path,
        "source_image":      img_path,
        "source_summary":    summary_path,
        "gt_objects":        gt_objects,
        "free_text":         free_text,
        "reasoning":         reasoning,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(annotation_path)),
        "reasoned"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_name}_reasoned.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved: {out_path}")


if __name__ == "__main__":
    main()