"""
summarize_human_annotations.py
──────────────────────────────
Collective human-annotation summary for a scene.

Reads ALL annotations in:
    annotations/<scene>/simple/*.json

For each of the 3 tracked objects (Object 1 / 2 / 3), aggregates what every
human annotator said and asks Qwen to produce ONE analysis paragraph per
object describing the spread of human perceptions — e.g. "3 annotators called
it white, 2 called it red; most agreed it was a bus; motion descriptions were
split between stationary and moving."

This is a COLLECTIVE summary (all annotators together), not per-annotation,
and it is comparative/descriptive — it does NOT speculate on *why* a human
perceived something a certain way.

Usage:
    python summarize_human_annotations.py scene-0103

Output:
    annotations/<scene>/summary/human_collective_summary.json
"""

import base64
import glob
import json
import os
import sys
import time

import requests


# ── EDIT THIS (used when no CLI arg is given) ───────────────────────────────
SCENE = "scene-0103"
# ─────────────────────────────────────────────────────────────────────────────


# ── Ollama config (mirrors reason_human_annotation.py) ──────────────────────
OLLAMA_URL  = "http://127.0.0.1:11434/api/chat"
QWEN_MODEL  = "qwen2.5vl:7b"
NUM_CTX     = 8192
NUM_PREDICT = 700
TEMPERATURE = 0.0
RETRY_COUNT = 2
RETRY_WAIT  = 10


# ── Path helpers ────────────────────────────────────────────────────────────
def find_representative_image(scene_dir: str) -> str | None:
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
            imgs = sorted(f for f in os.listdir(d) if f.endswith((".jpg", ".png")))
            if imgs:
                return os.path.join(d, imgs[0])
    return None


def find_cumulative_summary(scene_dir: str) -> str | None:
    candidates = [
        os.path.join(scene_dir, "top3", "summaries", "output_cumulative_top3.json"),
        os.path.join(scene_dir, "full", "summaries", "output_cumulative.json"),
        os.path.join(scene_dir, "full", "summaries", "output_cumulative_full.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_top3_identities(scene_dir: str) -> list[dict]:
    p = os.path.join(scene_dir, "top3", "top3_identities.json")
    if not os.path.exists(p):
        sys.exit(f"ERROR: top3_identities.json not found: {p}")
    with open(p) as f:
        return json.load(f)


def gt_for_top3(summary: dict, top3_identities: list[dict]) -> list[dict]:
    """Match objects_seen entries to the 3 locked identities, ordered by slot."""
    by_ident = {o.get("identity"): o for o in summary.get("objects_seen", [])}
    rows = []
    for ident in top3_identities:
        slot = ident.get("slot", 0)
        gt   = by_ident.get(ident.get("identity"), {})
        rows.append({
            "slot":        slot,
            "slot_label":  f"Object {slot + 1}",
            "object_type": gt.get("object_type", ident.get("object_type", "unknown")),
            "color":       gt.get("color",       ident.get("color", "unknown")),
            "action":      gt.get("action", "unknown"),
        })
    rows.sort(key=lambda r: r["slot"])
    return rows


# ── Annotation loading ──────────────────────────────────────────────────────
def load_human_annotations(scene_dir_annotations: str) -> list[dict]:
    """Load every *.json under annotations/<scene>/simple/."""
    simple_dir = os.path.join(scene_dir_annotations, "simple")
    if not os.path.isdir(simple_dir):
        sys.exit(f"ERROR: no simple/ annotations dir: {simple_dir}")
    files = sorted(glob.glob(os.path.join(simple_dir, "*.json")))
    anns  = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
            d["_source_file"] = os.path.basename(fp)
            anns.append(d)
        except Exception as e:
            print(f"  ⚠  skipped {fp}: {e}")
    return anns


# ── Prompt ──────────────────────────────────────────────────────────────────
def build_prompt(gt_rows: list[dict], free_texts: list[str]) -> str:
    gt_lines = "\n".join(
        f"{r['slot_label']} | type={r['object_type']} | color={r['color']} | motion={r['action']}"
        for r in gt_rows
    ) or "No GT available."

    ann_block = "\n\n".join(
        f"Annotator {i+1}:\n\"{t.strip()}\""
        for i, t in enumerate(free_texts) if t.strip()
    ) or "No annotations submitted."

    n = sum(1 for t in free_texts if t.strip())

    return f"""You are analysing how a GROUP of {n} human observers described the same traffic scene.

Each observer was asked to describe the SAME 3 labeled objects (Object 1, Object 2, Object 3) after watching the video. Below are the ground-truth facts for those 3 objects and the free-text each observer wrote.

──────────────────────────────────────
GROUND TRUTH — the 3 objects everyone was asked about:
{gt_lines}
──────────────────────────────────────
HUMAN ANNOTATIONS (one block per observer):
{ann_block}
──────────────────────────────────────

Write ONE analysis paragraph for EACH object (Object 1, Object 2, Object 3).
For each object:
- Aggregate what the observers collectively said about its TYPE, COLOR, and MOTION.
- State the spread explicitly with counts when observers disagree
  (e.g. "3 observers described it as white, 2 as red").
- Note where the group agreed and where it diverged.
- Briefly compare the collective human view to the ground truth for that object.
- Do NOT speculate about WHY an individual perceived something — only describe and compare what was said.

Use the attached image only to confirm the visible facts when describing agreement/disagreement.

Format your answer EXACTLY as:

Object 1: <paragraph>

Object 2: <paragraph>

Object 3: <paragraph>"""


# ── Qwen caller ─────────────────────────────────────────────────────────────
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
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.RequestException as e:
            print(f"  ⚠  Request error on attempt {attempt+1}: {e}")
        if attempt < RETRY_COUNT:
            print(f"     Retrying in {RETRY_WAIT}s …")
            time.sleep(RETRY_WAIT)
    return None


def split_into_objects(text: str) -> dict:
    """Parse 'Object N: ...' blocks into {Object 1: ..., ...}; fall back to raw."""
    out = {}
    if not text:
        return out
    import re
    parts = re.split(r"(Object\s+[123])\s*:", text)
    # parts = ['', 'Object 1', '<body>', 'Object 2', '<body>', ...]
    for i in range(1, len(parts) - 1, 2):
        key  = parts[i].strip()
        body = parts[i + 1].strip()
        out[key] = body
    return out


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else SCENE
    scene_out_dir     = os.path.join("output", scene)
    scene_annotations = os.path.join("annotations", scene)

    if not os.path.isdir(scene_out_dir):
        sys.exit(f"ERROR: no pipeline output: {scene_out_dir}")

    top3_identities = load_top3_identities(scene_out_dir)

    summary_path = find_cumulative_summary(scene_out_dir)
    if not summary_path:
        sys.exit(f"ERROR: no cumulative summary found under {scene_out_dir}")
    with open(summary_path) as f:
        summary = json.load(f)
    gt_rows = gt_for_top3(summary, top3_identities)

    img_path = find_representative_image(scene_out_dir)
    if not img_path:
        sys.exit(f"ERROR: no representative image under {scene_out_dir}/top3/")

    anns       = load_human_annotations(scene_annotations)
    free_texts = [a.get("free_text", "") for a in anns]
    n_valid    = sum(1 for t in free_texts if t.strip())
    if n_valid == 0:
        sys.exit("ERROR: no non-empty human annotations found.")

    print(f"Scene        : {scene}")
    print(f"Annotators   : {n_valid}")
    print(f"GT image     : {img_path}")
    print(f"GT summary   : {summary_path}")
    for r in gt_rows:
        print(f"  {r['slot_label']} | {r['object_type']} | {r['color']} | {r['action']}")
    print("\nCalling Qwen for collective per-object analysis …")

    prompt    = build_prompt(gt_rows, free_texts)
    analysis  = call_qwen_vision(img_path, prompt)
    if not analysis:
        sys.exit("ERROR: Qwen returned nothing after retries.")

    per_object = split_into_objects(analysis)
    print(f"\n{analysis}\n")

    output = {
        "annotator_type":   "human_collective",
        "scene_id":         scene,
        "num_annotators":   n_valid,
        "source_files":     [a["_source_file"] for a in anns],
        "source_image":     img_path,
        "source_summary":   summary_path,
        "gt_objects":       gt_rows,
        "raw_free_texts":   free_texts,
        "analysis_raw":     analysis,
        "per_object":       per_object,
    }

    out_dir  = os.path.join(scene_annotations, "summary")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "human_collective_summary.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
