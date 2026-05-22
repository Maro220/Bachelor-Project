"""
regen_top3_cumulative.py
------------------------
Regenerate output/<scene>/top3/summaries/output_cumulative_top3.json from the
existing per-keyframe Qwen outputs in top3/scene/, without re-running Phase 2.

Usage:
    python regen_top3_cumulative.py scene-0757
"""

import json
import os
import sys

if len(sys.argv) < 2:
    sys.exit("Usage: python regen_top3_cumulative.py <scene-name>")

scene_name   = sys.argv[1]
top3_out_dir = f"output/{scene_name}/top3"
if not os.path.isdir(top3_out_dir):
    sys.exit(f"Not found: {top3_out_dir}")

# Pull stable_fields from any existing top3 scene file
stable_fields = {"environment": "", "lighting": "", "nuscenes_description": ""}
for fn in sorted(os.listdir(os.path.join(top3_out_dir, "scene"))):
    if fn.startswith("output_sec_") and fn.endswith(".json"):
        with open(os.path.join(top3_out_dir, "scene", fn)) as f:
            d  = json.load(f)
        ss = d.get("scene_summary", {}) or {}
        stable_fields["environment"] = ss.get("environment", "") or stable_fields["environment"]
        stable_fields["lighting"]    = ss.get("lighting",    "") or stable_fields["lighting"]
        if stable_fields["environment"] and stable_fields["lighting"]:
            break

# Need top3_identities for the wrapper signature (not actually used for read).
rep_json = os.path.join(top3_out_dir, "representative_frame.json")
top3_identities = []
if os.path.exists(rep_json):
    with open(rep_json) as f:
        top3_identities = json.load(f).get("top3_identities", [])

# scene_annotator must be importable; it brings get_track_motion_summary into scope.
import scene_annotator  # noqa: F401
from top3_pipeline import generate_top3_cumulative

result = generate_top3_cumulative(top3_identities, top3_out_dir, stable_fields)
print(f"\nResult keys: {list(result.keys()) if result else '(empty)'}")
