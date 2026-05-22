"""
list_top3_candidates.py
-----------------------
After Phase 1 runs, print all candidate object identities for a scene so you
can pick which 3 to lock.

Usage:
    python list_top3_candidates.py scene-0757

Then:
    python scene_annotator.py scene-0757 --resume --top3 <id1>,<id2>,<id3>
    # optionally pin the representative second:
    python scene_annotator.py scene-0757 --resume --top3 <id1>,<id2>,<id3> --sec 17
"""

import glob
import json
import os
import sys

if len(sys.argv) < 2:
    sys.exit("Usage: python list_top3_candidates.py <scene-name>")

scene = sys.argv[1]
kf_dir = f"output/{scene}/full/keyframes"
if not os.path.isdir(kf_dir):
    sys.exit(f"Not found: {kf_dir} — run Phase 1 first.")


def _bbox_area(bb):
    return max(0, bb.get("x2", 0) - bb.get("x1", 0)) * \
           max(0, bb.get("y2", 0) - bb.get("y1", 0))


def _identity_of(obj):
    ident = obj.get("instance_token") or obj.get("gt_instance_token")
    if ident:
        return ident
    tid = obj.get("track_id")
    return f"yolo_{tid}" if tid is not None else None


agg = {}
for kf_path in sorted(glob.glob(os.path.join(kf_dir, "keyframe_*.json"))):
    with open(kf_path) as f:
        data = json.load(f)
    sec = data.get("sample_idx")
    if sec is None:
        try:
            sec = int(os.path.basename(kf_path).replace("keyframe_", "").replace(".json", ""))
        except ValueError:
            sec = -1

    for obj in data.get("scene_summary", {}).get("detected_objects", []) or []:
        ident = _identity_of(obj)
        if not ident:
            continue
        if ident not in agg:
            agg[ident] = {
                "identity":   ident,
                "type":       obj.get("object_type") or obj.get("type", "object"),
                "gt_sourced": obj.get("gt_sourced", False),
                "total_area": 0,
                "seconds":    [],
            }
        agg[ident]["total_area"] += _bbox_area(obj.get("bounding_box", {}))
        agg[ident]["seconds"].append(int(sec))
        if obj.get("gt_sourced"):
            agg[ident]["gt_sourced"] = True

rows = sorted(agg.values(), key=lambda x: x["total_area"], reverse=True)
print(f"\n{len(rows)} unique identities in {scene}:\n")
print(f"{'#':>3}  {'IDENTITY':<40}  {'TYPE':<14}  {'GT':<3}  {'AREA px²':>10}  SECONDS SEEN")
print("-" * 100)
for i, r in enumerate(rows, 1):
    secs = sorted(set(r["seconds"]))
    secs_str = (", ".join(map(str, secs[:10])) + (f"  …(+{len(secs)-10})" if len(secs) > 10 else ""))
    print(f"{i:>3}  {r['identity']:<40}  {r['type']:<14}  "
          f"{'✓' if r['gt_sourced'] else '':<3}  {r['total_area']:>10,}  [{secs_str}]")

# Suggest seconds where any 3 are co-visible
co_visible = {}
for r in rows:
    for s in set(r["seconds"]):
        co_visible.setdefault(s, []).append(r["identity"])
best = sorted(co_visible.items(), key=lambda kv: -len(kv[1]))[:5]
if best:
    print(f"\nSeconds with most identities co-visible (helpful for --sec):")
    for s, idents in best:
        print(f"  sec {s}: {len(idents)} identities")
