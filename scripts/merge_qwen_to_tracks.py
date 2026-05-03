#!/usr/bin/env python3
"""
Merge Qwen per-second attributes (color/size) into tracker output using detection_id mapping.

Usage: python scripts/merge_qwen_to_tracks.py
Writes updated tracks to output/tracks_enriched.json
"""
import json
import os
from collections import defaultdict

TRACKS_PATH = "output/tracks.json"
OUT_PATH = "output/tracks_enriched.json"
PER_SEC_DIR = "output"


def load_tracks(path):
    with open(path) as f:
        return json.load(f)


def build_detection_attr_map():
    # scan per-second files for detected_objects with id/object_id/detection_id
    mapping = {}
    for fname in os.listdir(PER_SEC_DIR):
        if not fname.startswith("output_sec_") or not fname.endswith(".json"):
            continue
        p = os.path.join(PER_SEC_DIR, fname)
        try:
            d = json.load(open(p))
        except Exception:
            continue
        ss = d.get("scene_summary", {})
        for obj in ss.get("detected_objects", []) or []:
            det_id = obj.get("id") or obj.get("object_id") or obj.get("detection_id")
            if det_id is None:
                continue
            mapping[str(det_id)] = {
                "color": obj.get("color"),
                "size": obj.get("size"),
                "object_type": obj.get("object_type") or obj.get("type")
            }
    return mapping


def enrich_tracks():
    tracks = load_tracks(TRACKS_PATH)
    det_map = build_detection_attr_map()

    for t in tracks.get("tracks", []) or []:
        # accumulate colors/sizes seen for this track
        colors = []
        sizes = []
        for f in t.get("frames", []):
            did = f.get("detection_id")
            if did is None:
                continue
            attrs = det_map.get(str(did))
            if not attrs:
                continue
            if attrs.get("color"):
                f["color"] = attrs.get("color")
                colors.append(attrs.get("color"))
            if attrs.get("size"):
                f["size"] = attrs.get("size")
                sizes.append(attrs.get("size"))

        # add aggregated attributes on the track
        if colors:
            # majority color
            freq = defaultdict(int)
            for c in colors:
                freq[c] += 1
            maj_color = max(freq.items(), key=lambda kv: kv[1])[0]
            t["dominant_color"] = maj_color
        else:
            t.setdefault("dominant_color", None)

        if sizes:
            freq = defaultdict(int)
            for s in sizes:
                freq[s] += 1
            maj_size = max(freq.items(), key=lambda kv: kv[1])[0]
            t["typical_size"] = maj_size
        else:
            t.setdefault("typical_size", None)

    with open(OUT_PATH, "w") as f:
        json.dump(tracks, f, indent=2)

    print(f"✅ Wrote enriched tracks to {OUT_PATH}")


if __name__ == "__main__":
    enrich_tracks()
