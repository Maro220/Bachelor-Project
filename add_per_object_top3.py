

import json
import os
import sys


SCENE = "scene-0103"


def cumulative_path(scene: str) -> str:
    return os.path.join("output", scene, "top3", "summaries",
                        "output_cumulative_top3.json")


def _first_track_id(obj: dict) -> int:
    tids = obj.get("track_ids") or []
    return min(tids) if tids else 10**9


def _motion_phrase(obj: dict, tmov_by_ident: dict) -> str:
    """Prefer temporal_movements (direction + speed); fall back to action."""
    ident = obj.get("identity")
    tm    = tmov_by_ident.get(ident)
    action = (obj.get("action") or "").strip()

    if tm:
        direction = (tm.get("direction") or "").strip()
        speed     = (tm.get("speed") or "").strip()
        act       = (tm.get("action") or action or "").strip()
        # e.g. action="turning right", direction="moving right", speed="slow"
        bits = []
        if act and act not in ("unknown", "stationary", "static"):
            bits.append(act)
        extra = []
        if speed and speed not in ("unknown",):
            extra.append(speed)
        if direction and direction not in ("unknown", "stationary"):
            extra.append(direction)
        phrase = " ".join(bits) if bits else "moving"
        if extra:
            phrase += f" ({', '.join(extra)})"
        return phrase

    # No temporal movement entry → static / parked object
    if action and action not in ("unknown",):
        return action
    return "stationary"


def _sentence(slot_label: str, obj: dict, tmov_by_ident: dict) -> str:
    color = (obj.get("color") or "").strip()
    otype = (obj.get("object_type") or "object").strip()

    descriptors = [d for d in (color,)
                   if d and d not in ("unknown", "")]
    desc_str = (" ".join(descriptors) + " ") if descriptors else ""

    motion = _motion_phrase(obj, tmov_by_ident)

    article = "an" if (desc_str or otype)[:1].lower() in "aeiou" else "a"
    return f"{slot_label}: {article} {desc_str}{otype} {motion}.".replace("  ", " ")


def build_per_object(cumulative: dict) -> dict:
    objs = list(cumulative.get("objects_seen", []))
    tmov_by_ident = {
        t.get("identity"): t
        for t in cumulative.get("temporal_movements", [])
    }

    # Order by track_id ascending → Object 1, 2, 3
    objs.sort(key=_first_track_id)

    per_object = {}
    for i, obj in enumerate(objs[:3]):
        label = f"Object {i+1}"
        per_object[label] = _sentence(label, obj, tmov_by_ident)
    return per_object


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else SCENE
    path  = cumulative_path(scene)
    if not os.path.exists(path):
        sys.exit(f"ERROR: cumulative not found: {path}")

    with open(path) as f:
        cumulative = json.load(f)

    per_object = build_per_object(cumulative)
    if not per_object:
        sys.exit("ERROR: no objects_seen to build from.")

    cumulative["per_object"] = per_object

    with open(path, "w") as f:
        json.dump(cumulative, f, indent=2, ensure_ascii=False)

    print(f"✓ Added per_object to {path}\n")
    for label in sorted(per_object):
        print(f"  {per_object[label]}")


if __name__ == "__main__":
    main()
