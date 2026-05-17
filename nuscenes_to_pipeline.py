
# Convert a NuScenes scene (CAM_FRONT) to MP4 video.

#Usage:
   # python nuscenes_to_pipeline.py --scene 0
   # python nuscenes_to_pipeline.py --scene 0 --camera CAM_FRONT
  #  python nuscenes_to_pipeline.py --list

import argparse
import os
import cv2
import sys

from nuscenes.nuscenes import NuScenes


def list_scenes(nusc):
    
    print(f"\n{'#':<5} {'Name':<40} {'Description'}")
    print("-" * 80)
    for i, scene in enumerate(nusc.scene):
        print(f"{i:<5} {scene['name']:<40} {scene['description'][:50]}")


def scene_to_video(nusc, scene_idx: int, camera: str, out_path: str, fps: int = 2):
    #Extract NuScenes scene camera frames and save as MP4
    scene = nusc.scene[scene_idx]
    sample = nusc.get("sample", scene["first_sample_token"])
    frames = []

    # Collect all frames
    while True:
        cam_data = nusc.get("sample_data", sample["data"][camera])
        img_path = os.path.join(nusc.dataroot, cam_data["filename"])
        if os.path.exists(img_path):
            frames.append(img_path)
        if sample["next"] == "":
            break
        sample = nusc.get("sample", sample["next"])

    if not frames:
        sys.exit(f"No frames found for scene {scene_idx}, camera {camera}")

    # Write MP4
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    for img_path in frames:
        img = cv2.imread(img_path)
        if img is not None:
            writer.write(img)
    writer.release()
    
    print(f"✅ Saved {len(frames)} frames → {out_path}  ({w}x{h} @ {fps}fps)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/v1.0-mini")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", type=int, default=None, help="Scene index")
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--list", action="store_true", help="List scenes and exit")
    args = parser.parse_args()

    print(f"Loading NuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    if args.list:
        list_scenes(nusc)
        return

    # Pick scene interactively if not specified
    if args.scene is None:
        list_scenes(nusc)
        try:
            args.scene = int(input(f"\nEnter scene number (0–{len(nusc.scene)-1}): ").strip())
        except (ValueError, KeyboardInterrupt):
            sys.exit("Cancelled.")

    scene_name = nusc.scene[args.scene]["name"]
    out_dir = os.path.join("output_nuscenes", scene_name)
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, f"{scene_name}_{args.camera}.mp4")

    print(f"\nConverting scene {args.scene} ({scene_name}) — {args.camera}")
    scene_to_video(nusc, args.scene, args.camera, out_video, fps=args.fps)


if __name__ == "__main__":
    main()
