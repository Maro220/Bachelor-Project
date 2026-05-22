"""
serve_form.py
-------------
Serve the annotation forms for a scene that has already been processed
by scene_annotator.py. No YOLO, no Qwen, no heavy computation.

Usage
-----
    python serve_form.py scene-0757
    python serve_form.py scene-0916 --port 7861

Forms available after start
----------------------------
    http://localhost:7860/simple?scene_id=scene-0757   ← simple 3-object form
    http://localhost:7860/?scene_id=scene-0757          ← full annotation form

Annotations saved to
--------------------
    annotations/scene-0757/simple/<name>_simple.json   ← JSON backup
    Google Sheet → "Simple Annotations" tab             ← same sheet, Scene ID column
"""

import sys
import os
import time

# ── Validate scene arg ───────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python serve_form.py <scene-name> [--port PORT]")
    print("Example: python serve_form.py scene-0757")
    sys.exit(1)

scene_name = sys.argv[1]

# Parse optional --port
port = 7860
for i, arg in enumerate(sys.argv[2:], 2):
    if arg == "--port" and i + 1 < len(sys.argv):
        try:
            port = int(sys.argv[i + 1])
        except ValueError:
            pass

# ── Verify pipeline output exists ───────────────────────────────────────────
scene_root = f"output/{scene_name}"
top3_rep   = os.path.join(scene_root, "top3", "representative_frame.json")

if not os.path.exists(scene_root):
    print(f"\n❌ No pipeline output found for '{scene_name}'.")
    print(f"   Expected: {scene_root}/")
    print(f"\n   Run the pipeline first:")
    print(f"     python scene_annotator.py {scene_name}")
    sys.exit(1)

if not os.path.exists(top3_rep):
    print(f"\n⚠  Top-3 output not found for '{scene_name}' ({top3_rep}).")
    print(f"   The simple form will show an error until Phase 2 completes.")
    print(f"   Re-run: python scene_annotator.py {scene_name}")

# ── Import scene_annotator and set active scene globals ──────────────────────
print(f"\nLoading scene_annotator for '{scene_name}'…")
import scene_annotator as _sa

_sa._SCENE_OUT_DIR     = scene_root
_sa._ACTIVE_SCENE_NAME = scene_name

# ── Ensure spreadsheet exists (creates it on first run) ──────────────────────
_sa._ensure_spreadsheet()

# ── Start server ─────────────────────────────────────────────────────────────
print(f"\n  Scene       : {scene_name}")
print(f"  Output root : {scene_root}/")
print(f"  Annotations : annotations/{scene_name}/simple/")

_sa.start_server(port=port)

print(f"\n  Simple form  : http://localhost:{port}/simple?scene_id={scene_name}")
print(f"  Full form    : http://localhost:{port}/?scene_id={scene_name}")
print(f"\n  Press Ctrl+C to stop.\n")

try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    print("\nStopping.")
    _sa._cleanup_ngrok()
