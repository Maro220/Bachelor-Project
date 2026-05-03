# 🎯 TWO-STAGE ANNOTATION PIPELINE - VISUAL FLOW

## Image Embedding (Part 1: How Images Appear)

```
┌─────────────────────────────────────────────────────────┐
│ Form 1 (& Form 2) - Image Display                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Upload image.jpg to Google Drive                       │
│         ↓                                               │
│  Get file_id from Drive                                 │
│         ↓                                               │
│  Make file publicly readable                            │
│         ↓                                               │
│  Create embed URL:                                      │
│  https://drive.google.com/uc?export=view&id=FILE_ID    │
│         ↓                                               │
│  Add to Google Form as imageItem (NOT text link)        │
│         ↓                                               │
│  ✓ Image appears embedded in form                       │
│  (full width, actual image visible)                     │
│                                                         │
│  ✗ Removed: "Direct Image Link" fallback text          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## AI Listening Flow (Part 2: Form 1 → Form 2 Generation)

```
┌──────────────────────────────────────────────────────────────────┐
│                    FORM 1 SUBMISSION                             │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                  ┌───────────────────────┐
                  │  Human fills Form 1:  │
                  │ Q1: Name              │
                  │ Q2: Scene Description │
                  └───────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ [PHASE 2] COLLECT FORM 1 RESPONSE                                │
│ ├─ Fetch responses from Google Forms                             │
│ ├─ Extract: name_text, description_text                          │
│ ├─ Validate: both fields have content                            │
│ └─ Return: (participant_name, description_text)                  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
         Description Text: "I see 3 red cars at an 
                           intersection. Bright daylight.
                           Wet road. 2 pedestrians."
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ [PHASE 3] AI LISTENS & EXTRACTS FIELDS                           │
│                                                                  │
│ Function: extract_schema_fields_from_text()                      │
│ LLM: Qwen2.5 VL (text-only)                                      │
│                                                                  │
│ Input:                                                           │
│  ├─ Human description (text only)                                │
│  └─ YOLO reference data (numbers, colors, sizes from AI)         │
│                                                                  │
│ Process (Qwen does NLP):                                         │
│  ├─ Parse: "3 red cars" → 3 vehicles, color red                  │
│  ├─ Parse: "intersection" → environment: intersection            │
│  ├─ Parse: "bright daylight" → lighting: daylight                │
│  ├─ Parse: "wet road" → road_condition: wet                      │
│  ├─ Parse: "2 pedestrians" → total_pedestrians: 2                │
│  └─ NOT MENTIONED: traffic_lights, sizes, positions              │
│                                                                  │
│ Output: JSON with extracted_fields = {                           │
│   "environment": {"value": "intersection", "confidence": 0.95},   │
│   "lighting": {"value": "daylight", "confidence": 0.92},         │
│   "road_condition": {"value": "wet", "confidence": 0.88},        │
│   "total_vehicles_detected": {"value": 3, "confidence": 0.90},   │
│   "total_pedestrians_detected": {"value": 2, "confidence": 0.89},│
│   "detected_objects": [                                          │
│     {"object_id": 1, "color": "red", ...}                        │
│   ]                                                              │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ [PHASE 3b] CROSS-REFERENCE WITH AI OUTPUT                        │
│ ├─ Compare extracted with YOLO detections                        │
│ ├─ Mark AI-only objects with source: "ai_only"                   │
│ └─ Don't override human answers with AI                          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ [PHASE 4] AI IDENTIFIES MISSING FIELDS                           │
│ Function: identify_missing_fields()                              │
│                                                                  │
│ Algorithm: Compare extracted fields to ALL required fields       │
│                                                                  │
│ Schema requires:                          Extracted:             │
│ ├─ environment ✓✓ (user mentioned)      ✓ intersection          │
│ ├─ lighting ✓✓ (user mentioned)         ✓ daylight              │
│ ├─ road_condition ✓✓ (user mentioned)   ✓ wet                   │
│ ├─ total_vehicles_detected ✓✓           ✓ 3                     │
│ ├─ total_pedestrians_detected ✓✓        ✓ 2                     │
│ ├─ total_objects_detected ✗ MISSING     ✗ not mentioned         │
│ ├─ total_traffic_lights_detected ✗ MISSING  ✗ not mentioned     │
│ └─ objects:                                                      │
│    └─ Object 1 (car):                                            │
│       ├─ object_type ✓✓ (car)                                    │
│       ├─ color ✓✓ (red)                                          │
│       ├─ size ✗ MISSING (not mentioned)                          │
│       └─ position ✗ MISSING (not mentioned)                      │
│                                                                  │
│ Also checks: AI detected objects user didn't mention?            │
│ ├─ Traffic light detected by AI but NOT mentioned by human       │
│ └─ → Add to Form 2: "Did you see any traffic lights?"            │
│                                                                  │
│ missing_fields = {                                               │
│   "scene_fields": ["total_objects_detected",                     │
│                   "total_traffic_lights_detected"],              │
│   "object_fields": {1: ["size", "position"]}                     │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                    ╔═════════════════╗
                    ║ CONDITIONAL     ║
                    ║ if missing_fields║
                    ╚═════════════════╝
                         ↙   ↖
                        /       \
                    YES /         \ NO
                      /             \
                     ↙               ↘
         ┌────────────────────┐  [No Form 2]
         │  FORM 2 NEEDED     │  All fields
         │  Create Form 2     │  extracted
         │  with required     │  ✓ Skip to
         │  questions for     │    Phase 7
         │  missing fields    │
         └────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────────┐
│ [PHASE 5] CREATE FORM 2 (AUTO-GENERATED)                         │
│ Function: create_targeted_followup_form()                        │
│                                                                  │
│ Form 2 Questions (ALL REQUIRED ★):                               │
│                                                                  │
│ SCENE-LEVEL:                                                     │
│ ├─ "How many total objects?" ★ (required)                        │
│ ├─ "How many traffic lights?" ★ (required)                       │
│ └─ [Image embedded in form]                                      │
│                                                                  │
│ OBJECT-LEVEL:                                                    │
│ ├─ "[Object 1] How large is this car?" ★ (required)              │
│ ├─ "[Object 1] Where is this car positioned?" ★ (required)       │
│ └─ "[New Object] Did you see any traffic lights?" ★ (required)   │
│                                                                  │
│ Save: form2_url_scene_X.txt (human clicks to fill Form 2)         │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                 ┌─────────────────────────┐
                 │ Human fills Form 2:     │
                 │ Q1: "3 objects total"   │
                 │ Q2: "2 traffic lights"  │
                 │ Q3: "medium" (car size) │
                 │ Q4: "center" (position) │
                 └─────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ [PHASE 6] COLLECT FORM 2 RESPONSE                                │
│ Function: collect_second_form_response()                         │
│                                                                  │
│ Fetch Form 2 responses:                                          │
│ form2_answers = {                                                │
│   "total_objects_detected": 3,                                   │
│   "total_traffic_lights_detected": 2,                            │
│   1: {"size": "medium", "position": "center"}                    │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ [PHASE 7] MERGE RESPONSES (100% HUMAN ONLY)                      │
│ Function: merge_form_responses()                                 │
│                                                                  │
│ Merge Priority:                                                  │
│ 1. Form 2 answers (latest human) ← HIGHEST                       │
│ 2. Form 1 extraction (original description)                      │
│ 3. NOT_ANSWERED (error flag, should never happen)                │
│                                                                  │
│ Example merges:                                                  │
│                                                                  │
│ total_traffic_lights_detected:                                   │
│  ├─ Form 2: "2" ← ✓ USE THIS                                     │
│  ├─ Form 1: (not mentioned)                                      │
│  └─ Final: 2                                                     │
│                                                                  │
│ size (Object 1):                                                 │
│  ├─ Form 2: "medium" ← ✓ USE THIS                                │
│  ├─ Form 1: (not mentioned)                                      │
│  └─ Final: "medium"                                              │
│                                                                  │
│ color (Object 1):                                                │
│  ├─ Form 2: (not asked, not needed)                              │
│  ├─ Form 1: "red" ← ✓ USE THIS                                   │
│  └─ Final: "red"                                                 │
│                                                                  │
│ ✓✓ NO AI FALLBACK (no AI values ever used)                       │
│                                                                  │
│ final_annotation = {                                             │
│   "participant_name": "Marina_001",                              │
│   "frame": "static",                                             │
│   "scene_summary": {                                             │
│     "environment": "intersection",    ← From Form 1              │
│     "lighting": "daylight",           ← From Form 1              │
│     "road_condition": "wet",          ← From Form 1              │
│     "total_vehicles_detected": 3,     ← From Form 1              │
│     "total_pedestrians_detected": 2,  ← From Form 1              │
│     "total_objects_detected": 3,      ← From Form 2 ✓            │
│     "total_traffic_lights_detected": 2, ← From Form 2 ✓          │
│     "detected_objects": [                                         │
│       {                                                          │
│         "object_id": 1,                                          │
│         "object_type": "car",        ← From Form 1               │
│         "color": "red",              ← From Form 1               │
│         "size": "medium",            ← From Form 2 ✓             │
│         "position": "center"         ← From Form 2 ✓             │
│       }                                                          │
│     ]                                                            │
│   }                                                              │
│ }                                                                │
│                                                                  │
│ Save: human_annotation_scene_static.json                         │
│ ✓ 100% HUMAN-ONLY OUTPUT (no AI values)                          │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Reference: How AI "Listens"

| Phase | What Happens | Key Code | Input | Output |
|-------|--------------|----------|-------|--------|
| 1 | Human fills Form 1 | `collect_first_form_response()` | Description text | `(name, description)` |
| 2 | AI extracts fields | `extract_schema_fields_from_text()` | Description + YOLO ref | `extracted_fields` JSON |
| 3 | AI identifies gaps | `identify_missing_fields()` | Extracted fields | `missing_fields` dict |
| 4 | Conditional trigger | `if missing_fields:` | Missing dict | Form 2 create (Y/N) |
| 5 | Form 2 created | `create_targeted_followup_form()` | Missing fields | `(form2_id, form2_url)` |
| 6 | Human fills Form 2 | `collect_second_form_response()` | Form 2 responses | `form2_answers` dict |
| 7 | Merge all data | `merge_form_responses()` | Form1 + Form2 + Name | Final JSON (100% human) |

## What "AI Listening" Means

✅ **AI LISTENS to Form 1:**
1. Read human's description text
2. Extract what they mentioned (LLM NLP)
3. Identify what they DIDN'T mention
4. Create Form 2 with ONLY those gaps

❌ **AI DOES NOT:**
- Fill in the form (humans do)
- Override human answers (Form 2 wins)
- Provide AI values in output (100% human only)
- Make assumptions (only asks explicit gaps)

The AI is a **smart question generator**, not a form filler!
