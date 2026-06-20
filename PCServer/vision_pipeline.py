"""
vision_pipeline.py

Changes from previous version:
- TaskStep gains a `manipulation_orientations` field (parallel to
  `manipulation_tags`). For each tag, GPT picks one of six orientations
  ("up", "down", "front", "back", "left", "right") that describes the best
  side of the object to place an arrow indicator, chosen to avoid clipping
  into the object itself or the surrounding environment.
- Orientation is passed through to fetch_step_segmentations results so
  app.py can include it in the JSON sent to the Quest.
"""

import base64
import os
import threading
from pydantic import BaseModel, Field
from openai import OpenAI
from ultralytics import YOLOE, SAM

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

_model_lock = threading.Lock()
_yoloe_model = None
_sam_model = None


def _get_models():
    global _yoloe_model, _sam_model
    with _model_lock:
        if _yoloe_model is None:
            print("[vision_pipeline] Loading YOLOE (first request only)...")
            _yoloe_model = YOLOE("yoloe-v8l-seg.pt")
        if _sam_model is None:
            print("[vision_pipeline] Loading SAM 2 (first request only)...")
            _sam_model = SAM("sam2.1_s.pt")
    return _yoloe_model, _sam_model


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class TaskStep(BaseModel):
    instruction: str = Field(
        description="A clear, single-action instruction or answer customised to the layout of the scene."
    )
    manipulation_tags: list[str] = Field(
        description="Purely visual, singular noun phrases for the exact objects to highlight for this step. "
                    "Empty list if no specific object needs highlighting."
    )
    manipulation_orientations: list[str] = Field(
        description="For each entry in manipulation_tags (same order, same length), the best direction "
                    "to place an arrow indicator next to that object.\n\n"
                    "*** STRICT ORIENTATION HIERARCHY ***\n"
                    "You MUST use 'up' or 'front' in 95% of cases. ONLY use the others if physically impossible.\n\n"
                    "PRIMARY DEFAULTS:\n"
                    "- 'up'    : DEFAULT for almost all objects (items on tables/floors, held items). Arrow floats above.\n"
                    "- 'front' : DEFAULT for large, flat-faced objects (TVs, wardrobes, closed doors, appliances).\n\n"
                    "LAST RESORT (Only if 'up'/'front' are completely blocked by walls/ceilings):\n"
                    "- 'down'  : ONLY if object is mounted completely flat to the ceiling.\n"
                    "- 'back'  : ONLY if object is exclusively interacted with from the rear.\n"
                    "- 'left'  : ONLY if object is wedged tight against a right wall blocking top/front access.\n"
                    "- 'right' : ONLY if object is wedged tight against a left wall blocking top/front access.\n"
    )


class PipelineData(BaseModel):
    intent: str = Field(
        description="'task' if the user wants a procedure broken into steps, "
                    "'query' if the user is asking where something is or wants to identify/find an object."
    )
    steps: list[TaskStep] = Field(
        description="The chronological list of steps (for tasks) or a single-element list with the answer "
                    "and the object to highlight (for queries)."
    )


# ---------------------------------------------------------------------------
# GPT planning
# ---------------------------------------------------------------------------

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _fetch_step_plan(user_prompt: str, image_path: str) -> PipelineData:
    base64_image = _encode_image(image_path)
    system_prompt = """
    You are a spatial computing assistant bridging multimodal vision to a zero-shot object detector
    and an augmented-reality arrow placement system.
    Look closely at the provided image of the user's current environment.

    STEP 1 — CLASSIFY INTENT:
    Decide whether the user's request is a:
    - "task": they want a procedure (e.g. "clean your room", "make a sandwich", "pack your bag")
    - "query": they want to find, locate or identify something (e.g. "where is my passport?",
               "find my keys", "which one is the HDMI cable?")

    Set the `intent` field accordingly.

    STEP 2 — PLAN:

    For TASKS: break the task into chronological steps. For each step, write a clear
    single-action instruction and list the physical objects the user needs to interact
    with for that specific step.

    For QUERIES: produce exactly ONE step. The `instruction` should directly answer the
    question based on what you can see ("Your passport is on the desk near the lamp.").
    The `manipulation_tags` should contain the object(s) to highlight so the user can
    see exactly what you are referring to.

    STEP 3 — ORIENTATION (STRICT RULES):
    For every entry in `manipulation_tags`, provide a matching entry in
    `manipulation_orientations` (same order, same length).

    You MUST heavily prioritize "up" and "front". Do NOT use "left", "right", "down", 
    or "back" unless "up" or "front" would physically clip inside a wall or ceiling.
    
    - "up"    (PRIMARY) : Use for almost everything. Small items, items on surfaces, floors, etc.
    - "front" (PRIMARY) : Use for large furniture, appliances, and screens.
    - "down"  (RARE)    : Use ONLY for ceiling-mounted fixtures.
    - "left"/"right" (RARE) : Use ONLY if the top and front are physically obstructed by the environment.

    CRITICAL TAGGING RULES FOR YOLOE (apply to both task and query):
    1. Visual Nouns Only: tags must be concrete, physical, and visible in this image.
    2. Strategic Adjectives: include colours, materials, or distinct shapes ("blue jacket", "wooden chair").
    3. No Abstractions or Prepositions: never include location text or verbs ("pillow", NOT "pillow on bed").
    4. Singular Form Only: always use singular nouns ("shoe" not "shoes").
    5. Contextual Accuracy: only tag items actually visible in the image.
    6. If nothing relevant is visible, return empty lists for both tags and orientations.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Request: {user_prompt}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ],
        response_format=PipelineData,
    )

    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

VALID_ORIENTATIONS = {"up", "down", "front", "back", "left", "right"}


def _safe_orientation(orientations: list[str], index: int) -> str:
    """
    Returns the orientation at `index`, falling back to "up" if the list is
    short (GPT occasionally returns fewer orientations than tags) or contains
    an unrecognised value.
    """
    if index < len(orientations):
        value = orientations[index].strip().lower()
        if value in VALID_ORIENTATIONS:
            return value
        print(f"[vision_pipeline] Unrecognised orientation '{value}', falling back to 'up'.")
    return "up"


def fetch_step_segmentations(user_prompt: str, image_path: str):
    """
    Runs the full GPT -> YOLOE -> SAM pipeline and returns, for each step, the
    list of detected objects with their SAM masks and GPT-chosen orientations.

    Each detection dict now contains an "orientation" key (str) in addition to
    "label", "bbox", and "mask".

    Returns:
        [
          {
            "step_number": 1,
            "instruction": "Pick up the blue jacket.",
            "detections": [
              {
                "label": "blue jacket",
                "orientation": "up",
                "bbox": [x1, y1, x2, y2],
                "mask": <HxW bool ndarray>
              },
              ...
            ]
          },
          ...
        ]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    plan = _fetch_step_plan(user_prompt, image_path)
    print(f"[vision_pipeline] Intent classified as: '{plan.intent}'")

    if not plan.steps:
        return []

    yolo, sam = _get_models()

    # Collect unique tags across all steps
    unique_classes = []
    for step in plan.steps:
        for tag in step.manipulation_tags:
            if tag not in unique_classes:
                unique_classes.append(tag)

    # Run YOLOE only if there are tags to detect
    detected_bboxes_map = {}
    if unique_classes:
        try:
            yolo.set_classes(unique_classes, yolo.get_text_pe(unique_classes))
        except AttributeError:
            yolo.set_classes(unique_classes)

        detection_results = yolo.predict(image_path, verbose=False)

        detected_bboxes_map = {name: [] for name in unique_classes}
        for box in detection_results[0].boxes:
            class_id   = int(box.cls[0])
            class_name = unique_classes[class_id]
            coordinates = box.xyxy[0].tolist()
            detected_bboxes_map[class_name].append(coordinates)

    results_per_step = []

    for i, step in enumerate(plan.steps, 1):
        if not step.manipulation_tags:
            print(f"[vision_pipeline] Step {i} has no manipulation tags — instruction only.")
            results_per_step.append({
                "step_number": i,
                "instruction": step.instruction,
                "detections":  [],
            })
            continue

        step_bboxes       = []
        step_labels       = []
        step_orientations = []

        for tag_idx, tag in enumerate(step.manipulation_tags):
            orientation = _safe_orientation(step.manipulation_orientations, tag_idx)
            for bbox in detected_bboxes_map.get(tag, []):
                step_bboxes.append(bbox)
                step_labels.append(tag)
                step_orientations.append(orientation)

        step_detections = []

        if step_bboxes:
            sam_results = sam(image_path, bboxes=step_bboxes, verbose=False)

            if sam_results[0].masks is not None:
                masks = sam_results[0].masks.data.cpu().numpy()
                for mask, bbox, label, orientation in zip(
                    masks, step_bboxes, step_labels, step_orientations
                ):
                    step_detections.append({
                        "label":       label,
                        "orientation": orientation,
                        "bbox":        bbox,
                        "mask":        (mask > 0.5),
                    })
        else:
            print(f"[vision_pipeline] Step {i}: tags {step.manipulation_tags!r} not detected by YOLOE.")

        results_per_step.append({
            "step_number": i,
            "instruction": step.instruction,
            "detections":  step_detections,
        })

    return results_per_step