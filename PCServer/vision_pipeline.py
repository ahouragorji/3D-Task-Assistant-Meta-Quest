"""
vision_pipeline.py

Two-call GPT architecture:

  CALL 1 — Planner (_fetch_step_plan)
    Receives: image + user request.
    Returns:  SemanticPlan — steps with instructions and per-object tags.
    Each object carries:
      - tag:       rich GDINO-friendly phrase for detection
      - user_view: how the object appears in the captured image (visual POV)
    The planner owns everything visual: scene description, object naming,
    spatial orientation. It has the image, so it can answer these accurately.

  CALL 2 — Annotator (_fetch_object_annotations)
    Receives: plan text only — NO image.
    Returns:  ObjectAnnotationPlan — simple_noun + action per object per step.
    The annotator owns everything semantic: what the user does to each object.
    No image needed — the action is inferable from the instruction verb alone.
    No tool/gesture decisions here — those are made by our Python code.

  TOOL SELECTION — _select_tool()
    Pure Python. Uses action + user_view → guidance_tool + gesture + placement_rule.
    Deterministic, no GPT, easy to tune.
"""

import json
import base64
import os
import threading
import requests
import pycocotools.mask as mask_util
from typing import List, Literal

from pydantic import BaseModel, Field
from openai import OpenAI
from ultralytics import YOLOE, SAM

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

_model_lock  = threading.Lock()
_yoloe_model = None
_sam_model   = None


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
# Shared enums
# ---------------------------------------------------------------------------

ACTION_ENUM = Literal[
    "press",    # tap, poke, push, flip — object stays fixed, point contact
    "pick_up",  # lift, grab, take, collect — object leaves its surface
    "wipe",     # scrub, clean, dust, sweep — friction across a flat surface
    "place",    # put down, set — object is placed somewhere
    "open",     # open a door, drawer, lid
    "close",    # close a door, drawer, lid
    "other",    # locate, look at, identify — no strong hand gesture
]

VIEW_ENUM = Literal[
    "front",        # camera faces the object's main surface straight on
    "front_left",   # camera sees the object slightly from its left side
    "front_right",  # camera sees the object slightly from its right side
    "front_top",    # camera looks slightly downward at the object's face
    "front_bottom", # camera looks slightly upward at the object's face
    "top",          # camera looks straight down onto the object
    "bottom",       # camera looks up at the object (ceiling-mounted)
    # "left",         # camera sees the object's left side profile
    # "right",        # camera sees the object's right side profile
]


# ---------------------------------------------------------------------------
# CALL 1 — Planner schema
# ---------------------------------------------------------------------------

class TaggedObject(BaseModel):
    tag: str = Field(
        description=(
            "Rich GDINO-friendly phrase describing this specific object as it appears in the scene. "
            "Include colour, material, spatial position, or part details. "
            "Examples: 'dark blue ceramic mug on the right side of the wooden desk', "
            "'silver handle of the top drawer', 'scattered white socks near the wardrobe'. "
            "No verbs, no actions, no abstract nouns."
        )
    )
    user_view: VIEW_ENUM = Field(
        description=(
            "The direction from which this object is seen IN THE CAPTURED IMAGE. "
            "Determine how the camera is oriented relative to this specific object:\n"
            "  front        — camera faces the object's main surface straight on, and only that surface is visible\n"
            "  front_left   — camera sees the object slightly from its left side, and both sides are visible.\n"
            "  front_right  — camera sees the object slightly from its right side, and both sides are visible\n"
            "  front_top    — camera looks slightly downward at the object's face, and both front and upper sides of the object are visible. Switch to front if upper side is occupied by another object\n"
            "  front_bottom — camera looks slightly upward at the object's face, and the bottom is visible (haging on a ceiling, for instance)\n"
            "  top          — camera looks straight down onto the object (flat on floor/table)\n"
            "  bottom       — camera looks up at the object (ceiling-mounted fixture) and only the bottom of the object is visible\n"
            # "  left         — camera only sees the object's left side profile\n"
            # "  right        — camera only sees the object's right side profile"
        )
    )


class SemanticStep(BaseModel):
    instruction: str = Field(
        description=(
            "One clear, concrete, single-action instruction for this step. "
            "Start with a verb. Be specific to what you see in the image. "
            "Under 15 words. Example: 'Pick up the white pillow from the floor near the wardrobe.'"
        )
    )
    objects: List[TaggedObject] = Field(
        description=(
            "One TaggedObject per physical object the user must interact with in this step. "
            "Empty list if no object needs to be highlighted."
        )
    )


class SemanticPlan(BaseModel):
    intent: str = Field(
        description="'task' for step-by-step procedures, 'query' for locate/identify requests."
    )
    steps: List[SemanticStep] = Field(
        description=(
            "Chronological steps. Each step is one atomic action. "
            "For tasks: include every step the user needs — don't skip obvious ones. "
            "For queries: exactly ONE step whose instruction directly answers the question."
        )
    )


PLANNER_SYSTEM_PROMPT = """You are an expert spatial assistant helping a user in their real environment.
You receive a photo of their space and their spoken request.
Your ONLY job is to write an excellent step-by-step action plan.

DO NOT think about AR tools or overlays. Just plan.

PLANNING RULES (STRICTLY ENFORCED):
  • Be concise: Most tasks need 2–5 steps. Only generate steps strictly necessary to complete the goal.
  • No micro-steps: Combine fluid motions. "Pick up the mug" not "Reach for the mug" then "Grab it".
  • No duplicates: Never repeat the same action or goal across steps.
  • Each step = one atomic action. Never combine two unrelated actions.
  • Be specific to what you see. Name objects by appearance:
    "the blue mug on the left side of the desk", not just "the mug".
  • Use action verbs: Pick up, Place, Press, Wipe, Fold, Open, Close, etc.
  • For cleaning tasks: SEPARATE picking up items FROM wiping surfaces.
    "Pick up the book" and "Wipe the desk" are two different steps.
  • For QUERIES: produce exactly ONE step that directly answers the question
    from visual evidence in the image.

OBJECT TAGGING RULES (Optimized for GroundingDINO Detection)

    PRIMARY GOAL:
    Describe the object exactly as it appears in the image so it can be uniquely identified from visual evidence alone.

    GOOD OBJECT DESCRIPTIONS:
    ✓ Start with the object's category:
        "mug", "drawer handle", "power button", "chair"

    ✓ Add distinctive visual attributes:
        color:
            "dark blue mug"
            "red cable"

        material:
            "wooden drawer"
            "metal handle"

        size/state:
            "small white bottle"
            "open laptop"
            "folded towel"

    ✓ Use ONE strong spatial anchor when needed:
        "the blue mug beside the silver laptop"
        "the leftmost black shoe"
        "the top drawer handle"

    ✓ Object parts are encouraged when they are the interaction target:
        "the silver handle of the top drawer"
        "the power button on the monitor"
        "the trigger of the spray bottle"

    ✓ Groups are allowed when the user interacts with multiple items:
        "the scattered shoes"
        "the pile of books"

    SPATIAL RULES:
    ✓ Prefer simple positional words:
        leftmost
        rightmost
        top
        bottom
        center
        front
        back

    ✓ Prefer a single nearby reference object:
        "the mug beside the laptop"

    ✗ Avoid multi-hop relationships:
        BAD:
            "the mug beside the laptop near the cables next to the monitor"

        GOOD:
            "the mug beside the laptop"

    VISUAL PRIORITY ORDER:
    1. Object category
    2. Color
    3. Material
    4. State (open, closed, folded, broken)
    5. Position
    6. Single spatial anchor

    EXAMPLES:

    GOOD:
    ✓ "dark blue ceramic mug"
    ✓ "silver microwave handle"
    ✓ "open black laptop"
    ✓ "leftmost white sneaker"
    ✓ "red charging cable beside the laptop"
    ✓ "top drawer handle"
    ✓ "folded gray towel"
    ✓ "green spray bottle on the counter"

    BAD:
    ✗ "pick up the mug"
    ✗ "the thing near the desk"
    ✗ "the clutter"
    ✗ "the object I want"
    ✗ "the mug beside the laptop near the monitor next to the keyboard"
    ✗ "please grab the blue cup"
    ✗ "the area to clean"

    NEVER:
    ✗ Include actions or verbs
    ✗ Include user instructions
    ✗ Use abstract concepts
    ✗ Use pronouns
    ✗ Write full sentences
    ✗ Describe more than one target object in a single tag

OUTPUT FORMAT:
Return a short noun phrase optimized for visual grounding, not a sentence.

USER VIEW RULES (look at the image for each object):
  Determine how the camera is oriented relative to each object.
  This is the visual POV of the photo, not the intended approach direction.
  • Item lying flat on floor/table → "top"
  • Button on a panel facing the camera → "front"
  • Wardrobe photographed from slightly left → "front_left"
  • Ceiling fixture → "bottom"
"""


# ---------------------------------------------------------------------------
# CALL 2 — Annotator schema (no image, text-only)
# ---------------------------------------------------------------------------

class ObjectAnnotation(BaseModel):
    original_tag: str = Field(
        description=(
            "Copy the object tag exactly as written in the plan. "
            "Do not rephrase or shorten it."
        )
    )
    simple_noun: str = Field(
        description=(
            "The absolute bare singular noun for this object — one word, no adjectives. "
            "Used internally for logging and future rule extensions. "
            "Examples: 'mug', 'button', 'shelf', 'drawer', 'cable', 'pillow', 'sock'."
        )
    )
    action: ACTION_ENUM = Field(
        description=(
            "The physical action the user performs on this object in this step.\n"
            "  press    — tap, poke, push, flip; object stays fixed in place\n"
            "  pick_up  — lift, grab, take, collect; object leaves its surface\n"
            "  wipe     — scrub, clean, dust, sweep; friction across the surface\n"
            "  place    — put down or set the object somewhere\n"
            "  open     — open a door, drawer, lid, or container\n"
            "  close    — close a door, drawer, lid, or container\n"
            "  other    — locate, look at, identify; no strong hand gesture"
        )
    )


class AnnotatedStep(BaseModel):
    step_number: int = Field(description="1-based index matching the plan step number.")
    objects: List[ObjectAnnotation] = Field(
        description=(
            "One ObjectAnnotation per object listed in this step. "
            "Empty list if the step has no objects."
        )
    )


class ObjectAnnotationPlan(BaseModel):
    annotated_steps: List[AnnotatedStep] = Field(
        description="One AnnotatedStep per step in the plan, in order."
    )


ANNOTATOR_SYSTEM_PROMPT = """You are an object annotation engine for an AR system.
You receive a step-by-step plan. Each step lists objects the user interacts with.

Your ONLY job for each object:
  1. Copy its tag exactly (original_tag)
  2. Extract the bare noun (simple_noun) — one word, no adjectives
  3. Classify the action the user performs on it in this step

You make NO decisions about AR tools, prefabs, arrows, or gestures.
You do NOT rewrite instructions. You do NOT add or remove steps or objects.

ACTION RULES — pick the single best match:
  press    → user pokes, taps, pushes, or flips; object stays in place
  pick_up  → user lifts, grabs, takes, or collects; object leaves the surface
  wipe     → user rubs, scrubs, dusts, or sweeps; friction across the surface
  place    → user puts it down or sets it somewhere
  open     → user opens it (door, drawer, lid)
  close    → user closes it (door, drawer, lid)
  other    → user locates, looks at, or identifies; no strong hand gesture

SIMPLE NOUN RULES:
  ✓ One bare singular noun: "mug", "button", "shelf", "drawer", "cable"
  ✗ No adjectives, colours, or locations — never more than one word
"""


# ---------------------------------------------------------------------------
# TOOL SELECTION — pure Python, no GPT
# ---------------------------------------------------------------------------

_GHOST_HAND_ACTIONS = {"press", "pick_up", "wipe", "open", "close"}

_ACTION_TO_GESTURE = {
    "press":   "poke",
    "open":    "grab",
    "close":   "poke",
    "pick_up": "grab",
    "wipe":    "clean",
    "clean": "clean"
}

def normalize_action(action: str) -> str:
    action = action.lower().replace(" ", "_")
    return ACTION_ALIASES.get(action, "other")

ACTION_ALIASES = {
    # ───── PRESS / POKE ─────
    "press": "press",
    "push": "press",
    "tap": "press",
    "poke": "press",
    "click": "press",
    "flip": "press",
    "toggle": "press",
    "switch": "press",
    "activate": "press",
    "turn_on": "press",
    "turn_off": "press",

    # ───── PICK UP / GRAB ─────
    "pick_up": "pick_up",
    "grab": "pick_up",
    "take": "pick_up",
    "lift": "pick_up",
    "collect": "pick_up",
    "retrieve": "pick_up",
    "carry": "pick_up",
    "hold": "pick_up",
    "remove": "pick_up",

    # ───── PLACE ─────
    "place": "place",
    "put": "place",
    "set": "place",
    "drop": "place",
    "return": "place",
    "store": "place",

    # ───── CLEAN / WIPE ─────
    "wipe": "wipe",
    "clean": "wipe",
    "scrub": "wipe",
    "dust": "wipe",
    "polish": "wipe",
    "sweep": "wipe",
    "sanitize": "wipe",

    # ───── OPEN ─────
    "open": "open",
    "uncover": "open",
    "unlock": "open",

    # ───── CLOSE ─────
    "close": "close",
    "shut": "close",
    "lock": "close",

    # ───── POINT / LOCATE ─────
    "find": "other",
    "locate": "other",
    "identify": "other",
    "look": "other",
    "inspect": "other",
    "check": "other",
    "observe": "other",

    # ───── MOVE / REPOSITION ─────
    "move": "pick_up",
    "reposition": "pick_up",
    "relocate": "pick_up",
    "shift": "pick_up",

    # ───── ORGANIZING ─────
    "organize": "pick_up",
    "sort": "pick_up",
    "arrange": "pick_up",
    "stack": "pick_up",
}

_VIEW_TO_PLACEMENT = {
  "front":        "front",
    "front_left":   "front",
    "front_right":  "front",
    "front_top":    "up",
    "front_bottom": "front",
    "top":          "up",
    "bottom":       "front",
    "left":         "front",
    "right":        "front",
}

# Wipe special case: even if viewed from "top" (desk surface), the cleaning
# hand should appear on the front-facing side, not floating directly above.
# _WIPE_VIEW_TO_PLACEMENT = {
#     "front":        "front",
#     "front_left":   "front",
#     "front_right":  "front",
#     "front_top":    "up",
#     "front_bottom": "front",
#     "top":          "up",
#     "bottom":       "front",
#     "left":         "front",
#     "right":        "front",
# }


def _select_tool(action: str, user_view: str) -> dict:
    """
    Deterministically maps (action, user_view) → guidance_tool + settings.
    Returns {"guidance_tool": str, "tool_settings": dict}.
    """

    if _VIEW_TO_PLACEMENT[user_view] == "front":
        return {
            "guidance_tool": "indicator_arrow",
            "tool_settings": {"placement_rule":"front"},
        }

    if action == "pick_up":
        return {
            "guidance_tool": "indicator_arrow",
            "tool_settings": {"placement_rule":"up"},
        }
# so the top view is visible, sicne if it wasn't we would have had front

    elif action in _GHOST_HAND_ACTIONS:
        action_alias   = normalize_action(action)
        gesture = _ACTION_TO_GESTURE[action_alias]

        if gesture == "poke":
                return {
            "guidance_tool": "ghost_hand",
            "tool_settings": {"gesture": gesture, "placement_rule": "front"},
        }

        if gesture == "wipe":
                return {
            "guidance_tool": "ghost_hand",
            "tool_settings": {"gesture": gesture, "placement_rule": "up"},
        }

        placement = (
            _VIEW_TO_PLACEMENT.get(user_view, "up")
        )

        return {
            "guidance_tool": "ghost_hand",
            "tool_settings": {"gesture": gesture, "placement_rule": placement},
        }
    
    else:
        # "place" and "other" → indicator_arrow
        placement = _VIEW_TO_PLACEMENT.get(user_view, "up")
        return {
            "guidance_tool": "indicator_arrow",
            "tool_settings": {"placement_rule": placement},
        }


# ---------------------------------------------------------------------------
# GPT helpers
# ---------------------------------------------------------------------------

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _fetch_step_plan(user_prompt: str, image_path: str) -> SemanticPlan:
    """Call 1: Planner sees the image. Returns SemanticPlan with tagged objects."""
    base64_image = _encode_image(image_path)

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": f"User request: {user_prompt}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ],
        response_format=SemanticPlan,
        temperature=0.2,
    )

    plan = response.choices[0].message.parsed
    print(f"[vision_pipeline] Planner: intent='{plan.intent}', steps={len(plan.steps)}")
    for i, s in enumerate(plan.steps, 1):
        tags = [(o.tag, o.user_view) for o in s.objects]
        print(f"  Step {i}: {s.instruction} | objects: {tags}")
    return plan


def _fetch_object_annotations(plan: SemanticPlan) -> ObjectAnnotationPlan:
    """
    Call 2: Annotator receives plan text only (no image).
    Adds simple_noun + action per object. No tool decisions.
    """
    plan_lines = [f"intent: {plan.intent}", ""]
    for i, step in enumerate(plan.steps, 1):
        plan_lines.append(f"Step {i}: {step.instruction}")
        if step.objects:
            for obj in step.objects:
                plan_lines.append(f"  - {obj.tag}")
        else:
            plan_lines.append("  (no objects)")
        plan_lines.append("")
    plan_text = "\n".join(plan_lines)

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ANNOTATOR_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Plan to annotate:\n\n{plan_text}"},
        ],
        response_format=ObjectAnnotationPlan,
        temperature=0.0,
    )

    result = response.choices[0].message.parsed
    print(f"[vision_pipeline] Annotator: {len(result.annotated_steps)} steps annotated.")
    for s in result.annotated_steps:
        for obj in s.objects:
            print(f"  Step {s.step_number} | noun='{obj.simple_noun}' action='{obj.action}'")
    return result


# ---------------------------------------------------------------------------
# Tag matching (GDINO class_name → our tag)
# ---------------------------------------------------------------------------

def _match_tag(phrase: str, unique_tags: list[str]) -> str | None:
    """Word-overlap matching with tiebreak by shorter tag (more specific)."""
    phrase_words = set(phrase.lower().split())
    best_tag, best_score = None, 0
    for tag in unique_tags:
        tag_words = set(tag.lower().split())
        overlap   = len(tag_words & phrase_words)
        if overlap > best_score or (
            overlap == best_score and best_tag and len(tag) < len(best_tag)
        ):
            best_score = overlap
            best_tag   = tag
    return best_tag if best_score > 0 else None

def _match_dino_class(dino_class: str, unique_tags: list[str]) -> str | None:
    """
    Matches DINO's output back to our rich tag. 
    Prevents "anchor bleed" by ignoring detections that only appear 
    AFTER spatial prepositions in the rich tag.
    """
    dino_class = dino_class.lower().strip()
    
    # Common spatial prepositions used to anchor objects
    prepositions = [" near ", " on ", " next to ", " beside ", " under ", " in ", " by ", " above ", " with "]
    
    best_tag = None
    best_score = -1

    for rich_tag in unique_tags:
        rich_tag_lower = rich_tag.lower()
        
        # 1. Isolate the "Target" half of the sentence
        target_phrase = rich_tag_lower
        for prep in prepositions:
            if prep in target_phrase:
                # Split at the preposition and keep only the first half
                target_phrase = target_phrase.split(prep)[0]
                
        # 2. STRICT FILTER: Is DINO's word in the target half?
        # If DINO found "laptop", but "laptop" is only in the second half of the phrase,
        # this will fail, successfully ignoring the anchor!
        if dino_class not in target_phrase:
            continue
            
        # 3. If it is the target, calculate overlap to find the best match
        dino_words = set(dino_class.split())
        rich_words = set(rich_tag_lower.split())
        overlap = len(rich_words & dino_words)
        
        if overlap > best_score:
            best_score = overlap
            best_tag = rich_tag
            
    return best_tag
# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def fetch_step_segmentations(
    user_prompt: str,
    image_path: str,
    detector: Literal["yoloe", "gdino_server"] = "gdino_server",
    server_url: str = "http://localhost:8000/predict",
) -> list:
    """
    Runs Planner → Annotator → Tool Selection → Detection → Results.

    Return shape per step:
        {
          "step_number": int,
          "instruction": str,
          "detections": [
            {
              "guidance_tool": str,
              "tool_settings": dict,       # {"gesture": ..., "placement_rule": ...}
              "label":         str,        # tag used for detection; "" for floating
              "bbox":          list,       # [x1, y1, x2, y2]; [] for floating
              "mask":          np.ndarray | None,
            },
            ...
          ]
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Call 1: Plan (image → steps + tagged objects with user_view) ──────────
    plan = _fetch_step_plan(user_prompt, image_path)
    if not plan.steps:
        return []

    # ── Call 2: Annotate (text only → simple_noun + action per object) ────────
    annotation_plan = _fetch_object_annotations(plan)

    # Build lookup: step_number → list[ObjectAnnotation]
    annotation_map: dict[int, list[ObjectAnnotation]] = {
        s.step_number: s.objects for s in annotation_plan.annotated_steps
    }

    # Build lookup: tag string → user_view (from Call 1, which saw the image)
    # We need user_view at tool-selection time, keyed by tag.
    tag_to_view: dict[str, str] = {}
    for step in plan.steps:
        for obj in step.objects:
            if obj.tag not in tag_to_view:
                tag_to_view[obj.tag] = obj.user_view

                
    # Build lookup: tag string → simple_noun (from Call 2)
    tag_to_noun: dict[str, str] = {}
    for s in annotation_plan.annotated_steps:
        for obj in s.objects:
            if obj.original_tag not in tag_to_noun:
                tag_to_noun[obj.original_tag] = obj.simple_noun

    # ── Collect unique tags for detection ─────────────────────────────────────
    unique_tags: list[str] = []
    for step in plan.steps:
        for obj in step.objects:
            if obj.tag and obj.tag not in unique_tags:
                unique_tags.append(obj.tag)

    # ── Detection ─────────────────────────────────────────────────────────────
    detected_objects_map: dict[str, list] = {t: [] for t in unique_tags}

    if unique_tags:
        if detector == "gdino_server":
            text_prompt = " . ".join(unique_tags)
            try:
                with open(image_path, "rb") as f:
                    resp = requests.post(
                        server_url,
                        files={"file": f},
                        data={"text_prompt": text_prompt, "multimask_output": False},
                    )
                if resp.status_code == 200:
                    for ann in resp.json().get("annotations", []):
                        matched = _match_dino_class(ann["class_name"].lower(), unique_tags)
                        if not matched:
                            print(f"[vision_pipeline] GDINO '{ann['class_name']}' matched no tag — skipped.")
                            continue
                        rle = ann["segmentation_rle"]
                        rle["counts"] = rle["counts"].encode("utf-8")
                        mask = mask_util.decode(rle).astype(bool)
                        detected_objects_map[matched].append({"bbox": ann["bbox"], "mask": mask})
                else:
                    print(f"[vision_pipeline] GDINO server error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as e:
                print(f"[vision_pipeline] GDINO server unreachable: {e}")

        elif detector == "yoloe":
            yolo, sam = _get_models()
            unique_nouns = list(set(tag_to_noun.get(t, t) for t in unique_tags))
            try:
                yolo.set_classes(unique_nouns, yolo.get_text_pe(unique_nouns))
            except AttributeError:
                yolo.set_classes(unique_nouns)

            results    = yolo.predict(image_path, verbose=False)
            all_bboxes = []
            all_refs   = []
            for box in results[0].boxes:
                tag  = unique_nouns [int(box.cls[0])]
                bbox = box.xyxy[0].tolist()
                all_bboxes.append(bbox)
                all_refs.append((tag, bbox))

            if all_bboxes:
                sam_results = sam(image_path, bboxes=all_bboxes, verbose=False)
                masks = (
                    sam_results[0].masks.data.cpu().numpy()
                    if sam_results[0].masks is not None
                    else [None] * len(all_bboxes)
                )
                for (tag, bbox), mask in zip(all_refs, masks):
                    detected_objects_map[tag].append({
                        "bbox": bbox,
                        "mask": (mask > 0.5) if mask is not None else None,
                    })

    # ── Build results per step ────────────────────────────────────────────────
    results_per_step = []

    for i, step in enumerate(plan.steps, 1):
        ann_objects = annotation_map.get(i, [])

        if not ann_objects:
            results_per_step.append({
                "step_number": i,
                "instruction": step.instruction,
                "detections":  [],
            })
            continue

        step_detections = []

        for ann_obj in ann_objects:
            tag       = ann_obj.original_tag
            user_view = tag_to_view.get(tag, "front")  # from Call 1

            # Tool selection: pure Python
            decision      = _select_tool(ann_obj.action, user_view)
            guidance_tool = decision["guidance_tool"]
            tool_settings = decision["tool_settings"]

            print(f"[vision_pipeline] Step {i} | '{ann_obj.simple_noun}' "
                  f"action={ann_obj.action} view={user_view} "
                  f"→ {guidance_tool} {tool_settings}")

            if not tag:
                # Floating overlay — no detection needed
                step_detections.append({
                    "guidance_tool": guidance_tool,
                    "tool_settings": tool_settings,
                    "label": "", "bbox": [], "mask": None,
                })
                continue

            detected_items = detected_objects_map.get(tag, [])
            if not detected_items:
                print(f"[vision_pipeline] Step {i}: '{tag}' not found by {detector}.")
                continue

            for item in detected_items:
                step_detections.append({
                    "guidance_tool": guidance_tool,
                    "tool_settings": tool_settings,
                    "label":         tag,
                    "bbox":          item["bbox"],
                    "mask":          item["mask"],
                })

        results_per_step.append({
            "step_number": i,
            "instruction": step.instruction,
            "detections":  step_detections,
        })

    return results_per_step