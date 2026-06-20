"""
app.py — unchanged except that each placement now carries an "orientation"
field (e.g. "up", "front") forwarded from the vision pipeline so the Quest
knows which arrow prefab to spawn and in which direction to offset it.
"""

import json
import traceback
import cv2
import os
import numpy as np
from flask import Flask, request, jsonify

from vision_pipeline import fetch_step_segmentations
from mask_reprojection import resolve_mask_world_point

app = Flask(__name__)

DEBUG_MASKS = os.environ.get("DEBUG_MASKS", "0") == "1"


def _load_metadata(meta_path: str) -> dict:
    with open(meta_path, "r") as f:
        return json.load(f)


def save_debug_masks(rgb_path, step_results, capture_id):
    img = cv2.imread(rgb_path)
    if img is None:
        print(f"[debug] Could not load RGB image at {rgb_path} for debugging.")
        return

    overlay = img.copy()
    for step in step_results:
        for det in step["detections"]:
            mask  = det["mask"]
            color = np.random.randint(100, 255, (3,), dtype=np.uint8).tolist()
            overlay[mask] = color

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    for step in step_results:
        for det in step["detections"]:
            label = f"Step {step['step_number']}: {det['label']} [{det.get('orientation','?')}]"
            bbox  = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    dir_name  = os.path.dirname(rgb_path)
    save_path = os.path.join(dir_name, f"DebugMasks_{capture_id}.jpg")
    cv2.imwrite(save_path, img)
    print(f"[debug] Saved visual mask verification to: {save_path}")


def _load_depth_map(depth_path: str, meta: dict) -> np.ndarray:
    width  = meta.get("depthWidth",  0)
    height = meta.get("depthHeight", 0)

    if width <= 0 or height <= 0:
        raise ValueError("Missing depth dimensions.")

    raw         = np.fromfile(depth_path, dtype=np.float32).reshape(height, width)
    clean_depth = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    near = meta.get("depthNearZ", 0.1)
    far  = meta.get("depthFarZ",  20.0)

    if "depthNearZ" not in meta:
        print("[server] WARNING: depthNearZ not in metadata, using fallback 0.1")
    if "depthFarZ" not in meta:
        print("[server] WARNING: depthFarZ not in metadata, using fallback 20.0")

    if np.isinf(far) or far > 10000.0:
        linear_depth = np.where(clean_depth > 0.0001, near / clean_depth, 0.0)
    else:
        linear_depth = np.where(
            clean_depth > 0.0001,
            (near * far) / (near + clean_depth * (far - near)),
            0.0
        )

    return linear_depth


@app.route("/process", methods=["POST"])
def process():
    body = request.get_json(force=True)

    required_fields = ["id", "rgbPath", "depthPath", "metaPath", "command"]
    missing = [f for f in required_fields if f not in body or not body[f]]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    capture_id = body["id"]
    rgb_path   = body["rgbPath"]
    depth_path = body["depthPath"]
    meta_path  = body["metaPath"]
    command    = body["command"]

    try:
        meta = _load_metadata(meta_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load metadata: {e}"}), 400

    try:
        depth_map = _load_depth_map(depth_path, meta)
    except Exception as e:
        return jsonify({"error": f"Failed to load depth map: {e}"}), 400

    try:
        step_results = fetch_step_segmentations(command, rgb_path)
        if DEBUG_MASKS:
            save_debug_masks(rgb_path, step_results, capture_id)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Vision pipeline failed: {e}"}), 500

    placements = []
    skipped    = 0

    for step in step_results:
        step_number = step["step_number"]
        instruction = step["instruction"]

        if not step["detections"]:
            # No objects detected — emit a text-only placement so the Quest
            # still displays the instruction for this step.
            placements.append({
                "step":        step_number,
                "instruction": instruction,
                "label":       "",
                "orientation": "",
                "worldX": 0.0, "worldY": 0.0, "worldZ": 0.0,
                "bboxCorners": [],
            })
            continue

        for detection in step["detections"]:
            world_point, bbox_corners = resolve_mask_world_point(
                detection["mask"], depth_map, meta, detection["bbox"]
            )
            if world_point is None:
                skipped += 1
                continue

            formatted_corners = [
                {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])}
                for pt in bbox_corners
            ]
            placements.append({
                "step":        step_number,
                "instruction": instruction,
                "label":       detection["label"],
                "orientation": detection.get("orientation", "up"), 
                "worldX": float(world_point[0]),
                "worldY": float(world_point[1]),
                "worldZ": float(world_point[2]),
                "bboxCorners": formatted_corners,
            })

    print(f"[server] '{capture_id}': {len(placements)} placements resolved, "
          f"{skipped} skipped (no valid depth).")

    return jsonify({"id": capture_id, "placements": placements})


if __name__ == "__main__":
    print("Starting detection + reprojection server on http://127.0.0.1:5000")
    print("Ensure OPENAI_API_KEY is set in your environment.")
    print(f"Debug mask saving: {'ON' if DEBUG_MASKS else 'OFF'} (set DEBUG_MASKS=1 to enable)")
    app.run(host="127.0.0.1", port=5000, debug=False)