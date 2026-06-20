"""
mask_reprojection.py

For each detected object's SAM mask, finds the 3D world-space point that
should anchor an instruction ball. The approach:

  1. Find all (u, v) pixel coordinates belonging to the mask.
  2. Compute the mask's centroid (u_c, v_c) in RGB pixel space.
  3. Sample up to MAX_SAMPLES points spread across the mask (not just the
     centroid pixel itself) and run each through reprojection.py's
     get_depth_at_rgb_pixel, which iteratively finds the matching depth-image
     pixel and returns the true depth at that location.
  4. Take the MEDIAN of the successfully resolved depths. This is robust to:
       - holes in the mask (occlusion, thin structures)
       - depth-sensor noise at object edges
       - the centroid pixel itself landing on a depth dead-zone (returns None)
     A single point (just the centroid) has no such robustness; averaging
     all mask pixels would be needlessly expensive for large masks and
     vulnerable to outliers from edge pixels bleeding onto the background.
  5. Unproject the centroid pixel using the median depth, then transform the
     resulting camera-space point into Unity world space.

Why sample a bounded subset rather than every mask pixel: a large mask
(e.g. "bed") can cover tens of thousands of pixels. Running the iterative
reprojection search (up to 5 iterations each) on every single one would be
slow for no real benefit — a few hundred well-spread samples already give a
stable median.
"""

import numpy as np
from reprojection import get_depth_at_rgb_pixel
from camera_math import build_depth_to_rgb_transform, unproject_pixel, rgb_camera_point_to_unity_world

MAX_SAMPLES = 200


def _sample_mask_pixels(mask: np.ndarray, max_samples: int = MAX_SAMPLES):
    """
    Returns up to max_samples (v, u) coordinates spread across the mask using
    uniform random sampling without replacement. Uniform random sampling
    (rather than e.g. every Nth pixel in scan order) avoids accidentally
    aliasing onto a periodic pattern in the mask shape.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.array([]), np.array([])

    if len(xs) <= max_samples:
        return ys, xs

    rng = np.random.default_rng(seed=42)  # deterministic across requests, easier to debug
    idx = rng.choice(len(xs), size=max_samples, replace=False)
    return ys[idx], xs[idx]


def mask_centroid(mask: np.ndarray):
    """Returns (u_centroid, v_centroid) in pixel coordinates."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    return float(np.mean(xs)), float(np.mean(ys))

def resolve_mask_world_point(mask: np.ndarray, depth_map: np.ndarray, meta: dict, bbox: list):
    """
    Now also accepts the 2D `bbox` [x1, y1, x2, y2].
    Returns (world_point, bbox_corners_world) or (None, None).
    """
    u_c, v_c = mask_centroid(mask)
    if u_c is None:
        return None, None

    R_d2r, T_d2r = build_depth_to_rgb_transform(meta)

    rgb_K_full = np.array([
        [meta["fx"], 0,          meta["cx"]],
        [0,          meta["fy"], meta["cy"]],
        [0,          0,          1.0       ],
    ])
    depth_K_full = np.array([
        [meta["depth_fx"], 0,                meta["depth_cx"]],
        [0,                meta["depth_fy"], meta["depth_cy"]],
        [0,                0,                1.0              ],
    ])

    ys, xs = _sample_mask_pixels(mask)
    resolved_depths = []

    for v, u in zip(ys, xs):
        z = get_depth_at_rgb_pixel(
            int(u), int(v), depth_map,
            rgb_K_full, depth_K_full, R_d2r, T_d2r,
        )
        if z is not None and z > 0.0:
            resolved_depths.append(z)

    if not resolved_depths:
        return None, None

    median_depth = float(np.median(resolved_depths))
    
   # In mask_reprojection.py
    min_depth = float(np.percentile(resolved_depths, 5))
    max_depth = min(float(np.percentile(resolved_depths, 95)), 5.0) # Cap at 5 meters

    # 1. Unproject the MASK CENTROID
    point_camera_space = unproject_pixel(
        u_c, v_c, median_depth,
        meta["fx"], meta["fy"], meta["cx"], meta["cy"],
    )
    world_point = rgb_camera_point_to_unity_world(point_camera_space, meta)

    # 2. Unproject the 3D Bounding Box
    x1, y1, x2, y2 = bbox
    # 4 corners of the 2D bounding box
    corners_2d = [
        (x1, y1), # Top-Left
        (x2, y1), # Top-Right
        (x2, y2), # Bottom-Right
        (x1, y2)  # Bottom-Left
    ]

    bbox_corners_world = []

    # Helper function to unproject a pixel at a specific depth and convert to World Space
    def get_world_corner(u, v, z):
        cam_pt = unproject_pixel(u, v, z, meta["fx"], meta["fy"], meta["cx"], meta["cy"])
        return rgb_camera_point_to_unity_world(cam_pt, meta)

    # Front face (Closest to camera: min_depth)
    for u, v in corners_2d:
        bbox_corners_world.append(get_world_corner(u, v, min_depth))

    # Back face (Furthest from camera: max_depth)
    for u, v in corners_2d:
        bbox_corners_world.append(get_world_corner(u, v, max_depth))
    print("bbox corners",bbox_corners_world)
    return world_point, bbox_corners_world