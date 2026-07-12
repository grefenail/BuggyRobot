"""
Lane detection pipeline — orchestrates the six step modules, always
in bird's-eye coordinates (see config.py / ipm.py).

  ipm         → warp camera view to bird's-eye
  step1_mask  → HSV white mask + grayscale
  step2_canny → Canny edge detection
  step3_roi   → trapezoid region-of-interest mask (bird's-eye space)
  step4_hough → HoughLinesP + left/right classification
  step5_fit   → polynomial fit + smoothing + sanity check
  step6_draw  → semi-transparent lane fill, warped back onto the camera view
"""

import cv2
from config import ROTATE_CW, LANE_FILL_ALPHA, CENTER_WAYPOINT_COUNT
from step1_mask  import apply_white_mask
from step2_canny import apply_canny
from step5_fit   import update_fit
from step6_draw  import lane_overlay, fill_lane, center_curve
from roi_search  import detect_with_stable_roi
from ipm         import warp_to_birdseye, warp_to_vehicle, draw_destination_overlay


def _blend_overlay(base, overlay):
    mask = overlay.any(axis=2)
    if not mask.any():
        return base
    blended = cv2.addWeighted(base, 1.0 - LANE_FILL_ALPHA, overlay, LANE_FILL_ALPHA, 0)
    out = base.copy()
    out[mask] = blended[mask]
    return out


def _run_detection(bird_frame):
    h, w = bird_frame.shape[:2]
    gray = apply_white_mask(bird_frame)
    edges = apply_canny(gray)
    lx, ly, rx, ry, hough_vis, roi, vertices = detect_with_stable_roi(edges, h, w)
    _, _, min_y, max_y, lc, rc = update_fit(lx, ly, rx, ry, h, w)
    return gray, edges, roi, hough_vis, vertices, min_y, max_y, lc, rc


def detect_lanes(frame_bgr):
    if ROTATE_CW:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    bird_frame = warp_to_birdseye(frame_bgr)
    _, _, _, _, _, _, _, lc, rc = _run_detection(bird_frame)
    overlay = warp_to_vehicle(lane_overlay(bird_frame.shape, lc, rc))
    return _blend_overlay(frame_bgr, overlay)


def _coords_from_detection(bird_frame, min_y, max_y, lc, rc):
    center = center_curve(lc, rc) if lc is not None and rc is not None else None
    center_waypoints = center_curve(lc, rc, CENTER_WAYPOINT_COUNT) \
        if lc is not None and rc is not None else None

    return {
        "bird_width": int(bird_frame.shape[1]),
        "bird_height": int(bird_frame.shape[0]),
        "min_y": int(min_y),
        "max_y": int(max_y),
        "left_curve": [[int(x), int(y)] for x, y in lc] if lc is not None else None,
        "right_curve": [[int(x), int(y)] for x, y in rc] if rc is not None else None,
        "center_curve": [[int(x), int(y)] for x, y in center] if center is not None else None,
        "center_waypoints_px": [[int(x), int(y)] for x, y in center_waypoints]
                               if center_waypoints is not None else None,
    }


def detect_lanes_with_coords(frame_bgr):
    if ROTATE_CW:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    bird_frame = warp_to_birdseye(frame_bgr)
    _, _, _, _, _, min_y, max_y, lc, rc = _run_detection(bird_frame)
    overlay = warp_to_vehicle(lane_overlay(bird_frame.shape, lc, rc))
    return _blend_overlay(frame_bgr, overlay), _coords_from_detection(
        bird_frame, min_y, max_y, lc, rc
    )


def detect_lanes_birdeye_coords(frame_bgr):
    """
    Return the fitted lane-boundary coordinates in bird's-eye pixels.

    The left/right curves are the same polylines used by Step 6 for the
    green fill and yellow borders, before they are warped back onto the
    camera view.
    """
    if ROTATE_CW:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    bird_frame = warp_to_birdseye(frame_bgr)
    _, _, _, _, _, min_y, max_y, lc, rc = _run_detection(bird_frame)
    return _coords_from_detection(bird_frame, min_y, max_y, lc, rc)


def detect_lanes_debug(frame_bgr):
    """Same pipeline but returns each intermediate image for the debug windows."""
    if ROTATE_CW:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    bird_frame = warp_to_birdseye(frame_bgr)
    gray, edges, roi, hough_vis, vertices, min_y, max_y, lc, rc = _run_detection(bird_frame)

    bird = fill_lane(bird_frame.copy(), lc, rc, min_y, max_y)
    cv2.polylines(bird, vertices, isClosed=True, color=(255, 0, 255), thickness=2)
    bird = draw_destination_overlay(bird)
    result = _blend_overlay(frame_bgr, warp_to_vehicle(lane_overlay(bird_frame.shape, lc, rc)))

    return result, gray, edges, roi, hough_vis, bird
