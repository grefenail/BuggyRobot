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
from config import ROTATE_CW, LANE_FILL_ALPHA
from step1_mask  import apply_white_mask
from step2_canny import apply_canny
from step5_fit   import update_fit
from step6_draw  import lane_overlay, fill_lane
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
