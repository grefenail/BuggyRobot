"""Step 6 — Lane fill drawing.

Draws a semi-transparent filled polygon between the two fitted
lane boundaries, with crisp border lines on each edge. Each boundary
is a polyline (2 points for a straight fallback fit, more when the
concentric-arc fit bent it), not just a single straight segment.
"""
import cv2
import numpy as np
from config import LANE_FILL_COLOR, LANE_FILL_ALPHA, BORDER_COLOR, BORDER_WIDTH


def fill_lane(img, left_curve, right_curve, min_y, max_y):
    """
    Blend a filled polygon between the left and right lane boundary
    curves (each a list of (x, y) points, bottom-to-top).

    Returns the annotated image unchanged if either side is missing.
    """
    if left_curve is None or right_curve is None:
        return img

    # Polygon vertices: left boundary bottom->top, then right boundary top->bottom.
    pts = np.array(left_curve + right_curve[::-1], dtype=np.int32)

    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], LANE_FILL_COLOR)
    result = cv2.addWeighted(overlay, LANE_FILL_ALPHA, img, 1 - LANE_FILL_ALPHA, 0)

    cv2.polylines(result, [np.array(left_curve,  dtype=np.int32)], False, BORDER_COLOR, BORDER_WIDTH)
    cv2.polylines(result, [np.array(right_curve, dtype=np.int32)], False, BORDER_COLOR, BORDER_WIDTH)
    return result


def lane_overlay(shape, left_curve, right_curve):
    """Return a BGR overlay containing only the lane fill/borders."""
    overlay = np.zeros(shape, dtype=np.uint8)
    if left_curve is None or right_curve is None:
        return overlay

    pts = np.array(left_curve + right_curve[::-1], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], LANE_FILL_COLOR)
    cv2.polylines(overlay, [np.array(left_curve, dtype=np.int32)], False, BORDER_COLOR, BORDER_WIDTH)
    cv2.polylines(overlay, [np.array(right_curve, dtype=np.int32)], False, BORDER_COLOR, BORDER_WIDTH)
    return overlay
