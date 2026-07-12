"""Step 6 — Lane fill drawing.

Draws a semi-transparent filled polygon between the two fitted
lane boundaries, with crisp border lines on each edge. Each boundary
is a polyline (2 points for a straight fallback fit, more when the
concentric-arc fit bent it), not just a single straight segment.
"""
import cv2
import numpy as np
from config import (
    LANE_FILL_COLOR,
    LANE_FILL_ALPHA,
    BORDER_COLOR,
    BORDER_WIDTH,
    CENTER_COLOR,
    CENTER_WIDTH,
    CENTER_DOT_COLOR,
    CENTER_DOT_RADIUS,
    CENTER_LABEL_COLOR,
    CENTER_LABEL_SCALE,
    CENTER_LABEL_THICKNESS,
    CENTER_WAYPOINT_COUNT,
)


def center_curve(left_curve, right_curve, sample_count=None):
    """
    Return the lane centerline in bird's-eye pixels by averaging the
    left/right boundary x positions at matching y rows.
    """
    if left_curve is None or right_curve is None:
        return None

    sample_count = sample_count or max(len(left_curve), len(right_curve), 2)
    y_bot = max(left_curve[0][1], right_curve[0][1])
    y_top = min(left_curve[-1][1], right_curve[-1][1])
    ys = np.linspace(y_bot, y_top, sample_count)

    def interp_x(curve):
        pts = np.array(curve, dtype=np.float64)
        order = np.argsort(pts[:, 1])
        return np.interp(ys, pts[order, 1], pts[order, 0])

    left_x = interp_x(left_curve)
    right_x = interp_x(right_curve)
    center_x = (left_x + right_x) / 2.0
    return [(int(round(x)), int(round(y))) for x, y in zip(center_x, ys)]


def _draw_center(img, left_curve, right_curve):
    center = center_curve(left_curve, right_curve)
    if center is None:
        return

    cv2.polylines(img, [np.array(center, dtype=np.int32)], False, CENTER_COLOR, CENTER_WIDTH)

    waypoints = center_curve(left_curve, right_curve, CENTER_WAYPOINT_COUNT)
    if waypoints is None:
        return
    for i, point in enumerate(waypoints):
        cv2.circle(img, point, CENTER_DOT_RADIUS, CENTER_DOT_COLOR, -1)
        cv2.circle(img, point, CENTER_DOT_RADIUS + 2, (0, 0, 0), 2)

        label = str(i)
        pos = (point[0] + 14, point[1] - 12)
        cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    CENTER_LABEL_SCALE, (0, 0, 0),
                    CENTER_LABEL_THICKNESS + 3, cv2.LINE_AA)
        cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    CENTER_LABEL_SCALE, CENTER_LABEL_COLOR,
                    CENTER_LABEL_THICKNESS, cv2.LINE_AA)


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
    _draw_center(result, left_curve, right_curve)
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
    _draw_center(overlay, left_curve, right_curve)
    return overlay
