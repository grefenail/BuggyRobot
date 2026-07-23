"""Drawing and coordinate projection helpers for pipeline 2."""

import cv2
import numpy as np

from ipm import get_matrices
from scanline_lines import line_x_at_y

CENTER_WAYPOINT_COUNT = 10
LANE_FILL_COLOR = (0, 180, 0)
LANE_FILL_ALPHA = 0.28


def draw_line_clipped(img, line, color, thickness=2):
    if line is None:
        return
    h, w = img.shape[:2]
    pts = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in line]
    cv2.line(img, pts[0], pts[1], color, thickness, cv2.LINE_AA)


def center_waypoints_from_anchor(anchor_x, y_top, y_bottom, count=CENTER_WAYPOINT_COUNT):
    return [
        (int(round(anchor_x)), int(round(y)))
        for y in np.linspace(y_bottom, y_top, count)
    ]


def fallback_center_anchor(accepted, default_center_x):
    if not accepted:
        return default_center_x

    nearest_left, nearest_right, _ = max(
        accepted,
        key=lambda pair: (pair[0][1] + pair[1][1]) / 2.0,
    )
    return (nearest_left[0] + nearest_right[0]) / 2.0


def center_waypoints_from_lines(left_line, right_line, y_top, y_bottom, count=CENTER_WAYPOINT_COUNT):
    if left_line is None or right_line is None:
        return []

    points = []
    for y in np.linspace(y_bottom, y_top, count):
        lx = line_x_at_y(left_line, y)
        rx = line_x_at_y(right_line, y)
        if lx is None or rx is None:
            continue
        points.append((int(round((lx + rx) / 2.0)), int(round(y))))
    return points


def center_waypoints(left_line, right_line, accepted, default_center_x, y_top, y_bottom):
    points = center_waypoints_from_lines(left_line, right_line, y_top, y_bottom)
    if points:
        return points, "fit"

    return [], "fallback"


def draw_center_waypoints(img, points):
    if not points:
        return
    h, w = img.shape[:2]
    clipped = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in points]

    if len(clipped) >= 2:
        cv2.polylines(img, [np.asarray(clipped, dtype=np.int32)], False, (0, 165, 255), 3, cv2.LINE_AA)

    for idx, point in enumerate(clipped):
        cv2.circle(img, point, 8, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.circle(img, point, 8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            str(idx),
            (point[0] + 10, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            str(idx),
            (point[0] + 10, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def blend_lane_fill(img, polygon_points, color=LANE_FILL_COLOR, alpha=LANE_FILL_ALPHA):
    if len(polygon_points) < 3:
        return img
    overlay = img.copy()
    pts = np.asarray(polygon_points, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)


def blend_nonzero_overlay(img, overlay, alpha=LANE_FILL_ALPHA):
    mask = np.any(overlay != 0, axis=2)
    if not np.any(mask):
        return img
    blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    out = img.copy()
    out[mask] = blended[mask]
    return out


def lane_polygon_from_lines(left_line, right_line, y_top, y_bottom):
    if left_line is None or right_line is None:
        return []

    left_bottom = line_x_at_y(left_line, y_bottom)
    left_top = line_x_at_y(left_line, y_top)
    right_top = line_x_at_y(right_line, y_top)
    right_bottom = line_x_at_y(right_line, y_bottom)
    if None in (left_bottom, left_top, right_top, right_bottom):
        return []

    return [
        (int(round(left_bottom)), int(round(y_bottom))),
        (int(round(left_top)), int(round(y_top))),
        (int(round(right_top)), int(round(y_top))),
        (int(round(right_bottom)), int(round(y_bottom))),
    ]


def project_bird_points_to_vehicle(points, width, height):
    if not points:
        return []
    _, bird_to_vehicle = get_matrices(width, height)
    pts = np.array([[[float(x), float(y)]] for x, y in points], dtype=np.float32)
    projected = cv2.perspectiveTransform(pts, bird_to_vehicle)[:, 0, :]
    return [(int(round(x)), int(round(y))) for x, y in projected]


def label_panel(img, text):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (min(w, 220), 28), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out
