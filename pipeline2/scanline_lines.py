"""Line fitting and lane geometry helpers for pipeline 2."""

import numpy as np

MIN_STRAIGHT_LINE_INLIERS = 8
STRAIGHT_LINE_TOLERANCE_PX = 12
MAX_LINE_ANGLE_FROM_VERTICAL_DEG = 8.0
BOTTOM_WIDTH_REFERENCE_FRAC = 0.45
MAX_WIDTH_DELTA_FROM_BOTTOM_FRAC = 0.20
BOTTOM_TRUST_MIN_Y_FRAC = 0.45
MIN_BOTTOM_TRUST_PAIRS = 4


def fit_line(points, y_min, y_max):
    if len(points) < 2:
        return None
    pts = np.asarray(points, dtype=np.float32)
    poly = np.poly1d(np.polyfit(pts[:, 1], pts[:, 0], deg=1))
    return [(int(poly(y_max)), int(y_max)), (int(poly(y_min)), int(y_min))]


def fit_consensus_straight_line(points, y_min, y_max):
    if len(points) < MIN_STRAIGHT_LINE_INLIERS:
        return None, []

    pts = np.asarray(points, dtype=np.float32)
    best_inliers = []

    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts)):
            y0, y1 = pts[i, 1], pts[j, 1]
            if abs(y1 - y0) < 1e-6:
                continue

            slope = (pts[j, 0] - pts[i, 0]) / (y1 - y0)
            intercept = pts[i, 0] - slope * y0
            predicted_x = slope * pts[:, 1] + intercept
            errors = np.abs(pts[:, 0] - predicted_x)
            inlier_idx = np.flatnonzero(errors <= STRAIGHT_LINE_TOLERANCE_PX)

            if len(inlier_idx) > len(best_inliers):
                best_inliers = inlier_idx

    if len(best_inliers) < MIN_STRAIGHT_LINE_INLIERS:
        return None, []

    inlier_points = [points[int(i)] for i in best_inliers]
    return fit_line(inlier_points, y_min, y_max), inlier_points


def line_x_at_y(line, y):
    if line is None:
        return None
    (x0, y0), (x1, y1) = line
    if abs(y1 - y0) < 1e-6:
        return None
    return x0 + (x1 - x0) * ((y - y0) / (y1 - y0))


def line_from_top_bottom_x(top_x, bottom_x, y_top, y_bottom):
    return [
        (int(round(bottom_x)), int(y_bottom)),
        (int(round(top_x)), int(y_top)),
    ]


def line_angle_from_vertical_deg(line):
    if line is None:
        return None
    (x0, y0), (x1, y1) = line
    dy = abs(y1 - y0)
    if dy < 1e-6:
        return 90.0
    return float(np.degrees(np.arctan2(abs(x1 - x0), dy)))


def line_is_plausible(line):
    angle = line_angle_from_vertical_deg(line)
    return angle is not None and angle <= MAX_LINE_ANGLE_FROM_VERTICAL_DEG


def lane_center_geometry(left_line, right_line, y_top, y_bottom):
    if left_line is None or right_line is None:
        return None

    left_top = line_x_at_y(left_line, y_top)
    left_bottom = line_x_at_y(left_line, y_bottom)
    right_top = line_x_at_y(right_line, y_top)
    right_bottom = line_x_at_y(right_line, y_bottom)
    if None in (left_top, left_bottom, right_top, right_bottom):
        return None

    return {
        "top_center": (left_top + right_top) / 2.0,
        "bottom_center": (left_bottom + right_bottom) / 2.0,
        "top_width": right_top - left_top,
        "bottom_width": right_bottom - left_bottom,
    }


def filter_pairs_by_bottom_width(accepted):
    if len(accepted) < MIN_STRAIGHT_LINE_INLIERS:
        return accepted

    bottom_count = max(
        MIN_STRAIGHT_LINE_INLIERS,
        int(round(len(accepted) * BOTTOM_WIDTH_REFERENCE_FRAC)),
    )
    bottom_pairs = sorted(accepted, key=lambda pair: pair[0][1], reverse=True)[:bottom_count]
    bottom_widths = [right[0] - left[0] for left, right, _ in bottom_pairs]
    bottom_width = float(np.median(bottom_widths))
    if bottom_width <= 0:
        return accepted

    max_delta = bottom_width * MAX_WIDTH_DELTA_FROM_BOTTOM_FRAC
    return [
        (left, right, score)
        for left, right, score in accepted
        if abs((right[0] - left[0]) - bottom_width) <= max_delta
    ]


def bottom_scanline_pairs(accepted, y_top=None, y_bottom=None):
    if y_top is not None and y_bottom is not None:
        min_y = y_top + (y_bottom - y_top) * BOTTOM_TRUST_MIN_Y_FRAC
        bottom_region = [pair for pair in accepted if pair[0][1] >= min_y]
        if len(bottom_region) >= MIN_BOTTOM_TRUST_PAIRS:
            return bottom_region

    if len(accepted) < MIN_STRAIGHT_LINE_INLIERS:
        return accepted

    bottom_count = max(
        MIN_STRAIGHT_LINE_INLIERS,
        int(round(len(accepted) * BOTTOM_WIDTH_REFERENCE_FRAC)),
    )
    return sorted(accepted, key=lambda pair: pair[0][1], reverse=True)[:bottom_count]


def line_pair_widths(left_line, right_line, y_top, y_bottom):
    if left_line is None or right_line is None:
        return None
    left_top = line_x_at_y(left_line, y_top)
    left_bottom = line_x_at_y(left_line, y_bottom)
    right_top = line_x_at_y(right_line, y_top)
    right_bottom = line_x_at_y(right_line, y_bottom)
    if None in (left_top, left_bottom, right_top, right_bottom):
        return None
    return right_top - left_top, right_bottom - left_bottom


def fitted_width_mismatch(left_line, right_line, y_top, y_bottom):
    widths = line_pair_widths(left_line, right_line, y_top, y_bottom)
    if widths is None:
        return False
    top_width, bottom_width = widths
    if bottom_width <= 0:
        return True
    return abs(top_width - bottom_width) > bottom_width * MAX_WIDTH_DELTA_FROM_BOTTOM_FRAC


def fit_from_pairs(pairs, y_top, y_bottom):
    left_points = [left for left, _, _ in pairs]
    right_points = [right for _, right, _ in pairs]
    left_line, left_inliers = fit_consensus_straight_line(left_points, y_top, y_bottom)
    right_line, right_inliers = fit_consensus_straight_line(right_points, y_top, y_bottom)
    return left_line, right_line, left_inliers, right_inliers


def fit_direct_from_pairs(pairs, y_top, y_bottom):
    left_points = [left for left, _, _ in pairs]
    right_points = [right for _, right, _ in pairs]
    if len(left_points) < 2 or len(right_points) < 2:
        return None, None, [], []
    return (
        fit_line(left_points, y_top, y_bottom),
        fit_line(right_points, y_top, y_bottom),
        left_points,
        right_points,
    )


def fit_center_width_from_pairs(pairs, y_top, y_bottom):
    if len(pairs) < 2:
        return None, None, [], []

    center_points = [
        ((left[0] + right[0]) / 2.0, left[1])
        for left, right, _ in pairs
    ]
    widths = [right[0] - left[0] for left, right, _ in pairs]
    lane_width = float(np.median(widths))
    if lane_width <= 0:
        return None, None, [], []

    center_line = fit_line(center_points, y_top, y_bottom)
    if center_line is None:
        return None, None, [], []

    center_top = line_x_at_y(center_line, y_top)
    center_bottom = line_x_at_y(center_line, y_bottom)
    if center_top is None or center_bottom is None:
        return None, None, [], []

    half_width = lane_width / 2.0
    return (
        line_from_top_bottom_x(center_top - half_width, center_bottom - half_width, y_top, y_bottom),
        line_from_top_bottom_x(center_top + half_width, center_bottom + half_width, y_top, y_bottom),
        [left for left, _, _ in pairs],
        [right for _, right, _ in pairs],
    )
