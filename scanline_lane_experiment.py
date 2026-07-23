"""Standalone scanline lane experiment.

This does not replace the main pipeline. It tests a simpler idea:
sample many horizontal bands in bird's-eye view, find x positions where
white pixels spike, pair left/right peaks by expected lane width, and
draw artificial lane lines through the most confident pairs.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

import config
from ipm import get_matrices, warp_to_birdseye
from step1_mask import apply_white_mask_relative

VIDEO_DIR = ROOT / "vids"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
EDGE_REJECT_MARGIN_PX = 5
DEFAULT_SCANLINE_COUNT = 30
MAX_SCANLINE_X_JUMP_PX = 35
SCANLINE_X_JUMP_PENALTY = 5.0
REGISTERED_PAIR_MIN_PEAKS = 3
REGISTERED_PAIR_WIDTH_PENALTY = 0.25
MAX_SCANLINE_ANCHOR_DRIFT_PX = 90
SCANLINE_ANCHOR_PENALTY = 1.5
# Disabled: center-pair preference. It rejects pairs whose midpoint is too
# far from the configured lane center, but it is not reliable when the
# bird's-eye center itself is off or neighboring lane lines are also centered.
# MAX_PAIR_CENTER_ERROR_FRAC = 0.18
# PAIR_CENTER_PENALTY = 2.5
MIN_STRAIGHT_LINE_INLIERS = 8
STRAIGHT_LINE_TOLERANCE_PX = 12
MAX_LINE_ANGLE_FROM_VERTICAL_DEG = 8.0
BOTTOM_WIDTH_REFERENCE_FRAC = 0.45
MAX_WIDTH_DELTA_FROM_BOTTOM_FRAC = 0.20
BOTTOM_TRUST_MIN_Y_FRAC = 0.45
MIN_BOTTOM_TRUST_PAIRS = 4
HOUGH_FALLBACK_MIN_Y_SPAN_FRAC = 0.18
HOUGH_FALLBACK_MIN_SEPARATION_FRAC = 0.45
HOUGH_FALLBACK_MAX_ANGLE_DEG = 10.0
COMPONENT_FALLBACK_MIN_AREA = 80
COMPONENT_FALLBACK_MIN_Y_SPAN_FRAC = 0.16
COMPONENT_FALLBACK_MAX_ANGLE_DEG = 10.0
RELATIVE_WHITE_TOP_PERCENT = 5
CENTER_WAYPOINT_COUNT = 10
LANE_FILL_COLOR = (0, 180, 0)
LANE_FILL_ALPHA = 0.28
LAST_GOOD_LANE_WIDTH_PX = None
LAST_GOOD_LANE_CENTER = None


def resolve_video(name):
    if name is None:
        videos = sorted(p for p in VIDEO_DIR.iterdir()
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            raise FileNotFoundError(f"No videos found in {VIDEO_DIR}")
        return videos[0]
    p = Path(name)
    if p.exists():
        return p.resolve()
    p2 = VIDEO_DIR / name
    if p2.exists():
        return p2.resolve()
    raise FileNotFoundError(f"Cannot find '{name}'")


def processing_frame(frame_bgr):
    scale = float(config.PROCESS_SCALE)
    if abs(scale - 1.0) < 1e-6:
        return frame_bgr
    return cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def white_mask_hsv(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, config.WHITE_V_MIN), (180, config.WHITE_S_MAX, 255))


def white_mask_relative(frame_bgr):
    relative = apply_white_mask_relative(frame_bgr, RELATIVE_WHITE_TOP_PERCENT)
    return np.where(relative > 0, 255, 0).astype(np.uint8)


def smooth_1d(values, kernel_size=21):
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def local_peaks(values, min_score, edge_margin=0):
    peaks = []
    x_start = max(1, edge_margin)
    x_end = min(len(values) - 1, len(values) - edge_margin)

    x = x_start
    while x < x_end:
        if values[x] < min_score:
            x += 1
            continue

        run_start = x
        while x < x_end and values[x] >= min_score:
            x += 1
        run_end = x

        run = values[run_start:run_end]
        if len(run) == 0:
            continue
        peak_score = float(run.max())
        max_positions = np.flatnonzero(run == peak_score)
        peak_x = run_start + int(round(float(max_positions.mean())))
        peaks.append((peak_x, peak_score))

    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks


def best_lane_pair(
    profile,
    expected_width,
    width_tolerance,
    center_x,
    previous_pair=None,
    anchor_pair=None,
    edge_margin=EDGE_REJECT_MARGIN_PX,
):
    min_score = max(3.0, float(profile.max()) * 0.25)
    peaks = local_peaks(profile, min_score, edge_margin=edge_margin)[:12]
    peak_xs = [x for x, _ in peaks]
    # Disabled: see MAX_PAIR_CENTER_ERROR_FRAC above.
    # max_center_error = expected_width * MAX_PAIR_CENTER_ERROR_FRAC
    best = None

    if previous_pair is not None and len(peaks) >= REGISTERED_PAIR_MIN_PEAKS:
        registered_best = None
        for li, left_score in peaks:
            for ri, right_score in peaks:
                if ri <= li:
                    continue
                lane_width = ri - li
                width_error = abs(lane_width - expected_width)
                if width_error > width_tolerance:
                    continue

                left_jump = abs(li - previous_pair[0])
                right_jump = abs(ri - previous_pair[1])
                if left_jump > MAX_SCANLINE_X_JUMP_PX or right_jump > MAX_SCANLINE_X_JUMP_PX:
                    continue

                registered_error = left_jump + right_jump
                score = left_score + right_score - registered_error - width_error * REGISTERED_PAIR_WIDTH_PENALTY
                if registered_best is None or registered_error < registered_best["registered_error"] or (
                    registered_error == registered_best["registered_error"] and score > registered_best["score"]
                ):
                    registered_best = {
                        "left": li,
                        "right": ri,
                        "score": score,
                        "left_score": left_score,
                        "right_score": right_score,
                        "width": lane_width,
                        "peaks": peak_xs,
                        "registered_error": registered_error,
                    }

        if registered_best is not None:
            return registered_best

    for li, left_score in peaks:
        for ri, right_score in peaks:
            if ri <= li:
                continue
            lane_width = ri - li
            width_error = abs(lane_width - expected_width)
            if width_error > width_tolerance:
                continue
            pair_center = (li + ri) / 2.0
            center_error = abs(pair_center - center_x)
            # Disabled: center-pair hard reject.
            # if center_error > max_center_error:
            #     continue
            continuity_error = 0.0
            if previous_pair is not None:
                left_jump = abs(li - previous_pair[0])
                right_jump = abs(ri - previous_pair[1])
                if left_jump > MAX_SCANLINE_X_JUMP_PX or right_jump > MAX_SCANLINE_X_JUMP_PX:
                    continue
                continuity_error = left_jump + right_jump
            anchor_error = 0.0
            if anchor_pair is not None:
                left_anchor_jump = abs(li - anchor_pair[0])
                right_anchor_jump = abs(ri - anchor_pair[1])
                if (
                    left_anchor_jump > MAX_SCANLINE_ANCHOR_DRIFT_PX
                    or right_anchor_jump > MAX_SCANLINE_ANCHOR_DRIFT_PX
                ):
                    continue
                anchor_error = left_anchor_jump + right_anchor_jump
            score = (
                left_score + right_score
                - width_error * 0.6
                - center_error * 0.15
                - continuity_error * SCANLINE_X_JUMP_PENALTY
                - anchor_error * SCANLINE_ANCHOR_PENALTY
            )
            if best is None or score > best["score"]:
                best = {
                    "left": li,
                    "right": ri,
                    "score": score,
                    "left_score": left_score,
                    "right_score": right_score,
                    "width": lane_width,
                    "peaks": peak_xs,
                }

    return best


def best_single_side_peak(profile, expected_x, center_x, want_left, edge_margin=EDGE_REJECT_MARGIN_PX):
    min_score = max(3.0, float(profile.max()) * 0.25)
    peaks = local_peaks(profile, min_score, edge_margin=edge_margin)[:12]
    candidates = [
        (x, score)
        for x, score in peaks
        if (x < center_x if want_left else x > center_x)
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda item: item[1] - abs(item[0] - expected_x) * 0.05)


def fit_line(points, y_min, y_max):
    if len(points) < 2:
        return None
    pts = np.asarray(points, dtype=np.float32)
    poly = np.poly1d(np.polyfit(pts[:, 1], pts[:, 0], deg=1))
    return [(int(poly(y_max)), int(y_max)), (int(poly(y_min)), int(y_min))]


def fit_consensus_straight_line(points, y_min, y_max):
    """
    Find the straight line supported by the most scanline points.

    This keeps a mostly-correct lane straight even when a few scanlines
    lock onto a neighboring white line.
    """
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


def draw_line_clipped(img, line, color, thickness=2):
    if line is None:
        return
    h, w = img.shape[:2]
    pts = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in line]
    cv2.line(img, pts[0], pts[1], color, thickness, cv2.LINE_AA)


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


# Optional Hough/component missing-side fallback, disabled for now.
# Keep this block around for future tuning.
#
# def line_mean_x(line):
#     if line is None:
#         return None
#     return (line[0][0] + line[1][0]) / 2.0
#
#
# def hough_fallback_lines(mask, y_top, y_bottom):
#     edges = cv2.Canny(mask, 50, 150)
#     min_len = max(20, int((y_bottom - y_top) * HOUGH_FALLBACK_MIN_Y_SPAN_FRAC))
#     raw = cv2.HoughLinesP(
#         edges,
#         rho=2,
#         theta=np.pi / 180,
#         threshold=35,
#         minLineLength=min_len,
#         maxLineGap=18,
#     )
#     if raw is None:
#         return []
#
#     lines = []
#     for x1, y1, x2, y2 in np.asarray(raw).reshape(-1, 4):
#         if abs(y2 - y1) < min_len:
#             continue
#         segment = [(int(x1), int(y1)), (int(x2), int(y2))]
#         angle = line_angle_from_vertical_deg(segment)
#         if angle is None or angle > HOUGH_FALLBACK_MAX_ANGLE_DEG:
#             continue
#         full_line = fit_line([(x1, y1), (x2, y2)], y_top, y_bottom)
#         if full_line is None or not line_is_plausible(full_line):
#             continue
#         lines.append(full_line)
#
#     return lines
#
#
# def first_hough_line_on_missing_side(mask, left_line, right_line, y_top, y_bottom, expected_width):
#     candidates = hough_fallback_lines(mask, y_top, y_bottom)
#     if not candidates:
#         return left_line, right_line, None
#
#     width_ref = expected_width
#     if LAST_GOOD_LANE_CENTER is not None:
#         width_ref = (LAST_GOOD_LANE_CENTER["top_width"] + LAST_GOOD_LANE_CENTER["bottom_width"]) / 2.0
#     min_separation = max(40.0, width_ref * HOUGH_FALLBACK_MIN_SEPARATION_FRAC)
#
#     if left_line is not None and right_line is None:
#         left_bottom = line_x_at_y(left_line, y_bottom)
#         if left_bottom is None:
#             return left_line, right_line, None
#         valid = []
#         for candidate in candidates:
#             candidate_bottom = line_x_at_y(candidate, y_bottom)
#             if candidate_bottom is None or candidate_bottom <= left_bottom + min_separation:
#                 continue
#             valid.append((candidate_bottom - left_bottom, candidate))
#         if valid:
#             valid.sort(key=lambda item: item[0])
#             return left_line, valid[0][1], "hough-right"
#
#     if right_line is not None and left_line is None:
#         right_bottom = line_x_at_y(right_line, y_bottom)
#         if right_bottom is None:
#             return left_line, right_line, None
#         valid = []
#         for candidate in candidates:
#             candidate_bottom = line_x_at_y(candidate, y_bottom)
#             if candidate_bottom is None or candidate_bottom >= right_bottom - min_separation:
#                 continue
#             valid.append((right_bottom - candidate_bottom, candidate))
#         if valid:
#             valid.sort(key=lambda item: item[0])
#             return valid[0][1], right_line, "hough-left"
#
#     return left_line, right_line, None
#
#
# def component_fallback_lines(mask, y_top, y_bottom):
#     roi = mask[y_top:y_bottom + 1]
#     count, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
#     min_span = max(20, int((y_bottom - y_top) * COMPONENT_FALLBACK_MIN_Y_SPAN_FRAC))
#     bottom_min_y = y_top + (y_bottom - y_top) * BOTTOM_TRUST_MIN_Y_FRAC
#     lines = []
#
#     for label in range(1, count):
#         x, y, width, height, area = stats[label]
#         if area < COMPONENT_FALLBACK_MIN_AREA or height < min_span:
#             continue
#
#         points = np.column_stack(np.where(labels == label))
#         ys = points[:, 0] + y_top
#         xs = points[:, 1]
#         bottom_mask = ys >= bottom_min_y
#         if np.count_nonzero(bottom_mask) < COMPONENT_FALLBACK_MIN_AREA:
#             continue
#
#         line = fit_line(list(zip(xs[bottom_mask], ys[bottom_mask])), y_top, y_bottom)
#         if line is None:
#             continue
#         angle = line_angle_from_vertical_deg(line)
#         if angle is None or angle > COMPONENT_FALLBACK_MAX_ANGLE_DEG:
#             continue
#         lines.append(line)
#
#     return lines
#
#
# def first_component_line_on_missing_side(mask, left_line, right_line, y_top, y_bottom, expected_width):
#     candidates = component_fallback_lines(mask, y_top, y_bottom)
#     if not candidates:
#         return left_line, right_line, None
#
#     width_ref = expected_width
#     if LAST_GOOD_LANE_CENTER is not None:
#         width_ref = (LAST_GOOD_LANE_CENTER["top_width"] + LAST_GOOD_LANE_CENTER["bottom_width"]) / 2.0
#     min_separation = max(40.0, width_ref * HOUGH_FALLBACK_MIN_SEPARATION_FRAC)
#
#     if left_line is not None and right_line is None:
#         left_bottom = line_x_at_y(left_line, y_bottom)
#         if left_bottom is None:
#             return left_line, right_line, None
#         valid = []
#         for candidate in candidates:
#             candidate_bottom = line_x_at_y(candidate, y_bottom)
#             if candidate_bottom is None or candidate_bottom <= left_bottom + min_separation:
#                 continue
#             valid.append((candidate_bottom - left_bottom, candidate))
#         if valid:
#             valid.sort(key=lambda item: item[0])
#             return left_line, valid[0][1], "component-right"
#
#     if right_line is not None and left_line is None:
#         right_bottom = line_x_at_y(right_line, y_bottom)
#         if right_bottom is None:
#             return left_line, right_line, None
#         valid = []
#         for candidate in candidates:
#             candidate_bottom = line_x_at_y(candidate, y_bottom)
#             if candidate_bottom is None or candidate_bottom >= right_bottom - min_separation:
#                 continue
#             valid.append((right_bottom - candidate_bottom, candidate))
#         if valid:
#             valid.sort(key=lambda item: item[0])
#             return valid[0][1], right_line, "component-left"
#
#     return left_line, right_line, None
#
#
# def align_fallback_side_from_bottom_width(left_line, right_line, mode, y_top, y_bottom):
#     if mode not in ("hough-right", "component-right", "hough-left", "component-left"):
#         return left_line, right_line
#
#     top_width, bottom_width = line_pair_widths(left_line, right_line, y_top, y_bottom) or (None, None)
#     if top_width is None or bottom_width is None or bottom_width <= 0:
#         return left_line, right_line
#     if abs(top_width - bottom_width) <= bottom_width * MAX_WIDTH_DELTA_FROM_BOTTOM_FRAC:
#         return left_line, right_line
#
#     if mode in ("hough-right", "component-right"):
#         left_top = line_x_at_y(left_line, y_top)
#         right_bottom = line_x_at_y(right_line, y_bottom)
#         if left_top is None or right_bottom is None:
#             return left_line, right_line
#         right_line = line_from_top_bottom_x(left_top + bottom_width, right_bottom, y_top, y_bottom)
#     else:
#         right_top = line_x_at_y(right_line, y_top)
#         left_bottom = line_x_at_y(left_line, y_bottom)
#         if right_top is None or left_bottom is None:
#             return left_line, right_line
#         left_line = line_from_top_bottom_x(right_top - bottom_width, left_bottom, y_top, y_bottom)
#
#     return left_line, right_line


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


def reconstruct_missing_line_from_widths(left_line, right_line, y_top, y_bottom):
    if LAST_GOOD_LANE_CENTER is None:
        return left_line, right_line, "fallback"

    top_width = LAST_GOOD_LANE_CENTER["top_width"]
    bottom_width = LAST_GOOD_LANE_CENTER["bottom_width"]
    if top_width <= 0 or bottom_width <= 0:
        return left_line, right_line, "fallback"

    if left_line is not None and right_line is None:
        left_top = line_x_at_y(left_line, y_top)
        left_bottom = line_x_at_y(left_line, y_bottom)
        if left_top is None or left_bottom is None:
            return left_line, right_line, "fallback"
        right_line = line_from_top_bottom_x(
            left_top + top_width,
            left_bottom + bottom_width,
            y_top,
            y_bottom,
        )
        return left_line, right_line, "width-reconstruct-left"

    if right_line is not None and left_line is None:
        right_top = line_x_at_y(right_line, y_top)
        right_bottom = line_x_at_y(right_line, y_bottom)
        if right_top is None or right_bottom is None:
            return left_line, right_line, "fallback"
        left_line = line_from_top_bottom_x(
            right_top - top_width,
            right_bottom - bottom_width,
            y_top,
            y_bottom,
        )
        return left_line, right_line, "width-reconstruct-right"

    return left_line, right_line, "fallback"


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
    trusted = [
        (left, right, score)
        for left, right, score in accepted
        if abs((right[0] - left[0]) - bottom_width) <= max_delta
    ]

    return trusted


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


def fit_lane_lines_from_scanlines(
    accepted,
    single_left_points,
    single_right_points,
    y_top,
    y_bottom,
    expected_width,
    mask=None,
):
    # Bottom-width filtering disabled for comparison.
    # accepted = filter_pairs_by_bottom_width(accepted)
    left_line, right_line, left_inliers, right_inliers = fit_from_pairs(accepted, y_top, y_bottom)

    if left_line is None:
        left_points = [left for left, _, _ in accepted]
        fallback_left_line, fallback_left_inliers = fit_consensus_straight_line(
            left_points + single_left_points,
            y_top,
            y_bottom,
        )
        if fallback_left_line is not None:
            left_line = fallback_left_line
            left_inliers = fallback_left_inliers

    if right_line is None:
        right_points = [right for _, right, _ in accepted]
        fallback_right_line, fallback_right_inliers = fit_consensus_straight_line(
            right_points + single_right_points,
            y_top,
            y_bottom,
        )
        if fallback_right_line is not None:
            right_line = fallback_right_line
            right_inliers = fallback_right_inliers

    # Bottom-only refit disabled for comparison.
    # if left_line is not None and right_line is not None and fitted_width_mismatch(left_line, right_line, y_top, y_bottom):
    #     bottom_pairs = bottom_scanline_pairs(accepted, y_top, y_bottom)
    #     if len(bottom_pairs) >= MIN_STRAIGHT_LINE_INLIERS:
    #         bottom_left_line, bottom_right_line, bottom_left_inliers, bottom_right_inliers = fit_from_pairs(
    #             bottom_pairs,
    #             y_top,
    #             y_bottom,
    #         )
    #     else:
    #         bottom_left_line, bottom_right_line, bottom_left_inliers, bottom_right_inliers = fit_center_width_from_pairs(
    #             bottom_pairs,
    #             y_top,
    #             y_bottom,
    #         )
    #     if bottom_left_line is not None and bottom_right_line is not None:
    #         left_line = bottom_left_line
    #         right_line = bottom_right_line
    #         left_inliers = bottom_left_inliers
    #         right_inliers = bottom_right_inliers

    if left_line is not None and not line_is_plausible(left_line):
        left_line = None
        left_inliers = []
    if right_line is not None and not line_is_plausible(right_line):
        right_line = None
        right_inliers = []

    hough_mode = None
    # Optional missing-side fallback, disabled for now.
    # This tries Hough first, then connected white-mask components, to find
    # the first consistent line on the missing side. It was too aggressive
    # in some frames, so leave it commented for future tuning.
    #
    # if mask is not None and ((left_line is None) != (right_line is None)):
    #     left_line, right_line, hough_mode = first_hough_line_on_missing_side(
    #         mask,
    #         left_line,
    #         right_line,
    #         y_top,
    #         y_bottom,
    #         expected_width,
    #     )
    #     if hough_mode is None:
    #         left_line, right_line, hough_mode = first_component_line_on_missing_side(
    #             mask,
    #             left_line,
    #             right_line,
    #             y_top,
    #             y_bottom,
    #             expected_width,
    #         )

    if left_line is not None and right_line is not None:
        mode = hough_mode if hough_mode is not None else "fit"
    else:
        left_line, right_line, mode = reconstruct_missing_line_from_widths(
            left_line,
            right_line,
            y_top,
            y_bottom,
        )

    return left_line, right_line, left_inliers, right_inliers, mode


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


def reset_scanline_state():
    global LAST_GOOD_LANE_WIDTH_PX, LAST_GOOD_LANE_CENTER
    LAST_GOOD_LANE_WIDTH_PX = None
    LAST_GOOD_LANE_CENTER = None


def detect_scanline_birdeye_coords(frame_bgr, scan_count=DEFAULT_SCANLINE_COUNT, band_height=9):
    global LAST_GOOD_LANE_WIDTH_PX, LAST_GOOD_LANE_CENTER

    proc = processing_frame(frame_bgr)
    bird = warp_to_birdseye(proc)
    mask = white_mask_relative(bird)
    h, w = mask.shape[:2]

    expected_width = (max(x for x, _ in config.IPM_DST_FRAC) -
                      min(x for x, _ in config.IPM_DST_FRAC)) * w
    width_tolerance = expected_width * 0.28
    center_x = ((min(x for x, _ in config.IPM_DST_FRAC) +
                 max(x for x, _ in config.IPM_DST_FRAC)) / 2.0) * w

    y_top = int(h * config.LINE_TOP_FRAC)
    y_bottom = int(h * config.LINE_BOTTOM_FRAC)
    ys = np.linspace(y_bottom, y_top, scan_count).astype(int)

    accepted = []
    single_left_points = []
    single_right_points = []
    previous_pair = None
    anchor_pair = None

    for y in ys:
        y0 = max(0, y - band_height // 2)
        y1 = min(h, y + band_height // 2 + 1)
        band = mask[y0:y1, :]
        profile = smooth_1d(np.count_nonzero(band, axis=0), kernel_size=max(7, w // 35))
        pair = best_lane_pair(
            profile,
            expected_width,
            width_tolerance,
            center_x,
            previous_pair=previous_pair,
            anchor_pair=anchor_pair,
        )
        if pair is None:
            left_peak = best_single_side_peak(
                profile,
                center_x - expected_width / 2.0,
                center_x,
                want_left=True,
            )
            right_peak = best_single_side_peak(
                profile,
                center_x + expected_width / 2.0,
                center_x,
                want_left=False,
            )
            if left_peak is not None:
                single_left_points.append((left_peak[0], y))
            if right_peak is not None:
                single_right_points.append((right_peak[0], y))
            continue

        left = (pair["left"], y)
        right = (pair["right"], y)
        accepted.append((left, right, pair["score"]))
        previous_pair = (pair["left"], pair["right"])
        if anchor_pair is None:
            anchor_pair = previous_pair

    left_line, right_line, left_inliers, right_inliers, mode = fit_lane_lines_from_scanlines(
        accepted,
        single_left_points,
        single_right_points,
        y_top,
        y_bottom,
        expected_width,
        mask=mask,
    )

    if mode == "fit" and accepted:
        LAST_GOOD_LANE_WIDTH_PX = float(np.median([right[0] - left[0] for left, right, _ in accepted]))
        LAST_GOOD_LANE_CENTER = lane_center_geometry(left_line, right_line, y_top, y_bottom)

    return {
        "left_curve": left_line,
        "right_curve": right_line,
        "mode": mode,
        "left_inliers": len(left_inliers),
        "right_inliers": len(right_inliers),
    }


def analyze_frame(frame_bgr, scan_count=DEFAULT_SCANLINE_COUNT, band_height=9):
    global LAST_GOOD_LANE_WIDTH_PX, LAST_GOOD_LANE_CENTER

    proc = processing_frame(frame_bgr)
    bird = warp_to_birdseye(proc)
    mask = white_mask_relative(bird)
    h, w = mask.shape[:2]

    expected_width = (max(x for x, _ in config.IPM_DST_FRAC) -
                      min(x for x, _ in config.IPM_DST_FRAC)) * w
    width_tolerance = expected_width * 0.28
    center_x = ((min(x for x, _ in config.IPM_DST_FRAC) +
                 max(x for x, _ in config.IPM_DST_FRAC)) / 2.0) * w

    y_top = int(h * config.LINE_TOP_FRAC)
    y_bottom = int(h * config.LINE_BOTTOM_FRAC)
    ys = np.linspace(y_bottom, y_top, scan_count).astype(int)

    original_vis = proc.copy()
    vis = bird.copy()
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    accepted = []
    single_left_points = []
    single_right_points = []
    previous_pair = None
    anchor_pair = None

    for y in ys:
        y0 = max(0, y - band_height // 2)
        y1 = min(h, y + band_height // 2 + 1)
        band = mask[y0:y1, :]
        profile = smooth_1d(np.count_nonzero(band, axis=0), kernel_size=max(7, w // 35))
        pair = best_lane_pair(
            profile,
            expected_width,
            width_tolerance,
            center_x,
            previous_pair=previous_pair,
            anchor_pair=anchor_pair,
        )
        if pair is None:
            left_peak = best_single_side_peak(
                profile,
                center_x - expected_width / 2.0,
                center_x,
                want_left=True,
            )
            right_peak = best_single_side_peak(
                profile,
                center_x + expected_width / 2.0,
                center_x,
                want_left=False,
            )
            if left_peak is not None:
                single_left_points.append((left_peak[0], y))
            if right_peak is not None:
                single_right_points.append((right_peak[0], y))

        cv2.line(vis, (0, y), (w - 1, y), (80, 80, 80), 1)
        cv2.line(mask_vis, (0, y), (w - 1, y), (80, 80, 80), 1)

        if pair is None:
            continue

        left = (pair["left"], y)
        right = (pair["right"], y)
        accepted.append((left, right, pair["score"]))
        previous_pair = (pair["left"], pair["right"])
        if anchor_pair is None:
            anchor_pair = previous_pair

    left_line, right_line, left_inliers, right_inliers, center_mode = fit_lane_lines_from_scanlines(
        accepted,
        single_left_points,
        single_right_points,
        y_top,
        y_bottom,
        expected_width,
        mask=mask,
    )

    if center_mode == "fit" and accepted:
        LAST_GOOD_LANE_WIDTH_PX = float(np.median([right[0] - left[0] for left, right, _ in accepted]))
        LAST_GOOD_LANE_CENTER = lane_center_geometry(left_line, right_line, y_top, y_bottom)

    draw_line_clipped(vis, left_line, (0, 255, 255), 3)
    draw_line_clipped(vis, right_line, (0, 255, 255), 3)
    draw_line_clipped(mask_vis, left_line, (0, 255, 255), 2)
    draw_line_clipped(mask_vis, right_line, (0, 255, 255), 2)

    center_points, base_center_mode = center_waypoints(
        left_line,
        right_line,
        accepted,
        center_x,
        y_top,
        y_bottom,
    )
    if center_mode == "fit" and base_center_mode != "fit":
        center_mode = base_center_mode

    lane_polygon = lane_polygon_from_lines(left_line, right_line, y_top, y_bottom)
    if lane_polygon:
        vis = blend_lane_fill(vis, lane_polygon)
        mask_vis = blend_lane_fill(mask_vis, lane_polygon)

    fill_overlay = np.zeros_like(original_vis)
    if lane_polygon:
        cv2.fillPoly(fill_overlay, [np.asarray(lane_polygon, dtype=np.int32)], LANE_FILL_COLOR)
        _, bird_to_vehicle = get_matrices(w, h)
        fill_overlay = cv2.warpPerspective(fill_overlay, bird_to_vehicle, (w, h), flags=cv2.INTER_LINEAR)
        original_vis = blend_nonzero_overlay(original_vis, fill_overlay)

    draw_line_clipped(vis, left_line, (0, 255, 255), 3)
    draw_line_clipped(vis, right_line, (0, 255, 255), 3)
    draw_line_clipped(mask_vis, left_line, (0, 255, 255), 2)
    draw_line_clipped(mask_vis, right_line, (0, 255, 255), 2)

    draw_center_waypoints(vis, center_points)
    draw_center_waypoints(mask_vis, center_points)
    vehicle_center_points = project_bird_points_to_vehicle(center_points, w, h)
    draw_center_waypoints(original_vis, vehicle_center_points)

    for point in left_inliers:
        cv2.circle(vis, point, 7, (0, 255, 255), 1)
        cv2.circle(mask_vis, point, 4, (0, 0, 255), -1)
    for point in right_inliers:
        cv2.circle(vis, point, 7, (0, 255, 255), 1)
        cv2.circle(mask_vis, point, 4, (255, 0, 0), -1)

    for left, right, _ in accepted:
        if left in left_inliers and right in right_inliers:
            cv2.circle(vis, left, 4, (0, 0, 255), -1)
            cv2.circle(vis, right, 4, (255, 0, 0), -1)
            cv2.line(vis, left, right, (0, 255, 0), 1)

    cv2.putText(
        vis,
        f"scanlines={scan_count} accepted={len(accepted)} "
        f"inliers L/R={len(left_inliers)}/{len(right_inliers)} "
        f"center_points={len(center_points)} {center_mode} "
        f"expected_width={expected_width:.0f}px edge_ignore={EDGE_REJECT_MARGIN_PX}px "
        f"max_jump={MAX_SCANLINE_X_JUMP_PX}px",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    original_vis = label_panel(original_vis, "Original + center points")
    vis = label_panel(vis, "Bird's-eye scanlines")
    mask_vis = label_panel(mask_vis, "White mask")

    return np.hstack([original_vis, vis, mask_vis])


def export_preview(video_path, output_path, max_frames=240):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    count = 0

    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        out = analyze_frame(frame)
        if writer is None:
            h, w = out.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open writer for: {output_path}")
        writer.write(out)
        count += 1
        if count % 30 == 0:
            print(f"  {count}/{max_frames} frames", end="\r")

    cap.release()
    if writer is not None:
        writer.release()
    print(f"\nWrote {count} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?")
    parser.add_argument("--output", default="scanline_lane_experiment_preview.mp4")
    parser.add_argument("--max-frames", type=int, default=240)
    args = parser.parse_args()

    export_preview(resolve_video(args.video), Path(args.output), args.max_frames)


if __name__ == "__main__":
    main()
