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

VIDEO_DIR = ROOT / "vids"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
EDGE_REJECT_MARGIN_PX = 5
DEFAULT_SCANLINE_COUNT = 30
MAX_SCANLINE_X_JUMP_PX = 35
SCANLINE_X_JUMP_PENALTY = 5.0
MIN_STRAIGHT_LINE_INLIERS = 8
STRAIGHT_LINE_TOLERANCE_PX = 12
CENTER_WAYPOINT_COUNT = 10
LANE_FILL_COLOR = (0, 180, 0)
LANE_FILL_ALPHA = 0.28
LAST_GOOD_LANE_WIDTH_PX = None


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


def smooth_1d(values, kernel_size=21):
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def local_peaks(values, min_score, edge_margin=0):
    peaks = []
    x_start = max(1, edge_margin)
    x_end = min(len(values) - 1, len(values) - edge_margin)
    for x in range(x_start, x_end):
        if values[x] >= min_score and values[x] >= values[x - 1] and values[x] >= values[x + 1]:
            peaks.append((x, float(values[x])))
    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks


def best_lane_pair(
    profile,
    expected_width,
    width_tolerance,
    center_x,
    previous_pair=None,
    edge_margin=EDGE_REJECT_MARGIN_PX,
):
    min_score = max(3.0, float(profile.max()) * 0.25)
    peaks = local_peaks(profile, min_score, edge_margin=edge_margin)[:12]
    best = None

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
            continuity_error = 0.0
            if previous_pair is not None:
                left_jump = abs(li - previous_pair[0])
                right_jump = abs(ri - previous_pair[1])
                if left_jump > MAX_SCANLINE_X_JUMP_PX or right_jump > MAX_SCANLINE_X_JUMP_PX:
                    continue
                continuity_error = left_jump + right_jump
            score = (
                left_score + right_score
                - width_error * 0.6
                - center_error * 0.15
                - continuity_error * SCANLINE_X_JUMP_PENALTY
            )
            if best is None or score > best["score"]:
                best = {
                    "left": li,
                    "right": ri,
                    "score": score,
                    "left_score": left_score,
                    "right_score": right_score,
                    "width": lane_width,
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


def offset_line(line, dx):
    if line is None:
        return None
    return [(int(round(x + dx)), int(y)) for x, y in line]


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

    anchor_x = fallback_center_anchor(accepted, default_center_x)
    return center_waypoints_from_anchor(anchor_x, y_top, y_bottom), "fallback"


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


def draw_vertical_fallback_fill(img, anchor_x, lane_width, y_top, y_bottom):
    half_width = lane_width / 2.0
    polygon = [
        (int(round(anchor_x - half_width)), int(round(y_bottom))),
        (int(round(anchor_x - half_width)), int(round(y_top))),
        (int(round(anchor_x + half_width)), int(round(y_top))),
        (int(round(anchor_x + half_width)), int(round(y_bottom))),
    ]
    return blend_lane_fill(img, polygon), polygon


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
    global LAST_GOOD_LANE_WIDTH_PX
    LAST_GOOD_LANE_WIDTH_PX = None


def detect_scanline_birdeye_coords(frame_bgr, scan_count=DEFAULT_SCANLINE_COUNT, band_height=9):
    global LAST_GOOD_LANE_WIDTH_PX

    proc = processing_frame(frame_bgr)
    bird = warp_to_birdseye(proc)
    mask = white_mask_hsv(bird)
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

    left_points = [left for left, _, _ in accepted]
    right_points = [right for _, right, _ in accepted]
    left_line, left_inliers = fit_consensus_straight_line(left_points, y_top, y_bottom)
    right_line, right_inliers = fit_consensus_straight_line(right_points, y_top, y_bottom)
    mode = "fit" if left_line is not None and right_line is not None else "fallback"

    if left_line is not None and right_line is not None and accepted:
        LAST_GOOD_LANE_WIDTH_PX = float(np.median([right[0] - left[0] for left, right, _ in accepted]))
    elif left_line is None and right_line is None:
        width_for_one_side = LAST_GOOD_LANE_WIDTH_PX if LAST_GOOD_LANE_WIDTH_PX is not None else expected_width
        single_left_line, single_left_inliers = fit_consensus_straight_line(
            single_left_points,
            y_top,
            y_bottom,
        )
        single_right_line, single_right_inliers = fit_consensus_straight_line(
            single_right_points,
            y_top,
            y_bottom,
        )

        if len(single_left_inliers) >= len(single_right_inliers) and single_left_line is not None:
            left_line = single_left_line
            right_line = offset_line(single_left_line, width_for_one_side)
            left_inliers = single_left_inliers
            right_inliers = []
            mode = "one-side-left"
        elif single_right_line is not None:
            right_line = single_right_line
            left_line = offset_line(single_right_line, -width_for_one_side)
            right_inliers = single_right_inliers
            left_inliers = []
            mode = "one-side-right"

    return {
        "left_curve": left_line,
        "right_curve": right_line,
        "mode": mode,
        "left_inliers": len(left_inliers),
        "right_inliers": len(right_inliers),
    }


def analyze_frame(frame_bgr, scan_count=DEFAULT_SCANLINE_COUNT, band_height=9):
    global LAST_GOOD_LANE_WIDTH_PX

    proc = processing_frame(frame_bgr)
    bird = warp_to_birdseye(proc)
    mask = white_mask_hsv(bird)
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
        cv2.circle(vis, left, 4, (0, 0, 255), -1)
        cv2.circle(vis, right, 4, (255, 0, 0), -1)
        cv2.line(vis, left, right, (0, 255, 0), 1)
        cv2.circle(mask_vis, left, 4, (0, 0, 255), -1)
        cv2.circle(mask_vis, right, 4, (255, 0, 0), -1)

    left_points = [left for left, _, _ in accepted]
    right_points = [right for _, right, _ in accepted]
    left_line, left_inliers = fit_consensus_straight_line(left_points, y_top, y_bottom)
    right_line, right_inliers = fit_consensus_straight_line(right_points, y_top, y_bottom)
    center_mode = "fit" if left_line is not None and right_line is not None else "fallback"

    if left_line is not None and right_line is not None and accepted:
        LAST_GOOD_LANE_WIDTH_PX = float(np.median([right[0] - left[0] for left, right, _ in accepted]))
    elif left_line is None and right_line is None:
        width_for_one_side = LAST_GOOD_LANE_WIDTH_PX if LAST_GOOD_LANE_WIDTH_PX is not None else expected_width
        single_left_line, single_left_inliers = fit_consensus_straight_line(
            single_left_points,
            y_top,
            y_bottom,
        )
        single_right_line, single_right_inliers = fit_consensus_straight_line(
            single_right_points,
            y_top,
            y_bottom,
        )

        if len(single_left_inliers) >= len(single_right_inliers) and single_left_line is not None:
            left_line = single_left_line
            right_line = offset_line(single_left_line, width_for_one_side)
            left_inliers = single_left_inliers
            right_inliers = []
            center_mode = "one-side-left"
        elif single_right_line is not None:
            right_line = single_right_line
            left_line = offset_line(single_right_line, -width_for_one_side)
            right_inliers = single_right_inliers
            left_inliers = []
            center_mode = "one-side-right"

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
    else:
        anchor_x = fallback_center_anchor(accepted, center_x)
        vis, lane_polygon = draw_vertical_fallback_fill(
            vis,
            anchor_x,
            expected_width,
            y_top,
            y_bottom,
        )
        mask_vis, _ = draw_vertical_fallback_fill(
            mask_vis,
            anchor_x,
            expected_width,
            y_top,
            y_bottom,
        )

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
    for point in right_inliers:
        cv2.circle(vis, point, 7, (0, 255, 255), 1)

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
