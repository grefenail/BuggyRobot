"""Pipeline 2 scanline lane detector orchestration.

The detector is split into focused modules:

- scanline_preprocess.py: resize, bird's-eye warp, white mask
- scanline_peaks.py: scanline profile peaks and left/right pairing
- scanline_lines.py: line fitting and lane geometry
- scanline_visual.py: overlays, center waypoints, preview drawing

This file keeps the public API used by run_pipeline2.py and detect_lanes.py.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import config
from ipm import get_matrices
from scanline_lines import (
    fit_consensus_straight_line,
    fit_from_pairs,
    lane_center_geometry,
    line_from_top_bottom_x,
    line_is_plausible,
    line_x_at_y,
)
from scanline_peaks import (
    EDGE_REJECT_MARGIN_PX,
    MAX_SCANLINE_X_JUMP_PX,
    best_lane_pair,
    best_single_side_peak,
    smooth_1d,
)
from scanline_preprocess import (
    prepare_scanline_frame,
    processing_frame,
    white_mask_hsv,
    white_mask_relative,
)
from scanline_visual import (
    LANE_FILL_COLOR,
    blend_lane_fill,
    blend_nonzero_overlay,
    center_waypoints,
    center_waypoints_from_lines,
    draw_center_waypoints,
    draw_line_clipped,
    label_panel,
    lane_polygon_from_lines,
    project_bird_points_to_vehicle,
)

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "vids"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
DEFAULT_SCANLINE_COUNT = 30
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


def scanline_geometry(mask_shape, scan_count):
    h, w = mask_shape[:2]
    expected_width = (max(x for x, _ in config.IPM_DST_FRAC) -
                      min(x for x, _ in config.IPM_DST_FRAC)) * w
    width_tolerance = expected_width * 0.28
    center_x = ((min(x for x, _ in config.IPM_DST_FRAC) +
                 max(x for x, _ in config.IPM_DST_FRAC)) / 2.0) * w
    y_top = int(h * config.LINE_TOP_FRAC)
    y_bottom = int(h * config.LINE_BOTTOM_FRAC)

    return {
        "height": h,
        "width": w,
        "expected_width": expected_width,
        "width_tolerance": width_tolerance,
        "center_x": center_x,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "scanline_ys": np.linspace(y_bottom, y_top, scan_count).astype(int),
    }


def collect_scanline_points(mask, geometry, band_height=9, on_scanline=None):
    accepted = []
    single_left_points = []
    single_right_points = []
    previous_pair = None
    anchor_pair = None
    w = geometry["width"]

    for y in geometry["scanline_ys"]:
        y0 = max(0, y - band_height // 2)
        y1 = min(geometry["height"], y + band_height // 2 + 1)
        band = mask[y0:y1, :]
        profile = smooth_1d(np.count_nonzero(band, axis=0), kernel_size=max(7, w // 35))
        pair = best_lane_pair(
            profile,
            geometry["expected_width"],
            geometry["width_tolerance"],
            geometry["center_x"],
            previous_pair=previous_pair,
            anchor_pair=anchor_pair,
        )

        if pair is None:
            left_peak = best_single_side_peak(
                profile,
                geometry["center_x"] - geometry["expected_width"] / 2.0,
                geometry["center_x"],
                want_left=True,
            )
            right_peak = best_single_side_peak(
                profile,
                geometry["center_x"] + geometry["expected_width"] / 2.0,
                geometry["center_x"],
                want_left=False,
            )
            if left_peak is not None:
                single_left_points.append((left_peak[0], y))
            if right_peak is not None:
                single_right_points.append((right_peak[0], y))
        else:
            left = (pair["left"], y)
            right = (pair["right"], y)
            accepted.append((left, right, pair["score"]))
            previous_pair = (pair["left"], pair["right"])
            if anchor_pair is None:
                anchor_pair = previous_pair

        if on_scanline is not None:
            on_scanline(y, pair)

    return accepted, single_left_points, single_right_points


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


def fit_lane_lines_from_scanlines(
    accepted,
    single_left_points,
    single_right_points,
    y_top,
    y_bottom,
    expected_width,
    mask=None,
):
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

    if left_line is not None and not line_is_plausible(left_line):
        left_line = None
        left_inliers = []
    if right_line is not None and not line_is_plausible(right_line):
        right_line = None
        right_inliers = []

    if left_line is not None and right_line is not None:
        mode = "fit"
    else:
        left_line, right_line, mode = reconstruct_missing_line_from_widths(
            left_line,
            right_line,
            y_top,
            y_bottom,
        )

    return left_line, right_line, left_inliers, right_inliers, mode


def fit_detection_from_points(accepted, single_left_points, single_right_points, geometry, mask):
    left_line, right_line, left_inliers, right_inliers, mode = fit_lane_lines_from_scanlines(
        accepted,
        single_left_points,
        single_right_points,
        geometry["y_top"],
        geometry["y_bottom"],
        geometry["expected_width"],
        mask=mask,
    )

    return {
        "left_curve": left_line,
        "right_curve": right_line,
        "mode": mode,
        "left_inliers": left_inliers,
        "right_inliers": right_inliers,
    }


def update_lane_memory(detection, accepted, geometry):
    global LAST_GOOD_LANE_WIDTH_PX, LAST_GOOD_LANE_CENTER

    if detection["mode"] != "fit" or not accepted:
        return

    LAST_GOOD_LANE_WIDTH_PX = float(np.median([right[0] - left[0] for left, right, _ in accepted]))
    LAST_GOOD_LANE_CENTER = lane_center_geometry(
        detection["left_curve"],
        detection["right_curve"],
        geometry["y_top"],
        geometry["y_bottom"],
    )


def run_scanline_steps(frame_bgr, scan_count=DEFAULT_SCANLINE_COUNT, band_height=9, on_scanline=None):
    proc, bird, mask = prepare_scanline_frame(frame_bgr)
    geometry = scanline_geometry(mask.shape, scan_count)
    accepted, single_left_points, single_right_points = collect_scanline_points(
        mask,
        geometry,
        band_height=band_height,
        on_scanline=on_scanline,
    )
    detection = fit_detection_from_points(
        accepted,
        single_left_points,
        single_right_points,
        geometry,
        mask,
    )
    update_lane_memory(detection, accepted, geometry)
    return proc, bird, mask, geometry, accepted, detection


def reset_scanline_state():
    global LAST_GOOD_LANE_WIDTH_PX, LAST_GOOD_LANE_CENTER
    LAST_GOOD_LANE_WIDTH_PX = None
    LAST_GOOD_LANE_CENTER = None


def detect_scanline_birdeye_coords(frame_bgr, scan_count=DEFAULT_SCANLINE_COUNT, band_height=9):
    _, _, _, _, _, detection = run_scanline_steps(frame_bgr, scan_count, band_height)

    return {
        "left_curve": detection["left_curve"],
        "right_curve": detection["right_curve"],
        "mode": detection["mode"],
        "left_inliers": len(detection["left_inliers"]),
        "right_inliers": len(detection["right_inliers"]),
    }


def analyze_frame(frame_bgr, scan_count=DEFAULT_SCANLINE_COUNT, band_height=9):
    h = w = 0

    def draw_sampled_scanline(y, _pair):
        cv2.line(vis, (0, y), (w - 1, y), (80, 80, 80), 1)
        cv2.line(mask_vis, (0, y), (w - 1, y), (80, 80, 80), 1)

    proc, bird, mask = prepare_scanline_frame(frame_bgr)
    h, w = mask.shape[:2]
    original_vis = proc.copy()
    vis = bird.copy()
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    geometry = scanline_geometry(mask.shape, scan_count)
    accepted, single_left_points, single_right_points = collect_scanline_points(
        mask,
        geometry,
        band_height=band_height,
        on_scanline=draw_sampled_scanline,
    )
    detection = fit_detection_from_points(
        accepted,
        single_left_points,
        single_right_points,
        geometry,
        mask,
    )
    update_lane_memory(detection, accepted, geometry)

    y_top = geometry["y_top"]
    y_bottom = geometry["y_bottom"]
    left_line = detection["left_curve"]
    right_line = detection["right_curve"]
    left_inliers = detection["left_inliers"]
    right_inliers = detection["right_inliers"]
    center_mode = detection["mode"]

    draw_line_clipped(vis, left_line, (0, 255, 255), 3)
    draw_line_clipped(vis, right_line, (0, 255, 255), 3)
    draw_line_clipped(mask_vis, left_line, (0, 255, 255), 2)
    draw_line_clipped(mask_vis, right_line, (0, 255, 255), 2)

    center_points, base_center_mode = center_waypoints(
        left_line,
        right_line,
        accepted,
        geometry["center_x"],
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
        f"expected_width={geometry['expected_width']:.0f}px edge_ignore={EDGE_REJECT_MARGIN_PX}px "
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
