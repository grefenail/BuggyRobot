"""
Scanline lane pipeline.

This module exposes the same public API as ``pipeline/detect_lanes.py``,
but the detector backend is the scanline experiment. It is intended to be
used by a runner/ROS publisher exactly like the original pipeline:

  detect_lanes(frame_bgr)
  detect_lanes_with_coords(frame_bgr)
  detect_lanes_birdeye_coords(frame_bgr)
  detect_lanes_debug(frame_bgr)
"""

import cv2
import numpy as np

import config
from config import CENTER_WAYPOINT_COUNT, LANE_FILL_ALPHA, ROTATE_CW
from ipm import get_matrices, warp_to_birdseye, warp_to_vehicle

import scanline_backend as scanline


def _blend_nonzero_overlay(base, overlay, alpha=LANE_FILL_ALPHA):
    mask = np.any(overlay != 0, axis=2)
    if not np.any(mask):
        return base
    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
    out = base.copy()
    out[mask] = blended[mask]
    return out


def _processing_frame(frame_bgr):
    scale = float(config.PROCESS_SCALE)
    if scale <= 0:
        raise ValueError("PROCESS_SCALE must be greater than 0")
    if abs(scale - 1.0) < 1e-6:
        return frame_bgr
    return cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _line_points(line):
    if line is None:
        return None
    return [[int(x), int(y)] for x, y in line]


def _center_waypoints(left_line, right_line, y_top, y_bottom, count=CENTER_WAYPOINT_COUNT):
    return scanline.center_waypoints_from_lines(left_line, right_line, y_top, y_bottom, count)


def _lane_polygon(left_line, right_line, y_top, y_bottom):
    return scanline.lane_polygon_from_lines(left_line, right_line, y_top, y_bottom)


def _coords_from_scanline(bird_frame, detection):
    h, w = bird_frame.shape[:2]
    y_top = int(h * config.LINE_TOP_FRAC)
    y_bottom = int(h * config.LINE_BOTTOM_FRAC)
    left_line = detection.get("left_curve")
    right_line = detection.get("right_curve")
    center_waypoints = _center_waypoints(left_line, right_line, y_top, y_bottom)

    return {
        "bird_width": int(w),
        "bird_height": int(h),
        "process_scale": float(config.PROCESS_SCALE),
        "min_y": int(y_top),
        "max_y": int(y_bottom),
        "left_curve": _line_points(left_line),
        "right_curve": _line_points(right_line),
        "center_curve": [[int(x), int(y)] for x, y in center_waypoints] if center_waypoints else None,
        "center_waypoints_px": [[int(x), int(y)] for x, y in center_waypoints] if center_waypoints else None,
        "mode": detection.get("mode"),
        "left_inliers": int(detection.get("left_inliers", 0)),
        "right_inliers": int(detection.get("right_inliers", 0)),
    }


def _vehicle_overlay_for_detection(frame_shape, bird_shape, detection):
    h, w = bird_shape[:2]
    y_top = int(h * config.LINE_TOP_FRAC)
    y_bottom = int(h * config.LINE_BOTTOM_FRAC)
    polygon = _lane_polygon(detection.get("left_curve"), detection.get("right_curve"), y_top, y_bottom)

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if polygon:
        pts = np.asarray(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], config.LANE_FILL_COLOR)
        cv2.polylines(overlay, [np.asarray(detection["left_curve"], dtype=np.int32)], False, config.BORDER_COLOR, config.BORDER_WIDTH)
        cv2.polylines(overlay, [np.asarray(detection["right_curve"], dtype=np.int32)], False, config.BORDER_COLOR, config.BORDER_WIDTH)
        center = _center_waypoints(detection["left_curve"], detection["right_curve"], y_top, y_bottom)
        if len(center) >= 2:
            cv2.polylines(overlay, [np.asarray(center, dtype=np.int32)], False, config.CENTER_COLOR, config.CENTER_WIDTH)

    vehicle_overlay = warp_to_vehicle(overlay)
    target_h, target_w = frame_shape[:2]
    if vehicle_overlay.shape[:2] != (target_h, target_w):
        vehicle_overlay = cv2.resize(vehicle_overlay, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return vehicle_overlay


def _detect(frame_bgr):
    if ROTATE_CW:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    process_frame = _processing_frame(frame_bgr)
    bird_frame = warp_to_birdseye(process_frame)
    detection = scanline.detect_scanline_birdeye_coords(frame_bgr)
    return frame_bgr, process_frame, bird_frame, detection


def detect_lanes(frame_bgr):
    frame_bgr, _, bird_frame, detection = _detect(frame_bgr)
    overlay = _vehicle_overlay_for_detection(frame_bgr.shape, bird_frame.shape, detection)
    return _blend_nonzero_overlay(frame_bgr, overlay)


def detect_lanes_with_coords(frame_bgr):
    frame_bgr, _, bird_frame, detection = _detect(frame_bgr)
    overlay = _vehicle_overlay_for_detection(frame_bgr.shape, bird_frame.shape, detection)
    return _blend_nonzero_overlay(frame_bgr, overlay), _coords_from_scanline(bird_frame, detection)


def detect_lanes_birdeye_coords(frame_bgr):
    _, _, bird_frame, detection = _detect(frame_bgr)
    return _coords_from_scanline(bird_frame, detection)


def detect_lanes_debug(frame_bgr):
    if ROTATE_CW:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

    process_frame = _processing_frame(frame_bgr)
    bird_frame = warp_to_birdseye(process_frame)
    mask = scanline.white_mask_relative(bird_frame)
    edges = cv2.Canny(mask, config.CANNY_LOW, config.CANNY_HIGH)
    detection = scanline.detect_scanline_birdeye_coords(frame_bgr)
    overlay = _vehicle_overlay_for_detection(frame_bgr.shape, bird_frame.shape, detection)
    result = _blend_nonzero_overlay(frame_bgr, overlay)

    h, w = bird_frame.shape[:2]
    y_top = int(h * config.LINE_TOP_FRAC)
    y_bottom = int(h * config.LINE_BOTTOM_FRAC)
    bird_overlay = bird_frame.copy()
    polygon = _lane_polygon(detection.get("left_curve"), detection.get("right_curve"), y_top, y_bottom)
    if polygon:
        bird_overlay = scanline.blend_lane_fill(bird_overlay, polygon)
    scanline.draw_line_clipped(bird_overlay, detection.get("left_curve"), config.BORDER_COLOR, config.BORDER_WIDTH)
    scanline.draw_line_clipped(bird_overlay, detection.get("right_curve"), config.BORDER_COLOR, config.BORDER_WIDTH)
    center_points = _center_waypoints(detection.get("left_curve"), detection.get("right_curve"), y_top, y_bottom)
    scanline.draw_center_waypoints(bird_overlay, center_points)

    scanlines = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for y in np.linspace(y_bottom, y_top, scanline.DEFAULT_SCANLINE_COUNT).astype(int):
        cv2.line(scanlines, (0, int(y)), (w - 1, int(y)), (80, 80, 80), 1)
    scanline.draw_line_clipped(scanlines, detection.get("left_curve"), config.BORDER_COLOR, 2)
    scanline.draw_line_clipped(scanlines, detection.get("right_curve"), config.BORDER_COLOR, 2)
    scanline.draw_center_waypoints(scanlines, center_points)

    return result, mask, edges, scanlines, bird_overlay
