"""Inverse Perspective Mapping utilities."""

import cv2
import numpy as np
import config


def _scaled_points(points_frac, width, height):
    return np.array(
        [(x * width, y * height) for x, y in points_frac],
        dtype=np.float32,
    )


def birdseye_lane_center_frac():
    """
    X-fraction of the expected single-lane center in bird's-eye space,
    derived from IPM_DST_FRAC -- used to split Hough segments into
    left/right by proximity to the actual configured lane, instead of
    the raw image midpoint (which is only correct by coincidence when
    IPM_DST_FRAC happens to be symmetric, and doesn't help at all when
    another lane's line has drifted into a widened ROI).
    """
    xs = [p[0] for p in config.IPM_DST_FRAC]
    return (min(xs) + max(xs)) / 2.0


def get_points(width, height):
    """Return (source_points, destination_points) in pixel coordinates."""
    src = _scaled_points(config.IPM_SRC_FRAC, width, height)
    dst = _scaled_points(config.IPM_DST_FRAC, width, height)
    return src, dst


def get_matrices(width, height):
    """Return (vehicle_to_bird_matrix, bird_to_vehicle_matrix)."""
    src, dst = get_points(width, height)
    vehicle_to_bird = cv2.getPerspectiveTransform(src, dst)
    bird_to_vehicle = cv2.getPerspectiveTransform(dst, src)
    return vehicle_to_bird, bird_to_vehicle


def warp_to_birdseye(image):
    """Warp a vehicle-view image into bird's-eye view."""
    height, width = image.shape[:2]
    vehicle_to_bird, _ = get_matrices(width, height)
    return cv2.warpPerspective(image, vehicle_to_bird, (width, height), flags=cv2.INTER_LINEAR)


def warp_to_vehicle(image):
    """Warp a bird's-eye image back into vehicle view."""
    height, width = image.shape[:2]
    _, bird_to_vehicle = get_matrices(width, height)
    return cv2.warpPerspective(image, bird_to_vehicle, (width, height), flags=cv2.INTER_LINEAR)


def draw_destination_overlay(image):
    """Draw the bird's-eye destination rectangle for visual tuning."""
    height, width = image.shape[:2]
    _, dst = get_points(width, height)
    out = image.copy()
    cv2.polylines(out, [dst.astype(np.int32)], True, (255, 0, 255), 2)
    for i, (x, y) in enumerate(dst.astype(np.int32)):
        cv2.circle(out, (x, y), 5, (255, 0, 255), -1)
        cv2.putText(out, f"D{i}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), 1, cv2.LINE_AA)
    return out
