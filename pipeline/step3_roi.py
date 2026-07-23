"""Step 3 — Region of Interest masking (bird's-eye space).

Only the bird's-eye ROI is kept here -- the pipeline always runs in
bird's-eye coordinates (see config.py), so the camera-space hexagon,
the adaptive curve-hugging shape, and the full-frame fallback that
existed alongside it are unused and have been removed.
"""
import cv2
import numpy as np
import config


def build_birdseye_vertices(height, width):
    """Trapezoidal ROI for bird's-eye coordinates.

    In a perfectly-calibrated bird's-eye view, real lane lines would
    be parallel and a plain rectangle would be the right search shape.
    In practice IPM_SRC_FRAC is a single straight-lane approximation
    that doesn't perfectly hold on a curving section -- the real lines
    still drift somewhat instead of staying perfectly vertical, and
    that residual drift is worse near the bottom (closest to the
    camera) than the top. A trapezoid with a wider bottom margin than
    top hugs that actual behavior far better than a fixed-width
    rectangle, which either clips the bottom or wastes search area at
    the top.
    """
    xs = [p[0] for p in config.IPM_DST_FRAC]
    ys = [p[1] for p in config.IPM_DST_FRAC]
    top_margin = config.BIRDSEYE_ROI_TOP_MARGIN_FRAC
    bottom_margin = config.BIRDSEYE_ROI_BOTTOM_MARGIN_FRAC
    x_left_top = int(max(0.0, min(xs) - top_margin) * width)
    x_right_top = int(min(1.0, max(xs) + top_margin) * width)
    x_left_bottom = int(max(0.0, min(xs) - bottom_margin) * width)
    x_right_bottom = int(min(1.0, max(xs) + bottom_margin) * width)
    y_top = int(max(0.0, min(ys)) * height)
    y_bottom = int(min(1.0, max(ys)) * height)
    return np.array([[
        (x_left_bottom, y_bottom),
        (x_left_top, y_top),
        (x_right_top, y_top),
        (x_right_bottom, y_bottom),
    ]], dtype=np.int32)


def apply_birdseye_roi(edges, height, width):
    if config.DISABLE_ROI_CROP:
        # Full-frame rectangle -- no masking at all, just used so the debug
        # overlay (cv2.polylines on `vertices`) still draws something
        # meaningful instead of a stale trapezoid.
        full_vertices = np.array([[
            (0, height - 1),
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
        ]], dtype=np.int32)
        return edges, full_vertices
    vertices = build_birdseye_vertices(height, width)
    mask     = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask), vertices
