"""ROI selection for the buggyvis1 lane pipeline.

The pipeline always runs in bird's-eye space with a static (non-
adaptive) trapezoid ROI (see config.py: USE_ADAPTIVE_ROI is off, and
the camera-space hexagon / full-frame fallback / recovery-hold
machinery that used to sit alongside this are unused and have been
removed).
"""

from step3_roi import apply_birdseye_roi
from step4_hough import detect_segments
from ipm import birdseye_lane_center_frac


def detect_with_stable_roi(edges, height, width):
    """Build the bird's-eye ROI and run Hough within it."""
    roi, vertices = apply_birdseye_roi(edges, height, width)
    center_x = birdseye_lane_center_frac() * width
    lx, ly, rx, ry, hough_vis = detect_segments(roi, width, center_x=center_x)
    return lx, ly, rx, ry, hough_vis, roi, vertices
