"""Step 4 — HoughLinesP detection and left/right classification.

Detects straight line segments in the ROI-masked (bird's-eye) edge
image, then splits them into left-lane and right-lane groups by
horizontal position relative to the lane center. (The camera-space
slope-sign classification used before bird's-eye was introduced has
been removed -- it's never reached now.)
"""
import cv2
import numpy as np
from config import (HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                    HOUGH_MIN_LENGTH, HOUGH_MAX_GAP,
                    MIN_SLOPE, LEFT_COLOR, RIGHT_COLOR)


def detect_segments(cropped, width, center_x=None):
    """
    Run Hough on the cropped edge image and classify segments.

    center_x -- the x position (pixels) to split left/right around.
    Defaults to width/2 (raw image center), but callers should pass
    the actual configured lane center (see ipm.birdseye_lane_center_frac)
    so a neighboring lane sitting in a widened ROI doesn't get
    misclassified as part of the current lane just because it happens
    to fall left/right of the image midpoint.

    Returns
    -------
    left_x, left_y   : point lists for left lane fit
    right_x, right_y : point lists for right lane fit
    hough_vis        : BGR debug image with coloured segments
    """
    lines = cv2.HoughLinesP(
        cropped,
        rho=HOUGH_RHO, theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD, lines=np.array([]),
        minLineLength=HOUGH_MIN_LENGTH, maxLineGap=HOUGH_MAX_GAP,
    )

    left_x,  left_y  = [], []
    right_x, right_y = [], []
    hough_vis = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    if lines is None:
        return left_x, left_y, right_x, right_y, hough_vis

    split_x = center_x if center_x is not None else width / 2

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1
            dy = y2 - y1

            # In bird's-eye view valid lane markings are close to
            # vertical, so classify by horizontal position instead of
            # slope sign.
            if abs(dy) < abs(dx) * MIN_SLOPE:
                cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                continue
            if (x1 + x2) / 2 < split_x:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
                cv2.line(hough_vis, (x1, y1), (x2, y2), LEFT_COLOR, 2)
            else:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])
                cv2.line(hough_vis, (x1, y1), (x2, y2), RIGHT_COLOR, 2)

    return left_x, left_y, right_x, right_y, hough_vis
