"""Step 1 — HSV white mask + grayscale.

Converts the frame to grayscale but only keeps pixels that are
white or light grey (lane paint, concrete curb). Everything else
(red track, green grass, sky, people) becomes black before Canny
so it cannot produce false edges.
"""
import cv2
import config
import numpy as np


def apply_white_mask(frame_bgr):
    """Return masked grayscale image — only white/light-grey pixels survive."""
    gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, config.WHITE_V_MIN), (180, config.WHITE_S_MAX, 255))
    return cv2.bitwise_and(gray, gray, mask=white_mask)

def apply_white_mask_relative(frame_bgr, top_percent=10):
    """
    Keep the brightest `top_percent` percent of pixels on each half of the image.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mid_x = gray.shape[1] // 2

    for x0, x1 in (
        (0, mid_x),
        (mid_x, gray.shape[1]),
    ):
        region = gray[:, x0:x1]
        flat = region.ravel()
        keep_count = int(flat.size * top_percent / 100)
        if keep_count <= 0:
            continue

        keep_count = min(keep_count, flat.size)

        # Pick exact pixel positions so equal-brightness ties do not keep too many pixels.
        keep_indices = np.argpartition(flat, flat.size - keep_count)[flat.size - keep_count:]

        ys, xs = np.unravel_index(keep_indices, region.shape)
        mask[ys, xs + x0] = 255

    if not np.any(mask):
        return np.zeros_like(gray)

    return cv2.bitwise_and(gray, gray, mask=mask)
