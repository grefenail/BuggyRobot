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

def apply_white_mask_relative(frame_bgr, top_percent=5):
    """
    Keep the brightest `top_percent` percent of pixels.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    flat = gray.ravel()
    keep_count = int(flat.size * top_percent / 100)
    if keep_count <= 0:
        return np.zeros_like(gray)

    keep_count = min(keep_count, flat.size)

    # Pick exact pixel positions so equal-brightness ties do not keep too many pixels.
    keep_indices = np.argpartition(flat, flat.size - keep_count)[flat.size - keep_count:]

    mask = np.zeros_like(flat, dtype=np.uint8)
    mask[keep_indices] = 255
    assert np.count_nonzero(mask) == keep_count
    mask = mask.reshape(gray.shape)

    return cv2.bitwise_and(gray, gray, mask=mask)
