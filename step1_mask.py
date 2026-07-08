"""Step 1 — HSV white mask + grayscale.

Converts the frame to grayscale but only keeps pixels that are
white or light grey (lane paint, concrete curb). Everything else
(red track, green grass, sky, people) becomes black before Canny
so it cannot produce false edges.
"""
import cv2
import config


def apply_white_mask(frame_bgr):
    """Return masked grayscale image — only white/light-grey pixels survive."""
    gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, config.WHITE_V_MIN), (180, config.WHITE_S_MAX, 255))
    return cv2.bitwise_and(gray, gray, mask=white_mask)
