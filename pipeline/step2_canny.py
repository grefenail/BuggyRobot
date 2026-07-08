"""Step 2 — Canny edge detection."""

import cv2
import config


def apply_canny(gray_masked):
    """Return binary edge image from the masked grayscale."""
    return cv2.Canny(gray_masked, config.CANNY_LOW, config.CANNY_HIGH)
