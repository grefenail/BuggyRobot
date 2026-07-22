"""Frame preparation for the pipeline 2 scanline detector."""

import cv2
import numpy as np

import config
from ipm import warp_to_birdseye
from step1_mask import apply_white_mask_relative

RELATIVE_WHITE_TOP_PERCENT = 5


def processing_frame(frame_bgr):
    scale = float(config.PROCESS_SCALE)
    if abs(scale - 1.0) < 1e-6:
        return frame_bgr
    return cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def white_mask_hsv(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, config.WHITE_V_MIN), (180, config.WHITE_S_MAX, 255))


def white_mask_relative(frame_bgr):
    relative = apply_white_mask_relative(frame_bgr, RELATIVE_WHITE_TOP_PERCENT)
    return np.where(relative > 0, 255, 0).astype(np.uint8)


def prepare_scanline_frame(frame_bgr):
    proc = processing_frame(frame_bgr)
    bird = warp_to_birdseye(proc)
    mask = white_mask_relative(bird)
    return proc, bird, mask
