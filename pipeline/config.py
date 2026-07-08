"""All tunable parameters for the buggyvis1 lane detection pipeline.

This is a cleaned-up copy of test2 with every currently-disabled
code path removed (camera-space ROI hexagon, adaptive ROI, the
HLS+Sobel color/gradient threshold alternative, yellow detection) --
only what's actually active is left. See test2/ for the fuller,
still-tunable version with those toggles in place.
"""
import numpy as np

ROTATE_CW = True

# Step 1 — HSV white mask
WHITE_V_MIN = 175   # min brightness  (lane paint ~240+, concrete ~180)
WHITE_S_MAX = 75    # max saturation  (pure white = low S, red track = high S) -- was 45, but that's
                    # tuned for bright/direct sunlight only. In shadow (which tends to correlate
                    # with turns on this track), the same white paint reads as more saturated in
                    # HSV, and 45 was filtering the real line out almost entirely. Verified: on a
                    # shadowed turning frame, 45 kept only 4.3k mask pixels (line nearly invisible),
                    # 75 kept 25k (both lines fully recovered).

# Step 2 — Canny
CANNY_LOW  = 50
CANNY_HIGH = 150

# Step 0 — Inverse Perspective Mapping (vehicle view <-> bird's-eye view).
# The whole pipeline runs in bird's-eye coordinates: Hough + fitting
# happen on the warped image, then the result is warped back onto the
# camera view for display.
USE_BIRDSEYE_DEBUG = True

# Four source points in the corrected vehicle-view image, as fractions
# of (width, height), ordered bottom-left, top-left, top-right,
# bottom-right. Tune these around straight parallel lane/track lines.
#
# The top edge (y=0.45) is deliberately kept close to the bottom edge
# (y=0.78) rather than reaching further up the frame (e.g. 0.34) --
# IPM assumes a single straight lane, and a curving track violates
# that assumption more the further away (higher up the frame) you
# look. Staying in the near field, where the track is still close to
# straight over that short a distance, keeps the calibration much more
# accurate at the cost of a shorter look-ahead.
IPM_SRC_FRAC = (
    (0.10, 0.78),
    (0.25, 0.45),
    (0.75, 0.45),
    (0.90, 0.78),
)

# Matching destination points in the bird's-eye output image, same
# ordering. This makes the source trapezoid become a rectangle.
IPM_DST_FRAC = (
    (0.25, 0.95),
    (0.25, 0.05),
    (0.75, 0.05),
    (0.75, 0.95),
)
BIRDSEYE_ROI_TOP_MARGIN_FRAC    = 0.05  # Padding around the IPM destination rectangle's top edge
                                        # (far from camera).
BIRDSEYE_ROI_BOTTOM_MARGIN_FRAC = 0.20  # Padding around the destination rectangle's bottom edge
                                        # (closest to camera) -- wider than the top margin, since
                                        # IPM_SRC_FRAC undershoots the real lane width more near the
                                        # bottom on a curving section than at the top. Was 0.50, but
                                        # combined with IPM_DST_FRAC already being 50% of frame width,
                                        # that clamped the bottom span to the full frame width (100%)
                                        # instead of being a real, bounded margin -- 0.20 stays under
                                        # that clamp point (0.25) so it's an actual margin again.

# Step 4 — HoughLinesP
HOUGH_RHO        = 2
HOUGH_THETA      = np.pi / 60
HOUGH_THRESHOLD  = 60
HOUGH_MIN_LENGTH = 20
HOUGH_MAX_GAP    = 10

# Step 4 — segment classification
MIN_SLOPE = 0.5    # reject near-horizontal segments

# Step 5 — fit + smoothing
USE_ARC_FIT        = True   # False = straight line fit per side always.
                             # True  = also try a quadratic per side, but only trust it (draw a
                             # bend) when it's a significantly better fit than a line -- see
                             # BEND_SIGNIFICANCE_FRAC. No circle/shared-center model involved
                             # anymore (see step5_fit.py docstring for why that was dropped).
SMOOTH_ALPHA       = 0.13  # exponential smoothing (0=frozen, 1=raw)
STALE_RESET_FRAMES = 150   # clear stale lock after this many failed frames (~5s at 30fps --
                           # holds the last known position on screen while detection is briefly lost)
KEEP_LAST_LINE_ON_MISS = True  # When Hough finds no usable lane in a frame, keep drawing the
                               # last accepted lane instead of dropping or straightening it.
LINE_TOP_FRAC      = 0.34   # where the drawn line starts (top)
LINE_BOTTOM_FRAC   = 0.87   # where the drawn line ends (bottom)
ARC_MAX_JUMP_PX    = 60    # reject a single-frame fit that would move an already-locked edge
                           # more than this many px (raw per-frame fits are noisy; better to
                           # skip a bad frame than blend the outlier in)
ARC_MAX_JUMP_PX_FRESH = 300 # looser jump tolerance used while a lock is still fresh (see
                           # ARC_JUMP_CONFIRM_FRAMES) -- a lock formed from the first few noisy
                           # frames can be genuinely wrong by more than ARC_MAX_JUMP_PX, and the
                           # real position (e.g. early in a turn, moving fast in bird's-eye
                           # space) can legitimately shift faster than that too. Without this,
                           # the strict gate rejects every correction toward the real position
                           # as "noise" and freezes on the bad initial value until
                           # STALE_RESET_FRAMES forces a full reset.
ARC_JUMP_CONFIRM_FRAMES = 30 # how many consecutive confirmed frames (see _confirm_ct) a lock
                           # must hold before the strict ARC_MAX_JUMP_PX gate applies -- before
                           # that, ARC_MAX_JUMP_PX_FRESH is used instead, letting a fresh lock
                           # correct quickly toward the real position

BEND_SIGNIFICANCE_FRAC = 0.3 # a side is only drawn as bent (quadratic) when it reduces the
                           # fit residual by more than this fraction vs. a straight line --
                           # noise alone always improves a quadratic's residual a little, so
                           # this has to be a meaningful margin, not just "any improvement"
MAX_BEND_DEVIATION_PX = 200 # secondary guard: reject a sampled curve whose middle bulges
                           # further than this from the straight chord between its endpoints
                           # (catches a quadratic that extrapolates wildly between sample points)
MIN_LANE_WIDTH_PX = 100    # reject a fit where left/right converge to less than this far apart
                           # at either endpoint -- real lane lines never narrow to near-zero width,
                           # so this catches contaminated point data that technically doesn't
                           # cross (passes the old left<right check) but is still nonsense

# Step 6 — drawing
LANE_FILL_COLOR = (0, 200, 0)    # green fill
LANE_FILL_ALPHA = 0.40           # opacity
BORDER_COLOR    = (0, 255, 255)  # yellow border
BORDER_WIDTH    = 4

# Debug visualisation colours
LEFT_COLOR  = (0, 0, 255)
RIGHT_COLOR = (255, 0, 0)
