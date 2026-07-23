"""All tunable parameters for the buggyvis1 lane detection pipeline.

This is a cleaned-up copy of test2 with every currently-disabled
code path removed (camera-space ROI hexagon, adaptive ROI, the
HLS+Sobel color/gradient threshold alternative, yellow detection) --
only what's actually active is left. See test2/ for the fuller,
still-tunable version with those toggles in place.
"""
import numpy as np

ROTATE_CW = False

# Resize frames before the expensive OpenCV pipeline. 0.5 turns 1280x720
# into 640x360, which is much easier for a Raspberry Pi to process.
PROCESS_SCALE = 1.0

GAUSSIAN_BLUR_KERNEL_SIZE = 3  # must be odd, >1. Larger = more smoothing, but slower and more

# The absolute-pixel constants below (HOUGH_MIN_LENGTH, MIN_LANE_WIDTH_PX,
# ARC_MAX_JUMP_PX, MAX_BEND_DEVIATION_PX, ...) were tuned by eye against a
# 640x360 processing frame (1280x720 native x 0.5 PROCESS_SCALE). They are
# NOT resolution-independent on their own: PROCESS_SCALE alone only
# compensates for changing that multiplier, not for the native camera feed
# itself changing resolution (e.g. a 640x360 native feed instead of
# 1280x720 produces a 320x180 processing frame at the same PROCESS_SCALE).
# _scaled_px() in step4_hough.py/step5_fit.py rescales these constants
# proportionally to the actual processing-frame width at runtime, using
# this as the reference they were tuned against.
REFERENCE_PROCESSING_WIDTH = 640

# Step 1 — HSV white mask

# WARNING: WE USE RELATIVE FILTER INSTEAD!
WHITE_V_MIN = 20   # min brightness  (lane paint ~240+, concrete ~180)
WHITE_S_MAX = 15    # max saturation  (pure white = low S, red track = high S) -- was 45, but that's
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
# The top edge is deliberately kept close to the bottom edge rather
# than reaching further up the frame -- IPM assumes a single straight
# lane, and a curving track violates that assumption more the further
# away (higher up the frame) you look. Staying in the near field,
# where the track is still close to straight over that short a
# distance, keeps the calibration much more accurate at the cost of a
# shorter look-ahead.
#
# Recalibrated for the Gazebo sim camera (640x360 native) by picking
# corners of a straight track section directly: bottom-left (128,356),
# top-left (289,205), top-right (342,205), bottom-right (485,356).
"""
IPM_SRC_FRAC = (
    (0.2000, 0.9889),
    (0.4516, 0.5694),
    (0.5344, 0.5694),
    (0.7578, 0.9889),
)
"""

# Widened horizontally from the original (128,356)-(289,205)-(342,205)-
# (485,356) calibration to bring more of the track's width into the
# bird's-eye view instead of discarding most of it -- same near/far
# look-ahead (y=356/205), wider corners: bottom (30,356)-(610,356),
# top (250,205)-(390,205).

# HIL:
"""
IPM_SRC_FRAC = (
    (0.0479, 0.9889),
    (0.4006, 0.5694),
    (0.5894, 0.5694),
    (0.9531, 0.9889),
)
"""
RESOLUTION = (640, 480)
IPM_SRC_ABS=(
    (12, 430),  
    (250, 200),
    (400, 200),
    (640, 430),
)

# real
IPM_SRC_FRAC = relative = tuple(
    (x / RESOLUTION[0], y / RESOLUTION[1])
    for x, y in IPM_SRC_ABS
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

# When True, skip the ROI trapezoid mask in step3_roi.py entirely and let
# Hough search the full post-IPM-warp edge image instead of just the band
# around BIRDSEYE_ROI_*_MARGIN_FRAC. Useful while (re)calibrating
# IPM_SRC_FRAC/IPM_DST_FRAC, since a stale/mismatched ROI can otherwise mask
# out a real line before you can tell whether the warp itself is right.
DISABLE_ROI_CROP = True

# Step 4 — HoughLinesP
HOUGH_RHO        = 2
HOUGH_THETA      = np.pi / 60
HOUGH_THRESHOLD  = 60
HOUGH_MIN_LENGTH = 20
HOUGH_MAX_GAP    = 10

# Step 4 — multi-line clustering
# Two Hough segments whose nearest endpoints are more than this many px
# apart are treated as different real-world lines rather than pieces of
# the same one. Matters once IPM_SRC_FRAC is wide enough to expose more
# than 2 lines at once (e.g. multiple track lanes), and in sharp curves
# where a single line's x position varies a lot across y -- without
# clustering, every segment on one side of center_x gets pooled into a
# single fit even if it comes from a different line, producing a
# crossing/nonsense fit (see step5_fit.py's crossing/width sanity check).
# Endpoint-proximity based (see step4_hough._cluster_by_adjacency), not
# tied to any single reference y, so it keeps a continuously-curving
# line's segments grouped correctly instead of splitting/merging them
# based on where they'd extrapolate to far away. Kept smaller than
# MIN_LANE_WIDTH_PX (below) so two adjacent real lines are never merged
# into a single cluster.
LANE_CLUSTER_ADJACENCY_PX = 30

# Step 4 — segment classification
MIN_SLOPE = 0.5    # reject near-horizontal segments

# Step 5 — fit + smoothing
USE_ARC_FIT        = True   # False = straight line fit per side always.
                             # True  = also try a quadratic per side, but only trust it (draw a
                             # bend) when it's a significantly better fit than a line (see
                             # BEND_SIGNIFICANCE_FRAC) AND the supporting points span enough of
                             # the target y-range to extrapolate safely (see
                             # MIN_ARC_Y_SPAN_FRAC). No circle/shared-center model involved
                             # anymore (see step5_fit.py docstring for why that was dropped).
                             # Was temporarily disabled after a 3-parameter quadratic fit through
                             # sparse, narrow-y-range Hough points swung wildly when evaluated at
                             # min_y/max_y well outside that range (observed: endpoints clamped
                             # to the frame edge, e.g. x=639, instead of a real curve). Re-enabled
                             # now that MIN_ARC_Y_SPAN_FRAC gates the bend on the data actually
                             # covering enough of the extrapolation range, not just point count.
MIN_ARC_Y_SPAN_FRAC = 0.5    # a quadratic bend is only trusted when (max(ys)-min(ys)) covers at
                             # least this fraction of (max_y-min_y) -- i.e. the points actually
                             # span half the near-to-far look-ahead distance before a curve is
                             # drawn, instead of a handful of nearby points getting extrapolated
                             # across the whole range. Below this, falls back to the line fit.
SMOOTH_ALPHA       = 0.8  # exponential smoothing (0=frozen, 1=raw)
STALE_RESET_FRAMES = 150   # clear stale lock after this many failed frames (~5s at 30fps --
                           # holds the last known position on screen while detection is briefly lost)
KEEP_LAST_LINE_ON_MISS = True  # When Hough finds no usable lane in a frame, keep drawing the
                               # last accepted lane instead of dropping or straightening it.
LINE_TOP_FRAC      = 0.05   # where the drawn line starts (top) -- match IPM_DST_FRAC top edge
LINE_BOTTOM_FRAC   = 0.95   # where the drawn line ends (bottom) -- match IPM_DST_FRAC bottom edge
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
MIN_LANE_WIDTH_PX = 80     # reject a fit where left/right converge to less than this far apart
                           # at either endpoint -- real lane lines never narrow to near-zero width,
                           # so this catches contaminated point data that technically doesn't
                           # cross (passes the old left<right check) but is still nonsense

# Step 6 — drawing
LANE_FILL_COLOR = (0, 200, 0)    # green fill
LANE_FILL_ALPHA = 0.40           # opacity
BORDER_COLOR    = (0, 255, 255)  # yellow border
BORDER_WIDTH    = 4
CENTER_COLOR    = (255, 255, 255)  # white centerline
CENTER_WIDTH    = 3
CENTER_DOT_COLOR = (0, 165, 255)   # orange sampled center waypoints
CENTER_DOT_RADIUS = 9
CENTER_LABEL_COLOR = (255, 255, 255)
CENTER_LABEL_SCALE = 0.85
CENTER_LABEL_THICKNESS = 2
CENTER_WAYPOINT_COUNT = 10          # sampled centerline points for coordinate export

# # Camera model for the Gazebo sim's front_camera sensor (640x360,
# # horizontal_fov=1.39626 rad/80 deg, mounted at chassis-local pose
# # 1.1 0 0.35 0 0 0 -- i.e. zero pitch/roll/yaw relative to the chassis).
# # Was previously calibrated for a real 720x1280 iPhone 16 at fx=fy=880 and
# # an assumed 15 deg downward pitch, neither of which applies to this sim
# # camera:
# #   fx = fy = (width/2) / tan(hfov/2) = (640/2) / tan(40 deg) = 381.36,
# #     derived directly from the sensor's own horizontal_fov -- the old
# #     880 was ~2.3x too large, uniformly shrinking every computed
# #     x_forward_m/y_left_m.
# #   pitch = 0, confirmed two ways: the SDF's sensor pose rotation is
# #     literally "0 0 0", and empirically the horizon (sky/track color
# #     transition) sits at pixel row 186 out of 360 in a captured frame,
# #     i.e. essentially exactly at the image's vertical center (180) as
# #     you'd expect for a level camera. The previous 15 deg assumption
# #     was worse than just a scale error -- it entered through
# #     scale = camera_height_m / -ray_robot[2], which is pitch-dependent
# #     in a position-varying way, and artificially compressed the
# #     computed distance for farther waypoints (small y_cam, near image
# #     center) much more than nearer ones -- collapsing the far end of
# #     the reported path toward y=0 regardless of the track's real shape.
# APPROX_CAMERA_FX_PX = 381.36
# APPROX_CAMERA_FY_PX = 381.36
# # Computed from the SDF chain (model -> chassis -> camera), with the
# # vehicle_blue model's 0.3 scale applied to the local offsets only (its
# # own world pose is not itself rescaled):
# #   model world z (0.325) + (chassis local z (0.175) + camera local z
# #   (0.35)) * scale (0.3) = 0.325 + 0.525*0.3 = 0.4825
# APPROX_CAMERA_HEIGHT_M = 0.4825
# APPROX_CAMERA_PITCH_DEG = 0.0      # downward pitch from horizontal -- confirmed via sensor pose "1.1 0 0.35 0 0 0" (roll pitch yaw all 0)

# Camera model for the REAL Logitech Brio front_camera (1280x720,
# diagonal FOV 90 deg / ~82 deg horizontal -- the Brio's widest setting,
# closest to the old sim's 80 deg horizontal_fov so sim->real stays comparable).
# Replaces the Gazebo sim values (640x360, hfov 80 deg, height 0.4825, pitch 0).
#   fx = fy = (diag/2) / tan(dfov/2)
#           = (sqrt(1280^2 + 720^2)/2) / tan(45 deg)
#           = (1468.6/2) / 1.0 = 734.3,
#     with principal point at the image center (cx, cy) = (640, 360).
#     NOTE: if you switch the Brio to a narrower FOV in Logi Options,
#     use fx=fy≈907 (78 deg) or ≈1153 (65 deg) instead.
APPROX_CAMERA_FX_PX = 734.3
APPROX_CAMERA_FY_PX = 734.3

# ---- MEASURE THESE ON THE ACTUAL CAR (they are not derivable) ----
# Height of the LENS CENTER above the ground plane, in meters.
# Put a ruler from the floor to the middle of the Brio's lens.
APPROX_CAMERA_HEIGHT_M = 0.15        # TODO: measure -- placeholder only

# Downward tilt of the camera from horizontal, in degrees (0 = level).
# Confirm empirically like the sim did: capture a frame, find the pixel
# row where the track/wall meets the far background. Row ~360 = level
# (pitch 0). If that horizon sits ABOVE center (row < 360), the camera
# is pitched DOWN by roughly: pitch_deg = atan((360 - row)/fy) in deg.
APPROX_CAMERA_PITCH_DEG = 0.0        # TODO: measure/confirm -- placeholder only

# ROS 2 publishing (optional -- see pipeline/ros_publish.py). Only used
# when run.py is started with --publish-ros; rclpy is not a hard
# dependency of the rest of the pipeline.
ROS_DEFAULT_TOPIC    = "/lane_waypoints"
ROS_DEFAULT_FRAME_ID = "base_link"
ROS_DEFAULT_IMAGE_TOPIC = "/lane_overlay/image_raw"
ROS_DEFAULT_IMAGE_FRAME_ID = "camera_optical_frame"
ROS_DEFAULT_DEBUG_IMAGE_TOPIC = "/lane_overlay/debug"
ROS_DEFAULT_INPUT_IMAGE_TOPIC = "/camera/image_raw"

# Debug visualisation colours
LEFT_COLOR  = (0, 0, 255)
RIGHT_COLOR = (255, 0, 0)
