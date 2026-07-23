"""Step 4 — HoughLinesP detection and left/right classification.

Detects straight line segments in the ROI-masked (bird's-eye) edge
image, then splits them into left-lane and right-lane groups by
horizontal position relative to the lane center. (The camera-space
slope-sign classification used before bird's-eye was introduced has
been removed -- it's never reached now.)
"""
import math

import cv2
import numpy as np
import config
from config import (HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                    HOUGH_MIN_LENGTH, HOUGH_MAX_GAP,
                    MIN_SLOPE, LEFT_COLOR, RIGHT_COLOR)


def _scaled_px(value, width):
    """Scale an absolute-pixel constant (tuned at REFERENCE_PROCESSING_WIDTH)
    to the actual processing frame width -- mirrors step5_fit._scaled_px.
    Ratio-based against the real runtime width (not just PROCESS_SCALE) so
    it stays correct even when the native camera resolution itself changes,
    not just the PROCESS_SCALE multiplier. Without this,
    HOUGH_MIN_LENGTH/HOUGH_MAX_GAP/HOUGH_THRESHOLD stay calibrated for
    whatever resolution they were tuned at, and a real lane segment that's
    now shorter in pixels can silently fall under the fixed threshold and
    get dropped."""
    return max(1, int(round(value * width / config.REFERENCE_PROCESSING_WIDTH)))


def _endpoint_distance(seg_a, seg_b):
    """Minimum distance between any endpoint of seg_a and any endpoint of
    seg_b."""
    ax1, ay1, ax2, ay2 = seg_a
    bx1, by1, bx2, by2 = seg_b
    a_pts = ((ax1, ay1), (ax2, ay2))
    b_pts = ((bx1, by1), (bx2, by2))
    return min(
        math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        for pa in a_pts
        for pb in b_pts
    )


def _cluster_by_adjacency(segments, adjacency_px):
    """Group near-vertical segments into clusters by endpoint proximity
    (union-find) -- each cluster is a candidate real-world line.

    A track/road can show more than 2 parallel lines at once (e.g. a
    multi-lane running track), and in a sharp curve a single continuous
    line's x position can vary hugely across the frame's y-range. An
    earlier version of this clustering extrapolated each segment out to
    one fixed reference y and grouped by that extrapolated x -- this
    works for straight parallel lines (left stays left, right stays
    right, at every y) but breaks in a sharp curve: extrapolating a
    near-field piece and a far-field piece of the *same* curving line out
    to one shared point can put them at wildly different x, splitting one
    real line into separate clusters, or making them coincide with a
    different, unrelated line's extrapolated position.

    Clustering by endpoint proximity instead never extrapolates anything:
    two segments join a cluster only if their nearest endpoints are
    actually close together in (x, y). This keeps a continuously-curving
    line's segments grouped correctly regardless of how much its x sweeps
    across y, since it only ever relies on local continuity between
    consecutive pieces (adjacency_px is chosen smaller than the minimum
    plausible lane width -- see config.LANE_CLUSTER_ADJACENCY_PX -- so
    two genuinely different, adjacent lines aren't merged together).
    """
    n = len(segments)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _endpoint_distance(segments[i], segments[j]) <= adjacency_px:
                union(i, j)

    grouped = {}
    for i, seg in enumerate(segments):
        grouped.setdefault(find(i), []).append(seg)
    return list(grouped.values())


def _cluster_near_field_x(cluster):
    """Representative x for a cluster, used to decide which side (left/
    right of center) it belongs to: the x of whichever endpoint across all
    its segments has the largest y -- i.e. the nearest-to-vehicle point
    actually observed, not an extrapolated one. Near-field position is
    both the most stable reference (least affected by curve sweep, since
    it's the point closest to where the curve *starts*) and the most
    relevant for an immediate steering decision."""
    best_x, best_y = None, -1.0
    for x1, y1, x2, y2 in cluster:
        for x, y in ((x1, y1), (x2, y2)):
            if y > best_y:
                best_y = y
                best_x = x
    return best_x


def _select_cluster(clusters, side_test, target_x):
    """Pick the cluster on the correct side of center whose near-field x
    is closest to target_x -- either the previous frame's locked position
    (preferred, keeps tracking the same physical line instead of jumping
    between candidates that are briefly equidistant from center), or
    center_x itself as a fallback when nothing was locked yet.

    side_test(near_field_x) -> bool restricts candidates to the correct
    side.
    """
    candidates = [
        c for c in clusters if side_test(_cluster_near_field_x(c))
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: abs(_cluster_near_field_x(c) - target_x),
    )


def detect_segments(cropped, width, center_x=None,
                     prefer_left_x=None, prefer_right_x=None):
    """
    Run Hough on the cropped edge image, group segments into candidate
    real-world lines, and pick the one immediately left and immediately
    right of the vehicle's own lane (see
    _cluster_by_adjacency/_select_cluster).

    center_x -- the x position (pixels) to split left/right around.
    Defaults to width/2 (raw image center), but callers should pass
    the actual configured lane center (see ipm.birdseye_lane_center_frac).
    prefer_left_x/prefer_right_x -- previous frame's locked x position
    (step5_fit's _left_pts[0]/_right_pts[0]) for each side, if any.

    Returns
    -------
    left_x, left_y   : point lists for left lane fit
    right_x, right_y : point lists for right lane fit
    hough_vis        : BGR debug image with coloured segments
    """
    lines = cv2.HoughLinesP(
        cropped,
        rho=_scaled_px(HOUGH_RHO, width), theta=HOUGH_THETA,
        threshold=_scaled_px(HOUGH_THRESHOLD, width), lines=np.array([]),
        minLineLength=_scaled_px(HOUGH_MIN_LENGTH, width), maxLineGap=_scaled_px(HOUGH_MAX_GAP, width),
    )

    left_x,  left_y  = [], []
    right_x, right_y = [], []
    hough_vis = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    if lines is None:
        return left_x, left_y, right_x, right_y, hough_vis

    split_x = center_x if center_x is not None else width / 2

    # Depending on the OpenCV build (and whether the optional output array is
    # reused), HoughLinesP may return either (N, 1, 4) or (N, 4).  Flatten the
    # segment dimensions so both representations are handled identically.
    kept = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx = x2 - x1
        dy = y2 - y1

        # In bird's-eye view valid lane markings are close to
        # vertical, so classify by horizontal position instead of
        # slope sign.
        if abs(dy) < abs(dx) * MIN_SLOPE:
            cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            continue
        kept.append((int(x1), int(y1), int(x2), int(y2)))

    if not kept:
        return left_x, left_y, right_x, right_y, hough_vis

    adjacency_px = _scaled_px(config.LANE_CLUSTER_ADJACENCY_PX, width)
    clusters = _cluster_by_adjacency(kept, adjacency_px)

    left_cluster = _select_cluster(
        clusters, lambda x: x < split_x,
        prefer_left_x if prefer_left_x is not None else split_x,
    )
    right_cluster = _select_cluster(
        clusters, lambda x: x >= split_x,
        prefer_right_x if prefer_right_x is not None else split_x,
    )

    for cluster, color, xs, ys in (
        (left_cluster, LEFT_COLOR, left_x, left_y),
        (right_cluster, RIGHT_COLOR, right_x, right_y),
    ):
        if cluster is None:
            continue
        for x1, y1, x2, y2 in cluster:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
            cv2.line(hough_vis, (x1, y1), (x2, y2), color, 2)

    # Draw candidate clusters that were seen but not selected (e.g. an
    # adjacent track lane) in a neutral color, so the debug view still
    # shows what else was there.
    for cluster in clusters:
        if cluster is left_cluster or cluster is right_cluster:
            continue
        for x1, y1, x2, y2 in cluster:
            cv2.line(hough_vis, (x1, y1), (x2, y2), (180, 180, 180), 1)

    return left_x, left_y, right_x, right_y, hough_vis
