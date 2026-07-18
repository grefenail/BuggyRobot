"""Step 5 — Significance-tested line/curve fit, exponential smoothing,
sanity check, stale reset.

Each side is fit independently with both a straight line (degree 1)
and a quadratic (degree 2). The quadratic is only trusted -- drawn as
a bend -- when it reduces the fit residual by a significant margin
over the line; otherwise the side is drawn straight.

This replaces an earlier joint concentric-circle model (both sides
sharing one center, radius r and r+lane_width). That model was
geometrically elegant but numerically fragile: distinguishing "radius
= 5000" from "radius = 250" from noisy points is an ill-conditioned
problem regardless of whether the input is perspective-corrected --
subtle real curvature and per-point measurement noise both nudge a
circle fit's radius down into the same range, so a radius threshold
alone can't tell them apart. Asking "is a curve a significantly
better fit than a line for THIS data" is a much better-conditioned
question, and needs no circle (or shared center) at all.
"""
import numpy as np
import config

# Module-level state (persists across frames)
_left_pts     = None   # smoothed [x_at_bottom, x_at_top]
_right_pts    = None
_no_detect_ct = 0
_left_curve   = None   # last drawn boundary polyline (bottom-to-top), 2 pts if straight
_right_curve  = None
_confirm_ct   = 0      # consecutive frames held since the last full lock loss

MIN_ARC_POINTS = 4  # minimum points per side to attempt a line/quadratic fit at all


def reset():
    """Force-clear all state (call when switching videos)."""
    global _left_pts, _right_pts, _no_detect_ct, _left_curve, _right_curve, _confirm_ct
    _left_pts = _right_pts = None
    _no_detect_ct = 0
    _left_curve = _right_curve = None
    _confirm_ct = 0


def get_debug_info():
    """
    Snapshot of the internal numbers worth watching live: whether each
    tracked boundary is currently bent (more than 2 curve points) or
    straight, and the stale/confirm counters. For on-screen overlay
    only -- not used by any detection logic.
    """
    left_bent  = _left_curve  is not None and len(_left_curve)  > 2
    right_bent = _right_curve is not None and len(_right_curve) > 2
    return {
        "bent": left_bent or right_bent,
        "left_bent": left_bent,
        "right_bent": right_bent,
        "no_detect_ct": _no_detect_ct,
        "confirm_ct": _confirm_ct,
        "left_pts": _left_pts,
        "right_pts": _right_pts,
    }


def _smooth(prev, new):
    if prev is None:
        return list(new)
    return [int(prev[i] * (1 - config.SMOOTH_ALPHA) + new[i] * config.SMOOTH_ALPHA)
            for i in range(len(new))]


def _scaled_px(value):
    return max(1, int(round(value * float(config.PROCESS_SCALE))))


def _line_fit(xs, ys, min_y, max_y, width):
    poly = np.poly1d(np.polyfit(ys, xs, deg=1))
    return [int(np.clip(poly(max_y), 0, width - 1)),
            int(np.clip(poly(min_y), 0, width - 1))]


def _fit_line_and_quad(xs, ys):
    """
    Fit both a line (degree 1) and a quadratic (degree 2) to the same
    points, and decide whether the quadratic is a significantly better
    fit -- not "does some curvature exist" (noise guarantees a
    quadratic always fits at least marginally better), but "does
    allowing curvature reduce the fit error by more than
    BEND_SIGNIFICANCE_FRAC compared to assuming a straight line."

    Returns (line_poly, quad_poly_or_None, is_significant).
    """
    ys_arr = np.asarray(ys, dtype=np.float64)
    xs_arr = np.asarray(xs, dtype=np.float64)

    c1, res1, *_ = np.polyfit(ys_arr, xs_arr, 1, full=True)
    line_poly = np.poly1d(c1)
    ssr_line = float(res1[0]) if len(res1) else float(np.sum((line_poly(ys_arr) - xs_arr) ** 2))

    if len(xs_arr) < MIN_ARC_POINTS:
        return line_poly, None, False

    c2, res2, *_ = np.polyfit(ys_arr, xs_arr, 2, full=True)
    quad_poly = np.poly1d(c2)
    ssr_quad = float(res2[0]) if len(res2) else float(np.sum((quad_poly(ys_arr) - xs_arr) ** 2))

    improvement = (ssr_line - ssr_quad) / ssr_line if ssr_line > 1e-6 else 0.0
    is_significant = improvement > config.BEND_SIGNIFICANCE_FRAC
    return line_poly, quad_poly, is_significant


CURVE_SAMPLES = 12   # points sampled along the curve for drawing


def _sample_curve(poly, y_bot, y_top, x_bot, x_top, width):
    """
    Sample points along the fitted quadratic between y_bot and y_top
    for drawing, rubber-banded so the curve passes exactly through the
    already-adopted (smoothed, gated) endpoints (x_bot, y_bot) and
    (x_top, y_top). The polynomial only supplies the bend shape; the
    endpoints stay whatever update_fit already decided, so there's no
    seam between this frame's curve and the tracked
    line-crossing/smoothing state.
    """
    ys = np.linspace(y_bot, y_top, CURVE_SAMPLES)
    raw = poly(ys)
    t = (ys - y_bot) / (y_top - y_bot)
    corrected = raw + (x_bot - raw[0]) * (1 - t) + (x_top - raw[-1]) * t
    return [(int(np.clip(x, 0, width - 1)), int(y)) for x, y in zip(corrected, ys)]


def _bend_is_reasonable(curve):
    """
    Reject a sampled curve whose middle bulges implausibly far from
    the straight chord between its own endpoints -- guards against a
    quadratic that was judged "significant" on noisy points but
    extrapolates wildly between them.
    """
    xs = np.array([p[0] for p in curve], dtype=np.float64)
    ys = np.array([p[1] for p in curve], dtype=np.float64)
    if ys[-1] == ys[0]:
        return True
    t = (ys - ys[0]) / (ys[-1] - ys[0])
    chord_x = xs[0] + t * (xs[-1] - xs[0])
    return np.abs(xs - chord_x).max() <= _scaled_px(config.MAX_BEND_DEVIATION_PX)


def _fit_side(xs, ys, min_y, max_y, width):
    """
    Fit one side independently: try line-vs-quadratic significance
    test first (if enough points), otherwise fall back to a plain
    line fit. Returns (new_pts, quad_poly_or_None).
    """
    if config.USE_ARC_FIT and len(xs) >= MIN_ARC_POINTS:
        line_poly, quad_poly, bent = _fit_line_and_quad(xs, ys)
        poly = quad_poly if (bent and quad_poly is not None) else line_poly
        try:
            x_bot = int(np.clip(poly(max_y), 0, width - 1))
            x_top = int(np.clip(poly(min_y), 0, width - 1))
        except Exception:
            return None, None
        return [x_bot, x_top], (quad_poly if bent else None)

    if len(xs) >= 2:
        try:
            return _line_fit(xs, ys, min_y, max_y, width), None
        except Exception:
            return None, None

    return None, None


def update_fit(left_x, left_y, right_x, right_y, height, width):
    """
    Fit both lane boundaries independently (line, or quadratic when
    significantly better), smooth, sanity-check, stale-reset.

    Returns
    -------
    left_pts   : [x_bottom, x_top] or None
    right_pts  : [x_bottom, x_top] or None
    min_y      : y coordinate of the top of the drawn line
    max_y      : y coordinate of the bottom (= frame height)
    left_curve : list of (x, y) points bottom-to-top for drawing (bent
                 when this frame's fit was a significant quadratic,
                 else just the 2 endpoints), or None
    right_curve: same as left_curve, for the right boundary
    """
    global _left_pts, _right_pts, _no_detect_ct, _left_curve, _right_curve, _confirm_ct

    min_y = int(height * config.LINE_TOP_FRAC)
    max_y = int(height * config.LINE_BOTTOM_FRAC)
    got_left = got_right = False
    prev_left_curve = _left_curve
    prev_right_curve = _right_curve

    new_left,  quad_left  = _fit_side(left_x,  left_y,  min_y, max_y, width)
    new_right, quad_right = _fit_side(right_x, right_y, min_y, max_y, width)

    # Outlier gate — a single-frame reading that jumps far past what's
    # already locked in is more likely noise than a real lane change.
    # Reject it and coast on the previous smoothed value instead of
    # letting one bad frame whipsaw the overlay.
    #
    # Use a much looser tolerance while the lock is still fresh (see
    # ARC_JUMP_CONFIRM_FRAMES): a lock formed from the first few noisy
    # frames can itself be wrong by more than the strict tolerance, and
    # without this, every subsequent correction toward the real
    # position gets rejected as "noise", freezing on the bad value
    # until a full stale-reset -- which can land on another bad lock
    # and repeat the cycle.
    jump_limit = _scaled_px(
        config.ARC_MAX_JUMP_PX if _confirm_ct >= config.ARC_JUMP_CONFIRM_FRAMES
        else config.ARC_MAX_JUMP_PX_FRESH
    )
    if new_left is not None and _left_pts is not None and \
            max(abs(new_left[i] - _left_pts[i]) for i in range(2)) > jump_limit:
        new_left = None
        quad_left = None
    if new_right is not None and _right_pts is not None and \
            max(abs(new_right[i] - _right_pts[i]) for i in range(2)) > jump_limit:
        new_right = None
        quad_right = None

    if new_left is not None:
        _left_pts = _smooth(_left_pts, new_left)
        got_left = True
    else:
        quad_left = None   # coasting on a stale _left_pts -- draw it straight
    if new_right is not None:
        _right_pts = _smooth(_right_pts, new_right)
        got_right = True
    else:
        quad_right = None

    # Sanity check — lines must not cross, or converge to near-nothing,
    # at top OR bottom. A fit that technically keeps left < right but
    # narrows to a sliver is just as wrong as one that crosses outright
    # -- real lane lines don't pinch down to near-zero width.
    if (_left_pts is not None and _right_pts is not None and
            (_right_pts[0] - _left_pts[0] < _scaled_px(config.MIN_LANE_WIDTH_PX) or
             _right_pts[1] - _left_pts[1] < _scaled_px(config.MIN_LANE_WIDTH_PX))):
        _left_pts = _right_pts = None
        got_left = got_right = False
        quad_left = quad_right = None

    # Stale reset — clear after too many consecutive failed frames.
    # This must fire regardless of KEEP_LAST_LINE_ON_MISS: that flag
    # only controls what gets drawn during a brief miss (last good
    # curve instead of blanking), not whether a persistently wrong
    # lock ever lets go. Gating this reset on it (as before) meant a
    # lock stuck on a wrong position -- e.g. one the outlier gate
    # below keeps rejecting every real correction against -- would
    # never clear and stayed frozen there forever instead of just
    # ~5s (STALE_RESET_FRAMES at 30fps).
    if got_left and got_right:
        _no_detect_ct = 0
    else:
        _no_detect_ct += 1
        if _no_detect_ct >= config.STALE_RESET_FRAMES:
            _left_pts = _right_pts = None
            _no_detect_ct = 0
            quad_left = quad_right = None

    # Confirmation counter -- counts consecutive frames held since the
    # last full lock loss (including frames coasting on a stale value
    # during the hold window above). Reset to 0 the instant the lock
    # actually drops to None.
    if _left_pts is not None and _right_pts is not None:
        _confirm_ct += 1
    else:
        _confirm_ct = 0

    left_curve = prev_left_curve if (
        config.KEEP_LAST_LINE_ON_MISS and not got_left and _left_pts is not None
    ) else None
    right_curve = prev_right_curve if (
        config.KEEP_LAST_LINE_ON_MISS and not got_right and _right_pts is not None
    ) else None

    if _left_pts is not None:
        if left_curve is None and quad_left is not None:
            left_curve = _sample_curve(quad_left, max_y, min_y,
                                        _left_pts[0], _left_pts[1], width)
            if left_curve is not None and not _bend_is_reasonable(left_curve):
                left_curve = None
        if left_curve is None:
            left_curve = [(_left_pts[0], max_y), (_left_pts[1], min_y)]
    if _right_pts is not None:
        if right_curve is None and quad_right is not None:
            right_curve = _sample_curve(quad_right, max_y, min_y,
                                         _right_pts[0], _right_pts[1], width)
            if right_curve is not None and not _bend_is_reasonable(right_curve):
                right_curve = None
        if right_curve is None:
            right_curve = [(_right_pts[0], max_y), (_right_pts[1], min_y)]

    _left_curve, _right_curve = left_curve, right_curve
    return _left_pts, _right_pts, min_y, max_y, left_curve, right_curve
