"""Scanline profile smoothing and lane-boundary peak selection."""

import numpy as np

EDGE_REJECT_MARGIN_PX = 5
MAX_SCANLINE_X_JUMP_PX = 35
SCANLINE_X_JUMP_PENALTY = 5.0
REGISTERED_PAIR_MIN_PEAKS = 3
REGISTERED_PAIR_WIDTH_PENALTY = 0.25
MAX_SCANLINE_ANCHOR_DRIFT_PX = 90
SCANLINE_ANCHOR_PENALTY = 1.5


def smooth_1d(values, kernel_size=21):
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def local_peaks(values, min_score, edge_margin=0):
    peaks = []
    x_start = max(1, edge_margin)
    x_end = min(len(values) - 1, len(values) - edge_margin)

    x = x_start
    while x < x_end:
        if values[x] < min_score:
            x += 1
            continue

        run_start = x
        while x < x_end and values[x] >= min_score:
            x += 1
        run_end = x

        run = values[run_start:run_end]
        if len(run) == 0:
            continue
        peak_score = float(run.max())
        max_positions = np.flatnonzero(run == peak_score)
        peak_x = run_start + int(round(float(max_positions.mean())))
        peaks.append((peak_x, peak_score))

    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks


def best_lane_pair(
    profile,
    expected_width,
    width_tolerance,
    center_x,
    previous_pair=None,
    anchor_pair=None,
    edge_margin=EDGE_REJECT_MARGIN_PX,
):
    min_score = max(3.0, float(profile.max()) * 0.25)
    peaks = local_peaks(profile, min_score, edge_margin=edge_margin)[:12]
    peak_xs = [x for x, _ in peaks]
    best = None

    if previous_pair is not None and len(peaks) >= REGISTERED_PAIR_MIN_PEAKS:
        registered_best = None
        for li, left_score in peaks:
            for ri, right_score in peaks:
                if ri <= li:
                    continue
                lane_width = ri - li
                width_error = abs(lane_width - expected_width)
                if width_error > width_tolerance:
                    continue

                left_jump = abs(li - previous_pair[0])
                right_jump = abs(ri - previous_pair[1])
                if left_jump > MAX_SCANLINE_X_JUMP_PX or right_jump > MAX_SCANLINE_X_JUMP_PX:
                    continue

                registered_error = left_jump + right_jump
                score = left_score + right_score - registered_error - width_error * REGISTERED_PAIR_WIDTH_PENALTY
                if registered_best is None or registered_error < registered_best["registered_error"] or (
                    registered_error == registered_best["registered_error"] and score > registered_best["score"]
                ):
                    registered_best = {
                        "left": li,
                        "right": ri,
                        "score": score,
                        "left_score": left_score,
                        "right_score": right_score,
                        "width": lane_width,
                        "peaks": peak_xs,
                        "registered_error": registered_error,
                    }

        if registered_best is not None:
            return registered_best

    for li, left_score in peaks:
        for ri, right_score in peaks:
            if ri <= li:
                continue
            lane_width = ri - li
            width_error = abs(lane_width - expected_width)
            if width_error > width_tolerance:
                continue
            pair_center = (li + ri) / 2.0
            center_error = abs(pair_center - center_x)
            continuity_error = 0.0
            if previous_pair is not None:
                left_jump = abs(li - previous_pair[0])
                right_jump = abs(ri - previous_pair[1])
                if left_jump > MAX_SCANLINE_X_JUMP_PX or right_jump > MAX_SCANLINE_X_JUMP_PX:
                    continue
                continuity_error = left_jump + right_jump
            anchor_error = 0.0
            if anchor_pair is not None:
                left_anchor_jump = abs(li - anchor_pair[0])
                right_anchor_jump = abs(ri - anchor_pair[1])
                if (
                    left_anchor_jump > MAX_SCANLINE_ANCHOR_DRIFT_PX
                    or right_anchor_jump > MAX_SCANLINE_ANCHOR_DRIFT_PX
                ):
                    continue
                anchor_error = left_anchor_jump + right_anchor_jump
            score = (
                left_score + right_score
                - width_error * 0.6
                - center_error * 0.15
                - continuity_error * SCANLINE_X_JUMP_PENALTY
                - anchor_error * SCANLINE_ANCHOR_PENALTY
            )
            if best is None or score > best["score"]:
                best = {
                    "left": li,
                    "right": ri,
                    "score": score,
                    "left_score": left_score,
                    "right_score": right_score,
                    "width": lane_width,
                    "peaks": peak_xs,
                }

    return best


def best_single_side_peak(profile, expected_x, center_x, want_left, edge_margin=EDGE_REJECT_MARGIN_PX):
    min_score = max(3.0, float(profile.max()) * 0.25)
    peaks = local_peaks(profile, min_score, edge_margin=edge_margin)[:12]
    candidates = [
        (x, score)
        for x, score in peaks
        if (x < center_x if want_left else x > center_x)
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda item: item[1] - abs(item[0] - expected_x) * 0.05)
