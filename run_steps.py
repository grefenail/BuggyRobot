"""
Step-by-step pipeline viewer for buggyvis1.

Run:
    python buggyvis1/run_steps.py
    python buggyvis1/run_steps.py IMG_6741.MP4

Controls:
    Right arrow / d  — next frame
    Left  arrow / a  — previous frame
    Space            — play / pause
    1-6              — toggle a step window on/off
    r                — restart
    q / Esc          — quit

    Trackbars in the "Bird's-Eye Controls" window update the ROI/IPM
    calibration live. Press 's' to save current trackbar values to
    config.py.
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import config
from step1_mask  import apply_white_mask
from step2_canny import apply_canny
from step5_fit   import update_fit, get_debug_info, MIN_ARC_POINTS, reset as reset_fit
from step6_draw  import fill_lane, lane_overlay
from roi_search  import detect_with_stable_roi
from ipm         import warp_to_birdseye, warp_to_vehicle, draw_destination_overlay

VIDEO_DIR        = ROOT / "vids"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

IPM_WIN  = "Bird's-Eye Controls  (s=save to config.py)"

STEPS = [
    "0 - Bird's-eye perspective",
    "1 — White Mask (HSV filter)",
    "2 — Canny Edges",
    "3 — ROI Mask",
    "4 — Hough Segments  red=left  blue=right  yellow=rejected",
    "5+6 — Final (fit + fill)",
]


def resolve_video(name):
    if name is None:
        videos = sorted(p for p in VIDEO_DIR.iterdir()
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            raise FileNotFoundError(f"No videos in {VIDEO_DIR}")
        return videos[0]
    p = Path(name)
    if p.exists():
        return p.resolve()
    p2 = VIDEO_DIR / name
    if p2.exists():
        return p2.resolve()
    raise FileNotFoundError(f"Cannot find '{name}'")


def label(img, text, is_gray=False):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if is_gray else img.copy()
    bar_w = min(len(text) * 10 + 16, out.shape[1])
    cv2.rectangle(out, (0, 0), (bar_w, 30), (20, 20, 20), -1)
    cv2.putText(out, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_debug_overlay(img, info, n_left, n_right):
    """Print the live internal numbers in the corner of the final
    view -- point counts per side, bent vs straight, and the
    stale/confirm counters -- so they can be watched frame by frame
    without running a separate script."""
    enough_left  = n_left  >= MIN_ARC_POINTS
    enough_right = n_right >= MIN_ARC_POINTS
    pts_line = f"points  left={n_left} ({'OK' if enough_left else f'<{MIN_ARC_POINTS} short'})"
    pts_line += "   "
    pts_line += f"right={n_right} ({'OK' if enough_right else f'<{MIN_ARC_POINTS} short'})"
    lines = [
        pts_line,
        f"bent: left={info['left_bent']}  right={info['right_bent']}",
        f"no_detect_ct: {info['no_detect_ct']}  confirm_ct: {info['confirm_ct']}",
        f"left_pts: {info['left_pts']}  right_pts: {info['right_pts']}",
    ]
    y = img.shape[0] - 14 * len(lines) - 10
    for i, line in enumerate(lines):
        if not line:
            continue
        color = (0, 255, 255)
        if i == 0:
            color = (0, 255, 0) if (enough_left and enough_right) else (0, 0, 255)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 14
    return img


def run_pipeline(frame_bgr):
    if config.ROTATE_CW:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    work_frame = warp_to_birdseye(frame_bgr)
    h, w = work_frame.shape[:2]

    gray  = apply_white_mask(work_frame)
    edges = apply_canny(gray)
    lx, ly, rx, ry, hough_vis, roi, vertices = detect_with_stable_roi(edges, h, w)
    _, _, min_y, max_y, lc, rc = update_fit(lx, ly, rx, ry, h, w)

    result = frame_bgr.copy()
    overlay = warp_to_vehicle(lane_overlay(work_frame.shape, lc, rc))
    mask = overlay.any(axis=2)
    if mask.any():
        blended = cv2.addWeighted(result, 1.0 - config.LANE_FILL_ALPHA,
                                  overlay, config.LANE_FILL_ALPHA, 0)
        result[mask] = blended[mask]
    bird = fill_lane(work_frame.copy(), lc, rc, min_y, max_y)
    cv2.polylines(bird, vertices, isClosed=True, color=(255, 0, 255), thickness=2)
    bird = draw_destination_overlay(bird)

    draw_debug_overlay(result, get_debug_info(), len(lx), len(rx))

    return [
        label(bird, STEPS[0]),
        label(gray,      STEPS[1], is_gray=True),
        label(edges,     STEPS[2], is_gray=True),
        label(roi,       STEPS[3], is_gray=True),
        label(hough_vis, STEPS[4]),
        label(result,    STEPS[5]),
    ]


def save_config():
    """Write current IPM/ROI values back into config.py."""
    cfg_path = ROOT / "config.py"
    text = cfg_path.read_text()
    replacements = {
        "LINE_BOTTOM_FRAC":                config.LINE_BOTTOM_FRAC,
        "BIRDSEYE_ROI_TOP_MARGIN_FRAC":    config.BIRDSEYE_ROI_TOP_MARGIN_FRAC,
        "BIRDSEYE_ROI_BOTTOM_MARGIN_FRAC": config.BIRDSEYE_ROI_BOTTOM_MARGIN_FRAC,
    }
    import re
    for key, val in replacements.items():
        text = re.sub(
            rf"^({key}\s*=\s*)[\d.]+",
            lambda m, v=val: f"{m.group(1)}{v:.2f}",
            text, flags=re.MULTILINE
        )

    src_lines = "\n".join(f"    ({x:.2f}, {y:.2f})," for x, y in config.IPM_SRC_FRAC)
    text = re.sub(
        r"IPM_SRC_FRAC = \([\s\S]*?\n\)",
        f"IPM_SRC_FRAC = (\n{src_lines}\n)",
        text,
    )

    cfg_path.write_text(text)
    print(f"\nSaved IPM/ROI values to config.py")


def play(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = max(1, int(1000 / fps))

    idx       = 0
    paused    = True
    visible   = [True, False, False, False, False, True]  # Bird's-eye + Final by default (rest are slow)
    frames_cache = {}
    last_roi  = None   # ROI/IPM values used in the last show() call

    # ── IPM / bird's-eye trackbar setup ─────────────────────────────────────────
    cv2.namedWindow(IPM_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(IPM_WIN, 400, 300)

    def _tb(val, attr):
        setattr(config, attr, val / 100)
        reset_fit()

    def _tb_ipm_src_x(val, point_index):
        pts = [list(p) for p in config.IPM_SRC_FRAC]
        pts[point_index][0] = val / 100
        config.IPM_SRC_FRAC = tuple(tuple(p) for p in pts)
        reset_fit()

    def _tb_ipm_src_y(val, point_indices):
        pts = [list(p) for p in config.IPM_SRC_FRAC]
        for i in point_indices:
            pts[i][1] = val / 100
        config.IPM_SRC_FRAC = tuple(tuple(p) for p in pts)
        reset_fit()

    # IPM_SRC_FRAC point order: 0=bottom-left, 1=top-left, 2=top-right, 3=bottom-right
    cv2.createTrackbar("Src Bot-Left X %",  IPM_WIN, int(config.IPM_SRC_FRAC[0][0] * 100), 100, lambda v: _tb_ipm_src_x(v, 0))
    cv2.createTrackbar("Src Top-Left X %",  IPM_WIN, int(config.IPM_SRC_FRAC[1][0] * 100), 100, lambda v: _tb_ipm_src_x(v, 1))
    cv2.createTrackbar("Src Top-Right X %", IPM_WIN, int(config.IPM_SRC_FRAC[2][0] * 100), 100, lambda v: _tb_ipm_src_x(v, 2))
    cv2.createTrackbar("Src Bot-Right X %", IPM_WIN, int(config.IPM_SRC_FRAC[3][0] * 100), 100, lambda v: _tb_ipm_src_x(v, 3))
    # Top/bottom Y -- how far up the frame (how far from the camera) the
    # source trapezoid reaches. Bringing "Src Top Y" down closer to
    # "Src Bottom Y" keeps the calibration in the near field, where a
    # curving track is still close enough to straight for the single-
    # lane IPM assumption to hold reasonably well.
    cv2.createTrackbar("Src Top Y %",       IPM_WIN, int(config.IPM_SRC_FRAC[1][1] * 100), 100, lambda v: _tb_ipm_src_y(v, (1, 2)))
    cv2.createTrackbar("Src Bottom Y %",    IPM_WIN, int(config.IPM_SRC_FRAC[0][1] * 100), 100, lambda v: _tb_ipm_src_y(v, (0, 3)))
    cv2.createTrackbar("Top Margin %",      IPM_WIN, int(config.BIRDSEYE_ROI_TOP_MARGIN_FRAC    * 100), 50, lambda v: _tb(v, "BIRDSEYE_ROI_TOP_MARGIN_FRAC"))
    cv2.createTrackbar("Bottom Margin %",   IPM_WIN, int(config.BIRDSEYE_ROI_BOTTOM_MARGIN_FRAC * 100), 50, lambda v: _tb(v, "BIRDSEYE_ROI_BOTTOM_MARGIN_FRAC"))
    cv2.createTrackbar("Line Bottom %",     IPM_WIN, int(config.LINE_BOTTOM_FRAC * 100), 100, lambda v: _tb(v, "LINE_BOTTOM_FRAC"))
    # ────────────────────────────────────────────────────────────────────────────

    def get_frame(i):
        if i in frames_cache:
            return frames_cache[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames_cache[i] = f
        return f if ok else None

    def current_roi_snapshot():
        return (config.LINE_BOTTOM_FRAC,
                config.IPM_SRC_FRAC,
                config.BIRDSEYE_ROI_TOP_MARGIN_FRAC, config.BIRDSEYE_ROI_BOTTOM_MARGIN_FRAC)

    def show(i):
        nonlocal last_roi
        frame = get_frame(i)
        if frame is None:
            return
        reset_fit()
        for j in range(i + 1):
            f = get_frame(j)
            if f is None:
                break
            imgs = run_pipeline(f)
        last_roi = current_roi_snapshot()
        for k, (win, img, vis) in enumerate(zip(STEPS, imgs, visible)):
            if vis:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.imshow(win, img)
            else:
                try:
                    cv2.destroyWindow(win)
                except cv2.error:
                    pass
        print(f"\rFrame {i}/{total-1}    ", end="", flush=True)

    print(f"\nStep viewer: {video_path.name}  ({total} frames @ {fps:.0f}fps)")
    print("Keys: ← → or a/d = prev/next   Space = play/pause")
    print("      1-6 = toggle window   s = save to config.py   r = restart   q = quit\n")

    show(idx)

    while True:
        wait = delay if not paused else 50
        key = cv2.waitKey(wait) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            idx = 0
            paused = True
            reset_fit()
            show(idx)
        elif key == ord('s'):
            save_config()
        elif key in (ord('d'), 83):
            if idx < total - 1:
                idx += 1
                show(idx)
        elif key in (ord('a'), 81):
            if idx > 0:
                idx -= 1
                show(idx)
        elif ord('1') <= key <= ord('6'):
            k = key - ord('1')
            visible[k] = not visible[k]
            show(idx)

        # Redraw whenever trackbars changed since last render
        if paused and current_roi_snapshot() != last_roi:
            show(idx)

        if not paused:
            if idx < total - 1:
                idx += 1
                show(idx)
            else:
                paused = True

    cap.release()
    cv2.destroyAllWindows()
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?")
    args = parser.parse_args()
    play(resolve_video(args.video))


if __name__ == "__main__":
    main()
