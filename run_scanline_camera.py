"""Camera-space scanline lane experiment.

This is a clean scanline-only test with no bird's-eye transform.

Run:
    python run_scanline_camera.py IMG_6743.MP4
    python run_scanline_camera.py IMG_6743.MP4 --headless --max-frames 120
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

import config
from step1_mask import apply_white_mask_relative


VIDEO_DIR = ROOT / "vids"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

SCANLINE_COUNT = 28
BAND_HEIGHT = 9
WHITE_TOP_PERCENT = 5
SCAN_TOP_FRAC = 0.45
SCAN_BOTTOM_FRAC = 0.95
MIN_PEAK_SCORE = 3.0
PEAK_SMOOTH_KERNEL_FRAC = 35
MAX_PEAKS_PER_ROW = 8
EXPECTED_WIDTH_FRAC = 0.24
WIDTH_TOLERANCE_FRAC = 0.45
MAX_PAIR_JUMP_PX = 45
MIN_FIT_POINTS = 5
LANE_FILL_COLOR = (0, 180, 0)
LANE_FILL_ALPHA = 0.28


def resolve_video(name):
    if name is None:
        videos = sorted(
            p for p in VIDEO_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise FileNotFoundError(f"No videos found in {VIDEO_DIR}")
        return videos[0]
    p = Path(name)
    if p.exists():
        return p.resolve()
    p2 = VIDEO_DIR / name
    if p2.exists():
        return p2.resolve()
    raise FileNotFoundError(f"Cannot find '{name}'")


def processing_frame(frame_bgr):
    scale = float(config.PROCESS_SCALE)
    if abs(scale - 1.0) < 1e-6:
        return frame_bgr
    return cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def white_mask(frame_bgr):
    gray = apply_white_mask_relative(frame_bgr, WHITE_TOP_PERCENT)
    return np.where(gray > 0, 255, 0).astype(np.uint8)


def smooth_1d(values, kernel_size):
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def local_peaks(profile, min_score):
    peaks = []
    x = 1
    x_end = len(profile) - 1
    while x < x_end:
        if profile[x] < min_score:
            x += 1
            continue

        start = x
        while x < x_end and profile[x] >= min_score:
            x += 1
        end = x
        run = profile[start:end]
        if len(run) == 0:
            continue

        score = float(run.max())
        max_pos = np.flatnonzero(run == score)
        peak_x = start + int(round(float(max_pos.mean())))
        peaks.append((peak_x, score))

    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks


def choose_pair(peaks, expected_width, width_tolerance, previous_pair=None):
    best = None
    for left_x, left_score in peaks:
        for right_x, right_score in peaks:
            if right_x <= left_x:
                continue

            width = right_x - left_x
            width_error = abs(width - expected_width)
            if width_error > width_tolerance:
                continue

            jump_error = 0.0
            if previous_pair is not None:
                left_jump = abs(left_x - previous_pair[0])
                right_jump = abs(right_x - previous_pair[1])
                if left_jump > MAX_PAIR_JUMP_PX or right_jump > MAX_PAIR_JUMP_PX:
                    continue
                jump_error = left_jump + right_jump

            score = left_score + right_score - width_error * 0.4 - jump_error * 2.0
            if best is None or score > best["score"]:
                best = {
                    "left": left_x,
                    "right": right_x,
                    "score": score,
                    "width": width,
                    "peaks": [x for x, _ in peaks],
                }
    return best


def fit_line(points, y_top, y_bottom):
    if len(points) < MIN_FIT_POINTS:
        return None
    pts = np.asarray(points, dtype=np.float32)
    poly = np.poly1d(np.polyfit(pts[:, 1], pts[:, 0], deg=1))
    return [(int(poly(y_bottom)), int(y_bottom)), (int(poly(y_top)), int(y_top))]


def line_x_at_y(line, y):
    if line is None:
        return None
    (x0, y0), (x1, y1) = line
    if abs(y1 - y0) < 1e-6:
        return None
    return x0 + (x1 - x0) * ((y - y0) / (y1 - y0))


def lane_polygon(left_line, right_line, y_top, y_bottom):
    if left_line is None or right_line is None:
        return []
    lb = line_x_at_y(left_line, y_bottom)
    lt = line_x_at_y(left_line, y_top)
    rt = line_x_at_y(right_line, y_top)
    rb = line_x_at_y(right_line, y_bottom)
    if None in (lb, lt, rt, rb):
        return []
    return [
        (int(round(lb)), int(y_bottom)),
        (int(round(lt)), int(y_top)),
        (int(round(rt)), int(y_top)),
        (int(round(rb)), int(y_bottom)),
    ]


def blend_fill(img, polygon):
    if len(polygon) < 3:
        return img
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.asarray(polygon, dtype=np.int32)], LANE_FILL_COLOR)
    return cv2.addWeighted(overlay, LANE_FILL_ALPHA, img, 1.0 - LANE_FILL_ALPHA, 0)


def draw_line(img, line, color, thickness):
    if line is None:
        return
    h, w = img.shape[:2]
    pts = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in line]
    cv2.line(img, pts[0], pts[1], color, thickness, cv2.LINE_AA)


def draw_center_points(img, left_line, right_line, y_top, y_bottom, count=10):
    if left_line is None or right_line is None:
        return
    points = []
    for y in np.linspace(y_bottom, y_top, count):
        lx = line_x_at_y(left_line, y)
        rx = line_x_at_y(right_line, y)
        if lx is None or rx is None:
            continue
        points.append((int(round((lx + rx) / 2.0)), int(round(y))))

    if len(points) >= 2:
        cv2.polylines(img, [np.asarray(points, dtype=np.int32)], False, (0, 165, 255), 3, cv2.LINE_AA)
    for idx, point in enumerate(points):
        cv2.circle(img, point, 8, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.circle(img, point, 8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(idx), (point[0] + 10, point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(idx), (point[0] + 10, point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def label_panel(img, text):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (min(w, 280), 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def analyze_frame(frame_bgr):
    frame = processing_frame(frame_bgr)
    mask = white_mask(frame)
    h, w = mask.shape[:2]
    y_top = int(h * SCAN_TOP_FRAC)
    y_bottom = int(h * SCAN_BOTTOM_FRAC)
    ys = np.linspace(y_bottom, y_top, SCANLINE_COUNT).astype(int)
    expected_width = w * EXPECTED_WIDTH_FRAC
    width_tolerance = expected_width * WIDTH_TOLERANCE_FRAC

    vis = frame.copy()
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    left_points = []
    right_points = []
    accepted = 0
    previous_pair = None

    for y in ys:
        y0 = max(0, y - BAND_HEIGHT // 2)
        y1 = min(h, y + BAND_HEIGHT // 2 + 1)
        band = mask[y0:y1, :]
        profile = smooth_1d(np.count_nonzero(band, axis=0), max(7, w // PEAK_SMOOTH_KERNEL_FRAC))
        min_score = max(MIN_PEAK_SCORE, float(profile.max()) * 0.25)
        peaks = local_peaks(profile, min_score)[:MAX_PEAKS_PER_ROW]
        pair = choose_pair(peaks, expected_width, width_tolerance, previous_pair)

        cv2.line(vis, (0, int(y)), (w - 1, int(y)), (80, 80, 80), 1)
        cv2.line(mask_vis, (0, int(y)), (w - 1, int(y)), (80, 80, 80), 1)

        selected = set()
        if pair is not None:
            left = (pair["left"], int(y))
            right = (pair["right"], int(y))
            left_points.append(left)
            right_points.append(right)
            previous_pair = (pair["left"], pair["right"])
            selected = {pair["left"], pair["right"]}
            accepted += 1
            cv2.line(vis, left, right, (0, 255, 0), 1)

        for peak_x, _ in peaks:
            if peak_x in selected:
                color = (0, 0, 255) if pair and peak_x == pair["left"] else (255, 0, 0)
            else:
                color = (0, 0, 0)
            cv2.circle(vis, (int(peak_x), int(y)), 4, color, -1)
            cv2.circle(mask_vis, (int(peak_x), int(y)), 4, color, -1)

    left_line = fit_line(left_points, y_top, y_bottom)
    right_line = fit_line(right_points, y_top, y_bottom)
    polygon = lane_polygon(left_line, right_line, y_top, y_bottom)
    if polygon:
        vis = blend_fill(vis, polygon)
        mask_vis = blend_fill(mask_vis, polygon)

    draw_line(vis, left_line, (0, 255, 255), 3)
    draw_line(vis, right_line, (0, 255, 255), 3)
    draw_line(mask_vis, left_line, (0, 255, 255), 2)
    draw_line(mask_vis, right_line, (0, 255, 255), 2)
    draw_center_points(vis, left_line, right_line, y_top, y_bottom)
    draw_center_points(mask_vis, left_line, right_line, y_top, y_bottom)

    cv2.putText(vis, f"camera scanlines accepted={accepted}",
                (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2, cv2.LINE_AA)

    return np.hstack([
        label_panel(vis, "Camera scanlines - no bird eye"),
        label_panel(mask_vis, "Camera white mask"),
    ])


def draw_hud(frame, paused, name, idx, total, source_fps, run_fps):
    status = "PAUSED" if paused else "PLAYING"
    h, w = frame.shape[:2]
    run_fps_text = "--" if run_fps is None else f"{run_fps:.1f}"
    text = (
        f"{name} | {status} | {idx}/{total} | src {source_fps:.1f} fps | "
        f"run {run_fps_text} fps | out {w}x{h} | Space pause  r restart  q quit"
    )
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (0, 0), (min(tw + 18, w), th + bl + 18), (0, 0, 0), -1)
    cv2.putText(frame, text, (9, th + 9), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def play(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = max(1, int(1000 / fps))
    paused = False
    idx = 0
    last_frame = None
    last_tick = None
    run_fps = None
    win = "Camera-space Scanline Experiment"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                idx = 0
                continue
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            last_frame = analyze_frame(frame)
            idx += 1
            now = time.perf_counter()
            if last_tick is not None:
                instant_fps = 1.0 / max(now - last_tick, 1e-6)
                run_fps = instant_fps if run_fps is None else (run_fps * 0.85 + instant_fps * 0.15)
            last_tick = now

        if last_frame is not None:
            cv2.imshow(win, draw_hud(last_frame.copy(), paused, video_path.name, idx, total, fps, run_fps))

        key = cv2.waitKey(delay if not paused else 50) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            time.sleep(0.1)
        elif key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            idx = 0
            paused = False

    cap.release()
    cv2.destroyAllWindows()


def process_headless(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = total if max_frames is None else min(total, max_frames)
    count = 0
    start = time.perf_counter()
    print(f"\nProcessing camera-space scanline experiment: {video_path.name}")
    while count < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        analyze_frame(frame)
        count += 1
        if count % 30 == 0:
            elapsed = max(time.perf_counter() - start, 1e-6)
            print(f"  {count}/{limit} frames | run {count / elapsed:.1f} fps", end="\r")
    cap.release()
    elapsed = max(time.perf_counter() - start, 1e-6)
    print(f"\nDone: {count} frames processed | run {count / elapsed:.1f} fps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    video_path = resolve_video(args.video)
    if args.headless:
        process_headless(video_path, args.max_frames)
    else:
        play(video_path)


if __name__ == "__main__":
    main()
