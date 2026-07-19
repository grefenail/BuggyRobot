"""Viewer/exporter for the standalone scanline lane experiment.

Run:
    python run_scanline.py
    python run_scanline.py IMG_6741.MP4
    python run_scanline.py IMG_6741.MP4 --export scanline_full.mp4
    python run_scanline.py IMG_6741.MP4 --headless

Controls:
    Space  pause/play
    r      restart
    q/Esc  quit
"""

import argparse
import time
from pathlib import Path

import cv2

from scanline_lane_experiment import analyze_frame, resolve_video


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

    win = "Scanline Lane Experiment"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"\nPlaying scanline experiment: {video_path.name}")
    print(f"  {total} frames @ {fps:.1f} fps")
    print("  Space=pause  r=restart  q/Esc=quit\n")

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


def export_video(video_path, output_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    count = 0
    start = time.perf_counter()

    print(f"\nExporting scanline experiment: {video_path.name} -> {output_path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        out = analyze_frame(frame)
        if writer is None:
            h, w = out.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open writer for: {output_path}")
        writer.write(out)
        count += 1
        if count % 30 == 0:
            elapsed = max(time.perf_counter() - start, 1e-6)
            print(f"  {count}/{total} frames | run {count / elapsed:.1f} fps", end="\r")

    cap.release()
    if writer is not None:
        writer.release()
    elapsed = max(time.perf_counter() - start, 1e-6)
    print(f"\nDone: {count} frames written to {output_path} | run {count / elapsed:.1f} fps")


def process_headless(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = total if max_frames is None else min(total, max_frames)
    count = 0
    start = time.perf_counter()

    print(f"\nProcessing scanline experiment headless: {video_path.name}")
    print(f"  {total} frames @ {fps:.1f} fps")
    if max_frames is not None:
        print(f"  Limit: {limit} frames")

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
    print(f"\nDone: {count} frames processed headless | run {count / elapsed:.1f} fps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?")
    parser.add_argument("--export", metavar="OUTPUT", help="Export full scanline output video")
    parser.add_argument("--headless", action="store_true", help="Process without opening the GUI window")
    parser.add_argument("--max-frames", type=int, help="Limit frames for headless processing")
    args = parser.parse_args()

    video_path = resolve_video(args.video)
    if args.export:
        export_video(video_path, Path(args.export))
    elif args.headless:
        process_headless(video_path, args.max_frames)
    else:
        play(video_path)


if __name__ == "__main__":
    main()
