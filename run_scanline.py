"""Viewer/exporter for the standalone scanline lane experiment.

Run:
    python run_scanline.py
    python run_scanline.py IMG_6741.MP4
    python run_scanline.py IMG_6741.MP4 --export scanline_full.mp4

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


def draw_hud(frame, paused, name, idx, total, source_fps, processing_fps):
    status = "PAUSED" if paused else "PLAYING"
    h, w = frame.shape[:2]
    text = (f"{name} | {status} | {idx}/{total} | video {source_fps:.1f} FPS | "
            f"processing {processing_fps:.1f} FPS | out {w}x{h} | Space pause  r restart  q quit")
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (0, 0), (min(tw + 18, w), th + bl + 18), (0, 0, 0), -1)
    cv2.putText(frame, text, (9, th + 9), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def play(video_path, headless=False):
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
    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    processing_fps = 0.0

    win = "Scanline Lane Experiment"
    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"\nPlaying scanline experiment: {video_path.name}")
    print(f"  {total} frames @ {fps:.1f} fps")
    print("  Space=pause  r=restart  q/Esc=quit\n")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if headless:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                idx = 0
                continue
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            last_frame = analyze_frame(frame)
            idx += 1
            fps_window_frames += 1
            fps_elapsed = time.perf_counter() - fps_window_start
            if fps_elapsed >= 0.5:
                processing_fps = fps_window_frames / fps_elapsed
                fps_window_start = time.perf_counter()
                fps_window_frames = 0
                if headless:
                    print(f"Processing frame {idx}/{total} | FPS: {processing_fps:.1f}", end="\r")

        if last_frame is not None and not headless:
            cv2.imshow(win, draw_hud(last_frame.copy(), paused, video_path.name, idx, total, fps, processing_fps))

        if headless:
            continue

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
    if not headless:
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
            print(f"  {count}/{total} frames", end="\r")

    cap.release()
    if writer is not None:
        writer.release()
    print(f"\nDone: {count} frames written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?")
    parser.add_argument("--export", metavar="OUTPUT", help="Export full scanline output video")
    parser.add_argument("--headless", action="store_true", help="Process without opening a display window")
    args = parser.parse_args()

    video_path = resolve_video(args.video)
    if args.export:
        export_video(video_path, Path(args.export))
    else:
        play(video_path, headless=args.headless)


if __name__ == "__main__":
    main()
