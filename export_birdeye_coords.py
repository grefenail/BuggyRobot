"""
Export the fitted lane-boundary coordinates in bird's-eye pixels.

Run:
    python export_birdeye_coords.py IMG_6743.MP4
    python export_birdeye_coords.py IMG_6743.MP4 --out coords.json

The exported left_curve/right_curve points are the same bird's-eye
polylines used to draw the green fill and yellow borders.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from run import resolve_video
from step5_fit import reset
from detect_lanes import detect_lanes_birdeye_coords
from waypoints import add_approx_ground_waypoints
import config


def export_coords(
    video_path,
    output_path,
    every=1,
    fx=config.APPROX_CAMERA_FX_PX,
    fy=config.APPROX_CAMERA_FY_PX,
    camera_height_m=config.APPROX_CAMERA_HEIGHT_M,
    pitch_deg=config.APPROX_CAMERA_PITCH_DEG,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    reset()
    frames = []
    frame_idx = 0
    exported = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every == 0:
            coords = detect_lanes_birdeye_coords(frame)
            coords = add_approx_ground_waypoints(
                coords, fx, fy, camera_height_m, pitch_deg
            )
            coords["frame_index"] = frame_idx
            coords["timestamp_ms"] = round((frame_idx / fps) * 1000.0, 3)
            frames.append(coords)
            exported += 1

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames", end="\r")

    cap.release()

    data = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_idx,
        "sample_every_n_frames": every,
        "coordinate_space": "birdseye_pixels_and_approx_robot_meters",
        "approx_calibration": {
            "status": "placeholder_not_calibrated",
            "camera_model": "approx_iPhone_16_1x_rear_wide_26mm_equiv",
            "fx_px": fx,
            "fy_px": fy,
            "cx_px": "image_width / 2",
            "cy_px": "image_height / 2",
            "camera_height_m": camera_height_m,
            "camera_pitch_down_deg": pitch_deg,
            "robot_frame": "x_forward_m, y_left_m",
        },
        "frames": frames,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nDone: exported {exported} coordinate records to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", help="Input video filename or path")
    parser.add_argument("--out", help="Output JSON path")
    parser.add_argument("--every", type=int, default=1,
                        help="Export every Nth frame only. Default: 1")
    parser.add_argument("--camera-height-m", type=float,
                        default=config.APPROX_CAMERA_HEIGHT_M,
                        help="Approximate camera height above ground. Default: 1.0")
    parser.add_argument("--camera-pitch-deg", type=float,
                        default=config.APPROX_CAMERA_PITCH_DEG,
                        help="Approximate downward pitch from horizontal. Default: 15")
    parser.add_argument("--fx", type=float, default=config.APPROX_CAMERA_FX_PX,
                        help="Approximate focal length in pixels.")
    parser.add_argument("--fy", type=float, default=config.APPROX_CAMERA_FY_PX,
                        help="Approximate focal length in pixels.")
    args = parser.parse_args()

    if args.every < 1:
        raise ValueError("--every must be >= 1")

    video_path = resolve_video(args.video)
    out_path = Path(args.out) if args.out else video_path.with_name(
        video_path.stem + "_birdeye_coords.json"
    )
    export_coords(
        video_path,
        out_path,
        args.every,
        args.fx,
        args.fy,
        args.camera_height_m,
        args.camera_pitch_deg,
    )


if __name__ == "__main__":
    main()
