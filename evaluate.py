"""Evaluate lane-boundary detection against the white-pixel mask.

For every video frame, 50 points are sampled along each fitted lane curve.
A point is a hit when a non-zero white-mask pixel exists within ``radius``
pixels of it.  Left and right scores are kept separate throughout the
report, then averaged per video and across videos.

Examples:
    python evaluate.py
    python evaluate.py IMG_6741.MP4 IMG_6742.MP4 --json scores.json
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

# The lane modules live in the sibling pipeline directory.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

from detect_lanes import detect_lanes_birdeye_coords
from ipm import warp_to_birdseye
from step1_mask import apply_white_mask
import config


VIDEO_DIR = ROOT / "vids"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def _sample_polyline(curve, count):
    """Return ``count`` evenly spaced pixel points along a polyline."""
    if not curve:
        return None
    points = np.asarray(curve, dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)

    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 0.0:
        return np.repeat(points[:1], count, axis=0)

    distances = np.linspace(0.0, total, count)
    sampled = []
    for distance in distances:
        segment = min(int(np.searchsorted(cumulative, distance, side="right") - 1),
                      len(points) - 2)
        span = cumulative[segment + 1] - cumulative[segment]
        fraction = 0.0 if span <= 0.0 else (distance - cumulative[segment]) / span
        sampled.append(points[segment] * (1.0 - fraction) + points[segment + 1] * fraction)
    return np.asarray(sampled, dtype=np.float32)


def _curve_score(mask, curve, count=50, radius=3):
    """Return matching-point count and score for one lane curve."""
    sampled = _sample_polyline(curve, count)
    if sampled is None:
        return 0, 0, 0.0

    height, width = mask.shape[:2]
    hits = 0
    for x_float, y_float in sampled:
        x, y = int(round(x_float)), int(round(y_float))
        x0, x1 = max(0, x - radius), min(width - 1, x + radius)
        y0, y1 = max(0, y - radius), min(height - 1, y + radius)
        if x0 <= x1 and y0 <= y1 and np.any(mask[y0:y1 + 1, x0:x1 + 1] > 0):
            hits += 1
    return hits, count, 100.0 * hits / count


def evaluate_video(video_path, points=50, radius=3):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    frame_count = 0
    left_scores, right_scores = [], []
    left_hits = right_hits = 0
    left_total = right_total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        coords = detect_lanes_birdeye_coords(frame)

        oriented = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if config.ROTATE_CW else frame
        mask = apply_white_mask(warp_to_birdseye(oriented))
        left_hit, left_n, left_score = _curve_score(mask, coords.get("left_curve"), points, radius)
        right_hit, right_n, right_score = _curve_score(mask, coords.get("right_curve"), points, radius)

        left_scores.append(left_score)
        right_scores.append(right_score)
        left_hits += left_hit
        right_hits += right_hit
        left_total += left_n
        right_total += right_n
        frame_count += 1

    cap.release()
    if frame_count == 0:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")

    left_average = float(np.mean(left_scores))
    right_average = float(np.mean(right_scores))
    return {
        "video": Path(video_path).name,
        "frames": frame_count,
        "left_score": round(left_average, 3),
        "right_score": round(right_average, 3),
        "video_score": round((left_average + right_average) / 2.0, 3),
        "left_hits": left_hits,
        "left_points": left_total,
        "right_hits": right_hits,
        "right_points": right_total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", help="Video paths or names in vids/")
    parser.add_argument("--points", type=int, default=50, help="Samples per side (default: 50)")
    parser.add_argument("--radius", type=int, default=3, help="White-pixel search radius (default: 3)")
    parser.add_argument("--json", metavar="PATH", help="Also write results as JSON")
    args = parser.parse_args()
    if args.points < 1 or args.radius < 0:
        parser.error("--points must be positive and --radius cannot be negative")

    paths = []
    for name in args.videos:
        path = Path(name)
        if not path.exists():
            path = VIDEO_DIR / name
        if not path.exists():
            parser.error(f"Cannot find video: {name}")
        paths.append(path)
    if not paths:
        paths = sorted(p for p in VIDEO_DIR.iterdir()
                       if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not paths:
        parser.error(f"No videos found in {VIDEO_DIR}")

    results = [evaluate_video(path, args.points, args.radius) for path in paths]
    overall = float(np.mean([item["video_score"] for item in results]))
    report = {
        "points_per_side": args.points,
        "radius_pixels": args.radius,
        "videos": results,
        "overall_score": round(overall, 3),
    }

    print("\nLane evaluation (score is percent)\n")
    for item in results:
        print(f"{item['video']}: left={item['left_score']:.2f}%  "
              f"right={item['right_score']:.2f}%  video={item['video_score']:.2f}%  "
              f"frames={item['frames']}")
    print(f"\nOverall score: {overall:.2f}%")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved JSON report to {args.json}")


if __name__ == "__main__":
    main()
