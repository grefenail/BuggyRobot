"""
Export a processed (lane-detected) video to a file you can download.

Run:
    python buggyvis1/export_video.py IMG_6743.MP4
    python buggyvis1/export_video.py IMG_6743.MP4 --out my_output.mp4

The input is looked up in vids/ if it isn't found as a direct path
(same lookup as run.py). The output defaults to "<input>_output.mp4"
saved right next to the input video.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run import resolve_video, export_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video filename or path")
    parser.add_argument("--out", help="Output file path (default: <input>_output.mp4)")
    args = parser.parse_args()

    video_path = resolve_video(args.video)
    out_path = Path(args.out) if args.out else video_path.with_name(video_path.stem + "_output.mp4")

    export_video(video_path, out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
