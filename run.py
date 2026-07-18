"""
Test 2 — mrhwick Medium article algorithm.

Run:
    python buggyvis1/run.py
    python buggyvis1/run.py IMG_6744.MP4
    python buggyvis1/run.py --headless IMG_6744.MP4
    python buggyvis1/run.py --live-input --headless --publish-ros

Controls: Space=pause  r=restart  d=debug steps  q/Esc=quit
"""

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

import cv2
import numpy as np
from detect_lanes import detect_lanes, detect_lanes_debug, detect_lanes_with_coords
from waypoints import add_approx_ground_waypoints
import config

VIDEO_DIR        = ROOT / "vids"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def resolve_video(name):
    if name is None:
        videos = sorted(p for p in VIDEO_DIR.iterdir()
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
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


def draw_hud(frame, paused, name, idx, total, debug):
    status = "PAUSED" if paused else "PLAYING"
    dbg    = " | DEBUG ON" if debug else ""
    text   = f"{name} | {status} | {idx}/{total} | Space: pause  r: restart  d: debug  q: quit{dbg}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (0, 0), (tw + 18, th + bl + 18), (0, 0, 0), -1)
    cv2.putText(frame, text, (9, th + 9), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def label(img, text, is_gray=False):
    """Add a step label to a debug image."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if is_gray else img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 10, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


DEBUG_WIN = "Debug Steps (0-4)"


def _tile_debug_images(imgs, cols=3, thumb_w=380):
    """
    Resize a list of BGR debug images to a common thumbnail width and
    arrange them into one grid image, so all pipeline steps show in a
    single window instead of one full-size OS window per step (which
    over X11 forwarding turns into a pile of oversized, overlapping
    windows you have to hunt down and close individually).
    """
    thumbs = []
    for img in imgs:
        h, w = img.shape[:2]
        thumbs.append(cv2.resize(img, (thumb_w, int(h * thumb_w / w))))
    thumb_h = max(t.shape[0] for t in thumbs)
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    padded = [
        np.vstack([t, np.zeros((thumb_h - t.shape[0], thumb_w, 3), dtype=np.uint8)])
        if t.shape[0] < thumb_h else t
        for t in thumbs
    ]
    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i:i + cols]
        row += [blank] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _format_waypoints(coords):
    waypoints = coords.get("center_waypoints_m_approx")
    if not waypoints:
        return "center waypoints: none"

    parts = []
    for i, item in enumerate(waypoints):
        ground = item.get("ground_m")
        bird_px = item.get("bird_px")
        if ground is None:
            parts.append(f"{i}:px={bird_px} m=None")
        else:
            parts.append(
                f"{i}:px={bird_px} x={ground['x_forward_m']:.2f}m "
                f"y={ground['y_left_m']:.2f}m"
            )
    return " | ".join(parts)


def play(video_path=None, print_waypoints=False, print_every=1, headless=False,
         live_input=False, live_topic=config.ROS_DEFAULT_INPUT_IMAGE_TOPIC,
         camera_height_m=config.APPROX_CAMERA_HEIGHT_M,
         camera_pitch_deg=config.APPROX_CAMERA_PITCH_DEG,
         ros_publisher=None, ros_every=1):
    if live_input:
        from ros_input import RosImageSource
        source = RosImageSource(live_topic)
        fps = 30.0
        total = 0
        source_name = live_topic
    else:
        source = cv2.VideoCapture(str(video_path))
        if not source.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")
        source.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        fps = source.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        source_name = video_path.name
    delay = max(1, int(1000 / fps))

    paused     = False
    debug      = False
    last_frame = None
    last_steps = None
    idx        = 0

    WIN_MAIN = "Test 2 — mrhwick Algorithm"
    if not headless:
        cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)

    mode = "Processing" if headless else "Playing"
    print(f"\n{mode}: {source_name}")
    if live_input:
        print("  Live ROS image input")
    else:
        print(f"  {total} frames @ {fps:.1f} fps")
    if headless:
        print("  Headless mode: display and keyboard controls disabled")
    else:
        print("  Space=pause  r=restart  d=debug steps  q/Esc=quit")
    if print_waypoints:
        print(f"  Printing center waypoints every {print_every} frame(s)")
        print("  Dots on the image are numbered to match terminal point indexes")
    if ros_publisher is not None:
        print(f"  Publishing center waypoints to ROS 2 every {ros_every} frame(s)")
    print()

    need_coords = print_waypoints or ros_publisher is not None

    frame_count = 0
    while True:
        if not paused:
            ok, frame = source.read()
            print(f"Fetching frame {frame_count}", end="\r")
            frame_count += 1
            if not ok:
                if live_input or headless:
                    break
                source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                idx = 0
                continue

            # add gaussian blur to reduce flicker of waypoints between frames
            frame = cv2.GaussianBlur(frame, (11, 11), 0)

            coords = None
            if live_input:
                #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                pass
            else:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if need_coords and not debug:
                last_frame, coords = detect_lanes_with_coords(frame)
            elif debug:
                last_frame, *last_steps = detect_lanes_debug(frame)
            else:
                last_frame = detect_lanes(frame)
            idx += 1

            if coords is not None:
                coords = add_approx_ground_waypoints(
                    coords,
                    camera_height_m=camera_height_m,
                    pitch_deg=camera_pitch_deg,
                )
                if print_waypoints and idx % print_every == 0:
                    ts_ms = round((idx - 1) / fps * 1000.0, 1)
                    print(f"frame={idx - 1} time_ms={ts_ms} {_format_waypoints(coords)}")
            if ros_publisher is not None and idx % ros_every == 0:
                ros_publisher.publish(coords, last_frame)

        if not headless and last_frame is not None:
            total_label = total if total else "live"
            display = draw_hud(last_frame.copy(), paused, source_name,
                               idx, total_label, debug)
            cv2.imshow(WIN_MAIN, display)

        if not headless and debug and last_steps is not None:
            gray, edges, roi, hough, bird = last_steps
            imgs = [
                label(bird,   "Step 0 - Bird's-eye perspective"),
                label(gray,   "Step 1 — Grayscale (white mask applied)", is_gray=True),
                label(edges,  "Step 2 — Canny Edges",  is_gray=True),
                label(roi,    "Step 3 — ROI Mask",     is_gray=True),
                label(hough,  "Step 4 — Hough  red=left  blue=right  yellow=rejected"),
            ]
            cv2.namedWindow(DEBUG_WIN, cv2.WINDOW_NORMAL)
            cv2.imshow(DEBUG_WIN, _tile_debug_images(imgs))
        elif not headless and not debug:
            try:
                cv2.destroyWindow(DEBUG_WIN)
            except cv2.error:
                pass

        if headless:
            continue

        key = cv2.waitKey(delay if not paused else 50) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
            time.sleep(0.1)
        elif key == ord("r") and not live_input:
            source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            idx = 0
            paused = False
        elif key == ord("d"):
            debug = not debug
            if not debug:
                last_steps = None
            print(f"Debug {'ON' if debug else 'OFF'}")

    if live_input:
        source.close()
    else:
        source.release()
    if not headless:
        cv2.destroyAllWindows()


def export_video(video_path, output_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read any frames from: {video_path}")
    # Ignore unreliable MP4 rotation metadata and orient the camera explicitly.
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    first = detect_lanes(frame)
    h, w  = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for: {output_path}")

    print(f"Exporting {video_path.name} -> {output_path}")
    writer.write(first)
    count = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        writer.write(detect_lanes(frame))
        count += 1
        if count % 30 == 0:
            print(f"  {count}/{total} frames", end="\r")

    cap.release()
    writer.release()
    print(f"\nDone: {count} frames written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?")
    parser.add_argument("--export", metavar="OUTPUT",
                         help="Export processed video to OUTPUT instead of playing it")
    parser.add_argument("--headless", action="store_true",
                        help="Process without opening display windows (for servers/containers)")
    parser.add_argument("--live-input", action="store_true",
                        help="Read frames from a ROS image topic instead of a video file")
    parser.add_argument("--live-topic", default=config.ROS_DEFAULT_INPUT_IMAGE_TOPIC,
                        help=f"ROS input image topic. Default: {config.ROS_DEFAULT_INPUT_IMAGE_TOPIC}")
    parser.add_argument("--print-waypoints", action="store_true",
                        help="Print center waypoint pixels/meters while playing")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Print every Nth frame when --print-waypoints is set")
    parser.add_argument("--camera-height-m", type=float,
                        default=config.APPROX_CAMERA_HEIGHT_M,
                        help="Approximate camera height for printed meter waypoints")
    parser.add_argument("--camera-pitch-deg", type=float,
                        default=config.APPROX_CAMERA_PITCH_DEG,
                        help="Approximate camera downward pitch for printed meter waypoints")
    parser.add_argument("--publish-ros", action="store_true",
                        help="Publish center waypoints as a nav_msgs/Path over ROS 2 (requires rclpy)")
    parser.add_argument("--ros-topic", default=config.ROS_DEFAULT_TOPIC,
                        help=f"ROS 2 topic to publish on. Default: {config.ROS_DEFAULT_TOPIC}")
    parser.add_argument("--ros-frame-id", default=config.ROS_DEFAULT_FRAME_ID,
                        help=f"frame_id for published poses. Default: {config.ROS_DEFAULT_FRAME_ID}")
    parser.add_argument("--ros-image-topic", default=config.ROS_DEFAULT_IMAGE_TOPIC,
                        help=f"ROS 2 lane-overlay image topic. Default: {config.ROS_DEFAULT_IMAGE_TOPIC}")
    parser.add_argument("--ros-image-frame-id", default=config.ROS_DEFAULT_IMAGE_FRAME_ID,
                        help=f"frame_id for the overlay image. Default: {config.ROS_DEFAULT_IMAGE_FRAME_ID}")
    parser.add_argument("--ros-every", type=int, default=1,
                        help="Publish every Nth frame when --publish-ros is set")
    args = parser.parse_args()
    if args.print_every < 1:
        raise ValueError("--print-every must be >= 1")
    if args.ros_every < 1:
        raise ValueError("--ros-every must be >= 1")
    if args.live_input and args.video:
        parser.error("the video argument cannot be used with --live-input")
    if args.live_input and args.export:
        parser.error("--export cannot be used with --live-input")
    video_path = None if args.live_input else resolve_video(args.video)
    if args.export:
        export_video(video_path, Path(args.export))
        return

    ros_publisher = None
    if args.publish_ros:
        from ros_publish import WaypointPublisher
        ros_publisher = WaypointPublisher(
            topic=args.ros_topic,
            frame_id=args.ros_frame_id,
            image_topic=args.ros_image_topic,
            image_frame_id=args.ros_image_frame_id,
        )

    try:
        play(
            video_path,
            print_waypoints=args.print_waypoints,
            print_every=args.print_every,
            headless=args.headless,
            live_input=args.live_input,
            live_topic=args.live_topic,
            camera_height_m=args.camera_height_m,
            camera_pitch_deg=args.camera_pitch_deg,
            ros_publisher=ros_publisher,
            ros_every=args.ros_every,
        )
    finally:
        if ros_publisher is not None:
            ros_publisher.close()


if __name__ == "__main__":
    main()
