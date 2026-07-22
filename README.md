# BuggyRobot — Bird's-Eye Lane Detection

A lane/track-boundary detection pipeline built with OpenCV. It warps
the camera view into a bird's-eye (top-down) perspective, detects the
two boundary lines of the current lane, fits a line or gentle curve to
each side, and overlays the result back onto the original video.

## How it works

The pipeline runs in six steps, all defined in their own module:

1. **`ipm.py`** — warps the camera frame into bird's-eye space using a
   fixed perspective transform (`IPM_SRC_FRAC` / `IPM_DST_FRAC` in
   `config.py`).
2. **`step1_mask.py`** — HSV color mask that keeps only white/light-grey
   (lane paint) pixels, discarding everything else before edge detection.
3. **`step2_canny.py`** — Canny edge detection on the masked image.
4. **`step3_roi.py`** + **`roi_search.py`** — restricts detection to a
   trapezoidal region of interest in bird's-eye space, sized to hug the
   expected lane width.
5. **`step4_hough.py`** — `HoughLinesP` finds straight segments within
   the ROI and classifies them left/right by position relative to the
   lane center.
6. **`step5_fit.py`** — fits a line (or a quadratic, only when
   significantly better than a line) to each side independently, with
   smoothing, an outlier-jump gate, and a stale-lock reset so the
   tracked lane recovers on its own if detection is lost.
7. **`step6_draw.py`** — fills the area between the two boundaries and
   warps the result back onto the original camera view.

## Pipeline 2: scanline detector

Pipeline 2 is a scanline-based alternative to the original Hough/curve-fit
pipeline. It uses the same bird's-eye coordinate idea, but samples horizontal
rows in bird's-eye space and looks for left/right white-pixel peaks that match
the expected lane width.

The processing order is:

1. **Resize** the camera frame with `PROCESS_SCALE`.
2. **Warp to bird's-eye view** using the configured perspective transform.
3. **Build a relative white mask** from the bird's-eye image.
4. **Sample scanlines** from the lower lane area toward the top.
5. **Find and pair peaks** that look like left/right lane boundaries.
6. **Fit straight lane lines** through the accepted scanline points.
7. **Sample center waypoints** between the two fitted boundaries.

Pipeline 2 is split across focused modules:

```
pipeline2/
    scanline_preprocess.py   resize, bird's-eye warp, white mask
    scanline_peaks.py        scanline smoothing, peak finding, pair scoring
    scanline_lines.py        line fitting and lane geometry helpers
    scanline_visual.py       lane fill, center dots, debug drawing
    scanline_backend.py      orchestration and public detector API
    detect_lanes.py          runner-facing pipeline wrapper
```

Run it with:

```
python run_pipeline2.py IMG_6744.MP4
python run_pipeline2.py IMG_6744.MP4 --headless
python run_pipeline2.py IMG_6744.MP4 --headless --print-waypoints --print-every 30
```

Pipeline 2 can also publish the center waypoints and overlay image to ROS 2:

```
python run_pipeline2.py IMG_6744.MP4 --headless --publish-ros
```

That requires a sourced ROS 2 environment with `rclpy`, `nav_msgs`,
`geometry_msgs`, and `sensor_msgs` available.

## Evaluation and speed

The table below is a bounded local benchmark on the first 120 frames of each
full sample video (`IMG_6741.MP4` through `IMG_6744.MP4`). Each lane boundary
is sampled at 50 points, and a point counts as a hit if white-mask pixels are
found within a 3 px radius. This is a mask-hit proxy, not hand-labeled ground
truth.

| Detector | Average FPS | Overall score |
| --- | ---: | ---: |
| Original pipeline | `55.92` | `13.52%` |
| Pipeline 2 scanline | `18.95` | `9.44%` |

Per-video results:

| Detector | Video | FPS | Left score | Right score | Video score |
| --- | --- | ---: | ---: | ---: | ---: |
| Original pipeline | `IMG_6741.MP4` | `53.71` | `0.02%` | `12.18%` | `6.10%` |
| Original pipeline | `IMG_6742.MP4` | `55.19` | `1.62%` | `12.77%` | `7.19%` |
| Original pipeline | `IMG_6743.MP4` | `56.99` | `45.82%` | `29.47%` | `37.64%` |
| Original pipeline | `IMG_6744.MP4` | `60.63` | `0.07%` | `6.22%` | `3.14%` |
| Pipeline 2 scanline | `IMG_6741.MP4` | `19.55` | `0.00%` | `6.58%` | `3.29%` |
| Pipeline 2 scanline | `IMG_6742.MP4` | `19.36` | `13.55%` | `12.50%` | `13.03%` |
| Pipeline 2 scanline | `IMG_6743.MP4` | `17.50` | `21.98%` | `18.25%` | `20.12%` |
| Pipeline 2 scanline | `IMG_6744.MP4` | `19.84` | `0.00%` | `2.67%` | `1.33%` |

## Example output

![Lane detection tracking the track boundary through a curve](vids/IMG_6741_preview.gif)

The green fill tracks the current lane as the track curves, with the
yellow border marking the two detected boundary lines. This GIF is a
downsampled preview — the full-quality clip is at
[`vids/IMG_6741_preview.mp4`](vids/IMG_6741_preview.mp4) (first 300
frames / ~10s of `IMG_6741.MP4`, full resolution and frame rate).

## Coordinate output

The orange numbered spots mark sampled centerline waypoints. The table
below lists the matching coordinates for the spots shown on the line.

![Numbered centerline coordinate spots](vids/IMG_6743_waypoint_dots_check.png)

Example from [`vids/IMG_6743_waypoints_terminal_test.json`](vids/IMG_6743_waypoints_terminal_test.json),
frame `0`, timestamp `0.0 ms`:

| Spot | Bird's-eye pixel | Vehicle pixel | Approx robot meters |
| ---: | --- | --- | --- |
| `0` | `[382, 1216]` | `[395.2, 998.4]` | `x=1.32 m, y=-0.06 m` |
| `1` | `[373, 1088]` | `[379.5, 928.0]` | `x=1.53 m, y=-0.04 m` |
| `2` | `[364, 960]` | `[365.65, 865.88]` | `x=1.78 m, y=-0.01 m` |
| `3` | `[355, 832]` | `[353.33, 810.67]` | `x=2.05 m, y=0.02 m` |
| `4` | `[346, 704]` | `[342.32, 761.26]` | `x=2.37 m, y=0.05 m` |
| `5` | `[336, 576]` | `[331.2, 716.8]` | `x=2.75 m, y=0.10 m` |
| `6` | `[327, 448]` | `[322.29, 676.57]` | `x=3.20 m, y=0.14 m` |
| `7` | `[318, 320]` | `[314.18, 640.0]` | `x=3.73 m, y=0.20 m` |
| `8` | `[309, 192]` | `[306.78, 606.61]` | `x=4.39 m, y=0.27 m` |
| `9` | `[300, 64]` | `[300.0, 576.0]` | `x=5.22 m, y=0.36 m` |

## Requirements

```
pip install -r requirements.txt
```

Four sample input videos are included in [`vids/`](vids/)
(`IMG_6741.MP4`–`IMG_6744.MP4`). Drop any additional videos into that
same folder to run the pipeline on them.

## Usage

**Play a video live**, with an on-screen HUD (`d` toggles debug step
windows, `space` pauses, `r` restarts):
```
python run.py IMG_6743.MP4
python run.py IMG_6743.MP4 --print-waypoints --print-every 10 --camera-height-m 1.0
```
The waypoint print mode draws numbered orange dots on the centerline
and prints the matching center waypoint pixels/meters in the terminal.

**Step through frame-by-frame** with live trackbars for tuning the
bird's-eye calibration and ROI margins (`s` saves the current trackbar
values back into `config.py`):
```
python run_steps.py IMG_6743.MP4
```

**Export a processed video file** to disk:
```
python export_video.py IMG_6743.MP4
python export_video.py IMG_6743.MP4 --out my_output.mp4
```

**Export bird's-eye lane coordinates** to JSON:
```
python export_birdeye_coords.py IMG_6743.MP4
python export_birdeye_coords.py IMG_6743.MP4 --out coords.json
python export_birdeye_coords.py IMG_6743.MP4 --every 10
python export_birdeye_coords.py IMG_6743.MP4 --camera-height-m 1.0
```
The JSON includes sampled `center_waypoints_px` and approximate
`center_waypoints_m_approx` for ROS-style path testing. The meter
values use placeholder iPhone 16 intrinsics and estimated mounting
extrinsics; replace them with real calibration before final testing.

All scripts fall back to the first video found in `vids/` if no
filename is given.

**Publish waypoints and the lane-overlay image to ROS 2** while playing
(requires `rclpy` — install
via a sourced ROS 2 distro, not pip):
```
python run.py IMG_6743.MP4 --publish-ros
python run.py IMG_6743.MP4 --publish-ros --ros-topic /lane_waypoints --ros-frame-id base_link
python run.py IMG_6743.MP4 --publish-ros --ros-image-topic /lane_overlay/image_raw
python run.py --live-input --headless --publish-ros
```
Publishes a `nav_msgs/Path` each processed frame, with each pose's
`x`/`y` set from the same `x_forward_m`/`y_left_m` ground waypoints
described above (`z` and orientation left at identity). The processed
camera frame, including the lane fill, boundaries, centerline, and numbered
waypoints, is published as a `sensor_msgs/Image` with `bgr8` encoding on
`/lane_overlay/image_raw`. `rclpy` is
only imported when `--publish-ros` is passed, so the rest of the
pipeline runs fine without ROS 2 installed. See
[`pipeline/ros_publish.py`](pipeline/ros_publish.py).

With `--live-input`, frames are read from `/camera/image_raw` instead of a
video file. Use `--live-topic` to select a different `sensor_msgs/Image`
topic. The subscriber accepts `bgr8`, `rgb8`, `bgra8`, `rgba8`, and `mono8`.

## Configuration

Original-pipeline tunables live in `pipeline/config.py`: perspective calibration,
ROI margins, Hough thresholds, and the fit/smoothing behavior. Pipeline 2
has the matching scanline tunables in `pipeline2/config.py`. See the comments
next to each value for what it controls and why it is set that way.

## Project structure

Entry-point scripts (what you actually run) live at the top level.
The original detector lives under `pipeline/`; the scanline detector
lives under `pipeline2/`.

```
run.py                 live video player with debug overlay
run_pipeline2.py       pipeline 2 scanline player, waypoint printing, ROS publishing
run_steps.py           step-by-step viewer with tuning trackbars
export_video.py        batch-process a video to a file
export_birdeye_coords.py
                       export fitted bird's-eye left/right lane coordinates

pipeline/
    config.py          tunable parameters for every step
    ipm.py              perspective warp (camera <-> bird's-eye)
    step1_mask.py        HSV white-paint mask
    step2_canny.py       Canny edge detection
    step3_roi.py         bird's-eye region-of-interest mask
    step4_hough.py       Hough line detection + left/right classification
    step5_fit.py         curve fitting, smoothing, stale-lock recovery
    step6_draw.py        lane fill + overlay drawing
    roi_search.py        ties the ROI + Hough steps together
    detect_lanes.py      orchestrates all the steps above into one pipeline

pipeline2/
    config.py             tunable parameters for pipeline 2
    ipm.py                perspective warp reused by pipeline 2
    step1_mask.py         relative white-mask helper
    scanline_preprocess.py
                          resize, bird's-eye warp, white mask
    scanline_peaks.py     scanline smoothing, peak finding, pair scoring
    scanline_lines.py     line fitting and lane geometry
    scanline_visual.py    lane fill, center dots, debug drawing
    scanline_backend.py   scanline orchestration and public backend API
    detect_lanes.py       runner-facing pipeline 2 wrapper
    ros_input.py          optional ROS 2 image subscriber
    ros_publish.py        optional ROS 2 waypoint/image publishers
    waypoints.py          pixel-to-ground waypoint conversion

vids/                  sample input videos + the example output clip
```
