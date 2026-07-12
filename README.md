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

## Configuration

All tunable parameters live in `pipeline/config.py` — perspective calibration,
ROI margins, Hough thresholds, and the fit/smoothing behavior. See the
comments next to each value for what it controls and why it's set the
way it is.

## Project structure

Entry-point scripts (what you actually run) live at the top level;
the pipeline's internals are grouped under `pipeline/`.

```
run.py                 live video player with debug overlay
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

vids/                  sample input videos + the example output clip
```
