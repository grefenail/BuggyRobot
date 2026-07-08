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
```

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

All three scripts fall back to the first video found in `vids/` if no
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
