# Demo App

## Purpose
The demo app shows how the selected RTMPose-S pose-estimation model can be integrated into a video-only squat technique analysis workflow. It is intended for project demonstration and report evidence, not for medical or professional coaching use.

## Scope
- Exercise: squat.
- Model: RTMPose-S (`rtmpose_s`).
- Backend: MMPose.
- Runtime: `.venv-mmpose`.
- Input type: video only.
- Inference strategy: top-down whole-frame bbox.

## Workflow
1. The user uploads a squat video in the Streamlit app.
2. The app saves the upload under project outputs.
3. The existing video pipeline processes every `frame_stride`-th frame.
4. RTMPose-S estimates COCO body keypoints.
5. The squat analysis computes knee angles, torso lean, phase, depth status, confidence validity, and warnings.
6. The app displays summary indicators, sampled annotated frames, optional annotated video preview, frame metrics, recent history, and download buttons.

## UI Parameters
- `frame_stride`: process every N-th frame.
- `max_frames`: optional cap on processed frames for faster manual smoke runs.
- `save annotated video`: save an annotated MP4 when the writer succeeds.
- `export Excel`: create an Excel report from the generated run artifacts.

## Output Indicators
- processed frames;
- valid keypoint frame ratio;
- usable knee frame ratio;
- reliable frame ratio;
- minimum and maximum selected knee angle;
- mean torso lean angle;
- depth status;
- phase counts: `top`, `middle`, `bottom`, `unknown`;
- warning summary.

## Saved Artifacts
Each run writes artifacts under `outputs/predictions/<run_id>/`:

- `video_summary.json`;
- `video_frames_metrics.csv`;
- `video_keypoints.json`;
- `sampled_frames/*.jpg`;
- `annotated_video.mp4`;
- `annotated_video_h264.mp4` when browser-compatible conversion succeeds;
- Excel report under `outputs/reports/` when enabled;
- JSONL history entry in `outputs/history/runs.jsonl`.

## Current Status
The video pipeline, squat analysis, JSON/CSV artifacts, sampled frames, annotated video, JSONL history, Excel export, and Streamlit UI are implemented. Manual squat pipeline smoke run `video_20260705T180520Z_c5b57c07` succeeded. Manual Streamlit smoke testing is pending.
