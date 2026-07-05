# Experiments

## Role
Concise report-facing experiment index. Full numeric metrics are stored in `outputs/metrics/*.csv` and `outputs/metrics/*.json`; verbose historical notes are archived under `docs/archive/`.

## Protocol Summary
- Dataset: COCO Keypoints compatible subset.
- Main split: 5000 train / 1000 validation / 1000 test images.
- Main fine-tuning budget: 30 epochs.
- Hardware target: RTX 3060.
- This is a resource-limited practice-project comparison, not a full COCO benchmark.

## Completed Training/Fine-Tuning
- YOLO Pose.
- RTMPose-S.
- HRNet-W32 UDP.
- Lite-HRNet-18.
- ViTPose-Small Simple Head.

Metrics sources:
- `outputs/metrics/training_runs.csv`;
- `outputs/metrics/model_comparison.csv`;
- `outputs/metrics/*_metrics.json`.

## Completed Evaluation
- Five-model 5-image smoke benchmark: `outputs/metrics/inference_speed_smoke.csv`.
- Five-model 100-image inference benchmark: `outputs/metrics/inference_speed.csv`.
- Final model selection: `outputs/metrics/final_model_selection.json`.

## Final Model
RTMPose-S (`rtmpose_s`) remains the selected final model. The selection is based on the recorded balance of keypoint quality, image inference speed, checkpoint size, integration practicality, and limitations.

## Manual Push-Up Prototype
- Status: technically processed by the video pipeline, but not suitable for final technique analysis.
- Observed run: 73 processed frames, `valid_keypoint_frame_ratio = 0.0`, mean critical confidence about `0.408`.
- Dominant warnings: `low_confidence`, `angle_not_reliable`, `partial_body_visible`, `unknown_phase`.
- Main reason: low side camera angle and horizontal body pose made wrists/elbows near the floor unreliable with whole-frame top-down RTMPose.
- Decision: switch final demo exercise to squats.
- This was not a benchmark and does not support any real-time claim.

## Manual Squat Smoke Run
- Run id: `video_20260705T180520Z_c5b57c07`.
- Input: `data/samples/squat_sample.mp4`.
- Artifacts: `outputs/predictions/video_20260705T180520Z_c5b57c07/`.
- Status: `ok`.
- Processed frames: 185 of 925 with `frame_stride = 5`.
- Ratios: valid keypoints `1.0`, usable knee `1.0`, reliable frames `1.0`.
- Squat summary: selected knee angle range `71.565` to `179.398`, mean torso lean `17.4967`, depth `sufficient`, warning frames `0`.
- Interpretation: the squat sample confirms the demo pipeline can produce reliable keypoints and useful squat indicators on at least one real manual smoke run.
- This was a smoke run, not a video benchmark.

## Remaining Experiment Work
- Example registry prepared in `docs/report/example_analysis.md` and `outputs/metrics/example_registry.csv`.
- Select/copy representative report frames for the registered examples.
- Save visualizations and explain observed error causes.
- Do not create fake examples.
