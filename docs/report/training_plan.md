# Training Plan

## Goal
Train or fine-tune five different pose-estimation architectures for topic 22 and record enough information for final comparison and demo selection.

## Completed architectures
- YOLO Pose.
- RTMPose-S.
- HRNet-W32 UDP.
- Lite-HRNet-18.
- ViTPose-Small Simple Head.

Mandatory progress: 5/5 complete.

## Protocol
- Dataset: COCO Keypoints subset 5k.
- Split: 5000 train / 1000 validation / 1000 test.
- Main runs: 30 epochs.
- Hardware target: RTX 3060.
- Frameworks: Ultralytics for YOLO Pose, MMPose for the other four models.

## Rules
- Real training commands are run manually by the user.
- The agent records completed results from provided logs and generated files.
- Missing values are recorded as `not measured`.
- Numeric training results live in `outputs/metrics/*.csv` and `outputs/metrics/*.json`.

## Completed outputs
- YOLO Pose extended run output under `runs/pose/outputs/models/yolo_pose/extended_yolo_pose_coco_subset_5k`.
- RTMPose-S output under `outputs/models/rtmpose`.
- HRNet-W32 UDP output under `outputs/models/hrnet`.
- Lite-HRNet-18 output under `outputs/models/litehrnet`.
- ViTPose-Small Simple Head output under `outputs/models/vitpose`.

