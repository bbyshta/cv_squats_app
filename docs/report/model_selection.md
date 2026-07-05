# Fine-Tuning Architecture Selection

## Goal
Select one final pose-estimation model for the exercise-technique video demo after comparing five different architectures by measured quality, speed, checkpoint size, integration practicality, video suitability, and limitations.

## Mandatory architecture candidates
1. YOLO Pose.
2. RTMPose.
3. HRNet.
4. Lite-HRNet.
5. ViTPose.

These are different architecture families, not merely different sizes of one model.

## Completed training status
All five mandatory candidates have completed real user-run training or fine-tuning. Numeric training metrics are stored in `outputs/metrics/training_runs.csv`, `outputs/metrics/model_comparison.csv`, and per-model JSON files.

## Completed benchmark status
Phase 7C 100-image benchmark is completed for all five models. Speed and checkpoint-size values are stored in `outputs/metrics/inference_speed.csv` and copied into `outputs/metrics/model_comparison.csv`.

## Final selection
The selected final demo model is **RTMPose-S** (`rtmpose_s`) with the MMPose backend.

Selection record: `outputs/metrics/final_model_selection.json`.

RTMPose-S is selected as the best practical demo model because it has the strongest overall balance of measured image inference speed, adequate COCO keypoint quality, moderate checkpoint size, and already verified MMPose training/benchmark compatibility. It is not selected only by highest AP: HRNet-W32 UDP has the highest recorded MMPose AP but is much slower and larger. It is not selected only by fastest FPS either: speed is considered together with quality, size, integration risk, and remaining demo requirements.

COCO quality metrics and exercise-video/demo stability observations are kept separate. Phase 7C measured image inference speed, not real video throughput. Real-time video performance is therefore not claimed.

## Exercise scope note
The selected model remains **RTMPose-S**. The final demo exercise changed from push-ups to squats after a manual push-up video prototype showed unreliable critical-keypoint confidence for technique analysis. Model selection and exercise selection are separate decisions: RTMPose-S is still the selected pose model, while squats are now the planned demo exercise because upright full-body pose is expected to be more stable for hips, knees, and ankles.

## Candidate interpretation
- HRNet-W32 UDP: strongest recorded MMPose keypoint AP, but too slow and large for the demo tradeoff.
- RTMPose-S: best measured practical speed/quality balance for the final demo.
- ViTPose-Small Simple Head: usable speed and slightly higher AP than RTMPose-S, but larger and more complex.
- YOLO Pose: small and fast, but its recorded pose metric is lower in this experiment and is not directly equivalent to MMPose AP.
- Lite-HRNet-18: smallest checkpoint, but not competitive on measured speed or MMPose AP in this run.

## Selection constraints
- The final demo app must use only the selected final model.
- Demo/app commands use `.venv-mmpose` as the single runtime environment.
- Demo scope is video-only.
- RTMPose-S is a top-down model; the demo passes each processed frame with `bbox_strategy = whole_frame`.
- Whole-frame bbox inference is a project limitation: if the person occupies a small part of the frame, keypoint quality may degrade.
- The removed early prototype backend must not be used as fallback.
- Video support must be evaluated at least for the selected model before final defense.
- Real-time video performance must not be claimed without a real video benchmark.
- Typical squat examples and errors are not collected yet and must be recorded from observed examples in Phase 9.
