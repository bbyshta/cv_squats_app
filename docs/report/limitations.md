# Limitations

- Squat technique analysis is MVP rule-based logic over detected 2D keypoints.
- 2D pose estimation cannot fully assess biomechanics, joint loading, depth correctness, or injury risk.
- The demo uses `bbox_strategy = whole_frame`; keypoint quality may degrade when the person is small, partially visible, horizontal, or near the frame edge.
- No video benchmark has been run, so the project does not claim real-time video performance.
- The app is not medical or professional coaching advice.
- Result quality depends on camera angle, lighting, clothing, occlusion, and full-body visibility.
- JSONL history and Excel reports summarize generated run artifacts; they are not benchmark records.
- Push-ups were rejected for the final demo because the tested side/low camera setup produced low confidence on critical keypoints.
