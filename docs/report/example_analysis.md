# Example Analysis Registry

## Purpose
This registry records six real squat demo runs for the final report: three successful examples and three failed or limited examples. It does not create new metrics, images, videos, or fake examples. Numeric values come from each run's `video_summary.json` when that file is present.

Registry CSV: `outputs/metrics/example_registry.csv`.

## Successful Examples

### GOOD-01
- Run id: `video_20260705T200108Z_ce5e8262`.
- Category: success.
- Summary metrics: 185 processed frames, valid keypoints `1.0`, usable knee `1.0`, reliable frames `1.0`, knee range `71.565` to `179.398`, mean torso lean `17.4967`, depth `sufficient`, warning frames `0`.
- Interpretation: strong success example with reliable pose tracking and clear squat depth.
- Reason for success: no frame warnings, full reliability, and clear top/middle/bottom motion.
- Artifacts: `outputs/predictions/video_20260705T200108Z_ce5e8262/`.
- Suggested figure: `outputs/figures/examples/GOOD-01.jpg`.

### GOOD-02
- Run id: `video_20260705T195844Z_af00c6de`.
- Category: success.
- Summary metrics: 109 processed frames, valid keypoints `1.0`, usable knee `1.0`, reliable frames `1.0`, knee range `88.321` to `180.0`, mean torso lean `13.6912`, depth `sufficient`, warning frames `0`.
- Interpretation: successful example with stable keypoints and balanced phase coverage.
- Reason for success: all processed frames are reliable and no warnings were recorded.
- Artifacts: `outputs/predictions/video_20260705T195844Z_af00c6de/`.
- Suggested figure: `outputs/figures/examples/GOOD-02.jpg`.

### GOOD-03
- Run id: `video_20260705T195618Z_aa082a89`.
- Category: success.
- Summary metrics: 92 processed frames, valid keypoints `0.9783`, usable knee `1.0`, reliable frames `0.9783`, knee range `96.818` to `179.64`, mean torso lean `14.5146`, depth `sufficient`, warning frames `12`.
- Dominant warnings: `possible_asymmetry:10`, `low_confidence:2`, `angle_tentative:2`.
- Interpretation: successful but imperfect run; useful for showing that the system keeps warnings even when the overall example is acceptable.
- Reason for success: the run reaches sufficient depth and keeps full usable-knee coverage despite minor warnings.
- Artifacts: `outputs/predictions/video_20260705T195618Z_aa082a89/`.
- Suggested figure: `outputs/figures/examples/GOOD-03.jpg`.

## Failed Or Limited Examples

### BAD-01
- Run id: `video_20260705T200353Z_f12d5240`.
- Category: technique_issue.
- Summary metrics: 76 processed frames, valid keypoints `1.0`, usable knee `1.0`, reliable frames `1.0`, knee range `113.526` to `178.601`, mean torso lean `10.5506`, depth `insufficient`, warning frames `76`.
- Dominant warnings: `insufficient_depth:76`.
- Interpretation: failed technique example with reliable keypoints but no frame reaching sufficient depth.
- Reason for failure: pose tracking is strong, but selected knee angle never reaches the configured depth threshold.
- Artifacts: `outputs/predictions/video_20260705T200353Z_f12d5240/`.
- Suggested figure: `outputs/figures/examples/BAD-01.jpg`.

### BAD-02
- Run id: `video_20260705T200306Z_99c8f7bc`.
- Category: technique_issue.
- Summary metrics: 53 processed frames, valid keypoints `0.0`, usable knee `0.5849`, reliable frames `0.0`, knee range `167.023` to `179.939`, mean torso lean `4.3328`, depth `insufficient`, warning frames `53`.
- Dominant warnings: `low_confidence:53`, `insufficient_depth:53`, `angle_tentative:31`, `knee_angle_unavailable:22`, `unknown_phase:22`.
- Interpretation: failed technique example where the run remains mostly top-phase or unknown and does not reach sufficient depth.
- Reason for failure: insufficient depth combined with low confidence and tentative or unavailable knee angles.
- Artifacts: `outputs/predictions/video_20260705T200306Z_99c8f7bc/`.
- Suggested figure: `outputs/figures/examples/BAD-02.jpg`.

### BAD-03
- Run id: `video_20260705T200227Z_b10ece35`.
- Category: technique_issue.
- Summary metrics: 65 processed frames, valid keypoints `0.0308`, usable knee `0.6462`, reliable frames `0.0308`, knee range `120.836` to `179.637`, mean torso lean `11.6901`, depth `insufficient`, warning frames `65`.
- Dominant warnings: `insufficient_depth:65`, `low_confidence:63`, `angle_tentative:40`, `knee_angle_unavailable:23`, `unknown_phase:23`, `possible_asymmetry:4`.
- Interpretation: failed technique/reliability example where low confidence prevents strong conclusions for most frames and depth remains insufficient.
- Reason for failure: low reliability, low confidence, and many tentative or unavailable angle estimates.
- Artifacts: `outputs/predictions/video_20260705T200227Z_b10ece35/`.
- Suggested figure: `outputs/figures/examples/BAD-03.jpg`.

## Next Step
Select representative sampled frames from each run and copy only the chosen report figures to `outputs/figures/examples/`.
