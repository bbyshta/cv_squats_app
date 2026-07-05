# Dataset Plan: COCO Keypoints Subset

## Selected Dataset

The planned fine-tuning dataset is a local COCO Keypoints / COCO-Pose compatible subset.

- Source: official COCO dataset, COCO Keypoints 2017.
- Purpose: training and evaluation of human pose estimation models.
- Annotation type: person keypoints.
- Annotation format: COCO JSON annotations.
- Object class used for pose estimation: `person`.
- Keypoint format: 17 COCO keypoints.
- Visibility format: each keypoint is stored as `x, y, v`, where `v` is the COCO visibility flag.

The 17 COCO keypoints are:

1. nose
2. left_eye
3. right_eye
4. left_ear
5. right_ear
6. left_shoulder
7. right_shoulder
8. left_elbow
9. right_elbow
10. left_wrist
11. right_wrist
12. left_hip
13. right_hip
14. left_knee
15. right_knee
16. left_ankle
17. right_ankle

## Why Push-Up Samples Are Not the Training Dataset

Push-up images and videos may be used for the demo, visual validation, successful and failed examples, and rule-based push-up technique analysis.

Push-up samples without manual keypoint annotations are not a full pose-estimation training dataset. They do not provide ground-truth `x, y, v` keypoints for supervised fine-tuning or reliable pose-estimation metrics. They must not be presented as the training dataset unless manual keypoint annotation is added and documented.

## Repository Storage Policy

Full COCO data is not committed to this repository because it is large and belongs under local data storage, not source control. The repository stores only configuration, validation code, documentation, and small sample/demo files.

The preparation script does not download COCO automatically. The user must place the required local COCO files before running validation or split generation.

Official COCO test annotations are not publicly available in the same local annotation form as train/val. For this educational subset, a local test split may be created from annotated local COCO train/val data.

## Expected Local Structure

Place COCO files under:

```text
data/raw/coco/
  annotations/
    person_keypoints_train2017.json
    person_keypoints_val2017.json
  train2017/
    *.jpg
  val2017/
    *.jpg
```

Required files before running the script:

- `data/raw/coco/annotations/person_keypoints_train2017.json`
- `data/raw/coco/annotations/person_keypoints_val2017.json`
- `data/raw/coco/train2017/`
- `data/raw/coco/val2017/`

## Script Checks

`src.data.prepare_coco_subset` validates local files and annotations with standard Python `json`; `pycocotools` is not required for basic validation.

The script checks:

- configured annotation JSON files exist;
- configured image directories exist;
- COCO JSON has `images`, `annotations`, and `categories`;
- a `person` category exists;
- person annotations with keypoints use the expected `17 * 3 = 51` keypoint values;
- visible keypoint counts can be computed from the COCO `v` flags;
- enough valid annotated images exist for the requested train/validation/test subset;
- split generation is reproducible from the configured seed.

In dry-run mode the script validates and prints the plan only. It does not create split files and does not copy images.

In normal mode it writes:

```text
data/processed/coco_pose_subset/splits/train.txt
data/processed/coco_pose_subset/splits/val.txt
data/processed/coco_pose_subset/splits/test.txt
outputs/metrics/dataset_summary.json
```

These files are generated only from local data that already exists on disk.
