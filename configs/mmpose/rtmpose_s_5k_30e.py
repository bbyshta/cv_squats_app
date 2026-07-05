"""RTMPose-S config adapted for the project COCO 5k/30e protocol.

Adapted from the local MMPose 1.3.2 package config:
/home/paulkrat/ml-project/.venv-mmpose/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py

This file must be used only for manual user-run training. The agent must not
download the checkpoint or start training. A future manual MMPose run may
download the verified checkpoint through MMPose/MMEngine.
"""

_base_ = [
    '/home/paulkrat/ml-project/.venv-mmpose/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py'  # noqa: E501
]

experiment_protocol = 'coco_keypoints_subset_5k_30e'
work_dir = 'outputs/models/rtmpose'

max_epochs = 30
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
randomness = dict(seed=42)

# Verified official RTMPose-S checkpoint. Do not download it from agent-run checks.
load_from = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
    'rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth'
)
resume = False

# The exported subset JSON stores file_name values such as
# train2017/000000000000.jpg or val2017/000000000000.jpg. Therefore data_root
# must stay empty and data_prefix.img must point at the common COCO root.
data_root = ''
coco_image_root = 'data/raw/coco/'
train_ann_file = (
    'data/processed/coco_pose_subset_5k/mmpose/annotations/'
    'person_keypoints_train_5k.json'
)
val_ann_file = (
    'data/processed/coco_pose_subset_5k/mmpose/annotations/'
    'person_keypoints_val_1k.json'
)
test_ann_file = (
    'data/processed/coco_pose_subset_5k/mmpose/annotations/'
    'person_keypoints_test_1k.json'
)

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=coco_image_root),
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=coco_image_root),
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=coco_image_root),
    ),
)

val_evaluator = dict(ann_file=val_ann_file)
test_evaluator = dict(ann_file=test_ann_file)
