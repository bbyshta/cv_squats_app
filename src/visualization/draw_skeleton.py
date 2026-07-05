"""Skeleton drawing helpers."""

from __future__ import annotations

from typing import Any

from src.inference.base import Keypoint, PoseResult


BODY_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def draw_skeleton(
    image: Any,
    pose_result: PoseResult,
    confidence_threshold: float = 0.5,
) -> Any:
    """Draw pose keypoints and main body connections on an OpenCV image."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for skeleton visualization. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    output = image.copy()
    visible = {
        keypoint.name: keypoint
        for keypoint in pose_result.keypoints
        if _is_visible(keypoint, confidence_threshold)
    }

    for start_name, end_name in BODY_CONNECTIONS:
        start = visible.get(start_name)
        end = visible.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(output, _point(start), _point(end), (0, 200, 255), 2)

    for keypoint in visible.values():
        cv2.circle(output, _point(keypoint), 4, (0, 255, 0), -1)

    return output


def _is_visible(keypoint: Keypoint, confidence_threshold: float) -> bool:
    return keypoint.confidence is None or keypoint.confidence >= confidence_threshold


def _point(keypoint: Keypoint) -> tuple[int, int]:
    return (int(round(keypoint.x)), int(round(keypoint.y)))

