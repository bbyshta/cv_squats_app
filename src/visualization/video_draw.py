"""Annotated video frame drawing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.inference.base import PoseResult
from src.visualization.draw_skeleton import draw_skeleton


class AnnotatedVideoWriter:
    """Small OpenCV VideoWriter wrapper with explicit failure checks."""

    def __init__(self, path: str | Path, fps: float, frame_size: tuple[int, int]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to write annotated video.") from exc

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open annotated video writer: {self.path}")

    def write(self, frame: Any) -> None:
        """Write one annotated frame."""
        self._writer.write(frame)

    def release(self) -> None:
        """Release the OpenCV writer."""
        self._writer.release()


def draw_annotated_video_frame(
    frame: Any,
    pose_result: PoseResult,
    frame_metrics: dict[str, Any],
    confidence_threshold: float,
) -> Any:
    """Draw skeleton and compact frame-level metrics on a video frame."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video annotation.") from exc

    output = draw_skeleton(frame, pose_result, confidence_threshold)
    phase = frame_metrics.get("squat_phase") or "unknown"
    selected_side = frame_metrics.get("selected_side") or "none"
    selected_knee = _format_optional_float(frame_metrics.get("selected_knee_angle"))
    torso_lean = _format_optional_float(frame_metrics.get("torso_lean_angle"))
    angle_quality = str(frame_metrics.get("angle_quality") or "unavailable")
    warnings = frame_metrics.get("warnings") or []
    warning_text = ",".join(warnings[:2]) if isinstance(warnings, list) else str(warnings)

    lines = [
        f"frame: {frame_metrics.get('frame_index')}",
        f"phase: {phase}  side: {selected_side}",
        f"knee: {selected_knee}  torso: {torso_lean}",
        f"angle quality: {angle_quality}",
    ]
    if warning_text:
        lines.append(f"warn: {warning_text}")

    x = 12
    y = 24
    for line in lines:
        cv2.putText(
            output,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return output


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"
