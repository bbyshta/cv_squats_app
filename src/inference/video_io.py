"""Video I/O helpers for stride-based demo processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg"}


@dataclass(frozen=True)
class VideoMetadata:
    """Basic video metadata needed for output artifacts."""

    width: int
    height: int
    fps: float
    frame_count_total: int


@dataclass(frozen=True)
class VideoFrame:
    """One decoded video frame."""

    frame_index: int
    image: Any


def validate_video_path(path: str | Path) -> Path:
    """Validate that a path points to a supported video file."""
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Input video path is not a file: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video extension '{video_path.suffix}'. Supported: {supported}")
    return video_path


def validate_frame_stride(frame_stride: int) -> int:
    """Validate and return a positive frame stride."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be greater than or equal to 1")
    return frame_stride


def should_process_frame(frame_index: int, frame_stride: int) -> bool:
    """Return whether a frame should be processed for the configured stride."""
    validate_frame_stride(frame_stride)
    if frame_index < 0:
        raise ValueError("frame_index must be greater than or equal to 0")
    return frame_index % frame_stride == 0


def read_video_metadata(path: str | Path) -> VideoMetadata:
    """Read basic video metadata with OpenCV."""
    video_path = validate_video_path(path)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video processing.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read valid frame size from video: {video_path}")
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps if fps > 0.0 else 25.0,
        frame_count_total=max(frame_count_total, 0),
    )


def iter_video_frames(
    path: str | Path,
    frame_stride: int,
    max_frames: int | None = None,
) -> Iterator[VideoFrame]:
    """Yield decoded frames whose indexes match ``frame_stride``.

    ``max_frames`` limits the number of processed frames, not the number of
    decoded source frames.
    """
    video_path = validate_video_path(path)
    validate_frame_stride(frame_stride)
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be greater than or equal to 1 when provided")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video processing.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_index = 0
    processed_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if should_process_frame(frame_index, frame_stride):
                yield VideoFrame(frame_index=frame_index, image=frame)
                processed_count += 1
                if max_frames is not None and processed_count >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
