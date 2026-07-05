"""Core pose inference and analysis data contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Keypoint:
    """Single body keypoint in image coordinates."""

    name: str
    x: float
    y: float
    confidence: float | None = None


@dataclass(frozen=True)
class PoseResult:
    """Pose estimator output for one image or video frame."""

    model_name: str
    keypoints: list[Keypoint]
    image_size: tuple[int, int]
    inference_time_ms: float
    source_path: str | None = None

    def get_keypoint(self, name: str) -> Keypoint | None:
        """Return a keypoint by name, or None if it is absent."""
        return next((keypoint for keypoint in self.keypoints if keypoint.name == name), None)

    def mean_confidence(self) -> float:
        """Return mean confidence over keypoints that provide confidence."""
        confidences = [
            keypoint.confidence
            for keypoint in self.keypoints
            if keypoint.confidence is not None
        ]
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    def detected_keypoints(self, confidence_threshold: float) -> int:
        """Count keypoints with confidence greater than or equal to threshold."""
        return sum(
            1
            for keypoint in self.keypoints
            if keypoint.confidence is not None and keypoint.confidence >= confidence_threshold
        )


@dataclass(frozen=True)
class PushupAnalysisResult:
    """Rule-based push-up analysis output."""

    angles: dict[str, float | None]
    pushup_phase: str
    rep_count: int | None
    technique_assessment: str
    warnings: list[str] = field(default_factory=list)


class PoseEstimator(ABC):
    """Common interface for pose estimation backends."""

    model_name: str

    @abstractmethod
    def load(self) -> None:
        """Load model resources required for inference."""

    @abstractmethod
    def predict_image(self, image: Any, source_path: str | None = None) -> PoseResult:
        """Run pose estimation on an OpenCV image or image path."""
