"""Metadata registry for pose model candidates.

This module intentionally does not load model weights or import heavy model
frameworks. Real loaders should be added only after evaluation selects the
final demo model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    """Static metadata for a pose estimation architecture candidate."""

    key: str
    name: str
    architecture_family: str
    pretrained_weights_source: str
    default_input_size: tuple[int, int] | None
    status: str = "metadata_only"


MODEL_REGISTRY: dict[str, ModelMetadata] = {
    "yolo_pose": ModelMetadata(
        key="yolo_pose",
        name="YOLO Pose",
        architecture_family="YOLO Pose",
        pretrained_weights_source="yolo11n-pose.pt for recorded YOLO runs",
        default_input_size=(640, 640),
    ),
    "rtmpose": ModelMetadata(
        key="rtmpose",
        name="RTMPose",
        architecture_family="RTMPose",
        pretrained_weights_source="Official MMPose RTMPose-S checkpoint",
        default_input_size=(256, 192),
    ),
    "hrnet": ModelMetadata(
        key="hrnet",
        name="HRNet",
        architecture_family="HRNet",
        pretrained_weights_source="Official MMPose HRNet-W32 UDP checkpoint",
        default_input_size=(256, 192),
    ),
    "litehrnet": ModelMetadata(
        key="litehrnet",
        name="Lite-HRNet",
        architecture_family="Lite-HRNet",
        pretrained_weights_source="Official MMPose Lite-HRNet-18 checkpoint",
        default_input_size=(256, 192),
    ),
    "vitpose": ModelMetadata(
        key="vitpose",
        name="ViTPose",
        architecture_family="ViTPose",
        pretrained_weights_source="Official MMPose ViTPose checkpoint to be finalized",
        default_input_size=(256, 192),
    ),
}


def get_model_metadata(model_key: str) -> ModelMetadata:
    """Return metadata for a registered model candidate."""
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model key '{model_key}'. Available: {available}") from exc


def list_model_metadata() -> list[ModelMetadata]:
    """Return all registered model metadata in deterministic key order."""
    return [MODEL_REGISTRY[key] for key in sorted(MODEL_REGISTRY)]
