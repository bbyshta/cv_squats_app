"""File I/O helpers for inference outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.inference.base import PoseResult


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image_path(path: str | Path) -> Path:
    """Validate that a path points to a supported image file."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Input path is not a file: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image extension '{image_path.suffix}'. Supported: {supported}")
    return image_path


def read_image(path: str | Path) -> Any:
    """Read an image with OpenCV."""
    image_path = validate_image_path(path)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to read images. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")
    return image


def write_image(path: str | Path, image: Any) -> Path:
    """Write an image with OpenCV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to write images. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Could not write image file: {output_path}")
    return output_path


def save_pose_result_json(pose_result: PoseResult, path: str | Path) -> Path:
    """Save pose result as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(pose_result)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return output_path

