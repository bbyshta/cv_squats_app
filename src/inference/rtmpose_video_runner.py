"""RTMPose-S frame inference adapter for the video-only demo runtime."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from src.inference.base import Keypoint, PoseResult
from src.utils.config import load_yaml_config
from src.utils.io import write_image
from src.visualization.draw_skeleton import draw_skeleton


COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

SELECTED_MODEL_STATUS = "final_model_selected"
SELECTED_MODEL_ID = "rtmpose_s"
SELECTED_ARCHITECTURE = "RTMPose"
SELECTED_BACKEND = "mmpose"
WHOLE_FRAME_BBOX = "whole_frame"


@dataclass(frozen=True)
class RTMPoseVideoConfig:
    """Selected RTMPose-S runtime configuration."""

    model_id: str
    architecture: str
    model_name: str
    backend: str
    config_path: Path
    checkpoint_path: Path
    bbox_strategy: str
    device: str
    confidence_threshold: float
    prediction_dir: Path
    runtime_environment: str


@dataclass(frozen=True)
class FrameInferenceOutput:
    """Serializable single-frame RTMPose-S output."""

    model_id: str
    model_name: str
    backend: str
    bbox_strategy: str
    frame_index: int | None
    keypoints: list[dict[str, float | str | None]]
    status: str
    error_message: str | None = None


class RTMPoseFrameAdapter:
    """Lazy-loaded RTMPose-S adapter for one OpenCV video frame."""

    def __init__(self, config: RTMPoseVideoConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def load(self) -> None:
        """Load MMPose model resources."""
        if self._model is not None:
            return

        try:
            from mmpose.apis import init_model
        except ImportError as exc:
            raise RuntimeError(
                "MMPose is required for RTMPose-S demo inference. "
                "Run this adapter with .venv-mmpose/bin/python."
            ) from exc

        self._model = init_model(
            str(self.config.config_path),
            str(self.config.checkpoint_path),
            device=self.config.device,
        )

    def predict_frame(self, frame: Any, frame_index: int | None = None) -> FrameInferenceOutput:
        """Run RTMPose-S on one OpenCV frame using a whole-frame bbox."""
        try:
            keypoints = self._predict_keypoints(frame)
        except Exception as exc:  # noqa: BLE001 - backend/runtime errors vary.
            return FrameInferenceOutput(
                model_id=self.config.model_id,
                model_name=self.config.model_name,
                backend=self.config.backend,
                bbox_strategy=self.config.bbox_strategy,
                frame_index=frame_index,
                keypoints=[],
                status="failed",
                error_message=str(exc),
            )

        return FrameInferenceOutput(
            model_id=self.config.model_id,
            model_name=self.config.model_name,
            backend=self.config.backend,
            bbox_strategy=self.config.bbox_strategy,
            frame_index=frame_index,
            keypoints=[asdict(keypoint) for keypoint in keypoints],
            status="ok",
        )

    def to_pose_result(
        self,
        output: FrameInferenceOutput,
        image_size: tuple[int, int],
        source_path: str | None = None,
    ) -> PoseResult:
        """Convert serializable frame output to the shared pose result contract."""
        return PoseResult(
            model_name=output.model_name,
            keypoints=[
                Keypoint(
                    name=str(item["name"]),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    confidence=None if item["confidence"] is None else float(item["confidence"]),
                )
                for item in output.keypoints
            ],
            image_size=image_size,
            inference_time_ms=0.0,
            source_path=source_path,
        )

    def _predict_keypoints(self, frame: Any) -> list[Keypoint]:
        self.load()
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("Frame must be a valid OpenCV image array")

        try:
            import numpy as np
            from mmpose.apis import inference_topdown
        except ImportError as exc:
            raise RuntimeError(
                "MMPose and NumPy are required for RTMPose-S frame inference. "
                "Run this adapter with .venv-mmpose/bin/python."
            ) from exc

        height, width = int(frame.shape[0]), int(frame.shape[1])
        bbox = np.array([[0, 0, width, height]], dtype=np.float32)
        results = inference_topdown(self._model, frame, bboxes=bbox, bbox_format="xyxy")
        return extract_keypoints_from_mmpose_results(results)


def load_rtmpose_video_config(config_path: str | Path) -> RTMPoseVideoConfig:
    """Load and validate the selected RTMPose-S video runtime config."""
    root = Path.cwd()
    config = load_yaml_config(config_path)

    selected_model = config.get("selected_model")
    if not isinstance(selected_model, dict):
        raise ValueError("Config must contain selected_model mapping")

    _require_value(selected_model, "status", SELECTED_MODEL_STATUS)
    _require_value(selected_model, "model_id", SELECTED_MODEL_ID)
    _require_value(selected_model, "architecture", SELECTED_ARCHITECTURE)
    _require_value(selected_model, "backend", SELECTED_BACKEND)
    _require_value(selected_model, "bbox_strategy", WHOLE_FRAME_BBOX)
    if selected_model.get("allow_model_selection_in_demo") is not False:
        raise ValueError("selected_model.allow_model_selection_in_demo must be false")

    model_name = _required_string(selected_model, "name")
    model_config_path = _required_existing_file(root / _required_string(selected_model, "config_path"))
    checkpoint_path = _required_existing_file(root / _selected_checkpoint_path(selected_model))

    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Config must contain runtime mapping")
    runtime_environment = runtime.get("environment", ".venv-mmpose")
    if runtime_environment != ".venv-mmpose":
        raise ValueError("runtime.environment must be .venv-mmpose for the demo")

    app = config.get("app")
    if not isinstance(app, dict):
        raise ValueError("Config must contain app mapping")

    return RTMPoseVideoConfig(
        model_id=SELECTED_MODEL_ID,
        architecture=SELECTED_ARCHITECTURE,
        model_name=model_name,
        backend=SELECTED_BACKEND,
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        bbox_strategy=WHOLE_FRAME_BBOX,
        device=str(runtime.get("device", "cuda:0")),
        confidence_threshold=float(runtime.get("confidence_threshold", 0.5)),
        prediction_dir=Path(str(app.get("prediction_dir", "outputs/predictions"))),
        runtime_environment=runtime_environment,
    )


def extract_keypoints_from_mmpose_results(results: Sequence[Any]) -> list[Keypoint]:
    """Extract the first detected person's COCO keypoints from MMPose results."""
    if not results:
        return []

    pred_instances = getattr(results[0], "pred_instances", None)
    if pred_instances is None:
        return []

    raw_keypoints = _first_person_keypoints(getattr(pred_instances, "keypoints", []))
    raw_scores = _first_person_scores(getattr(pred_instances, "keypoint_scores", []))

    keypoints: list[Keypoint] = []
    for index, point in enumerate(raw_keypoints):
        if index >= len(COCO_KEYPOINT_NAMES) or len(point) < 2:
            continue
        confidence = None
        if index < len(raw_scores):
            confidence = float(raw_scores[index])
        keypoints.append(
            Keypoint(
                name=COCO_KEYPOINT_NAMES[index],
                x=float(point[0]),
                y=float(point[1]),
                confidence=confidence,
            )
        )
    return keypoints


def read_video_frame(video_path: str | Path, frame_index: int) -> Any:
    """Read one frame from a video file with OpenCV."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Input video not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input video path is not a file: {path}")
    if frame_index < 0:
        raise ValueError("frame_index must be greater than or equal to 0")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to read video frames. Run with .venv-mmpose/bin/python."
        ) from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_index} from video: {path}")
    return frame


def save_frame_outputs(
    output: FrameInferenceOutput,
    frame: Any,
    runtime_config: RTMPoseVideoConfig,
    run_id: str,
    input_path: str | Path,
) -> Path:
    """Save keypoints, skeleton image, and metadata for one smoke/debug frame."""
    run_dir = runtime_config.prediction_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    keypoints_path = run_dir / "frame_keypoints.json"
    metadata_path = run_dir / "frame_metadata.json"
    skeleton_path = run_dir / "frame_skeleton.jpg"

    _write_json(keypoints_path, asdict(output))
    pose_result = RTMPoseFrameAdapter(runtime_config).to_pose_result(
        output,
        image_size=(int(frame.shape[1]), int(frame.shape[0])),
        source_path=str(input_path),
    )
    skeleton = draw_skeleton(frame, pose_result, runtime_config.confidence_threshold)
    write_image(skeleton_path, skeleton)
    _write_json(
        metadata_path,
        {
            "run_id": run_id,
            "input_path": str(input_path),
            "frame_index": output.frame_index,
            "model_id": output.model_id,
            "model_name": output.model_name,
            "backend": output.backend,
            "bbox_strategy": output.bbox_strategy,
            "status": output.status,
            "error_message": output.error_message,
            "keypoint_count": len(output.keypoints),
            "outputs": {
                "keypoints": str(keypoints_path),
                "skeleton": str(skeleton_path),
                "metadata": str(metadata_path),
            },
        },
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RTMPose-S on one video frame for manual smoke/debug checks."
    )
    parser.add_argument("--input", required=True, help="Path to a video file.")
    parser.add_argument("--frame-index", type=int, default=0, help="Zero-based frame index.")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to application config. Default: configs/app.yaml",
    )
    parser.add_argument("--run-id", default=None, help="Optional output run id.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_id = args.run_id or f"rtmpose_frame_{uuid.uuid4().hex[:12]}"

    try:
        runtime_config = load_rtmpose_video_config(args.config)
        frame = read_video_frame(args.input, args.frame_index)
        adapter = RTMPoseFrameAdapter(runtime_config)
        output = adapter.predict_frame(frame, frame_index=args.frame_index)
        if output.status != "ok":
            raise RuntimeError(output.error_message or "RTMPose-S frame inference failed")
        run_dir = save_frame_outputs(output, frame, runtime_config, run_id, args.input)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    print(json.dumps({"status": "ok", "run_dir": str(run_dir)}, ensure_ascii=False))


def _require_value(mapping: dict[str, Any], key: str, expected: str) -> None:
    if mapping.get(key) != expected:
        raise ValueError(f"selected_model.{key} must be {expected!r}")


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"selected_model.{key} must be a non-empty string")
    return value


def _selected_checkpoint_path(selected_model: dict[str, Any]) -> str:
    checkpoint_path = selected_model.get("checkpoint_path")
    weights_path = selected_model.get("weights_path")
    if checkpoint_path is None:
        return _required_string(selected_model, "weights_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("selected_model.checkpoint_path must be a non-empty string")
    if weights_path is not None and weights_path != checkpoint_path:
        raise ValueError("selected_model.checkpoint_path and weights_path must match")
    return checkpoint_path


def _required_existing_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required RTMPose-S file not found: {path}")
    return path


def _first_person_keypoints(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    first = value[0]
    if hasattr(first, "tolist"):
        first = first.tolist()
    if isinstance(first, (int, float)):
        return []
    if first and isinstance(first[0], (int, float)):
        return list(value)
    return list(first)


def _first_person_scores(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    first = value[0]
    if hasattr(first, "tolist"):
        first = first.tolist()
    if isinstance(first, (int, float)):
        return list(value)
    if first and isinstance(first[0], (int, float)):
        return list(first)
    return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
