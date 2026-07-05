from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.base import Keypoint
from src.inference.rtmpose_video_runner import (
    FrameInferenceOutput,
    RTMPoseFrameAdapter,
    build_parser,
    extract_keypoints_from_mmpose_results,
    load_rtmpose_video_config,
)
from src.utils.config import load_yaml_config


def test_app_config_records_selected_rtmpose_whole_frame() -> None:
    config = load_yaml_config("configs/app.yaml")

    assert config["selected_model"]["model_id"] == "rtmpose_s"
    assert config["selected_model"]["architecture"] == "RTMPose"
    assert config["selected_model"]["backend"] == "mmpose"
    assert config["selected_model"]["checkpoint_path"].endswith("best_coco_AP_epoch_30.pth")
    assert config["selected_model"]["bbox_strategy"] == "whole_frame"
    assert config["selected_model"]["allow_model_selection_in_demo"] is False
    assert config["runtime"]["environment"] == ".venv-mmpose"
    assert config["runtime"]["draw_min_conf"] == 0.15
    assert config["runtime"]["angle_min_conf"] == 0.25
    assert config["runtime"]["body_line_min_conf"] == 0.35
    assert config["runtime"]["reliable_min_conf"] == 0.50
    assert config["runtime"]["side_switch_margin"] == 0.08
    assert config["app"]["exercise"] == "squat"
    assert config["app"]["exercise_type"] == "squat"
    assert config["squat"]["phase_top_min_angle"] == 160
    assert config["squat"]["phase_bottom_max_angle"] == 110
    assert config["squat"]["depth_sufficient_max_angle"] == 110


def test_load_rtmpose_video_config_validates_whole_frame(tmp_path: Path) -> None:
    model_config = tmp_path / "rtmpose.py"
    checkpoint = tmp_path / "rtmpose.pth"
    model_config.write_text("# test config\n", encoding="utf-8")
    checkpoint.write_bytes(b"test")
    config_path = _write_app_config(tmp_path, model_config, checkpoint)

    runtime_config = load_rtmpose_video_config(config_path)

    assert runtime_config.model_id == "rtmpose_s"
    assert runtime_config.backend == "mmpose"
    assert runtime_config.bbox_strategy == "whole_frame"
    assert runtime_config.runtime_environment == ".venv-mmpose"


def test_load_rtmpose_video_config_rejects_other_bbox_strategy(tmp_path: Path) -> None:
    model_config = tmp_path / "rtmpose.py"
    checkpoint = tmp_path / "rtmpose.pth"
    model_config.write_text("# test config\n", encoding="utf-8")
    checkpoint.write_bytes(b"test")
    config_path = _write_app_config(
        tmp_path,
        model_config,
        checkpoint,
        bbox_strategy="detector",
    )

    with pytest.raises(ValueError, match="bbox_strategy"):
        load_rtmpose_video_config(config_path)


def test_load_rtmpose_video_config_reports_missing_checkpoint(tmp_path: Path) -> None:
    model_config = tmp_path / "rtmpose.py"
    model_config.write_text("# test config\n", encoding="utf-8")
    missing_checkpoint = tmp_path / "missing.pth"
    config_path = _write_app_config(tmp_path, model_config, missing_checkpoint)

    with pytest.raises(FileNotFoundError, match="Required RTMPose-S file not found"):
        load_rtmpose_video_config(config_path)


def test_adapter_output_schema_with_mocked_keypoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_config = tmp_path / "rtmpose.py"
    checkpoint = tmp_path / "rtmpose.pth"
    model_config.write_text("# test config\n", encoding="utf-8")
    checkpoint.write_bytes(b"test")
    runtime_config = load_rtmpose_video_config(_write_app_config(tmp_path, model_config, checkpoint))

    def fake_predict_keypoints(self: RTMPoseFrameAdapter, frame: object) -> list[Keypoint]:
        return [
            Keypoint(name="left_shoulder", x=10.0, y=20.0, confidence=0.9),
            Keypoint(name="left_elbow", x=15.0, y=35.0, confidence=0.8),
        ]

    monkeypatch.setattr(RTMPoseFrameAdapter, "_predict_keypoints", fake_predict_keypoints)
    output = RTMPoseFrameAdapter(runtime_config).predict_frame(object(), frame_index=7)

    assert isinstance(output, FrameInferenceOutput)
    assert output.model_id == "rtmpose_s"
    assert output.model_name == "RTMPose-S"
    assert output.backend == "mmpose"
    assert output.bbox_strategy == "whole_frame"
    assert output.frame_index == 7
    assert output.status == "ok"
    assert output.error_message is None
    assert output.keypoints == [
        {"name": "left_shoulder", "x": 10.0, "y": 20.0, "confidence": 0.9},
        {"name": "left_elbow", "x": 15.0, "y": 35.0, "confidence": 0.8},
    ]


def test_extract_keypoints_from_mmpose_like_result() -> None:
    class PredInstances:
        keypoints = [[[1.0, 2.0], [3.0, 4.0]]]
        keypoint_scores = [[0.9, 0.8]]

    class Result:
        pred_instances = PredInstances()

    keypoints = extract_keypoints_from_mmpose_results([Result()])

    assert keypoints == [
        Keypoint(name="nose", x=1.0, y=2.0, confidence=0.9),
        Keypoint(name="left_eye", x=3.0, y=4.0, confidence=0.8),
    ]


def test_video_runner_cli_does_not_introduce_image_upload_mode() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--input" in option_strings
    assert "--frame-index" in option_strings
    assert "--image" not in option_strings
    assert "video frame" in parser.description


def _write_app_config(
    tmp_path: Path,
    model_config: Path,
    checkpoint: Path,
    bbox_strategy: str = "whole_frame",
) -> Path:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        f"""
app:
  prediction_dir: "{tmp_path / 'predictions'}"
selected_model:
  status: "final_model_selected"
  model_id: "rtmpose_s"
  architecture: "RTMPose"
  name: "RTMPose-S"
  backend: "mmpose"
  config_path: "{model_config}"
  checkpoint_path: "{checkpoint}"
  weights_path: "{checkpoint}"
  bbox_strategy: "{bbox_strategy}"
  allow_model_selection_in_demo: false
runtime:
  environment: ".venv-mmpose"
  device: "cpu"
  confidence_threshold: 0.5
""".lstrip(),
        encoding="utf-8",
    )
    return config_path
