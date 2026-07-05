from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from src.analysis.squat import (
    SquatAnalysisConfig,
    aggregate_depth_status,
    analyze_squat_frame,
    calculate_torso_lean_angle,
    classify_squat_phase,
    default_squat_config,
)
from src.inference.base import Keypoint, PoseResult
from src.inference.rtmpose_video_runner import FrameInferenceOutput, RTMPoseVideoConfig
from src.inference.video_io import VideoFrame, VideoMetadata, should_process_frame
from src.inference.video_pipeline import (
    FRAME_METRICS_COLUMNS,
    analyze_video_frame,
    build_parser,
    build_video_summary,
    process_video,
)
from src.utils.video_export import BrowserVideoExportResult


def test_should_process_frame_uses_frame_stride() -> None:
    processed = [index for index in range(12) if should_process_frame(index, 5)]

    assert processed == [0, 5, 10]


def test_knee_angle_calculation_from_synthetic_points() -> None:
    output = _frame_output(_squat_keypoints("top", confidence=0.8), frame_index=1)

    metrics = analyze_video_frame(
        output=output,
        pose_result=_pose_result(output),
        thresholds=default_squat_config(),
    )

    assert metrics["left_knee_angle"] == pytest.approx(180.0)
    assert metrics["right_knee_angle"] == pytest.approx(180.0)
    assert metrics["selected_knee_angle"] == pytest.approx(180.0)
    assert metrics["knee_angle_valid"] is True


def test_torso_lean_calculation_from_synthetic_points() -> None:
    shoulder = Keypoint("left_shoulder", 1.0, 0.0, 0.9)
    hip = Keypoint("left_hip", 0.0, 1.0, 0.9)

    assert calculate_torso_lean_angle(shoulder, hip) == pytest.approx(45.0)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (170.0, "top"),
        (135.0, "middle"),
        (90.0, "bottom"),
        (None, "unknown"),
    ],
)
def test_squat_phase_classification(angle: float | None, expected: str) -> None:
    assert classify_squat_phase(angle, default_squat_config()) == expected


def test_depth_status_aggregation() -> None:
    config = default_squat_config()

    assert aggregate_depth_status([], config) == "unknown"
    assert aggregate_depth_status(
        [{"selected_knee_angle": 130.0, "knee_angle_valid": True}],
        config,
    ) == "insufficient"
    assert aggregate_depth_status(
        [{"selected_knee_angle": 105.0, "knee_angle_valid": True}],
        config,
    ) == "sufficient"


def test_asymmetry_warning_when_knee_angles_differ() -> None:
    output = _frame_output(_asymmetric_squat_keypoints(), frame_index=2)

    metrics = analyze_video_frame(
        output=output,
        pose_result=_pose_result(output),
        thresholds=default_squat_config(),
    )

    assert metrics["left_right_knee_angle_diff"] is not None
    assert metrics["left_right_knee_angle_diff"] > 20.0
    assert "possible_asymmetry" in metrics["warnings"]


def test_confidence_gating_for_knee_angle() -> None:
    output = _frame_output(_squat_keypoints("middle", confidence=0.2), frame_index=3)

    metrics = analyze_video_frame(
        output=output,
        pose_result=_pose_result(output),
        thresholds=default_squat_config(),
    )

    assert metrics["knee_angle_valid"] is False
    assert metrics["selected_knee_angle"] is None
    assert metrics["squat_phase"] == "unknown"
    assert "knee_angle_unavailable" in metrics["warnings"]


def test_low_ankle_confidence_does_not_block_torso_lean() -> None:
    keypoints = [
        Keypoint(point.name, point.x, point.y, 0.1)
        if point.name.endswith("_ankle")
        else point
        for point in _squat_keypoints("middle", confidence=0.8)
    ]
    output = _frame_output(keypoints, frame_index=4)

    metrics = analyze_video_frame(
        output=output,
        pose_result=_pose_result(output),
        thresholds=default_squat_config(),
    )

    assert metrics["knee_angle_valid"] is False
    assert metrics["torso_lean_valid"] is True
    assert metrics["torso_lean_angle"] is not None


def test_low_wrist_elbow_confidence_does_not_affect_squat_analysis() -> None:
    keypoints = _squat_keypoints("bottom", confidence=0.8)
    keypoints.extend(
        [
            Keypoint("left_elbow", 0.0, 0.0, 0.0),
            Keypoint("right_elbow", 0.0, 0.0, 0.0),
            Keypoint("left_wrist", 0.0, 0.0, 0.0),
            Keypoint("right_wrist", 0.0, 0.0, 0.0),
        ]
    )
    output = _frame_output(keypoints, frame_index=5)

    metrics = analyze_video_frame(
        output=output,
        pose_result=_pose_result(output),
        thresholds=default_squat_config(),
    )

    assert metrics["knee_angle_valid"] is True
    assert metrics["squat_phase"] == "bottom"
    assert metrics["angle_quality"] == "reliable"


def test_summary_aggregation_schema() -> None:
    summary = build_video_summary(
        run_id="run-1",
        input_path="data/samples/squat_sample.mp4",
        runtime_config=_runtime_config(Path(".")),
        frame_stride=5,
        frame_count_total=100,
        frame_metrics=[
            {
                "critical_keypoints_valid": True,
                "knee_angle_valid": True,
                "frame_reliable": True,
                "selected_knee_angle": 170.0,
                "torso_lean_angle": 5.0,
                "torso_lean_valid": True,
                "squat_phase": "top",
                "mean_critical_confidence": 0.9,
                "warnings": [],
            },
            {
                "critical_keypoints_valid": False,
                "knee_angle_valid": True,
                "frame_reliable": False,
                "selected_knee_angle": 105.0,
                "torso_lean_angle": None,
                "torso_lean_valid": False,
                "squat_phase": "bottom",
                "mean_critical_confidence": 0.2,
                "warnings": ["low_confidence", "torso_lean_unavailable"],
            },
        ],
        depth_status="sufficient",
        artifacts={"summary": "summary.json"},
        limitations=[],
        status="ok",
        error_message=None,
    )

    assert summary["processed_frame_count"] == 2
    assert summary["exercise_type"] == "squat"
    assert summary["usable_knee_frame_count"] == 2
    assert summary["usable_knee_frame_ratio"] == pytest.approx(1.0)
    assert summary["reliable_frame_count"] == 1
    assert summary["reliable_frame_ratio"] == pytest.approx(0.5)
    assert summary["min_selected_knee_angle"] == pytest.approx(105.0)
    assert summary["max_selected_knee_angle"] == pytest.approx(170.0)
    assert summary["mean_torso_lean_angle"] == pytest.approx(5.0)
    assert summary["depth_status"] == "sufficient"
    assert summary["phase_counts"] == {"top": 1, "bottom": 1}


def test_process_video_writes_squat_csv_and_json_schema_with_mocked_runtime(tmp_path: Path) -> None:
    input_path = _touch_video(tmp_path)
    config_path = _write_app_config(tmp_path)

    result = process_video(
        input_path=input_path,
        config_path=config_path,
        frame_stride=5,
        max_frames=2,
        output_root=tmp_path / "predictions",
        save_annotated_video=False,
        save_history=True,
        history_path=tmp_path / "history" / "runs.jsonl",
        metadata_reader=lambda _: VideoMetadata(width=640, height=480, fps=30.0, frame_count_total=20),
        frame_reader=lambda *_: [
            VideoFrame(frame_index=0, image=_FakeFrame()),
            VideoFrame(frame_index=5, image=_FakeFrame()),
        ],
        adapter_factory=lambda runtime_config: _FakeAdapter(runtime_config),
        frame_drawer=lambda frame, *_: frame,
        image_writer=_fake_image_writer,
    )

    with result.frame_metrics_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == list(FRAME_METRICS_COLUMNS)
        rows = list(reader)

    keypoints_payload = json.loads(result.keypoints_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert rows[0]["left_knee_angle"]
    assert "selected_knee_angle" in rows[0]
    assert "torso_lean_angle" in rows[0]
    assert "squat_phase" in rows[0]
    assert "depth_status" in rows[0]
    assert "sampled_frame_path" in rows[0]
    assert keypoints_payload["model_id"] == "rtmpose_s"
    assert "squat" in keypoints_payload["frames"][0]
    assert "validity" in keypoints_payload["frames"][0]
    assert summary_payload["processed_frame_count"] == 2
    assert "usable_knee_frame_count" in summary_payload
    assert "phase_counts" in summary_payload
    assert summary_payload["artifacts"]["annotated_video"] is None
    assert summary_payload["artifacts"]["history_jsonl"].endswith("runs.jsonl")
    assert result.history_path is not None
    assert result.history_path.is_file()
    assert (result.sampled_frames_dir / "frame_000000.jpg").is_file()


def test_annotated_video_writer_failure_does_not_fail_run(tmp_path: Path) -> None:
    input_path = _touch_video(tmp_path)
    config_path = _write_app_config(tmp_path)

    result = process_video(
        input_path=input_path,
        config_path=config_path,
        frame_stride=1,
        max_frames=1,
        output_root=tmp_path / "predictions",
        save_annotated_video=True,
        save_history=False,
        metadata_reader=lambda _: VideoMetadata(width=640, height=480, fps=30.0, frame_count_total=1),
        frame_reader=lambda *_: [VideoFrame(frame_index=0, image=_FakeFrame())],
        adapter_factory=lambda runtime_config: _FakeAdapter(runtime_config),
        video_writer_factory=_failing_writer_factory,
        frame_drawer=lambda frame, *_: frame,
        image_writer=_fake_image_writer,
    )

    assert result.summary["status"] == "ok"
    assert result.annotated_video_path is None
    assert result.summary["artifacts"]["annotated_video"] is None
    assert result.summary["artifacts"]["annotated_video_warning"] == "annotated_video_writer_unavailable"
    assert result.summary["limitations"]


def test_browser_video_export_failure_does_not_fail_video_pipeline(tmp_path: Path) -> None:
    input_path = _touch_video(tmp_path)
    config_path = _write_app_config(tmp_path)

    result = process_video(
        input_path=input_path,
        config_path=config_path,
        frame_stride=1,
        max_frames=1,
        output_root=tmp_path / "predictions",
        save_annotated_video=True,
        save_history=False,
        metadata_reader=lambda _: VideoMetadata(width=640, height=480, fps=30.0, frame_count_total=1),
        frame_reader=lambda *_: [VideoFrame(frame_index=0, image=_FakeFrame())],
        adapter_factory=lambda runtime_config: _FakeAdapter(runtime_config),
        video_writer_factory=_file_creating_writer_factory,
        frame_drawer=lambda frame, *_: frame,
        image_writer=_fake_image_writer,
        browser_video_exporter=_missing_browser_video_exporter,
    )

    summary_payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    artifacts = summary_payload["artifacts"]

    assert summary_payload["status"] == "ok"
    assert artifacts["annotated_video"].endswith("annotated_video.mp4")
    assert artifacts["annotated_video_browser"] is None
    assert artifacts["annotated_video_browser_compatible"] is False
    assert artifacts["video_export_warning"] == "ffmpeg_not_available"
    assert summary_payload["limitations"]


def test_excel_export_failure_does_not_fail_video_pipeline(tmp_path: Path) -> None:
    input_path = _touch_video(tmp_path)
    config_path = _write_app_config(tmp_path)

    result = process_video(
        input_path=input_path,
        config_path=config_path,
        frame_stride=1,
        max_frames=1,
        output_root=tmp_path / "predictions",
        save_annotated_video=False,
        save_history=False,
        export_excel=True,
        metadata_reader=lambda _: VideoMetadata(width=640, height=480, fps=30.0, frame_count_total=1),
        frame_reader=lambda *_: [VideoFrame(frame_index=0, image=_FakeFrame())],
        adapter_factory=lambda runtime_config: _FakeAdapter(runtime_config),
        frame_drawer=lambda frame, *_: frame,
        image_writer=_fake_image_writer,
        excel_exporter=_failing_excel_exporter,
    )

    summary_payload = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.summary["status"] == "ok"
    assert result.excel_report_path is None
    assert summary_payload["status"] == "ok"
    assert summary_payload["pipeline_warnings"][0].startswith("excel_export_failed")


def test_video_pipeline_cli_has_video_only_arguments() -> None:
    parser = build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert "--input" in option_strings
    assert "--config" in option_strings
    assert "--frame-stride" in option_strings
    assert "--max-frames" in option_strings
    assert "--output-root" in option_strings
    assert "--save-annotated-video" in option_strings
    assert "--save-history" in option_strings
    assert "--history-path" in option_strings
    assert "--export-excel" in option_strings
    assert "--image" not in option_strings


class _FakeFrame:
    shape = (480, 640, 3)


class _FakeAdapter:
    def __init__(self, runtime_config: RTMPoseVideoConfig) -> None:
        self.runtime_config = runtime_config
        self.load_count = 0

    def load(self) -> None:
        self.load_count += 1

    def predict_frame(self, frame: Any, frame_index: int | None = None) -> FrameInferenceOutput:
        return _frame_output(_squat_keypoints("bottom", confidence=0.8), frame_index=frame_index)


def _failing_writer_factory(path: str | Path, fps: float, frame_size: tuple[int, int]) -> Any:
    raise RuntimeError("test writer failure")


class _FileCreatingWriter:
    def __init__(self, path: str | Path, fps: float, frame_size: tuple[int, int]) -> None:
        self.path = Path(path)

    def write(self, frame: Any) -> None:
        pass

    def release(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"fake video")


def _file_creating_writer_factory(path: str | Path, fps: float, frame_size: tuple[int, int]) -> Any:
    return _FileCreatingWriter(path, fps, frame_size)


def _missing_browser_video_exporter(
    input_path: str | Path,
    output_path: str | Path | None,
) -> BrowserVideoExportResult:
    return BrowserVideoExportResult(
        output_path=None,
        browser_compatible=False,
        warning="ffmpeg_not_available",
    )


def _failing_excel_exporter(run_dir: str | Path, output_path: str | Path | None) -> Path:
    raise RuntimeError("test excel failure")


def _fake_image_writer(path: str | Path, image: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake image")
    return output_path


def _touch_video(tmp_path: Path) -> Path:
    path = tmp_path / "squat_sample.mp4"
    path.write_bytes(b"fake video")
    return path


def _write_app_config(tmp_path: Path) -> Path:
    model_config = tmp_path / "rtmpose.py"
    checkpoint = tmp_path / "rtmpose.pth"
    model_config.write_text("# test config\n", encoding="utf-8")
    checkpoint.write_bytes(b"test")
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        f"""
app:
  prediction_dir: "{tmp_path / 'predictions'}"
  exercise: "squat"
  exercise_type: "squat"
selected_model:
  status: "final_model_selected"
  model_id: "rtmpose_s"
  architecture: "RTMPose"
  name: "RTMPose-S"
  backend: "mmpose"
  config_path: "{model_config}"
  checkpoint_path: "{checkpoint}"
  weights_path: "{checkpoint}"
  bbox_strategy: "whole_frame"
  allow_model_selection_in_demo: false
runtime:
  environment: ".venv-mmpose"
  device: "cpu"
  confidence_threshold: 0.5
  draw_min_conf: 0.15
  angle_min_conf: 0.25
  body_line_min_conf: 0.35
  reliable_min_conf: 0.50
  side_switch_margin: 0.08
squat:
  phase_top_min_angle: 160
  phase_bottom_max_angle: 110
  depth_sufficient_max_angle: 110
  torso_lean_warn_angle: 45
  knee_symmetry_warn_diff: 20
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def _runtime_config(root: Path) -> RTMPoseVideoConfig:
    return RTMPoseVideoConfig(
        model_id="rtmpose_s",
        architecture="RTMPose",
        model_name="RTMPose-S",
        backend="mmpose",
        config_path=root / "rtmpose.py",
        checkpoint_path=root / "rtmpose.pth",
        bbox_strategy="whole_frame",
        device="cpu",
        confidence_threshold=0.5,
        prediction_dir=root / "predictions",
        runtime_environment=".venv-mmpose",
    )


def _frame_output(
    keypoints: list[Keypoint],
    frame_index: int | None,
) -> FrameInferenceOutput:
    return FrameInferenceOutput(
        model_id="rtmpose_s",
        model_name="RTMPose-S",
        backend="mmpose",
        bbox_strategy="whole_frame",
        frame_index=frame_index,
        keypoints=[
            {
                "name": keypoint.name,
                "x": keypoint.x,
                "y": keypoint.y,
                "confidence": keypoint.confidence,
            }
            for keypoint in keypoints
        ],
        status="ok",
    )


def _pose_result(output: FrameInferenceOutput) -> PoseResult:
    return PoseResult(
        model_name=output.model_name,
        keypoints=[
            Keypoint(
                name=str(item["name"]),
                x=float(item["x"]),
                y=float(item["y"]),
                confidence=float(item["confidence"]) if item["confidence"] is not None else None,
            )
            for item in output.keypoints
        ],
        image_size=(640, 480),
        inference_time_ms=0.0,
        source_path="test.mp4",
    )


def _squat_keypoints(phase: str, confidence: float) -> list[Keypoint]:
    ankle_by_phase = {
        "top": (0.0, 2.0),
        "middle": (0.866, 1.5),
        "bottom": (1.0, 1.0),
    }
    left_ankle = ankle_by_phase[phase]
    right_ankle = (left_ankle[0] + 4.0, left_ankle[1])
    return [
        Keypoint("left_shoulder", 0.0, -1.0, confidence),
        Keypoint("left_hip", 0.0, 0.0, confidence),
        Keypoint("left_knee", 0.0, 1.0, confidence),
        Keypoint("left_ankle", left_ankle[0], left_ankle[1], confidence),
        Keypoint("right_shoulder", 4.0, -1.0, confidence),
        Keypoint("right_hip", 4.0, 0.0, confidence),
        Keypoint("right_knee", 4.0, 1.0, confidence),
        Keypoint("right_ankle", right_ankle[0], right_ankle[1], confidence),
    ]


def _asymmetric_squat_keypoints() -> list[Keypoint]:
    keypoints = _squat_keypoints("top", confidence=0.9)
    return [
        Keypoint(point.name, 5.0, 1.0, point.confidence)
        if point.name == "right_ankle"
        else point
        for point in keypoints
    ]
