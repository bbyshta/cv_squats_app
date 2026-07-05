from __future__ import annotations

import json
from pathlib import Path

import src.app.streamlit_app as streamlit_app


def test_streamlit_module_imports_without_running_app_side_effects() -> None:
    assert callable(streamlit_app.main)


def test_config_model_info_loader_uses_selected_rtmpose_model(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
app:
  title: "Squat technique analysis"
  exercise_type: "squat"
  history_path: "outputs/history/runs.jsonl"
  report_dir: "outputs/reports"
  prediction_dir: "outputs/predictions"
selected_model:
  model_id: "rtmpose_s"
  name: "RTMPose-S"
  backend: "mmpose"
  bbox_strategy: "whole_frame"
runtime:
  environment: ".venv-mmpose"
""".lstrip(),
        encoding="utf-8",
    )

    model_info = streamlit_app.load_demo_model_info(config_path)

    assert model_info["exercise_type"] == "squat"
    assert model_info["model_id"] == "rtmpose_s"
    assert model_info["model_name"] == "RTMPose-S"
    assert model_info["backend"] == "mmpose"
    assert model_info["bbox_strategy"] == "whole_frame"
    assert model_info["runtime_environment"] == ".venv-mmpose"


def test_summary_formatting_with_mocked_summary() -> None:
    metrics = streamlit_app.format_summary_metrics(
        {
            "status": "ok",
            "run_id": "run-1",
            "processed_frame_count": 185,
            "valid_keypoint_frame_ratio": 1.0,
            "usable_knee_frame_ratio": 0.75,
            "reliable_frame_ratio": 0.5,
            "min_selected_knee_angle": 71.5654,
            "max_selected_knee_angle": 179.3981,
            "mean_torso_lean_angle": 17.4967,
            "depth_status": "sufficient",
            "phase_counts": {"top": 53},
            "dominant_warnings": {"low_confidence": 2},
        }
    )

    assert metrics["status"] == "ok"
    assert metrics["valid_keypoint_ratio"] == "100.0%"
    assert metrics["usable_knee_ratio"] == "75.0%"
    assert metrics["reliable_frame_ratio"] == "50.0%"
    assert metrics["min_selected_knee_angle"] == "71.565"
    assert metrics["phase_counts"] == {"top": 53}
    assert metrics["warnings"] == {"low_confidence": 2}


def test_history_loading_handles_missing_file(tmp_path: Path) -> None:
    assert streamlit_app.load_recent_history(tmp_path / "missing.jsonl") == []


def test_history_loading_returns_recent_entries_newest_first(tmp_path: Path) -> None:
    history_path = tmp_path / "runs.jsonl"
    history_path.write_text(
        "\n".join(
            json.dumps({"run_id": f"run-{index}", "status": "ok"})
            for index in range(3)
        ),
        encoding="utf-8",
    )

    entries = streamlit_app.load_recent_history(history_path, limit=2)

    assert [entry["run_id"] for entry in entries] == ["run-2", "run-1"]


def test_artifact_path_collection_handles_missing_optional_outputs(tmp_path: Path) -> None:
    summary_path = tmp_path / "video_summary.json"
    metrics_path = tmp_path / "video_frames_metrics.csv"
    keypoints_path = tmp_path / "video_keypoints.json"
    sampled_frame = tmp_path / "sampled_frames" / "frame_000000.jpg"
    sampled_frame.parent.mkdir(parents=True)
    sampled_frame.write_bytes(b"image")
    summary_path.write_text("{}", encoding="utf-8")
    metrics_path.write_text("frame_index\n0\n", encoding="utf-8")
    keypoints_path.write_text("{}", encoding="utf-8")

    artifacts = streamlit_app.collect_artifact_paths(
        {
            "artifacts": {
                "summary": str(summary_path),
                "frame_metrics_csv": str(metrics_path),
                "keypoints_json": str(keypoints_path),
                "sampled_frames_dir": str(sampled_frame.parent),
                "sampled_frames": [str(sampled_frame), str(tmp_path / "missing.jpg")],
                "annotated_video": str(tmp_path / "missing.mp4"),
                "excel_report": str(tmp_path / "missing.xlsx"),
            }
        }
    )

    assert artifacts["summary"] == summary_path
    assert artifacts["frame_metrics_csv"] == metrics_path
    assert artifacts["keypoints_json"] == keypoints_path
    assert artifacts["sampled_frames"] == [sampled_frame]
    assert artifacts["annotated_video"] is None
    assert artifacts["annotated_video_preview"] is None
    assert artifacts["annotated_video_browser"] is None
    assert artifacts["annotated_video_browser_compatible"] is False
    assert artifacts["excel_report"] is None


def test_artifact_collector_prefers_browser_compatible_video(tmp_path: Path) -> None:
    original_video = tmp_path / "annotated_video.mp4"
    browser_video = tmp_path / "annotated_video_h264.mp4"
    original_video.write_bytes(b"mp4v")
    browser_video.write_bytes(b"h264")

    artifacts = streamlit_app.collect_artifact_paths(
        {
            "artifacts": {
                "annotated_video": str(original_video),
                "annotated_video_browser": str(browser_video),
                "annotated_video_browser_compatible": True,
            }
        }
    )

    assert artifacts["annotated_video"] == original_video
    assert artifacts["annotated_video_browser"] == browser_video
    assert artifacts["annotated_video_preview"] == browser_video
    assert artifacts["annotated_video_browser_compatible"] is True
