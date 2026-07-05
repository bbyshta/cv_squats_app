from __future__ import annotations

import json
from pathlib import Path

from src.utils.history import append_jsonl_record, append_run_history, build_history_record


def test_jsonl_append_creates_valid_json_line(tmp_path: Path) -> None:
    history_path = tmp_path / "nested" / "runs.jsonl"

    append_jsonl_record(history_path, {"run_id": "run-1", "status": "ok"})

    lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"run_id": "run-1", "status": "ok"}


def test_history_handles_missing_file_and_directories(tmp_path: Path) -> None:
    summary = _summary()
    history_path = tmp_path / "history" / "runs.jsonl"

    written_path = append_run_history(summary, tmp_path / "predictions" / "run-1", history_path)

    assert written_path == history_path
    record = json.loads(history_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["run_id"] == "run-1"
    assert record["input_type"] == "video"
    assert record["exercise_type"] == "squat"
    assert record["usable_knee_frame_ratio"] == 1.0
    assert record["output_dir"].endswith("run-1")


def test_build_history_record_uses_required_fields(tmp_path: Path) -> None:
    record = build_history_record(
        _summary(),
        tmp_path / "predictions" / "run-1",
        created_at="2026-07-05T18:05:20Z",
    )

    assert set(record) == {
        "run_id",
        "created_at",
        "input_type",
        "exercise_type",
        "input_path",
        "model_id",
        "model_name",
        "backend",
        "bbox_strategy",
        "status",
        "frame_stride",
        "frame_count_total",
        "processed_frame_count",
        "valid_keypoint_frame_ratio",
        "usable_knee_frame_ratio",
        "reliable_frame_ratio",
        "min_selected_knee_angle",
        "max_selected_knee_angle",
        "mean_torso_lean_angle",
        "depth_status",
        "warning_frame_count",
        "output_dir",
        "artifacts",
    }
    assert record["created_at"] == "2026-07-05T18:05:20Z"


def _summary() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "exercise_type": "squat",
        "input_path": "data/samples/squat_sample.mp4",
        "model_id": "rtmpose_s",
        "model_name": "RTMPose-S",
        "backend": "mmpose",
        "bbox_strategy": "whole_frame",
        "status": "ok",
        "frame_stride": 5,
        "frame_count_total": 925,
        "processed_frame_count": 185,
        "valid_keypoint_frame_ratio": 1.0,
        "usable_knee_frame_ratio": 1.0,
        "reliable_frame_ratio": 1.0,
        "min_selected_knee_angle": 71.565,
        "max_selected_knee_angle": 179.398,
        "mean_torso_lean_angle": 17.49,
        "depth_status": "sufficient",
        "warning_frame_count": 0,
        "artifacts": {"summary": "video_summary.json"},
    }
