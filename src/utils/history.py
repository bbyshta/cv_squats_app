"""Compact JSONL history for video demo runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HISTORY_PATH = Path("outputs/history/runs.jsonl")


def append_run_history(
    summary: dict[str, Any],
    output_dir: str | Path,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
) -> Path:
    """Append one compact run record to the JSONL history file."""
    record = build_history_record(summary, output_dir)
    return append_jsonl_record(history_path, record)


def build_history_record(
    summary: dict[str, Any],
    output_dir: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the stable history payload from a video summary."""
    artifacts = summary.get("artifacts", {})
    return {
        "run_id": summary.get("run_id"),
        "created_at": created_at or _utc_now(),
        "input_type": "video",
        "exercise_type": summary.get("exercise_type", "squat"),
        "input_path": summary.get("input_path"),
        "model_id": summary.get("model_id"),
        "model_name": summary.get("model_name"),
        "backend": summary.get("backend"),
        "bbox_strategy": summary.get("bbox_strategy"),
        "status": summary.get("status"),
        "frame_stride": summary.get("frame_stride"),
        "frame_count_total": summary.get("frame_count_total"),
        "processed_frame_count": summary.get("processed_frame_count"),
        "valid_keypoint_frame_ratio": summary.get("valid_keypoint_frame_ratio"),
        "usable_knee_frame_ratio": summary.get("usable_knee_frame_ratio"),
        "reliable_frame_ratio": summary.get("reliable_frame_ratio"),
        "min_selected_knee_angle": summary.get("min_selected_knee_angle"),
        "max_selected_knee_angle": summary.get("max_selected_knee_angle"),
        "mean_torso_lean_angle": summary.get("mean_torso_lean_angle"),
        "depth_status": summary.get("depth_status"),
        "warning_frame_count": summary.get("warning_frame_count"),
        "output_dir": str(output_dir),
        "artifacts": artifacts,
    }


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> Path:
    """Append one JSON object as a single UTF-8 JSONL line."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        file.write("\n")
    return output_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
