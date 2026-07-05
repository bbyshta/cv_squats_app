"""Streamlit video-only squat demo app."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.inference.video_pipeline import VideoPipelineResult, process_video
from src.utils.config import load_yaml_config
from src.utils.history import DEFAULT_HISTORY_PATH


DEFAULT_CONFIG_PATH = Path("configs/app.yaml")
DEFAULT_UPLOAD_ROOT = Path("outputs/uploads")
DEFAULT_OUTPUT_ROOT = Path("outputs/predictions")


def load_demo_model_info(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the selected model and app metadata for display."""
    config = load_yaml_config(config_path)
    app_config = dict(config.get("app", {}))
    selected_model = dict(config.get("selected_model", {}))
    runtime = dict(config.get("runtime", {}))
    return {
        "title": app_config.get("title", "Squat technique analysis"),
        "exercise_type": app_config.get("exercise_type", app_config.get("exercise", "squat")),
        "history_path": app_config.get("history_path", str(DEFAULT_HISTORY_PATH)),
        "report_dir": app_config.get("report_dir", "outputs/reports"),
        "prediction_dir": app_config.get("prediction_dir", str(DEFAULT_OUTPUT_ROOT)),
        "model_id": selected_model.get("model_id"),
        "model_name": selected_model.get("name"),
        "backend": selected_model.get("backend"),
        "bbox_strategy": selected_model.get("bbox_strategy"),
        "runtime_environment": runtime.get("environment"),
    }


def save_uploaded_video(uploaded_file: Any, upload_root: str | Path = DEFAULT_UPLOAD_ROOT) -> Path:
    """Persist a Streamlit uploaded video under outputs/uploads."""
    original_name = Path(getattr(uploaded_file, "name", "uploaded_video.mp4")).name
    suffix = Path(original_name).suffix or ".mp4"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    upload_dir = Path(upload_root) / f"upload_{timestamp}_{uuid.uuid4().hex[:8]}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_path = upload_dir / f"input{suffix}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def format_summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Return compact summary values for UI display."""
    return {
        "status": summary.get("status"),
        "run_id": summary.get("run_id"),
        "processed_frames": summary.get("processed_frame_count"),
        "valid_keypoint_ratio": _format_ratio(summary.get("valid_keypoint_frame_ratio")),
        "usable_knee_ratio": _format_ratio(summary.get("usable_knee_frame_ratio")),
        "reliable_frame_ratio": _format_ratio(summary.get("reliable_frame_ratio")),
        "min_selected_knee_angle": _format_number(summary.get("min_selected_knee_angle")),
        "max_selected_knee_angle": _format_number(summary.get("max_selected_knee_angle")),
        "mean_torso_lean_angle": _format_number(summary.get("mean_torso_lean_angle")),
        "depth_status": summary.get("depth_status"),
        "phase_counts": summary.get("phase_counts", {}),
        "warnings": summary.get("dominant_warnings", {}),
    }


def load_recent_history(
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Load recent JSONL history entries, newest first."""
    path = Path(history_path)
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-limit:][::-1]


def collect_artifact_paths(summary: dict[str, Any]) -> dict[str, Any]:
    """Collect required and optional artifact paths from a video summary."""
    artifacts = dict(summary.get("artifacts", {}))
    sampled_frames = [
        Path(path)
        for path in artifacts.get("sampled_frames", [])
        if path and Path(path).is_file()
    ]
    annotated_video = _existing_path_or_none(artifacts.get("annotated_video"))
    browser_video = _existing_path_or_none(artifacts.get("annotated_video_browser"))
    browser_compatible = bool(artifacts.get("annotated_video_browser_compatible")) and browser_video is not None
    return {
        "summary": _path_or_none(artifacts.get("summary")),
        "frame_metrics_csv": _path_or_none(artifacts.get("frame_metrics_csv")),
        "keypoints_json": _path_or_none(artifacts.get("keypoints_json")),
        "sampled_frames_dir": _path_or_none(artifacts.get("sampled_frames_dir")),
        "sampled_frames": sampled_frames,
        "annotated_video": annotated_video,
        "annotated_video_browser": browser_video,
        "annotated_video_preview": browser_video or annotated_video,
        "annotated_video_browser_compatible": browser_compatible,
        "excel_report": _existing_path_or_none(artifacts.get("excel_report")),
    }


def main() -> None:
    """Render the Streamlit app."""
    import streamlit as st

    st.set_page_config(page_title="Squat technique analysis", layout="wide")
    model_info = load_demo_model_info(DEFAULT_CONFIG_PATH)

    st.title("Squat technique analysis")
    st.caption("Video-only demo using the selected RTMPose-S model. No real-time performance claim.")
    st.subheader("Selected model")
    st.write(
        {
            "model_id": model_info["model_id"],
            "model_name": model_info["model_name"],
            "backend": model_info["backend"],
            "bbox_strategy": model_info["bbox_strategy"],
            "runtime": model_info["runtime_environment"],
            "exercise": model_info["exercise_type"],
        }
    )

    uploaded_video = st.file_uploader(
        "Upload squat video",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
    )
    frame_stride = st.number_input("frame_stride", min_value=1, max_value=120, value=5, step=1)
    use_max_frames = st.checkbox("Limit processed frames", value=False)
    max_frames = None
    if use_max_frames:
        max_frames = int(st.number_input("max_frames", min_value=1, value=150, step=1))
    save_annotated_video = st.checkbox("Save annotated video", value=True)
    export_excel = st.checkbox("Export Excel", value=False)

    if st.button("Run analysis", type="primary", disabled=uploaded_video is None):
        if uploaded_video is None:
            st.warning("Upload a squat video first.")
        else:
            _run_analysis(
                st=st,
                uploaded_video=uploaded_video,
                frame_stride=int(frame_stride),
                max_frames=max_frames,
                save_annotated_video=save_annotated_video,
                export_excel=export_excel,
                model_info=model_info,
            )

    _render_history(st, model_info["history_path"])


def _run_analysis(
    st: Any,
    uploaded_video: Any,
    frame_stride: int,
    max_frames: int | None,
    save_annotated_video: bool,
    export_excel: bool,
    model_info: dict[str, Any],
) -> None:
    input_path = save_uploaded_video(uploaded_video)
    with st.spinner("Running squat video analysis..."):
        try:
            result = process_video(
                input_path=input_path,
                config_path=DEFAULT_CONFIG_PATH,
                frame_stride=frame_stride,
                max_frames=max_frames,
                output_root=model_info["prediction_dir"],
                save_annotated_video=save_annotated_video,
                save_history=True,
                history_path=model_info["history_path"],
                export_excel=export_excel,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(f"Analysis failed: {exc}")
            return
    _render_result(st, result)


def _render_result(st: Any, result: VideoPipelineResult) -> None:
    summary = result.summary
    metrics = format_summary_metrics(summary)
    artifacts = collect_artifact_paths(summary)

    st.success(f"Analysis status: {metrics['status']}")
    st.write(f"Run ID: `{metrics['run_id']}`")

    cols = st.columns(4)
    cols[0].metric("Processed frames", metrics["processed_frames"])
    cols[1].metric("Valid keypoints", metrics["valid_keypoint_ratio"])
    cols[2].metric("Usable knee", metrics["usable_knee_ratio"])
    cols[3].metric("Reliable frames", metrics["reliable_frame_ratio"])

    cols = st.columns(4)
    cols[0].metric("Min knee angle", metrics["min_selected_knee_angle"])
    cols[1].metric("Max knee angle", metrics["max_selected_knee_angle"])
    cols[2].metric("Mean torso lean", metrics["mean_torso_lean_angle"])
    cols[3].metric("Depth", metrics["depth_status"])

    st.write("Phase counts", metrics["phase_counts"] or {})
    st.write("Warnings", metrics["warnings"] or {"none": 0})

    _render_sampled_frames(st, artifacts["sampled_frames"])
    if artifacts["annotated_video_preview"] is not None:
        st.subheader("Annotated video")
        if not artifacts["annotated_video_browser_compatible"]:
            st.warning(
                "Annotated video was created but may not be browser-compatible. "
                "Use download or sampled frames."
            )
        _, video_column, _ = st.columns([1, 2, 1])
        with video_column:
            st.video(Path(artifacts["annotated_video_preview"]).read_bytes(), width="stretch")

    frame_metrics_path = artifacts["frame_metrics_csv"]
    if frame_metrics_path is not None and frame_metrics_path.is_file():
        import pandas as pd

        st.subheader("Frame metrics")
        st.dataframe(pd.read_csv(frame_metrics_path), width="stretch")

    _render_downloads(st, artifacts)


def _render_sampled_frames(st: Any, sampled_frames: list[Path]) -> None:
    st.subheader("Sampled annotated frames")
    if not sampled_frames:
        st.info("No sampled frames were saved.")
        return
    captions = [path.name for path in sampled_frames]
    st.image([str(path) for path in sampled_frames], caption=captions, width=240)


def _render_downloads(st: Any, artifacts: dict[str, Any]) -> None:
    st.subheader("Downloads")
    downloads = (
        ("video_summary.json", artifacts["summary"], "application/json"),
        ("video_frames_metrics.csv", artifacts["frame_metrics_csv"], "text/csv"),
        ("video_keypoints.json", artifacts["keypoints_json"], "application/json"),
        ("annotated_video.mp4", artifacts["annotated_video"], "video/mp4"),
        ("annotated_video_h264.mp4", artifacts["annotated_video_browser"], "video/mp4"),
        (
            "Excel report",
            artifacts["excel_report"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    )
    for label, path, mime in downloads:
        if path is None or not Path(path).is_file():
            continue
        st.download_button(
            label=label,
            data=Path(path).read_bytes(),
            file_name=Path(path).name,
            mime=mime,
        )


def _render_history(st: Any, history_path: str | Path) -> None:
    st.subheader("Recent history")
    entries = load_recent_history(history_path, limit=5)
    if not entries:
        st.info("No history entries yet.")
        return
    display_rows = [
        {
            "created_at": item.get("created_at"),
            "run_id": item.get("run_id"),
            "status": item.get("status"),
            "processed": item.get("processed_frame_count"),
            "usable_knee_ratio": item.get("usable_knee_frame_ratio"),
            "depth_status": item.get("depth_status"),
            "output_dir": item.get("output_dir"),
        }
        for item in entries
    ]
    st.dataframe(display_rows, width="stretch")


def _path_or_none(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _existing_path_or_none(value: Any) -> Path | None:
    path = _path_or_none(value)
    if path is None or not path.is_file():
        return None
    return path


def _format_ratio(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1%}"


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
