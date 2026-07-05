"""Stride-based RTMPose-S video pipeline for the video-only demo scope."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from src.analysis.squat import (
    SquatAnalysisConfig,
    aggregate_depth_status,
    analyze_squat_frame,
    default_squat_config,
    squat_config_from_app_config,
)
from src.inference.base import Keypoint, PoseResult
from src.inference.rtmpose_video_runner import (
    FrameInferenceOutput,
    RTMPoseFrameAdapter,
    RTMPoseVideoConfig,
    load_rtmpose_video_config,
)
from src.inference.video_io import (
    VideoFrame,
    VideoMetadata,
    iter_video_frames,
    read_video_metadata,
    validate_frame_stride,
    validate_video_path,
)
from src.utils.config import load_yaml_config
from src.utils.history import DEFAULT_HISTORY_PATH, append_run_history
from src.utils.io import write_image
from src.utils.report_excel import export_video_report
from src.utils.video_export import BrowserVideoExportResult, export_browser_compatible_mp4
from src.visualization.video_draw import AnnotatedVideoWriter, draw_annotated_video_frame


SAMPLED_FRAME_LIMIT = 20
FRAME_METRICS_COLUMNS = (
    "frame_index",
    "inference_status",
    "selected_side",
    "critical_keypoints_valid",
    "mean_critical_confidence",
    "detected_keypoints",
    "left_knee_angle",
    "right_knee_angle",
    "selected_knee_angle",
    "torso_lean_angle",
    "squat_phase",
    "depth_status",
    "left_right_knee_angle_diff",
    "knee_angle_valid",
    "left_knee_angle_valid",
    "right_knee_angle_valid",
    "torso_lean_valid",
    "frame_reliable",
    "angle_quality",
    "technique_assessment",
    "warning_count",
    "warnings",
    "sampled_frame_path",
    "error_message",
)


@dataclass(frozen=True)
class VideoPipelineResult:
    """Completed video pipeline output."""

    run_id: str
    run_dir: Path
    summary_path: Path
    frame_metrics_path: Path
    keypoints_path: Path
    sampled_frames_dir: Path
    annotated_video_path: Path | None
    history_path: Path | None
    excel_report_path: Path | None
    summary: dict[str, Any]


def load_squat_analysis_config(config_path: str | Path) -> SquatAnalysisConfig:
    """Load squat analysis thresholds from application config."""
    return squat_config_from_app_config(load_yaml_config(config_path))


def process_video(
    input_path: str | Path,
    config_path: str | Path,
    frame_stride: int,
    max_frames: int | None = None,
    output_root: str | Path = "outputs/predictions",
    save_annotated_video: bool = True,
    save_history: bool = True,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    export_excel: bool = False,
    excel_output_path: str | Path | None = None,
    run_id: str | None = None,
    metadata_reader: Callable[[str | Path], VideoMetadata] = read_video_metadata,
    frame_reader: Callable[[str | Path, int, int | None], Iterable[VideoFrame]] = iter_video_frames,
    adapter_factory: Callable[[RTMPoseVideoConfig], Any] = RTMPoseFrameAdapter,
    video_writer_factory: Callable[[str | Path, float, tuple[int, int]], Any] = AnnotatedVideoWriter,
    frame_drawer: Callable[[Any, PoseResult, dict[str, Any], float], Any] = draw_annotated_video_frame,
    image_writer: Callable[[str | Path, Any], Path] = write_image,
    history_writer: Callable[[dict[str, Any], str | Path, str | Path], Path] = append_run_history,
    excel_exporter: Callable[[str | Path, str | Path | None], Path] = export_video_report,
    browser_video_exporter: Callable[[str | Path, str | Path | None], BrowserVideoExportResult] = (
        export_browser_compatible_mp4
    ),
) -> VideoPipelineResult:
    """Process a video and write future demo artifacts."""
    video_path = validate_video_path(input_path)
    validate_frame_stride(frame_stride)
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be greater than or equal to 1 when provided")

    runtime_config = load_rtmpose_video_config(config_path)
    thresholds = load_squat_analysis_config(config_path)
    metadata = metadata_reader(video_path)
    current_run_id = run_id or _new_run_id()
    run_dir = Path(output_root) / current_run_id
    sampled_frames_dir = run_dir / "sampled_frames"
    sampled_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_metrics_path = run_dir / "video_frames_metrics.csv"
    keypoints_path = run_dir / "video_keypoints.json"
    summary_path = run_dir / "video_summary.json"
    annotated_video_path = run_dir / "annotated_video.mp4"

    adapter = adapter_factory(runtime_config)
    adapter.load()

    frame_metrics: list[dict[str, Any]] = []
    keypoint_frames: list[dict[str, Any]] = []
    sampled_frame_paths: list[str] = []
    limitations: list[str] = []
    writer_warning: str | None = None
    previous_selected_side: str | None = None
    writer = _open_writer_best_effort(
        save_annotated_video,
        annotated_video_path,
        metadata,
        frame_stride,
        video_writer_factory,
    )
    if save_annotated_video and writer is None:
        writer_warning = "annotated_video_writer_unavailable"
        limitations.append("Annotated video was not written because VideoWriter could not open.")

    for video_frame in frame_reader(video_path, frame_stride, max_frames):
        output = adapter.predict_frame(video_frame.image, frame_index=video_frame.frame_index)
        pose_result = _pose_result_from_output(
            output,
            image_size=_frame_image_size(video_frame.image, metadata),
            source_path=str(video_path),
        )
        metrics = analyze_video_frame(
            output=output,
            pose_result=pose_result,
            thresholds=thresholds,
            previous_selected_side=previous_selected_side,
        )
        previous_selected_side = metrics["selected_side"] or previous_selected_side
        keypoint_frames.append(
            {
                "frame_index": video_frame.frame_index,
                "status": output.status,
                "error_message": output.error_message,
                "keypoints": output.keypoints,
            }
        )
        annotated_frame = frame_drawer(
            video_frame.image,
            pose_result,
            metrics,
            thresholds.draw_min_conf,
        )
        if len(sampled_frame_paths) < SAMPLED_FRAME_LIMIT:
            sample_path = sampled_frames_dir / f"frame_{video_frame.frame_index:06d}.jpg"
            image_writer(sample_path, annotated_frame)
            sampled_frame_paths.append(str(sample_path))
            metrics["sampled_frame_path"] = str(sample_path)
        else:
            metrics["sampled_frame_path"] = ""
        frame_metrics.append(metrics)
        if writer is not None:
            try:
                writer.write(annotated_frame)
            except Exception as exc:  # noqa: BLE001 - OpenCV writer failures vary.
                writer_warning = f"annotated_video_write_failed: {exc}"
                limitations.append("Annotated video writing failed before all processed frames were saved.")
                _release_writer(writer)
                writer = None

    if writer is not None:
        _release_writer(writer)

    artifacts = {
        "summary": str(summary_path),
        "frame_metrics_csv": str(frame_metrics_path),
        "keypoints_json": str(keypoints_path),
        "sampled_frames_dir": str(sampled_frames_dir),
        "sampled_frames": sampled_frame_paths,
        "annotated_video": str(annotated_video_path)
        if save_annotated_video and writer_warning is None
        else None,
        "annotated_video_browser": None,
        "annotated_video_browser_compatible": False,
    }
    if writer_warning is not None:
        artifacts["annotated_video_warning"] = writer_warning
    if artifacts["annotated_video"] and annotated_video_path.is_file():
        browser_video_path = run_dir / "annotated_video_h264.mp4"
        browser_export_result = browser_video_exporter(annotated_video_path, browser_video_path)
        if browser_export_result.browser_compatible and browser_export_result.output_path is not None:
            artifacts["annotated_video_browser"] = str(browser_export_result.output_path)
            artifacts["annotated_video_browser_compatible"] = True
        else:
            artifacts["video_export_warning"] = browser_export_result.warning or "video_export_failed"
            limitations.append(
                "Browser-compatible annotated video conversion failed; "
                "the original video and sampled frames remain available."
            )

    status = "ok" if frame_metrics else "failed"
    error_message = None if frame_metrics else "No frames were processed from the input video."
    if not frame_metrics:
        limitations.append("No frames were processed; check the input video and frame_stride.")
    run_depth_status = aggregate_depth_status(frame_metrics, thresholds)
    if run_depth_status == "insufficient":
        for metrics in frame_metrics:
            if "insufficient_depth" not in metrics["warnings"]:
                metrics["warnings"].append("insufficient_depth")
            metrics["warning_count"] = len(metrics["warnings"])
    for metrics in frame_metrics:
        metrics["depth_status"] = run_depth_status

    keypoint_frames = [
        _keypoint_frame_payload(item, frame_metrics[index])
        for index, item in enumerate(keypoint_frames)
    ]

    summary = build_video_summary(
        run_id=current_run_id,
        input_path=video_path,
        runtime_config=runtime_config,
        frame_stride=frame_stride,
        frame_count_total=metadata.frame_count_total,
        frame_metrics=frame_metrics,
        depth_status=run_depth_status,
        artifacts=artifacts,
        limitations=limitations,
        status=status,
        error_message=error_message,
    )
    _write_frame_metrics_csv(frame_metrics_path, frame_metrics)
    _write_json(
        keypoints_path,
        {
            "run_id": current_run_id,
            "input_path": str(video_path),
            "model_id": runtime_config.model_id,
            "model_name": runtime_config.model_name,
            "backend": runtime_config.backend,
            "bbox_strategy": runtime_config.bbox_strategy,
            "frames": keypoint_frames,
        },
    )
    _write_json(summary_path, summary)

    pipeline_warnings: list[str] = []
    excel_report_path: Path | None = None
    history_output_path: Path | None = None
    if export_excel and status == "ok":
        try:
            excel_report_path = excel_exporter(run_dir, excel_output_path)
            artifacts["excel_report"] = str(excel_report_path)
        except Exception as exc:  # noqa: BLE001 - report export must not fail inference.
            warning = f"excel_export_failed: {exc}"
            pipeline_warnings.append(warning)
            limitations.append("Excel report export failed after video artifacts were written.")
    if save_history and status == "ok":
        try:
            history_output_path = history_writer(summary, run_dir, history_path)
            artifacts["history_jsonl"] = str(history_output_path)
        except Exception as exc:  # noqa: BLE001 - history is best-effort for demo runs.
            warning = f"history_write_failed: {exc}"
            pipeline_warnings.append(warning)
            limitations.append("JSONL history write failed after video artifacts were written.")
    if pipeline_warnings or excel_report_path is not None or history_output_path is not None:
        summary["artifacts"] = artifacts
        summary["limitations"] = _deduplicate(limitations)
        summary["pipeline_warnings"] = pipeline_warnings
        _write_json(summary_path, summary)

    return VideoPipelineResult(
        run_id=current_run_id,
        run_dir=run_dir,
        summary_path=summary_path,
        frame_metrics_path=frame_metrics_path,
        keypoints_path=keypoints_path,
        sampled_frames_dir=sampled_frames_dir,
        annotated_video_path=annotated_video_path if artifacts["annotated_video"] else None,
        history_path=history_output_path,
        excel_report_path=excel_report_path,
        summary=summary,
    )


def analyze_video_frame(
    output: FrameInferenceOutput,
    pose_result: PoseResult,
    thresholds: SquatAnalysisConfig | None = None,
    previous_selected_side: str | None = None,
) -> dict[str, Any]:
    """Create confidence-aware metrics for one processed video frame."""
    current_thresholds = thresholds or default_squat_config()
    if output.status != "ok":
        return _frame_metrics(
            output=output,
            selected_side=None,
            critical_keypoints_valid=False,
            mean_critical_confidence=0.0,
            detected_keypoints=0,
            left_knee_angle=None,
            right_knee_angle=None,
            selected_knee_angle=None,
            torso_lean_angle=None,
            squat_phase="unknown",
            depth_status="unknown",
            left_right_knee_angle_diff=None,
            knee_angle_valid=False,
            left_knee_angle_valid=False,
            right_knee_angle_valid=False,
            torso_lean_valid=False,
            frame_reliable=False,
            angle_quality="unavailable",
            technique_assessment="not_assessed_inference_failed",
            warnings=["inference_failed", "knee_angle_unavailable", "unknown_phase"],
            sampled_frame_path="",
        )

    analysis = analyze_squat_frame(
        pose_result,
        current_thresholds,
        previous_selected_side=previous_selected_side,
    )
    return _frame_metrics(
        output=output,
        selected_side=analysis["selected_side"],
        critical_keypoints_valid=bool(analysis["critical_keypoints_valid"]),
        mean_critical_confidence=float(analysis["mean_critical_confidence"]),
        detected_keypoints=pose_result.detected_keypoints(current_thresholds.draw_min_conf),
        left_knee_angle=analysis["left_knee_angle"],
        right_knee_angle=analysis["right_knee_angle"],
        selected_knee_angle=analysis["selected_knee_angle"],
        torso_lean_angle=analysis["torso_lean_angle"],
        squat_phase=str(analysis["squat_phase"]),
        depth_status=str(analysis["depth_status"]),
        left_right_knee_angle_diff=analysis["left_right_knee_angle_diff"],
        knee_angle_valid=bool(analysis["knee_angle_valid"]),
        left_knee_angle_valid=bool(analysis["left_knee_angle_valid"]),
        right_knee_angle_valid=bool(analysis["right_knee_angle_valid"]),
        torso_lean_valid=bool(analysis["torso_lean_valid"]),
        frame_reliable=bool(analysis["frame_reliable"]),
        angle_quality=str(analysis["angle_quality"]),
        technique_assessment=_squat_assessment(analysis),
        warnings=list(analysis["warnings"]),
        sampled_frame_path="",
    )


def build_video_summary(
    run_id: str,
    input_path: str | Path,
    runtime_config: RTMPoseVideoConfig,
    frame_stride: int,
    frame_count_total: int,
    frame_metrics: list[dict[str, Any]],
    depth_status: str,
    artifacts: dict[str, Any],
    limitations: list[str],
    status: str,
    error_message: str | None,
) -> dict[str, Any]:
    """Aggregate per-frame metrics into the required video summary schema."""
    processed_count = len(frame_metrics)
    valid_count = sum(1 for item in frame_metrics if item.get("critical_keypoints_valid") is True)
    usable_knee_count = sum(1 for item in frame_metrics if item.get("knee_angle_valid") is True)
    reliable_count = sum(1 for item in frame_metrics if item.get("frame_reliable") is True)
    selected_knee_angles = [
        float(item["selected_knee_angle"])
        for item in frame_metrics
        if item.get("selected_knee_angle") is not None and item.get("knee_angle_valid") is True
    ]
    torso_lean_angles = [
        float(item["torso_lean_angle"])
        for item in frame_metrics
        if item.get("torso_lean_angle") is not None and item.get("torso_lean_valid") is True
    ]
    phase_counter: Counter[str] = Counter(str(item.get("squat_phase", "unknown")) for item in frame_metrics)
    confidences = [
        float(item["mean_critical_confidence"])
        for item in frame_metrics
        if item.get("mean_critical_confidence") is not None
    ]
    warning_counter: Counter[str] = Counter()
    for item in frame_metrics:
        for warning in item.get("warnings", []):
            warning_counter[str(warning)] += 1

    return {
        "run_id": run_id,
        "input_type": "video",
        "exercise_type": "squat",
        "input_path": str(input_path),
        "model_id": runtime_config.model_id,
        "model_name": runtime_config.model_name,
        "backend": runtime_config.backend,
        "bbox_strategy": runtime_config.bbox_strategy,
        "frame_stride": frame_stride,
        "frame_count_total": frame_count_total,
        "processed_frame_count": processed_count,
        "valid_keypoint_frame_count": valid_count,
        "valid_keypoint_frame_ratio": _ratio(valid_count, processed_count),
        "usable_knee_frame_count": usable_knee_count,
        "usable_knee_frame_ratio": _ratio(usable_knee_count, processed_count),
        "reliable_frame_count": reliable_count,
        "reliable_frame_ratio": _ratio(reliable_count, processed_count),
        "min_selected_knee_angle": min(selected_knee_angles) if selected_knee_angles else None,
        "max_selected_knee_angle": max(selected_knee_angles) if selected_knee_angles else None,
        "mean_torso_lean_angle": _mean(torso_lean_angles) if torso_lean_angles else None,
        "depth_status": depth_status,
        "phase_counts": dict(phase_counter),
        "mean_critical_confidence": _mean(confidences),
        "warning_frame_count": sum(1 for item in frame_metrics if item.get("warnings")),
        "dominant_warnings": dict(warning_counter.most_common()),
        "artifacts": artifacts,
        "limitations": limitations,
        "status": status,
        "error_message": error_message,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the selected RTMPose-S video pipeline with frame_stride sampling."
    )
    parser.add_argument("--input", required=True, help="Path to an input video file.")
    parser.add_argument(
        "--config",
        default="configs/app.yaml",
        help="Path to application config. Default: configs/app.yaml",
    )
    parser.add_argument("--frame-stride", type=int, required=True, help="Process every Nth frame.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of processed frames.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/predictions",
        help="Output root for prediction artifacts. Default: outputs/predictions",
    )
    parser.add_argument(
        "--save-annotated-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Best-effort annotated MP4 output. Default: true.",
    )
    parser.add_argument(
        "--save-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a compact JSONL run-history record after successful runs. Default: true.",
    )
    parser.add_argument(
        "--history-path",
        default=str(DEFAULT_HISTORY_PATH),
        help=f"JSONL history path. Default: {DEFAULT_HISTORY_PATH}",
    )
    parser.add_argument(
        "--export-excel",
        action="store_true",
        help="Export an XLSX report from the generated run artifacts.",
    )
    parser.add_argument(
        "--excel-output",
        default=None,
        help="Optional XLSX output path. Default: outputs/reports/video_report_<run_id>.xlsx",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = process_video(
            input_path=args.input,
            config_path=args.config,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            output_root=args.output_root,
            save_annotated_video=args.save_annotated_video,
            save_history=args.save_history,
            history_path=args.history_path,
            export_excel=args.export_excel,
            excel_output_path=args.excel_output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    print(
        json.dumps(
            {
                "status": result.summary["status"],
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "summary": str(result.summary_path),
            },
            ensure_ascii=False,
        )
    )


def _frame_metrics(
    output: FrameInferenceOutput,
    selected_side: str | None,
    critical_keypoints_valid: bool,
    mean_critical_confidence: float,
    detected_keypoints: int,
    left_knee_angle: float | None,
    right_knee_angle: float | None,
    selected_knee_angle: float | None,
    torso_lean_angle: float | None,
    squat_phase: str,
    depth_status: str,
    left_right_knee_angle_diff: float | None,
    knee_angle_valid: bool,
    left_knee_angle_valid: bool,
    right_knee_angle_valid: bool,
    torso_lean_valid: bool,
    frame_reliable: bool,
    angle_quality: str,
    technique_assessment: str,
    warnings: list[str],
    sampled_frame_path: str,
) -> dict[str, Any]:
    return {
        "frame_index": output.frame_index,
        "inference_status": output.status,
        "selected_side": selected_side,
        "critical_keypoints_valid": critical_keypoints_valid,
        "mean_critical_confidence": mean_critical_confidence,
        "detected_keypoints": detected_keypoints,
        "left_knee_angle": _round_optional(left_knee_angle),
        "right_knee_angle": _round_optional(right_knee_angle),
        "selected_knee_angle": _round_optional(selected_knee_angle),
        "torso_lean_angle": _round_optional(torso_lean_angle),
        "squat_phase": squat_phase,
        "depth_status": depth_status,
        "left_right_knee_angle_diff": _round_optional(left_right_knee_angle_diff),
        "knee_angle_valid": knee_angle_valid,
        "left_knee_angle_valid": left_knee_angle_valid,
        "right_knee_angle_valid": right_knee_angle_valid,
        "torso_lean_valid": torso_lean_valid,
        "frame_reliable": frame_reliable,
        "angle_quality": angle_quality,
        "technique_assessment": technique_assessment,
        "warning_count": len(warnings),
        "warnings": warnings,
        "sampled_frame_path": sampled_frame_path,
        "error_message": output.error_message,
    }


def _pose_result_from_output(
    output: FrameInferenceOutput,
    image_size: tuple[int, int],
    source_path: str,
) -> PoseResult:
    return PoseResult(
        model_name=output.model_name,
        keypoints=[
            Keypoint(
                name=str(item["name"]),
                x=float(item["x"]),
                y=float(item["y"]),
                confidence=None if item.get("confidence") is None else float(item["confidence"]),
            )
            for item in output.keypoints
        ],
        image_size=image_size,
        inference_time_ms=0.0,
        source_path=source_path,
    )


def _open_writer_best_effort(
    save_annotated_video: bool,
    path: Path,
    metadata: VideoMetadata,
    frame_stride: int,
    writer_factory: Callable[[str | Path, float, tuple[int, int]], Any],
) -> Any | None:
    if not save_annotated_video:
        return None
    try:
        fps = max(metadata.fps / frame_stride, 1.0)
        return writer_factory(path, fps, (metadata.width, metadata.height))
    except Exception:
        return None


def _release_writer(writer: Any) -> None:
    try:
        writer.release()
    except Exception:
        pass


def _write_frame_metrics_csv(path: Path, frame_metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_METRICS_COLUMNS)
        writer.writeheader()
        for item in frame_metrics:
            row = {column: item.get(column) for column in FRAME_METRICS_COLUMNS}
            row["warnings"] = ";".join(item.get("warnings", []))
            writer.writerow(row)


def _keypoint_frame_payload(base_frame: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        **base_frame,
        "selected_side": metrics["selected_side"],
        "critical_keypoints_valid": metrics["critical_keypoints_valid"],
        "mean_critical_confidence": metrics["mean_critical_confidence"],
        "squat": {
            "left_knee_angle": metrics["left_knee_angle"],
            "right_knee_angle": metrics["right_knee_angle"],
            "selected_knee_angle": metrics["selected_knee_angle"],
            "torso_lean_angle": metrics["torso_lean_angle"],
            "squat_phase": metrics["squat_phase"],
            "depth_status": metrics["depth_status"],
            "left_right_knee_angle_diff": metrics["left_right_knee_angle_diff"],
        },
        "validity": {
            "knee_angle": metrics["knee_angle_valid"],
            "left_knee_angle": metrics["left_knee_angle_valid"],
            "right_knee_angle": metrics["right_knee_angle_valid"],
            "torso_lean": metrics["torso_lean_valid"],
            "frame_reliable": metrics["frame_reliable"],
        },
        "angle_quality": metrics["angle_quality"],
        "warnings": metrics["warnings"],
    }


def _squat_assessment(analysis: dict[str, Any]) -> str:
    if analysis["angle_quality"] == "unavailable":
        return "not_assessed_knee_angle_unavailable"
    if analysis["angle_quality"] == "tentative":
        return "tentative_squat_pose_low_confidence"
    if "excessive_forward_lean" in analysis["warnings"]:
        return "torso_lean_warning"
    if "possible_asymmetry" in analysis["warnings"]:
        return "knee_asymmetry_warning"
    return "no_confident_issue_detected"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _frame_image_size(frame: Any, fallback: VideoMetadata) -> tuple[int, int]:
    if hasattr(frame, "shape") and len(frame.shape) >= 2:
        return (int(frame.shape[1]), int(frame.shape[0]))
    return (fallback.width, fallback.height)


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"video_{timestamp}_{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
    main()
