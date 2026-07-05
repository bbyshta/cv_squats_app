"""Confidence-aware squat analysis for video frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.analysis.angles import calculate_angle
from src.inference.base import Keypoint, PoseResult


SIDES = ("left", "right")
KNEE_PARTS = ("hip", "knee", "ankle")
TORSO_PARTS = ("shoulder", "hip")
CRITICAL_PARTS = ("shoulder", "hip", "knee", "ankle")


@dataclass(frozen=True)
class SquatAnalysisConfig:
    """Thresholds for confidence-aware squat analysis."""

    draw_min_conf: float = 0.15
    angle_min_conf: float = 0.25
    body_line_min_conf: float = 0.35
    reliable_min_conf: float = 0.50
    side_switch_margin: float = 0.08
    phase_top_min_angle: float = 160.0
    phase_bottom_max_angle: float = 110.0
    depth_sufficient_max_angle: float = 110.0
    torso_lean_warn_angle: float = 45.0
    knee_symmetry_warn_diff: float = 20.0


@dataclass(frozen=True)
class SquatSideFeatures:
    """Computed squat features for one body side."""

    side: str
    knee_angle: float | None
    knee_angle_valid: bool
    knee_angle_reliable: bool
    knee_confidence: float
    critical_confidence: float
    critical_keypoints_valid: bool
    torso_lean_angle: float | None
    torso_lean_valid: bool
    torso_lean_reliable: bool
    torso_lean_confidence: float


def default_squat_config() -> SquatAnalysisConfig:
    """Return default squat analysis thresholds."""
    return SquatAnalysisConfig()


def squat_config_from_app_config(config: dict[str, Any]) -> SquatAnalysisConfig:
    """Build squat analysis config from app YAML data."""
    app = config.get("app")
    if not isinstance(app, dict):
        raise ValueError("Config must contain app mapping")
    exercise = app.get("exercise_type", app.get("exercise"))
    if exercise != "squat":
        raise ValueError("app.exercise must be squat for the Phase 8D demo")

    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Config must contain runtime mapping")
    squat = config.get("squat", {})
    if not isinstance(squat, dict):
        raise ValueError("Config squat section must be a mapping when provided")

    defaults = default_squat_config()
    result = SquatAnalysisConfig(
        draw_min_conf=_float_value(runtime, "draw_min_conf", defaults.draw_min_conf),
        angle_min_conf=_float_value(runtime, "angle_min_conf", defaults.angle_min_conf),
        body_line_min_conf=_float_value(
            runtime,
            "body_line_min_conf",
            defaults.body_line_min_conf,
        ),
        reliable_min_conf=_float_value(runtime, "reliable_min_conf", defaults.reliable_min_conf),
        side_switch_margin=_float_value(runtime, "side_switch_margin", defaults.side_switch_margin),
        phase_top_min_angle=_float_value(
            squat,
            "phase_top_min_angle",
            defaults.phase_top_min_angle,
        ),
        phase_bottom_max_angle=_float_value(
            squat,
            "phase_bottom_max_angle",
            defaults.phase_bottom_max_angle,
        ),
        depth_sufficient_max_angle=_float_value(
            squat,
            "depth_sufficient_max_angle",
            defaults.depth_sufficient_max_angle,
        ),
        torso_lean_warn_angle=_float_value(
            squat,
            "torso_lean_warn_angle",
            defaults.torso_lean_warn_angle,
        ),
        knee_symmetry_warn_diff=_float_value(
            squat,
            "knee_symmetry_warn_diff",
            defaults.knee_symmetry_warn_diff,
        ),
    )
    validate_squat_config(result)
    return result


def validate_squat_config(config: SquatAnalysisConfig) -> None:
    """Validate squat analysis thresholds."""
    for name in ("draw_min_conf", "angle_min_conf", "body_line_min_conf", "reliable_min_conf"):
        value = getattr(config, name)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
    if config.side_switch_margin < 0.0:
        raise ValueError("side_switch_margin must be greater than or equal to 0.0")
    if config.angle_min_conf > config.reliable_min_conf:
        raise ValueError("angle_min_conf must not exceed reliable_min_conf")
    if config.body_line_min_conf > config.reliable_min_conf:
        raise ValueError("body_line_min_conf must not exceed reliable_min_conf")
    if config.phase_bottom_max_angle >= config.phase_top_min_angle:
        raise ValueError("phase_bottom_max_angle must be less than phase_top_min_angle")


def analyze_squat_frame(
    pose_result: PoseResult,
    config: SquatAnalysisConfig,
    previous_selected_side: str | None = None,
) -> dict[str, Any]:
    """Analyze squat state for one pose-estimated video frame."""
    features = {side: _side_features(pose_result, side, config) for side in SIDES}
    selected_side = _select_stable_side(features, previous_selected_side, config.side_switch_margin)
    selected = features.get(selected_side) if selected_side is not None else None
    phase_features = _best_knee_features(features, selected_side)

    selected_knee_angle = phase_features.knee_angle if phase_features is not None else None
    knee_angle_valid = bool(phase_features and phase_features.knee_angle_valid)
    frame_reliable = bool(
        phase_features
        and phase_features.knee_angle_reliable
        and phase_features.critical_keypoints_valid
    )
    phase = classify_squat_phase(selected_knee_angle, config)
    depth_status = "unknown"

    torso_features = selected if selected is not None else phase_features
    torso_lean_angle = torso_features.torso_lean_angle if torso_features is not None else None
    torso_lean_valid = bool(torso_features and torso_features.torso_lean_valid)
    left = features["left"]
    right = features["right"]
    angle_diff = _angle_diff(left.knee_angle, right.knee_angle)
    mean_critical_confidence = selected.critical_confidence if selected is not None else 0.0
    critical_keypoints_valid = bool(selected and selected.critical_keypoints_valid)

    warnings = _warnings(
        phase=phase,
        knee_angle_valid=knee_angle_valid,
        frame_reliable=frame_reliable,
        torso_lean_angle=torso_lean_angle,
        torso_lean_valid=torso_lean_valid,
        angle_diff=angle_diff,
        config=config,
    )
    return {
        "selected_side": selected_side,
        "critical_keypoints_valid": critical_keypoints_valid,
        "mean_critical_confidence": mean_critical_confidence,
        "left_knee_angle": _round_optional(left.knee_angle),
        "right_knee_angle": _round_optional(right.knee_angle),
        "selected_knee_angle": _round_optional(selected_knee_angle),
        "torso_lean_angle": _round_optional(torso_lean_angle),
        "squat_phase": phase,
        "depth_status": depth_status,
        "left_right_knee_angle_diff": _round_optional(angle_diff),
        "knee_angle_valid": knee_angle_valid,
        "left_knee_angle_valid": left.knee_angle_valid,
        "right_knee_angle_valid": right.knee_angle_valid,
        "torso_lean_valid": torso_lean_valid,
        "frame_reliable": frame_reliable,
        "angle_quality": _angle_quality(knee_angle_valid, frame_reliable),
        "warnings": warnings,
        "confidence": {
            "left_knee": left.knee_confidence,
            "right_knee": right.knee_confidence,
            "torso_lean": torso_features.torso_lean_confidence if torso_features else 0.0,
            "mean_critical": mean_critical_confidence,
        },
    }


def classify_squat_phase(knee_angle: float | None, config: SquatAnalysisConfig) -> str:
    """Classify squat phase from selected knee angle."""
    if knee_angle is None:
        return "unknown"
    if knee_angle >= config.phase_top_min_angle:
        return "top"
    if knee_angle <= config.phase_bottom_max_angle:
        return "bottom"
    return "middle"


def aggregate_depth_status(
    frame_metrics: list[dict[str, Any]],
    config: SquatAnalysisConfig,
) -> str:
    """Return run-level squat depth status from selected knee angles."""
    angles = [
        float(item["selected_knee_angle"])
        for item in frame_metrics
        if item.get("selected_knee_angle") is not None and item.get("knee_angle_valid") is True
    ]
    if not angles:
        return "unknown"
    if min(angles) <= config.depth_sufficient_max_angle:
        return "sufficient"
    return "insufficient"


def calculate_torso_lean_angle(shoulder: Keypoint, hip: Keypoint) -> float:
    """Return shoulder-hip line angle relative to image vertical in degrees."""
    dx = float(shoulder.x) - float(hip.x)
    dy = float(shoulder.y) - float(hip.y)
    if dx == 0.0 and dy == 0.0:
        raise ValueError("Cannot calculate torso lean with identical shoulder and hip points")
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _side_features(
    pose_result: PoseResult,
    side: str,
    config: SquatAnalysisConfig,
) -> SquatSideFeatures:
    knee_confidence = _mean_confidence_for_parts(pose_result, side, KNEE_PARTS)
    knee_angle_valid = _parts_pass_threshold(pose_result, side, KNEE_PARTS, config.angle_min_conf)
    knee_angle = None
    if knee_angle_valid:
        knee_angle = _calculate_knee_angle(pose_result, side)
    knee_angle_reliable = _parts_pass_threshold(
        pose_result,
        side,
        KNEE_PARTS,
        config.reliable_min_conf,
    )

    torso_lean_confidence = _mean_confidence_for_parts(pose_result, side, TORSO_PARTS)
    torso_lean_valid = _parts_pass_threshold(
        pose_result,
        side,
        TORSO_PARTS,
        config.body_line_min_conf,
    )
    torso_lean_angle = None
    if torso_lean_valid:
        shoulder = pose_result.get_keypoint(f"{side}_shoulder")
        hip = pose_result.get_keypoint(f"{side}_hip")
        if shoulder is not None and hip is not None:
            torso_lean_angle = calculate_torso_lean_angle(shoulder, hip)
    torso_lean_reliable = _parts_pass_threshold(
        pose_result,
        side,
        TORSO_PARTS,
        config.reliable_min_conf,
    )

    critical_confidence = _mean_confidence_for_parts(pose_result, side, CRITICAL_PARTS)
    critical_keypoints_valid = _parts_pass_threshold(
        pose_result,
        side,
        CRITICAL_PARTS,
        config.reliable_min_conf,
    )
    return SquatSideFeatures(
        side=side,
        knee_angle=knee_angle,
        knee_angle_valid=knee_angle_valid,
        knee_angle_reliable=knee_angle_reliable,
        knee_confidence=knee_confidence,
        critical_confidence=critical_confidence,
        critical_keypoints_valid=critical_keypoints_valid,
        torso_lean_angle=torso_lean_angle,
        torso_lean_valid=torso_lean_valid,
        torso_lean_reliable=torso_lean_reliable,
        torso_lean_confidence=torso_lean_confidence,
    )


def _select_stable_side(
    features: dict[str, SquatSideFeatures],
    previous_selected_side: str | None,
    side_switch_margin: float,
) -> str | None:
    candidates = [features[side] for side in SIDES if side in features]
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (item.knee_angle_valid, item.knee_confidence))
    if best.knee_confidence <= 0.0:
        return None
    if previous_selected_side in features:
        previous = features[previous_selected_side]
        if previous.knee_confidence > 0.0:
            confidence_gap = best.knee_confidence - previous.knee_confidence
            if confidence_gap < side_switch_margin:
                return previous.side
    return best.side


def _best_knee_features(
    features: dict[str, SquatSideFeatures],
    selected_side: str | None,
) -> SquatSideFeatures | None:
    if selected_side in features and features[selected_side].knee_angle_valid:
        return features[selected_side]
    valid = [item for item in features.values() if item.knee_angle_valid]
    if not valid:
        return None
    return max(valid, key=lambda item: item.knee_confidence)


def _warnings(
    phase: str,
    knee_angle_valid: bool,
    frame_reliable: bool,
    torso_lean_angle: float | None,
    torso_lean_valid: bool,
    angle_diff: float | None,
    config: SquatAnalysisConfig,
) -> list[str]:
    warnings: list[str] = []
    if not frame_reliable:
        warnings.append("low_confidence")
    if knee_angle_valid and not frame_reliable:
        warnings.append("angle_tentative")
    if not knee_angle_valid:
        warnings.append("knee_angle_unavailable")
    if not torso_lean_valid:
        warnings.append("torso_lean_unavailable")
    if torso_lean_valid and torso_lean_angle is not None and torso_lean_angle > config.torso_lean_warn_angle:
        warnings.append("excessive_forward_lean")
    if angle_diff is not None and angle_diff > config.knee_symmetry_warn_diff:
        warnings.append("possible_asymmetry")
    if phase == "unknown":
        warnings.append("unknown_phase")
    return _deduplicate(warnings)


def _calculate_knee_angle(pose_result: PoseResult, side: str) -> float:
    hip = pose_result.get_keypoint(f"{side}_hip")
    knee = pose_result.get_keypoint(f"{side}_knee")
    ankle = pose_result.get_keypoint(f"{side}_ankle")
    return calculate_angle(hip, knee, ankle)


def _parts_pass_threshold(
    pose_result: PoseResult,
    side: str,
    parts: tuple[str, ...],
    threshold: float,
) -> bool:
    for part in parts:
        keypoint = pose_result.get_keypoint(f"{side}_{part}")
        if not _valid_keypoint(keypoint):
            return False
        if keypoint is None or keypoint.confidence is None:
            return False
        if float(keypoint.confidence) < threshold:
            return False
    return True


def _mean_confidence_for_parts(
    pose_result: PoseResult,
    side: str,
    parts: tuple[str, ...],
) -> float:
    confidences: list[float] = []
    for part in parts:
        keypoint = pose_result.get_keypoint(f"{side}_{part}")
        if not _valid_keypoint(keypoint) or keypoint is None or keypoint.confidence is None:
            confidences.append(0.0)
        else:
            confidences.append(float(keypoint.confidence))
    return sum(confidences) / len(confidences) if confidences else 0.0


def _valid_keypoint(keypoint: Keypoint | None) -> bool:
    if keypoint is None:
        return False
    return math.isfinite(float(keypoint.x)) and math.isfinite(float(keypoint.y))


def _angle_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(left - right)


def _angle_quality(knee_angle_valid: bool, frame_reliable: bool) -> str:
    if not knee_angle_valid:
        return "unavailable"
    if frame_reliable:
        return "reliable"
    return "tentative"


def _float_value(mapping: dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config value {key} must be numeric") from exc


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
