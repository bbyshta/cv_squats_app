"""Confidence-aware push-up technique rules for one image."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from src.analysis.angles import calculate_angle
from src.inference.base import Keypoint, PoseResult


MISSING_ASSESSMENT = "Недостаточно данных для уверенной оценки техники."
BOTTOM_ASSESSMENT = "Нижняя фаза отжимания определена, глубина достаточная по заданному правилу."
DEPTH_WARNING_ASSESSMENT = "Возможна недостаточная глубина отжимания."
TRANSITION_ASSESSMENT = "Положение похоже на переходную фазу. Оценка является ориентировочной."
TOP_ASSESSMENT = "Верхняя фаза отжимания определена по заданному правилу."


@dataclass(frozen=True)
class SideAnalysis:
    """Intermediate analysis result for one body side."""

    side: str
    elbow_angle: float | None
    body_line_angle: float | None
    valid_keypoints: int
    mean_confidence: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PushupImageAnalysis:
    """Serializable push-up analysis result for one image."""

    selected_side: str | None
    angles: dict[str, float | None]
    pushup_phase: str
    technique_assessment: str
    warnings: list[str]
    confidence_threshold: float
    side_scores: dict[str, dict[str, float | int | str | None]]

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_pushup_image(pose_result: PoseResult, rules_config: dict) -> PushupImageAnalysis:
    """Analyze one image pose result with configurable push-up rules."""
    confidence_threshold = _confidence_threshold(rules_config)
    top_min_angle = _float_rule(
        rules_config, ("phase_detection", "top_elbow_angle_min_degrees"), default=155.0
    )
    bottom_max_angle = _float_rule(
        rules_config, ("phase_detection", "bottom_elbow_angle_max_degrees"), default=95.0
    )
    max_body_deviation = _float_rule(
        rules_config,
        ("technique", "max_body_line_angle_deviation_degrees"),
        default=12.0,
    )

    left = _analyze_side(pose_result, "left", confidence_threshold)
    right = _analyze_side(pose_result, "right", confidence_threshold)
    selected = _select_side(left, right)

    warnings = [*left.warnings, *right.warnings]
    for side_analysis in (left, right):
        if side_analysis.body_line_angle is None:
            continue
        deviation = abs(180.0 - side_analysis.body_line_angle)
        if deviation > max_body_deviation:
            warnings.append(
                f"{_side_label(side_analysis.side)} линия тела отклоняется от прямой "
                f"на {deviation:.1f}°: возможны провисание или поднятие таза."
            )

    selected_elbow = selected.elbow_angle if selected is not None else None
    phase = _detect_phase(selected_elbow, top_min_angle, bottom_max_angle)
    assessment = _assess(selected, phase, bottom_max_angle)

    return PushupImageAnalysis(
        selected_side=selected.side if selected is not None else None,
        angles={
            "left_elbow": left.elbow_angle,
            "right_elbow": right.elbow_angle,
            "left_body_line": left.body_line_angle,
            "right_body_line": right.body_line_angle,
            "selected_elbow": selected_elbow,
            "selected_body_line": selected.body_line_angle if selected is not None else None,
        },
        pushup_phase=phase,
        technique_assessment=assessment,
        warnings=warnings,
        confidence_threshold=confidence_threshold,
        side_scores={
            "left": _side_score(left),
            "right": _side_score(right),
        },
    )


def _analyze_side(
    pose_result: PoseResult,
    side: str,
    confidence_threshold: float,
) -> SideAnalysis:
    shoulder = _valid_keypoint(pose_result, f"{side}_shoulder", confidence_threshold)
    elbow = _valid_keypoint(pose_result, f"{side}_elbow", confidence_threshold)
    wrist = _valid_keypoint(pose_result, f"{side}_wrist", confidence_threshold)
    hip = _valid_keypoint(pose_result, f"{side}_hip", confidence_threshold)
    ankle = _valid_keypoint(pose_result, f"{side}_ankle", confidence_threshold)
    knee = _valid_keypoint(pose_result, f"{side}_knee", confidence_threshold)

    keypoints = {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "hip": hip,
        "ankle": ankle,
        "knee": knee,
    }
    valid_points = [point for point in keypoints.values() if point is not None]
    warnings: list[str] = []

    elbow_angle = None
    if shoulder is not None and elbow is not None and wrist is not None:
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
    else:
        warnings.append(f"{_side_label(side, masculine=True)} локоть не рассчитан: недостаточно надежных keypoints.")

    body_line_angle = None
    lower_point = ankle if ankle is not None else knee
    if shoulder is not None and hip is not None and lower_point is not None:
        body_line_angle = calculate_angle(shoulder, hip, lower_point)
        if ankle is None and knee is not None:
            warnings.append(f"{_side_label(side)} линия тела рассчитана по колену вместо голеностопа.")
    else:
        warnings.append(f"{_side_label(side)} линия тела не рассчитана: недостаточно надежных keypoints.")

    if len(valid_points) < len(keypoints):
        missing = [name for name, point in keypoints.items() if point is None]
        warnings.append(
            f"{_side_label(side)} сторона: слабые или отсутствующие keypoints: {', '.join(missing)}."
        )

    return SideAnalysis(
        side=side,
        elbow_angle=elbow_angle,
        body_line_angle=body_line_angle,
        valid_keypoints=len(valid_points),
        mean_confidence=_mean_confidence(valid_points),
        warnings=warnings,
    )


def _valid_keypoint(
    pose_result: PoseResult,
    name: str,
    confidence_threshold: float,
) -> Keypoint | None:
    keypoint = pose_result.get_keypoint(name)
    if keypoint is None:
        return None
    if keypoint.x is None or keypoint.y is None:
        return None
    if not math.isfinite(float(keypoint.x)) or not math.isfinite(float(keypoint.y)):
        return None
    if keypoint.confidence is not None and keypoint.confidence < confidence_threshold:
        return None
    return keypoint


def _select_side(left: SideAnalysis, right: SideAnalysis) -> SideAnalysis | None:
    elbow_candidates = [side for side in (left, right) if side.elbow_angle is not None]
    if elbow_candidates:
        return max(elbow_candidates, key=lambda side: (side.valid_keypoints, side.mean_confidence))

    body_candidates = [side for side in (left, right) if side.body_line_angle is not None]
    if body_candidates:
        return max(body_candidates, key=lambda side: (side.valid_keypoints, side.mean_confidence))
    return None


def _detect_phase(
    selected_elbow_angle: float | None,
    top_min_angle: float,
    bottom_max_angle: float,
) -> str:
    if selected_elbow_angle is None:
        return "unknown"
    if selected_elbow_angle >= top_min_angle:
        return "top"
    if selected_elbow_angle <= bottom_max_angle:
        return "bottom"
    return "transition"


def _assess(
    selected: SideAnalysis | None,
    phase: str,
    bottom_max_angle: float,
) -> str:
    if selected is None or selected.elbow_angle is None:
        return MISSING_ASSESSMENT
    if phase == "bottom" and selected.elbow_angle <= bottom_max_angle:
        return BOTTOM_ASSESSMENT
    if phase != "top" and selected.elbow_angle > bottom_max_angle:
        return DEPTH_WARNING_ASSESSMENT
    if phase == "top":
        return TOP_ASSESSMENT
    return TRANSITION_ASSESSMENT


def _confidence_threshold(rules_config: dict) -> float:
    return _float_rule(
        rules_config,
        ("confidence", "min_critical_keypoint_confidence"),
        default=0.5,
    )


def _float_rule(rules_config: dict, path: tuple[str, str], default: float) -> float:
    section = rules_config.get(path[0], {})
    if not isinstance(section, dict):
        return default
    value = section.get(path[1], default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config value {'.'.join(path)} must be numeric") from exc


def _mean_confidence(keypoints: list[Keypoint]) -> float:
    confidences = [point.confidence for point in keypoints if point.confidence is not None]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def _side_score(side: SideAnalysis) -> dict[str, float | int | str | None]:
    return {
        "side": side.side,
        "valid_keypoints": side.valid_keypoints,
        "mean_confidence": side.mean_confidence,
        "elbow_angle": side.elbow_angle,
        "body_line_angle": side.body_line_angle,
    }


def _side_label(side: str, masculine: bool = False) -> str:
    if side == "left":
        return "Левый" if masculine else "Левая"
    if side == "right":
        return "Правый" if masculine else "Правая"
    return side
