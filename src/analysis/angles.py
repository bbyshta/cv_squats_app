"""Geometry helpers for pose analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


Point2D = tuple[float, float]


def calculate_angle(a: Any, b: Any, c: Any) -> float:
    """Return angle ABC in degrees using 2D image coordinates.

    Point ``b`` is the joint center. Inputs may be ``(x, y)`` sequences,
    mappings with ``x``/``y`` keys, or objects exposing ``x``/``y`` attributes.
    """
    point_a = _to_point(a, "a")
    point_b = _to_point(b, "b")
    point_c = _to_point(c, "c")

    ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])
    ba_length = math.hypot(*ba)
    bc_length = math.hypot(*bc)
    if ba_length == 0.0 or bc_length == 0.0:
        raise ValueError("Cannot calculate angle with a zero-length vector")

    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (ba_length * bc_length)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _to_point(point: Any, label: str) -> Point2D:
    if point is None:
        raise ValueError(f"Point {label} is missing")

    x: Any
    y: Any
    if isinstance(point, Mapping):
        x = point.get("x")
        y = point.get("y")
    elif _is_sequence_point(point):
        x = point[0]
        y = point[1]
    else:
        x = getattr(point, "x", None)
        y = getattr(point, "y", None)

    if x is None or y is None:
        raise ValueError(f"Point {label} must contain x and y coordinates")

    try:
        result = (float(x), float(y))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Point {label} coordinates must be numeric") from exc

    if not math.isfinite(result[0]) or not math.isfinite(result[1]):
        raise ValueError(f"Point {label} coordinates must be finite")
    return result


def _is_sequence_point(point: Any) -> bool:
    return (
        isinstance(point, Sequence)
        and not isinstance(point, (str, bytes, bytearray))
        and len(point) >= 2
    )
