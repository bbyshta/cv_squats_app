import pytest

from src.analysis.angles import calculate_angle


def test_calculate_angle_straight_line() -> None:
    assert calculate_angle((0, 0), (1, 0), (2, 0)) == pytest.approx(180.0)


def test_calculate_angle_right_angle() -> None:
    assert calculate_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)


def test_calculate_angle_rejects_missing_point() -> None:
    with pytest.raises(ValueError, match="missing"):
        calculate_angle(None, (0, 0), (1, 1))


def test_calculate_angle_rejects_zero_length_vector() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        calculate_angle((0, 0), (0, 0), (1, 1))
