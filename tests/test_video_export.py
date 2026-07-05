from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from src.utils.video_export import export_browser_compatible_mp4


def test_ffmpeg_missing_returns_warning_without_exception(tmp_path: Path) -> None:
    input_path = tmp_path / "annotated_video.mp4"
    input_path.write_bytes(b"video")

    result = export_browser_compatible_mp4(
        input_path,
        tmp_path / "annotated_video_h264.mp4",
        runner=_missing_ffmpeg_runner,
    )

    assert result.browser_compatible is False
    assert result.output_path is None
    assert result.warning == "ffmpeg_not_available"


def test_successful_conversion_reports_browser_compatible_output(tmp_path: Path) -> None:
    input_path = tmp_path / "annotated_video.mp4"
    output_path = tmp_path / "annotated_video_h264.mp4"
    input_path.write_bytes(b"video")

    result = export_browser_compatible_mp4(
        input_path,
        output_path,
        runner=_successful_runner,
    )

    assert result.browser_compatible is True
    assert result.output_path == output_path
    assert output_path.read_bytes() == b"h264"


def _missing_ffmpeg_runner(*args: Any, **kwargs: Any) -> None:
    raise FileNotFoundError("ffmpeg")


def _successful_runner(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    Path(command[-1]).write_bytes(b"h264")
    return subprocess.CompletedProcess(command, 0)
