"""Best-effort video export helpers for browser playback."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class BrowserVideoExportResult:
    """Result of browser-compatible video conversion."""

    output_path: Path | None
    browser_compatible: bool
    warning: str | None = None


def export_browser_compatible_mp4(
    input_path: str | Path,
    output_path: str | Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> BrowserVideoExportResult:
    """Convert an MP4 to H.264/yuv420p for browser playback when ffmpeg is available."""
    source_path = Path(input_path)
    if not source_path.is_file():
        return BrowserVideoExportResult(
            output_path=None,
            browser_compatible=False,
            warning="annotated_video_missing",
        )
    target_path = Path(output_path) if output_path is not None else source_path.with_name(
        f"{source_path.stem}_h264.mp4"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    try:
        runner(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return BrowserVideoExportResult(
            output_path=None,
            browser_compatible=False,
            warning="ffmpeg_not_available",
        )
    except subprocess.CalledProcessError as exc:
        stderr = _decode_stderr(exc)
        message = f"ffmpeg_conversion_failed: {stderr}" if stderr else "ffmpeg_conversion_failed"
        return BrowserVideoExportResult(
            output_path=None,
            browser_compatible=False,
            warning=message,
        )
    if not target_path.is_file():
        return BrowserVideoExportResult(
            output_path=None,
            browser_compatible=False,
            warning="ffmpeg_output_missing",
        )
    return BrowserVideoExportResult(
        output_path=target_path,
        browser_compatible=True,
        warning=None,
    )


def _decode_stderr(exc: subprocess.CalledProcessError) -> str:
    stderr = exc.stderr
    if not stderr:
        return ""
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", errors="replace").strip()
    return str(stderr).strip()
