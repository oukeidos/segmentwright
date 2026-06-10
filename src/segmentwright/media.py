from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def require_ffmpeg() -> None:
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise RuntimeError(f"{command} is required but was not found on PATH")


def normalize_to_wav(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
    )


def split_wav(input_wav: Path, split_times: list[float], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("audio_*.wav"):
        stale.unlink()

    if not split_times:
        output = output_dir / "audio_000.wav"
        run_command(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(input_wav),
                "-c",
                "copy",
                str(output),
            ]
        )
        return [output]

    formatted = ",".join(f"{time_sec:.6f}" for time_sec in split_times)
    pattern = output_dir / "audio_%03d.wav"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(input_wav),
            "-f",
            "segment",
            "-segment_times",
            formatted,
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            "-segment_start_number",
            "0",
            str(pattern),
        ]
    )
    return sorted(output_dir.glob("audio_*.wav"))


def extract_wav_segment(
    input_wav: Path,
    *,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> None:
    if end_sec <= start_sec:
        raise ValueError(f"Invalid segment range: {start_sec}..{end_sec}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start_sec:.6f}",
            "-to",
            f"{end_sec:.6f}",
            "-i",
            str(input_wav),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
    )


def probe_duration(path: Path) -> float:
    completed = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    value = completed.stdout.strip()
    if not value or value == "N/A":
        raise RuntimeError(f"Could not probe duration for {path}")
    return float(value)


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(args)}\n{completed.stderr}"
        )
    return completed
