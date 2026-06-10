from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

NOISY_LOGGERS = (
    "NVIDIA",
    "absl",
    "asyncio",
    "datasets",
    "fireredvad",
    "huggingface_hub",
    "librosa",
    "matplotlib",
    "nemo",
    "numba",
    "numexpr",
    "pytorch_lightning",
    "torch",
    "torchaudio",
    "transformers",
    "urllib3",
)


def configure_logging(verbose: bool) -> None:
    """Configure Python logging for concise normal output and detailed verbose output."""

    logging.captureWarnings(True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s" if verbose else "%(levelname)s %(message)s",
        force=True,
    )
    segmentwright_logger = logging.getLogger("segmentwright")
    segmentwright_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    noisy_level = logging.NOTSET if verbose else logging.ERROR
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(noisy_level)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remaining:.0f}s"


def format_count(count: int, singular: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


@dataclass
class StageRecord:
    index: int
    name: str
    status: str
    elapsed_sec: float
    details: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "status": self.status,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "details": self.details,
        }


class ConsoleReporter:
    def __init__(
        self,
        *,
        verbose: bool,
        artifacts_dir: Path,
        stream: TextIO | None = None,
    ) -> None:
        self.verbose = verbose
        self.artifacts_dir = artifacts_dir
        self.stream = stream if stream is not None else sys.stderr
        self.started_at_monotonic = time.monotonic()
        self.current_stage: str | None = None
        self._current_stage_start: float | None = None
        self._current_stage_index = 0
        self._next_stage_index = 1
        self.stage_records: list[StageRecord] = []
        self.warning_count = 0
        self._last_progress: dict[str, tuple[int, int]] = {}

    @property
    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started_at_monotonic

    def run_start(
        self,
        *,
        input_path: Path,
        output_path: Path,
        mode: str,
        preset: str,
        artifacts_dir: Path | None,
    ) -> None:
        self._write("Segmentwright")
        self._write(f"Mode: {mode}")
        self._write(f"Preset: {preset}")
        self._write(f"Input: {input_path}")
        self._write(f"Output: {output_path}")
        if artifacts_dir is not None:
            self._write(f"Artifacts: {artifacts_dir}")

    def stage_start(self, name: str, details: str | None = None) -> None:
        if self.current_stage is not None:
            self.stage_end(self.current_stage, status="done")
        index = self._next_stage_index
        self._next_stage_index += 1
        self._current_stage_index = index
        self.current_stage = name
        self._current_stage_start = time.monotonic()
        suffix = f" ({details})" if details else ""
        self._write(f"[{index}] {name}...{suffix}")

    def stage_progress(
        self,
        name: str,
        current: int,
        total: int,
        details: str | None = None,
        unit: str = "chunk",
    ) -> None:
        if total <= 1 and not details:
            return
        key = (name, current, total)
        if self._last_progress.get(name) == key[1:]:
            return
        self._last_progress[name] = key[1:]
        suffix = f": {details}" if details else ""
        self._write(f"  {unit} {current}/{total}{suffix}")

    def stage_end(
        self,
        name: str,
        *,
        status: str = "ok",
        details: str | None = None,
    ) -> None:
        if self.current_stage != name:
            return
        elapsed = (
            time.monotonic() - self._current_stage_start
            if self._current_stage_start is not None
            else 0.0
        )
        self.stage_records.append(
            StageRecord(
                index=self._current_stage_index,
                name=name,
                status=status,
                elapsed_sec=elapsed,
                details=details,
            )
        )
        detail_suffix = f" ({details})" if details else ""
        self._write(f"[{self._current_stage_index}] {name}: {status}{detail_suffix}")
        self.current_stage = None
        self._current_stage_start = None

    def warning(self, message: str) -> None:
        self.warning_count += 1
        self._write(f"Warning: {message}")

    def run_success(
        self,
        *,
        output_path: Path,
        artifacts_dir: Path | None,
        cue_count: int,
    ) -> None:
        self._write(
            "Success: "
            f"wrote {format_count(cue_count, 'cue')} to {output_path} "
            f"in {format_duration(self.elapsed_sec)}"
        )
        if artifacts_dir is not None:
            self._write(f"Artifacts: {artifacts_dir}")

    def run_failure(
        self,
        *,
        error: BaseException,
        artifacts_dir: Path | None,
        verbose: bool,
    ) -> None:
        if self.current_stage is not None:
            self.stage_end(self.current_stage, status="failed", details=str(error))
        self._write(f"Failed: {type(error).__name__}: {error}")
        if artifacts_dir is not None:
            self._write(f"Details: {artifacts_dir / 'errors.json'}")
        else:
            self._write("Debug artifacts were temporary and will be removed.")
            self._write("Rerun with --artifacts PATH to preserve diagnostic files.")
        if verbose:
            traceback.print_exception(error, file=self.stream)
        else:
            self._write("Run again with --verbose for full logs.")

    def _write(self, message: str) -> None:
        print(message, file=self.stream, flush=True)


@contextlib.contextmanager
def capture_external_output(*, enabled: bool, path: Path):
    """Redirect process-level stdout/stderr to an artifact log while noisy code runs."""

    if not enabled:
        yield
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with path.open("a", encoding="utf-8") as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            yield
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
