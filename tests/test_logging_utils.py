from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from segmentwright.logging_utils import (
    ConsoleReporter,
    capture_external_output,
    configure_logging,
)


def test_configure_logging_normal_mode_suppresses_noisy_loggers() -> None:
    configure_logging(verbose=False)

    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger("segmentwright").level == logging.INFO
    assert logging.getLogger("nemo").level == logging.ERROR
    assert logging.getLogger("torch").level == logging.ERROR


def test_configure_logging_verbose_mode_enables_debug_logs() -> None:
    configure_logging(verbose=True)

    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("segmentwright").level == logging.DEBUG
    assert logging.getLogger("nemo").level == logging.NOTSET


def test_console_reporter_writes_run_stage_warning_and_success(tmp_path: Path) -> None:
    stream = io.StringIO()
    reporter = ConsoleReporter(verbose=False, artifacts_dir=tmp_path, stream=stream)

    reporter.run_start(
        input_path=tmp_path / "in.wav",
        output_path=tmp_path / "out.srt",
        mode="external transcript",
        preset="lite-experimental",
        artifacts_dir=tmp_path / "artifacts",
    )
    reporter.stage_start("Test stage", details="2 chunks")
    reporter.stage_progress("Test stage", 1, 2, details="id=chunk_0001")
    reporter.warning("sample warning")
    reporter.stage_end("Test stage", details="2 chunks")
    reporter.run_success(
        output_path=tmp_path / "out.srt",
        artifacts_dir=tmp_path / "artifacts",
        cue_count=3,
    )

    output = stream.getvalue()
    assert "Segmentwright" in output
    assert "Mode: external transcript" in output
    assert "[1] Test stage..." in output
    assert "chunk 1/2: id=chunk_0001" in output
    assert "Warning: sample warning" in output
    assert "Success: wrote 3 cues" in output
    assert reporter.warning_count == 1
    assert len(reporter.stage_records) == 1


def test_capture_external_output_redirects_stdout_and_stderr(tmp_path: Path) -> None:
    log_path = tmp_path / "external.log"

    with capture_external_output(enabled=True, path=log_path):
        print("stdout line")
        os.write(2, b"stderr line\n")

    captured = log_path.read_text(encoding="utf-8")
    assert "stdout line" in captured
    assert "stderr line" in captured
