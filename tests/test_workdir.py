from __future__ import annotations

import os
from pathlib import Path

from segmentwright.workdir import (
    TEMP_WORKDIR_MARKER,
    TEMP_WORKDIR_PREFIX,
    cleanup_stale_temp_workdirs,
    temporary_artifacts_dir,
    write_temp_workdir_marker,
)


def test_temporary_artifacts_dir_creates_marker_and_removes_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")

    with temporary_artifacts_dir(input_path=input_path) as workdir:
        captured = workdir
        assert workdir.exists()
        assert workdir.name.startswith(TEMP_WORKDIR_PREFIX)
        assert (workdir / TEMP_WORKDIR_MARKER).is_file()

    assert captured.exists() is False


def test_cleanup_stale_temp_workdirs_removes_only_marked_old_dirs(tmp_path: Path) -> None:
    old_dir = tmp_path / f"{TEMP_WORKDIR_PREFIX}old"
    recent_dir = tmp_path / f"{TEMP_WORKDIR_PREFIX}recent"
    unmarked_dir = tmp_path / f"{TEMP_WORKDIR_PREFIX}unmarked"
    unrelated_dir = tmp_path / "other-old"
    for path in (old_dir, recent_dir, unmarked_dir, unrelated_dir):
        path.mkdir()

    write_temp_workdir_marker(old_dir)
    write_temp_workdir_marker(recent_dir)
    old_time = 100.0
    now = old_time + 25 * 60 * 60
    recent_time = now
    os.utime(old_dir / TEMP_WORKDIR_MARKER, (old_time, old_time))
    os.utime(recent_dir / TEMP_WORKDIR_MARKER, (recent_time, recent_time))

    removed = cleanup_stale_temp_workdirs(
        temp_root=tmp_path,
        max_age_sec=24 * 60 * 60,
        now=now,
    )

    assert removed == [old_dir]
    assert old_dir.exists() is False
    assert recent_dir.exists()
    assert unmarked_dir.exists()
    assert unrelated_dir.exists()
