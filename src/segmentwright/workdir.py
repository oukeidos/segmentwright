from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

TEMP_WORKDIR_PREFIX = "segmentwright-"
TEMP_WORKDIR_MARKER = ".segmentwright-workdir.json"
DEFAULT_STALE_WORKDIR_AGE_SEC = 24 * 60 * 60


@contextmanager
def temporary_artifacts_dir(*, input_path: Path | None = None) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=TEMP_WORKDIR_PREFIX) as temp_name:
        workdir = Path(temp_name)
        write_temp_workdir_marker(workdir, input_path=input_path)
        yield workdir


def cleanup_stale_temp_workdirs(
    *,
    temp_root: Path | None = None,
    max_age_sec: float = DEFAULT_STALE_WORKDIR_AGE_SEC,
    now: float | None = None,
) -> list[Path]:
    root = temp_root or Path(tempfile.gettempdir())
    current_time = time.time() if now is None else now
    removed: list[Path] = []
    if not root.exists():
        return removed
    for candidate in root.glob(f"{TEMP_WORKDIR_PREFIX}*"):
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        marker = candidate / TEMP_WORKDIR_MARKER
        if not marker.is_file():
            continue
        try:
            marker_mtime = marker.stat().st_mtime
        except OSError:
            continue
        if current_time - marker_mtime < max_age_sec:
            continue
        try:
            shutil.rmtree(candidate)
        except OSError:
            continue
        removed.append(candidate)
    return removed


def write_temp_workdir_marker(workdir: Path, *, input_path: Path | None = None) -> None:
    marker = workdir / TEMP_WORKDIR_MARKER
    payload = {
        "schema": "segmentwright.temp_workdir.v1",
        "created_at_unix": time.time(),
        "pid": os.getpid(),
        "input_path_sha256": hash_path_string(input_path) if input_path is not None else None,
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def hash_path_string(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
