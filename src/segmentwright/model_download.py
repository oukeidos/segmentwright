from __future__ import annotations

import fcntl
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import AppConfig

FIRERED_VAD_REPO_ID = "FireRedTeam/FireRedVAD"
FIRERED_VAD_REPO_PREFIX = "VAD"
FIRERED_VAD_REQUIRED_FILES = ("model.pth.tar", "cmvn.ark")


class ModelDownloadError(RuntimeError):
    """Raised when model preparation cannot complete safely."""


@dataclass(frozen=True)
class FireRedVadModelStatus:
    model_dir: Path
    required_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    empty_files: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_files and not self.empty_files


def ensure_firered_vad_model(config: AppConfig) -> Path:
    model_dir = config.firered_vad_model_dir.expanduser()
    status = inspect_firered_vad_model_dir(model_dir)
    if status.complete:
        return model_dir
    if model_dir.exists():
        raise_incomplete_model_dir(status)
    if not config.firered_vad_auto_download:
        raise FileNotFoundError(
            "FireRedVAD model directory not found. Download model weights separately "
            f"or enable auto-download: {model_dir}"
        )
    return download_firered_vad_model(model_dir)


def validate_firered_vad_model_dir(model_dir: Path) -> None:
    status = inspect_firered_vad_model_dir(model_dir.expanduser())
    if not status.complete:
        raise_incomplete_model_dir(status)


def inspect_firered_vad_model_dir(model_dir: Path) -> FireRedVadModelStatus:
    required = tuple(FIRERED_VAD_REQUIRED_FILES)
    if not model_dir.exists():
        return FireRedVadModelStatus(
            model_dir=model_dir,
            required_files=required,
            missing_files=required,
            empty_files=(),
        )
    missing: list[str] = []
    empty: list[str] = []
    for file_name in required:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing.append(file_name)
            continue
        if file_path.stat().st_size <= 0:
            empty.append(file_name)
    return FireRedVadModelStatus(
        model_dir=model_dir,
        required_files=required,
        missing_files=tuple(missing),
        empty_files=tuple(empty),
    )


def download_firered_vad_model(model_dir: Path) -> Path:
    model_dir = model_dir.expanduser()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir.parent / f"{model_dir.name}.download.lock"
    staging_dir = model_dir.parent / f"{model_dir.name}.download-tmp"
    with locked_file(lock_path):
        status = inspect_firered_vad_model_dir(model_dir)
        if status.complete:
            return model_dir
        if model_dir.exists():
            raise_incomplete_model_dir(status)

        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=False)
        try:
            for file_name in FIRERED_VAD_REQUIRED_FILES:
                downloaded_path = hf_download(
                    repo_id=FIRERED_VAD_REPO_ID,
                    filename=f"{FIRERED_VAD_REPO_PREFIX}/{file_name}",
                )
                shutil.copy2(downloaded_path, staging_dir / file_name)
            validate_firered_vad_model_dir(staging_dir)
            write_download_manifest(staging_dir)
            staging_dir.replace(model_dir)
            return model_dir
        except Exception as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            if isinstance(exc, ModelDownloadError):
                raise
            raise ModelDownloadError(f"Failed to download FireRedVAD model files: {exc}") from exc


def hf_download(*, repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModelDownloadError(
            "huggingface_hub is required for FireRedVAD auto-download. "
            "Install product dependencies with `uv sync`."
        ) from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def write_download_manifest(model_dir: Path) -> None:
    payload = {
        "schema": "segmentwright.firered_vad_download_manifest.v1",
        "repo_id": FIRERED_VAD_REPO_ID,
        "repo_prefix": FIRERED_VAD_REPO_PREFIX,
        "files": list(FIRERED_VAD_REQUIRED_FILES),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "tool": "huggingface_hub.hf_hub_download",
    }
    (model_dir / "download_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def raise_incomplete_model_dir(status: FireRedVadModelStatus) -> None:
    details: list[str] = []
    if status.missing_files:
        details.append(f"missing: {', '.join(status.missing_files)}")
    if status.empty_files:
        details.append(f"empty: {', '.join(status.empty_files)}")
    detail_text = "; ".join(details) if details else "unknown issue"
    raise FileNotFoundError(
        f"Incomplete FireRedVAD model directory at {status.model_dir} ({detail_text}). "
        "Remove the directory and retry auto-download, or pass a complete model directory."
    )


@contextmanager
def locked_file(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
