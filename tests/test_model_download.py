from __future__ import annotations

from pathlib import Path

import pytest

import segmentwright.model_download as model_download
from segmentwright.config import AppConfig
from segmentwright.model_download import (
    FIRERED_VAD_REQUIRED_FILES,
    ensure_firered_vad_model,
    inspect_firered_vad_model_dir,
    validate_firered_vad_model_dir,
)


def test_existing_complete_firered_model_dir_passes_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = make_complete_model_dir(tmp_path)
    monkeypatch.setattr(
        model_download,
        "hf_download",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    assert ensure_firered_vad_model(make_config(tmp_path, model_dir)) == model_dir


def test_missing_model_dir_downloads_required_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for file_name in FIRERED_VAD_REQUIRED_FILES:
        (source_dir / file_name).write_text(f"{file_name}\n", encoding="utf-8")
    requested: list[str] = []

    def fake_download(*, repo_id: str, filename: str) -> Path:
        del repo_id
        requested.append(filename)
        return source_dir / Path(filename).name

    model_dir = tmp_path / "downloaded" / "VAD"
    monkeypatch.setattr(model_download, "hf_download", fake_download)

    assert ensure_firered_vad_model(make_config(tmp_path, model_dir)) == model_dir
    assert requested == ["VAD/model.pth.tar", "VAD/cmvn.ark"]
    for file_name in FIRERED_VAD_REQUIRED_FILES:
        assert (model_dir / file_name).read_text(encoding="utf-8") == f"{file_name}\n"
    assert (model_dir / "download_manifest.json").exists()


def test_auto_download_disabled_preserves_fail_fast_behavior(tmp_path: Path) -> None:
    model_dir = tmp_path / "missing" / "VAD"

    with pytest.raises(FileNotFoundError, match="auto-download"):
        ensure_firered_vad_model(make_config(tmp_path, model_dir, firered_vad_auto_download=False))


def test_existing_incomplete_model_dir_fails_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "FireRedVAD" / "VAD"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pth.tar").write_text("model\n", encoding="utf-8")
    monkeypatch.setattr(
        model_download,
        "hf_download",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    with pytest.raises(FileNotFoundError, match="Incomplete FireRedVAD model directory"):
        ensure_firered_vad_model(make_config(tmp_path, model_dir))

    assert (model_dir / "model.pth.tar").read_text(encoding="utf-8") == "model\n"
    assert not (model_dir / "cmvn.ark").exists()


def test_download_failure_removes_staging_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "downloaded" / "VAD"

    def fake_download(*, repo_id: str, filename: str) -> Path:
        del repo_id, filename
        raise RuntimeError("network down")

    monkeypatch.setattr(model_download, "hf_download", fake_download)

    with pytest.raises(model_download.ModelDownloadError, match="Failed to download"):
        ensure_firered_vad_model(make_config(tmp_path, model_dir))

    assert not model_dir.exists()
    assert not (model_dir.parent / "VAD.download-tmp").exists()


def test_validate_firered_model_dir_rejects_empty_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "FireRedVAD" / "VAD"
    model_dir.mkdir(parents=True)
    for file_name in FIRERED_VAD_REQUIRED_FILES:
        (model_dir / file_name).write_bytes(b"")

    status = inspect_firered_vad_model_dir(model_dir)

    assert status.empty_files == FIRERED_VAD_REQUIRED_FILES
    with pytest.raises(FileNotFoundError, match="empty"):
        validate_firered_vad_model_dir(model_dir)


def make_complete_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "FireRedVAD" / "VAD"
    model_dir.mkdir(parents=True)
    for file_name in FIRERED_VAD_REQUIRED_FILES:
        (model_dir / file_name).write_text(f"{file_name}\n", encoding="utf-8")
    return model_dir


def make_config(
    tmp_path: Path,
    model_dir: Path,
    *,
    firered_vad_auto_download: bool = True,
) -> AppConfig:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=test-key\n", encoding="utf-8")
    return AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=env_file,
        firered_vad_model_dir=model_dir,
        firered_vad_auto_download=firered_vad_auto_download,
    )
