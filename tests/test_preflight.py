from __future__ import annotations

from pathlib import Path

import pytest

import segmentwright.preflight as preflight
from segmentwright.config import AppConfig


def test_preflight_passes_when_runtime_requirements_are_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(preflight, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(preflight, "module_available", lambda module_name: True)
    monkeypatch.setattr(
        preflight,
        "ensure_firered_vad_model",
        lambda config: config.firered_vad_model_dir,
    )
    monkeypatch.setattr(preflight, "cuda_available", lambda: True)

    report = preflight.run_preflight_checks(config)

    assert report.ffmpeg_ok
    assert report.gemini_key_ok
    assert report.firered_model_dir_ok
    assert report.python_modules_ok
    assert report.cuda_ok


def test_preflight_fails_before_paid_gemini_when_ml_dependencies_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(preflight, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(preflight, "module_available", lambda module_name: module_name == "torch")
    monkeypatch.setattr(
        preflight,
        "ensure_firered_vad_model",
        lambda config: config.firered_vad_model_dir,
    )
    monkeypatch.setattr(preflight, "cuda_available", lambda: True)

    with pytest.raises(RuntimeError, match="before paid Gemini transcription"):
        preflight.run_preflight_checks(config)


def test_preflight_fails_when_firered_model_dir_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, create_model_dir=False)
    monkeypatch.setattr(preflight, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(preflight, "module_available", lambda module_name: True)
    monkeypatch.setattr(
        preflight,
        "ensure_firered_vad_model",
        lambda config: (_ for _ in ()).throw(
            FileNotFoundError("FireRedVAD model directory not found")
        ),
    )
    monkeypatch.setattr(preflight, "cuda_available", lambda: True)

    with pytest.raises(FileNotFoundError, match="FireRedVAD model directory not found"):
        preflight.run_preflight_checks(config)


def test_preflight_rejects_requested_cuda_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, device="cuda")
    monkeypatch.setattr(preflight, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(preflight, "module_available", lambda module_name: True)
    monkeypatch.setattr(
        preflight,
        "ensure_firered_vad_model",
        lambda config: config.firered_vad_model_dir,
    )
    monkeypatch.setattr(preflight, "cuda_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        preflight.run_preflight_checks(config)


def make_config(
    tmp_path: Path,
    *,
    create_model_dir: bool = True,
    device: str = "auto",
) -> AppConfig:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=test-key\n", encoding="utf-8")
    model_dir = tmp_path / "FireRedVAD" / "VAD"
    if create_model_dir:
        model_dir.mkdir(parents=True)
    return AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=env_file,
        firered_vad_model_dir=model_dir,
        device=device,
    )
