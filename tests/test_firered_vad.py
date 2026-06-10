from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from segmentwright.vad.firered import FireRedVadBackend


def test_firered_vad_backend_parses_runtime_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "FireRedVAD" / "VAD"
    model_dir.mkdir(parents=True)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")
    fake_module = types.ModuleType("fireredvad")
    fake_module.FireRedVad = FakeFireRedVad
    fake_module.FireRedVadConfig = FakeFireRedVadConfig
    monkeypatch.setitem(sys.modules, "fireredvad", fake_module)

    result = FireRedVadBackend(model_dir=model_dir, device="cpu").detect(audio_path)

    assert result.duration_sec == 2.0
    assert [(segment.start_sec, segment.end_sec) for segment in result.speech_segments] == [
        (0.1, 0.5)
    ]
    assert result.raw["engine"] == "fireredvad"
    assert result.raw["device"] == "cpu"


def test_firered_vad_backend_fails_clearly_when_model_dir_is_missing(
    tmp_path: Path,
) -> None:
    backend = FireRedVadBackend(model_dir=tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="FireRedVAD model directory not found"):
        backend.detect(tmp_path / "audio.wav")


class FakeFireRedVadConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeFireRedVad:
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        config: FakeFireRedVadConfig,
    ) -> FakeFireRedVad:
        assert model_dir.endswith("FireRedVAD/VAD")
        assert config.kwargs["speech_threshold"] == 0.4
        return cls()

    def detect(self, audio_path: str) -> tuple[dict, object]:
        assert audio_path.endswith("audio.wav")
        return (
            {
                "dur": 2.0,
                "timestamps": [(0.1, 0.5), (1.0, 0.9)],
                "wav_path": audio_path,
            },
            object(),
        )
