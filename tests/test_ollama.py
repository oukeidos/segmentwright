from dataclasses import replace
from pathlib import Path

import pytest

from segmentwright.config import AppConfig
from segmentwright.ollama import (
    ManagedOllamaServer,
    OllamaRuntimeConfig,
    is_local_ollama_url,
    model_available,
    model_names_from_tags,
)
from segmentwright.pipeline import build_ollama_runtime_config


def test_ollama_model_names_from_tags_deduplicates_name_and_model() -> None:
    tags = {
        "models": [
            {"name": "gemma3:12b-it-qat", "model": "gemma3:12b-it-qat"},
            {"name": "gemma3:4b-it-qat"},
            {"model": "other:latest"},
            "invalid",
        ]
    }

    assert model_names_from_tags(tags) == [
        "gemma3:12b-it-qat",
        "gemma3:4b-it-qat",
        "other:latest",
    ]


def test_ollama_model_available_accepts_exact_or_latest_for_untagged_name() -> None:
    assert model_available("gemma3:12b-it-qat", ["gemma3:12b-it-qat"])
    assert model_available("gemma3", ["gemma3:latest"])
    assert not model_available("gemma3:12b-it-qat", ["gemma3:4b-it-qat"])


def test_is_local_ollama_url() -> None:
    assert is_local_ollama_url("http://localhost:11434")
    assert is_local_ollama_url("http://127.0.0.1:11434")
    assert not is_local_ollama_url("http://192.168.0.2:11434")


def test_build_ollama_runtime_config_disabled_for_direct() -> None:
    config = app_config()

    runtime = build_ollama_runtime_config(config)

    assert runtime.enabled is False


def test_build_ollama_runtime_config_enabled_for_resegmentation() -> None:
    config = replace(
        app_config(),
        segmentation="resegmentation",
        ollama_auto_pull=True,
        ollama_start_timeout_sec=5.0,
    )

    runtime = build_ollama_runtime_config(config)

    assert runtime.enabled is True
    assert runtime.model == "gemma4:12b"
    assert runtime.auto_pull is True
    assert runtime.start_timeout_sec == 5.0


def test_managed_ollama_unloads_existing_server_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_unload(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("segmentwright.ollama.unload_ollama_model", fake_unload)
    manager = ManagedOllamaServer(
        config=OllamaRuntimeConfig(
            enabled=True,
            base_url="http://localhost:11434",
            model="gemma4:12b",
        ),
        log_dir=tmp_path,
    )
    manager.status.server_available = True
    manager.status.model_available = True
    manager.status.started_by_segmentwright = False

    manager.request_model_unload()

    assert len(calls) == 1
    assert manager.status.model_unloaded is True


def test_managed_ollama_unloads_owned_server(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_unload(*args, **kwargs):
        calls.append((args, kwargs))
        return {"done": True}

    monkeypatch.setattr("segmentwright.ollama.unload_ollama_model", fake_unload)
    manager = ManagedOllamaServer(
        config=OllamaRuntimeConfig(
            enabled=True,
            base_url="http://localhost:11434",
            model="gemma4:12b",
        ),
        log_dir=tmp_path,
    )
    manager.status.server_available = True
    manager.status.model_available = True
    manager.status.started_by_segmentwright = True

    manager.request_model_unload()

    assert len(calls) == 1
    assert manager.status.model_unloaded is True


def test_managed_ollama_cleans_up_owned_server_when_enter_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    process = FakeProcess()

    monkeypatch.setattr("segmentwright.ollama.fetch_ollama_tags", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "segmentwright.ollama.wait_for_ollama",
        lambda *_args, **_kwargs: {"models": []},
    )

    def fake_start_server(self, _log_path):
        self.process = process
        self.status.started_by_segmentwright = True

    monkeypatch.setattr(ManagedOllamaServer, "_start_server", fake_start_server)
    manager = ManagedOllamaServer(
        config=OllamaRuntimeConfig(
            enabled=True,
            base_url="http://localhost:11434",
            model="gemma4:12b",
        ),
        log_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="Ollama model 'gemma4:12b' is not installed"):
        manager.__enter__()

    assert process.terminated is True
    assert process.killed is False


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, *, timeout: float) -> None:
        del timeout


def app_config() -> AppConfig:
    return AppConfig(
        input_path=Path("input.wav"),
        output_path=Path("output.srt"),
        artifacts_dir=Path("artifacts"),
        env_file=Path(".env"),
    )
