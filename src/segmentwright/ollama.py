from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlparse


@dataclass(frozen=True)
class OllamaRuntimeConfig:
    enabled: bool
    base_url: str
    model: str
    auto_pull: bool = False
    start_timeout_sec: float = 30.0
    request_timeout_sec: float = 2.0


@dataclass
class OllamaRuntimeStatus:
    enabled: bool
    base_url: str
    model: str
    server_available: bool = False
    started_by_segmentwright: bool = False
    model_available: bool = False
    auto_pull: bool = False
    pulled_model: bool = False
    model_unloaded: bool = False
    log_path: str | None = None
    models: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ManagedOllamaServer:
    def __init__(
        self,
        *,
        config: OllamaRuntimeConfig,
        log_dir: Path,
    ) -> None:
        self.config = config
        self.log_dir = log_dir
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any | None = None
        self.status = OllamaRuntimeStatus(
            enabled=config.enabled,
            base_url=config.base_url,
            model=config.model,
            auto_pull=config.auto_pull,
        )

    def __enter__(self) -> OllamaRuntimeStatus:
        if not self.config.enabled:
            return self.status
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / "ollama.log"
            self.status.log_path = str(log_path)

            tags = fetch_ollama_tags(
                self.config.base_url,
                timeout_sec=self.config.request_timeout_sec,
            )
            if tags is None:
                self._start_server(log_path)
                tags = wait_for_ollama(
                    self.config.base_url,
                    timeout_sec=self.config.start_timeout_sec,
                    request_timeout_sec=self.config.request_timeout_sec,
                )
            self.status.server_available = True

            models = model_names_from_tags(tags)
            self.status.models = models
            if model_available(self.config.model, models):
                self.status.model_available = True
                return self.status

            if not self.config.auto_pull:
                raise RuntimeError(self._missing_model_message())

            self._pull_model(log_path)
            tags = fetch_ollama_tags(
                self.config.base_url,
                timeout_sec=self.config.request_timeout_sec,
            )
            if tags is None:
                raise RuntimeError("Ollama became unreachable after pulling the model")
            models = model_names_from_tags(tags)
            self.status.models = models
            self.status.pulled_model = True
            if not model_available(self.config.model, models):
                raise RuntimeError(
                    f"Ollama model {self.config.model!r} is still unavailable after pull"
                )
            self.status.model_available = True
            return self.status
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.request_model_unload()
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()

    def _start_server(self, log_path: Path) -> None:
        if not is_local_ollama_url(self.config.base_url):
            raise RuntimeError(
                f"Cannot auto-start Ollama for non-local URL {self.config.base_url!r}; "
                "start a compatible server for that URL or use a localhost Ollama URL."
            )
        executable = shutil.which("ollama")
        if executable is None:
            raise RuntimeError(
                "Ollama executable was not found. Install Ollama or start a compatible "
                "server at the configured URL."
            )
        self.log_handle = log_path.open("ab")
        self.process = subprocess.Popen(
            [executable, "serve"],
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.status.started_by_segmentwright = True

    def _pull_model(self, log_path: Path) -> None:
        executable = shutil.which("ollama")
        if executable is None:
            raise RuntimeError("Ollama executable was not found; cannot pull model automatically")
        with log_path.open("ab") as handle:
            subprocess.run(
                [executable, "pull", self.config.model],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def _missing_model_message(self) -> str:
        base = (
            f"Ollama model {self.config.model!r} is not installed. "
            f"Run `ollama pull {self.config.model}` or pass --ollama-auto-pull."
        )
        if self.status.started_by_segmentwright:
            return (
                f"{base} Segmentwright auto-started Ollama in the current user context; "
                "if your model is installed under a system-service Ollama, start that service "
                "before running Segmentwright, or pull the model in the current user context."
            )
        return base

    def request_model_unload(self) -> None:
        if (
            not self.config.enabled
            or not self.status.server_available
            or not self.status.model_available
            or self.status.model_unloaded
        ):
            return
        try:
            unload_ollama_model(
                self.config.base_url,
                model=self.config.model,
                timeout_sec=self.config.request_timeout_sec,
            )
        except (OSError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self.status.warnings.append(
                f"Could not unload Ollama model {self.config.model!r}: {exc}"
            )
            return
        self.status.model_unloaded = True


def fetch_ollama_tags(base_url: str, *, timeout_sec: float) -> dict[str, Any] | None:
    url = base_url.rstrip("/") + "/api/tags"
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def wait_for_ollama(
    base_url: str,
    *,
    timeout_sec: float,
    request_timeout_sec: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        tags = fetch_ollama_tags(base_url, timeout_sec=request_timeout_sec)
        if tags is not None:
            return tags
        time.sleep(0.25)
    raise TimeoutError(f"Ollama did not become ready within {timeout_sec:.1f}s")


def unload_ollama_model(
    base_url: str,
    *,
    model: str,
    timeout_sec: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/generate"
    body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_sec) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def model_names_from_tags(tags: dict[str, Any]) -> list[str]:
    models = tags.get("models")
    if not isinstance(models, list):
        return []
    names = []
    for model in models:
        if not isinstance(model, dict):
            continue
        for key in ("name", "model"):
            value = model.get(key)
            if isinstance(value, str) and value:
                names.append(value)
    return sorted(set(names))


def model_available(model: str, models: list[str]) -> bool:
    if model in models:
        return True
    if ":" not in model:
        return f"{model}:latest" in models
    return False


def is_local_ollama_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}
