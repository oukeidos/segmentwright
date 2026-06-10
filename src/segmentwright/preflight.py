from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .config import AppConfig
from .env import load_gemini_api_key
from .media import require_ffmpeg
from .model_download import ensure_firered_vad_model


@dataclass(frozen=True)
class PreflightReport:
    ffmpeg_ok: bool
    gemini_key_ok: bool
    firered_model_dir_ok: bool
    python_modules_ok: bool
    cuda_ok: bool


REQUIRED_RUNTIME_MODULES = (
    "fireredvad",
    "huggingface_hub",
    "torch",
    "torchaudio",
    "torchcodec",
    "nemo.collections.asr",
)


def run_preflight_checks(config: AppConfig) -> PreflightReport:
    require_ffmpeg()
    gemini_key_ok = False
    if config.transcript_path is None:
        # Load only to validate presence. The key value must never be logged or persisted.
        load_gemini_api_key(config.env_file)
        gemini_key_ok = True
    missing_modules = [
        module_name for module_name in REQUIRED_RUNTIME_MODULES if not module_available(module_name)
    ]
    if missing_modules:
        missing = ", ".join(missing_modules)
        context = (
            "paid Gemini transcription"
            if config.transcript_path is None
            else "Segmentwright processing"
        )
        raise RuntimeError(
            f"Missing ML runtime dependencies before {context}: "
            f"{missing}. Install product dependencies with `uv sync`."
        )
    ensure_firered_vad_model(config)
    if config.device == "cuda" and not cuda_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    return PreflightReport(
        ffmpeg_ok=True,
        gemini_key_ok=gemini_key_ok,
        firered_model_dir_ok=True,
        python_modules_ok=True,
        cuda_ok=True,
    )


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())
