from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .prompts import (
    PLAIN_TRANSCRIPT_PROMPT_PRESET,
    PROMPT_PRESETS,
    SUBTITLE_LINES_PROMPT_PRESET,
)
from .resegmentation.local_llm import (
    DEFAULT_RESEGMENTATION_PROMPT_PRESET,
    RESEGMENTATION_PROMPT_PRESETS,
)

SUPPORTED_INPUT_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".mp3",
    ".m4a",
    ".wav",
}

DEFAULT_ENV_FILE = Path("~/.segmentwright/.env")
DEFAULT_FIRE_RED_VAD_MODEL_DIR = Path("~/.cache/segmentwright/FireRedVAD/VAD")
DEFAULT_PRESET = "quality-baseline"
DEFAULT_LANGUAGE = "ja"
SUPPORTED_LANGUAGES = {"ja", "en"}
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_3_FLASH_PREVIEW_MODEL = "gemini-3-flash-preview"
GEMINI_31_FLASH_LITE_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_JAPANESE_ALIGNMENT_MODEL = "nvidia/parakeet-tdt_ctc-0.6b-ja"
DEFAULT_ENGLISH_ALIGNMENT_MODEL = "nvidia/parakeet-ctc-0.6b"
DEFAULT_ALIGNMENT_MODELS = {
    "ja": DEFAULT_JAPANESE_ALIGNMENT_MODEL,
    "en": DEFAULT_ENGLISH_ALIGNMENT_MODEL,
}
DEFAULT_TARGET_CHUNK_SEC = 180.0
DEFAULT_MAX_CHUNK_SEC = 180.0
DEFAULT_MIN_RETRY_SPLIT_SEC = 30.0
GEMINI_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
RESPONSE_FORMATS = {"json_lines", "plain_text"}
SEGMENTATION_MODES = {"direct", "resegmentation"}
DEFAULT_RESEGMENTATION_MODEL = "gemma4:12b"
DEFAULT_RESEGMENTATION_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ProductPreset:
    name: str
    gemini_model: str
    prompt_preset: str
    response_format: str
    segmentation: str
    resegmentation_prompt: str = DEFAULT_RESEGMENTATION_PROMPT_PRESET


PRODUCT_PRESETS: dict[str, ProductPreset] = {
    "quality-baseline": ProductPreset(
        name="quality-baseline",
        gemini_model=DEFAULT_GEMINI_MODEL,
        prompt_preset=SUBTITLE_LINES_PROMPT_PRESET,
        response_format="json_lines",
        segmentation="direct",
    ),
    "cheaper-candidate": ProductPreset(
        name="cheaper-candidate",
        gemini_model=GEMINI_3_FLASH_PREVIEW_MODEL,
        prompt_preset=PLAIN_TRANSCRIPT_PROMPT_PRESET,
        response_format="plain_text",
        segmentation="resegmentation",
    ),
    "lite-experimental": ProductPreset(
        name="lite-experimental",
        gemini_model=GEMINI_31_FLASH_LITE_MODEL,
        prompt_preset=PLAIN_TRANSCRIPT_PROMPT_PRESET,
        response_format="plain_text",
        segmentation="resegmentation",
    ),
}


@dataclass(frozen=True)
class AppConfig:
    input_path: Path
    output_path: Path
    artifacts_dir: Path
    env_file: Path
    artifacts_explicit: bool = False
    preset: str = DEFAULT_PRESET
    language: str = DEFAULT_LANGUAGE
    language_explicit: bool = False
    gemini_model: str = DEFAULT_GEMINI_MODEL
    alignment_model: str | None = None
    prompt_preset: str = SUBTITLE_LINES_PROMPT_PRESET
    prompt_text: str | None = None
    prompt_file: Path | None = None
    response_format: str = "json_lines"
    segmentation: str = "direct"
    gemini_thinking_level: str | None = None
    gemini_temperature: float | None = None
    transcript_path: Path | None = None
    strict_transcript_chunks: bool = False
    transcript_chunk_warning_sec: float = DEFAULT_MAX_CHUNK_SEC
    resegmentation_model: str = DEFAULT_RESEGMENTATION_MODEL
    resegmentation_ollama_url: str = DEFAULT_RESEGMENTATION_OLLAMA_URL
    resegmentation_prompt: str = DEFAULT_RESEGMENTATION_PROMPT_PRESET
    resegmentation_max_retries: int = 3
    ollama_auto_pull: bool = False
    ollama_start_timeout_sec: float = 30.0
    firered_vad_model_dir: Path = DEFAULT_FIRE_RED_VAD_MODEL_DIR
    firered_vad_auto_download: bool = True
    device: str = "auto"
    target_chunk_sec: float = DEFAULT_TARGET_CHUNK_SEC
    max_chunk_sec: float = DEFAULT_MAX_CHUNK_SEC
    min_retry_split_sec: float = DEFAULT_MIN_RETRY_SPLIT_SEC
    long_omission_sec: float = 60.0
    coverage_gap_merge_sec: float = 2.0
    max_retries: int = 1
    max_split_depth: int = 2
    max_retry_work_items: int | None = None
    cache_dir: Path | None = None
    cache_enabled: bool = True
    force_gemini: bool = False
    overwrite: bool = False
    verbose: bool = False

    @property
    def resolved_cache_dir(self) -> Path:
        if self.cache_dir is not None:
            return self.cache_dir
        return self.artifacts_dir / "cache"


def default_artifacts_dir(input_path: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    stem = input_path.stem or "input"
    return Path("artifacts") / f"{stem}-{stamp}"


def expand_path(path: Path) -> Path:
    return path.expanduser()


def default_alignment_model_for_language(language: str) -> str:
    try:
        return DEFAULT_ALIGNMENT_MODELS[language]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Invalid language {language!r}; expected one of: {supported}") from exc


def resolve_alignment_model(config: AppConfig) -> str:
    if config.alignment_model:
        return config.alignment_model
    return default_alignment_model_for_language(config.language)


def validate_config(config: AppConfig) -> None:
    if config.preset not in PRODUCT_PRESETS:
        supported = ", ".join(sorted(PRODUCT_PRESETS))
        raise ValueError(f"Invalid preset {config.preset!r}; expected one of: {supported}")

    if config.language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Invalid language {config.language!r}; expected one of: {supported}")

    if config.alignment_model is not None and not config.alignment_model.strip():
        raise ValueError("--alignment-model must not be empty")

    if not config.input_path.exists():
        raise ValueError(f"Input file does not exist: {config.input_path}")

    suffix = config.input_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise ValueError(f"Unsupported input extension {suffix!r}; supported: {supported}")

    if config.output_path.suffix.lower() != ".srt":
        raise ValueError(f"Output path must end with .srt: {config.output_path}")

    if config.output_path.exists() and config.input_path.samefile(config.output_path):
        raise ValueError(
            f"Output path must not point to the input file: {config.output_path}"
        )

    if config.device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Invalid device {config.device!r}; expected auto, cuda, or cpu")
    if (
        config.gemini_thinking_level is not None
        and config.gemini_thinking_level not in GEMINI_THINKING_LEVELS
    ):
        supported = ", ".join(sorted(GEMINI_THINKING_LEVELS))
        raise ValueError(
            f"Invalid Gemini thinking level {config.gemini_thinking_level!r}; "
            f"expected one of: {supported}"
        )
    if config.gemini_temperature is not None and not (
        0.0 <= config.gemini_temperature <= 2.0
    ):
        raise ValueError("--temperature must be between 0.0 and 2.0")
    if config.response_format not in RESPONSE_FORMATS:
        supported = ", ".join(sorted(RESPONSE_FORMATS))
        raise ValueError(
            f"Invalid response format {config.response_format!r}; expected one of: {supported}"
        )
    if config.prompt_preset not in PROMPT_PRESETS:
        supported = ", ".join(sorted(PROMPT_PRESETS))
        raise ValueError(
            f"Invalid prompt preset {config.prompt_preset!r}; expected one of: {supported}"
        )
    if config.segmentation not in SEGMENTATION_MODES:
        supported = ", ".join(sorted(SEGMENTATION_MODES))
        raise ValueError(
            f"Invalid segmentation mode {config.segmentation!r}; expected one of: {supported}"
        )
    if config.resegmentation_prompt not in RESEGMENTATION_PROMPT_PRESETS:
        supported = ", ".join(sorted(RESEGMENTATION_PROMPT_PRESETS))
        raise ValueError(
            f"Invalid resegmentation prompt {config.resegmentation_prompt!r}; "
            f"expected one of: {supported}"
        )
    if config.resegmentation_max_retries < 0:
        raise ValueError("--resegmentation-max-retries must be >= 0")
    if not isinstance(config.ollama_auto_pull, bool):
        raise ValueError("ollama_auto_pull must be a boolean")
    if config.ollama_start_timeout_sec <= 0:
        raise ValueError("--ollama-start-timeout-sec must be greater than 0")
    if config.prompt_text is not None and config.prompt_file is not None:
        raise ValueError("--prompt-file and --prompt-text are mutually exclusive")
    if config.prompt_file is not None and not config.prompt_file.exists():
        raise ValueError(f"Prompt file does not exist: {config.prompt_file}")
    if config.transcript_path is not None:
        if not config.transcript_path.exists():
            raise ValueError(f"Transcript file does not exist: {config.transcript_path}")
        if config.transcript_path.suffix.lower() != ".json":
            raise ValueError(f"Transcript file must be JSON for now: {config.transcript_path}")
        if config.gemini_thinking_level is not None:
            raise ValueError("--gemini-thinking-level cannot be used with --transcript")
        if config.gemini_temperature is not None:
            raise ValueError("--temperature cannot be used with --transcript")
        if config.prompt_text is not None or config.prompt_file is not None:
            raise ValueError("--prompt-file/--prompt-text cannot be used with --transcript")
        if config.force_gemini:
            raise ValueError("--force-gemini cannot be used with --transcript")
        if config.transcript_chunk_warning_sec <= 0:
            raise ValueError("--transcript-chunk-warning-sec must be greater than 0")
    if config.response_format == "plain_text" and config.segmentation == "direct":
        # This can work only if Gemini returns multiple lines. Keep the combination available for
        # manual experiments, but the product presets avoid it.
        pass
    if not isinstance(config.firered_vad_auto_download, bool):
        raise ValueError("firered_vad_auto_download must be a boolean")

    if config.target_chunk_sec <= 0:
        raise ValueError("--target-chunk-sec must be greater than 0")
    if config.max_chunk_sec <= 0:
        raise ValueError("--max-chunk-sec must be greater than 0")
    if config.min_retry_split_sec <= 0:
        raise ValueError("--min-retry-split-sec must be greater than 0")
    if config.target_chunk_sec > config.max_chunk_sec:
        raise ValueError("--target-chunk-sec must not exceed --max-chunk-sec")

    if config.long_omission_sec <= 0:
        raise ValueError("--long-omission-sec must be greater than 0")
    if config.coverage_gap_merge_sec < 0:
        raise ValueError("--coverage-gap-merge-sec must be >= 0")
    if config.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if config.max_split_depth < 0:
        raise ValueError("--max-split-depth must be >= 0")
    if config.max_retry_work_items is not None and config.max_retry_work_items <= 0:
        raise ValueError("--max-retry-work-items must be greater than 0")

    if config.output_path.exists() and not config.overwrite:
        raise ValueError(
            f"Output already exists; pass --overwrite to replace it: {config.output_path}"
        )
    if (
        config.artifacts_explicit
        and config.artifacts_dir.exists()
        and any(config.artifacts_dir.iterdir())
        and not config.overwrite
    ):
        raise ValueError(
            f"Artifacts directory is not empty; pass --overwrite to use it: {config.artifacts_dir}"
        )
