from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from .config import (
    DEFAULT_ALIGNMENT_MODELS,
    DEFAULT_ENV_FILE,
    DEFAULT_FIRE_RED_VAD_MODEL_DIR,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHUNK_SEC,
    DEFAULT_MIN_RETRY_SPLIT_SEC,
    DEFAULT_PRESET,
    DEFAULT_RESEGMENTATION_MODEL,
    DEFAULT_RESEGMENTATION_OLLAMA_URL,
    DEFAULT_RESEGMENTATION_PROMPT_PRESET,
    DEFAULT_TARGET_CHUNK_SEC,
    GEMINI_THINKING_LEVELS,
    PRODUCT_PRESETS,
    RESEGMENTATION_PROMPT_PRESETS,
    RESPONSE_FORMATS,
    SEGMENTATION_MODES,
    SUPPORTED_LANGUAGES,
    AppConfig,
    default_artifacts_dir,
    expand_path,
    validate_config,
)
from .logging_utils import configure_logging
from .pipeline import run_pipeline
from .prompts import PROMPT_PRESETS
from .workdir import cleanup_stale_temp_workdirs, temporary_artifacts_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config: AppConfig | None = None
    try:
        config = build_config(args)
        configure_logging(config.verbose)
        return run_with_artifacts_policy(config)
    except Exception as exc:  # noqa: BLE001 - CLI must turn failures into messages.
        if config is not None and config.verbose:
            traceback.print_exception(exc, file=sys.stderr)
            return 1
        print(f"segmentwright: {exc}", file=sys.stderr)
        return 1


def run_with_artifacts_policy(config: AppConfig) -> int:
    cleanup_stale_temp_workdirs()
    if config.artifacts_explicit:
        return run_validated_pipeline(config)
    with temporary_artifacts_dir(input_path=config.input_path) as artifacts_dir:
        runtime_config = replace(config, artifacts_dir=artifacts_dir)
        return run_validated_pipeline(runtime_config)


def run_validated_pipeline(config: AppConfig) -> int:
    try:
        validate_config(config)
        run_pipeline(config)
    except Exception:
        if (config.artifacts_dir / "errors.json").exists():
            return 1
        raise
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segmentwright",
        description="Generate aligned SRT subtitles from speech media.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)

    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--preset",
        choices=sorted(PRODUCT_PRESETS),
        default=DEFAULT_PRESET,
        help="Product preset. Defaults to quality-baseline.",
    )
    parser.add_argument(
        "--language",
        choices=sorted(SUPPORTED_LANGUAGES),
        default=None,
        help=(
            "Source speech language. Defaults to Japanese for built-in transcription; "
            "with --transcript, defaults to the transcript file language."
        ),
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help=f"Override the preset Gemini model. Default preset model: {DEFAULT_GEMINI_MODEL}.",
    )
    parser.add_argument(
        "--alignment-model",
        help=(
            "Override the forced-alignment model. Defaults by language: "
            + ", ".join(
                f"{language}={model}"
                for language, model in sorted(DEFAULT_ALIGNMENT_MODELS.items())
            )
            + "."
        ),
    )
    parser.add_argument(
        "--gemini-thinking-level",
        "--thinking-level",
        choices=sorted(GEMINI_THINKING_LEVELS),
        help="Optional Gemini thinkingLevel override for Gemini 3 models.",
    )
    parser.add_argument(
        "--gemini-temperature",
        "--temperature",
        type=float,
        help="Optional Gemini sampling temperature override.",
    )
    parser.add_argument(
        "--prompt-preset",
        choices=sorted(PROMPT_PRESETS),
        help="Override the preset built-in prompt.",
    )
    parser.add_argument("--prompt-file", type=Path, help="Read a custom prompt from a UTF-8 file.")
    parser.add_argument("--prompt-text", help="Use a custom prompt string.")
    parser.add_argument(
        "--response-format",
        choices=sorted(RESPONSE_FORMATS),
        help="Override the preset Gemini response format.",
    )
    parser.add_argument(
        "--segmentation",
        choices=sorted(SEGMENTATION_MODES),
        help="Override the preset segmentation mode.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help=(
            "Use a pre-generated chunk-level transcript JSON and skip built-in ASR."
        ),
    )
    parser.add_argument(
        "--strict-transcript-chunks",
        action="store_true",
        help="Treat imported transcript chunk validation warnings as errors.",
    )
    parser.add_argument(
        "--transcript-chunk-warning-sec",
        type=float,
        default=DEFAULT_MAX_CHUNK_SEC,
        help="Warn when an imported transcript chunk exceeds this duration.",
    )
    parser.add_argument(
        "--resegmentation-model",
        "--ollama-model",
        default=DEFAULT_RESEGMENTATION_MODEL,
        help="Local Ollama model used when --segmentation resegmentation is active.",
    )
    parser.add_argument(
        "--resegmentation-ollama-url",
        default=DEFAULT_RESEGMENTATION_OLLAMA_URL,
        help="Ollama base URL used when --segmentation resegmentation is active.",
    )
    parser.add_argument(
        "--resegmentation-prompt",
        choices=sorted(RESEGMENTATION_PROMPT_PRESETS),
        help=(
            "Local LLM segmentation prompt style. "
            f"Default preset value: {DEFAULT_RESEGMENTATION_PROMPT_PRESET}."
        ),
    )
    parser.add_argument(
        "--resegmentation-max-retries",
        type=int,
        default=3,
        help="Maximum local resegmentation retries after a quality-gate failure.",
    )
    parser.add_argument(
        "--ollama-auto-pull",
        action="store_true",
        help="Pull the configured Ollama model automatically if it is missing.",
    )
    parser.add_argument(
        "--ollama-start-timeout-sec",
        type=float,
        default=30.0,
        help="Seconds to wait for an auto-started Ollama server.",
    )
    parser.add_argument(
        "--firered-vad-model-dir",
        type=Path,
        default=DEFAULT_FIRE_RED_VAD_MODEL_DIR,
    )
    parser.add_argument("--no-firered-vad-auto-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    parser.add_argument("--target-chunk-sec", type=float, default=DEFAULT_TARGET_CHUNK_SEC)
    parser.add_argument("--max-chunk-sec", type=float, default=DEFAULT_MAX_CHUNK_SEC)
    parser.add_argument(
        "--min-retry-split-sec",
        type=float,
        default=DEFAULT_MIN_RETRY_SPLIT_SEC,
    )

    parser.add_argument("--long-omission-sec", type=float, default=60.0)
    parser.add_argument("--coverage-gap-merge-sec", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-split-depth", type=int, default=2)
    parser.add_argument(
        "--max-retry-work-items",
        type=int,
        default=None,
        help="Limit additional retry children only. Default: unlimited.",
    )

    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-gemini", action="store_true")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> AppConfig:
    input_path = expand_path(args.input)
    artifacts_dir = (
        expand_path(args.artifacts) if args.artifacts else default_artifacts_dir(input_path)
    )
    preset = PRODUCT_PRESETS[args.preset]
    transcript_path = expand_path(args.transcript) if args.transcript else None
    language = args.language or DEFAULT_LANGUAGE
    segmentation = (
        args.segmentation
        or ("resegmentation" if transcript_path is not None else preset.segmentation)
    )
    return AppConfig(
        input_path=input_path,
        output_path=expand_path(args.output),
        artifacts_dir=expand_path(artifacts_dir),
        env_file=expand_path(args.env_file),
        artifacts_explicit=args.artifacts is not None,
        preset=args.preset,
        language=language,
        language_explicit=args.language is not None,
        gemini_model=args.gemini_model or preset.gemini_model,
        alignment_model=args.alignment_model,
        gemini_thinking_level=args.gemini_thinking_level,
        gemini_temperature=args.gemini_temperature,
        prompt_preset=args.prompt_preset or preset.prompt_preset,
        prompt_file=expand_path(args.prompt_file) if args.prompt_file else None,
        prompt_text=args.prompt_text,
        response_format=args.response_format or preset.response_format,
        segmentation=segmentation,
        transcript_path=transcript_path,
        strict_transcript_chunks=args.strict_transcript_chunks,
        transcript_chunk_warning_sec=args.transcript_chunk_warning_sec,
        resegmentation_model=args.resegmentation_model,
        resegmentation_ollama_url=args.resegmentation_ollama_url,
        resegmentation_prompt=args.resegmentation_prompt or preset.resegmentation_prompt,
        resegmentation_max_retries=args.resegmentation_max_retries,
        ollama_auto_pull=args.ollama_auto_pull,
        ollama_start_timeout_sec=args.ollama_start_timeout_sec,
        firered_vad_model_dir=expand_path(args.firered_vad_model_dir),
        firered_vad_auto_download=not args.no_firered_vad_auto_download,
        device=args.device,
        target_chunk_sec=args.target_chunk_sec,
        max_chunk_sec=args.max_chunk_sec,
        min_retry_split_sec=args.min_retry_split_sec,
        long_omission_sec=args.long_omission_sec,
        coverage_gap_merge_sec=args.coverage_gap_merge_sec,
        max_retries=args.max_retries,
        max_split_depth=args.max_split_depth,
        max_retry_work_items=args.max_retry_work_items,
        cache_dir=expand_path(args.cache_dir) if args.cache_dir else None,
        cache_enabled=not args.no_cache,
        force_gemini=args.force_gemini,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    raise SystemExit(main())
