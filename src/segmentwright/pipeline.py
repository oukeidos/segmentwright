from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

from .aligners.base import LineSpan
from .aligners.parakeet import ParakeetAlignerBackend
from .cache import TranscriptCache, TranscriptCacheKey, hash_file
from .chunking import ChunkPlannerConfig, PlannedChunk, plan_chunks
from .config import PRODUCT_PRESETS, SUPPORTED_LANGUAGES, AppConfig, resolve_alignment_model
from .env import load_gemini_api_key
from .gemini import GeminiTranscriber, GeminiTranscriptionError, GeminiTranscriptionResult
from .lifecycle import ModelLifecycleRecorder
from .logging_utils import ConsoleReporter, capture_external_output, format_count
from .media import (
    extract_wav_segment,
    normalize_to_wav,
    probe_duration,
    split_wav,
)
from .ollama import ManagedOllamaServer, OllamaRuntimeConfig, OllamaRuntimeStatus
from .preflight import run_preflight_checks
from .prompts import PromptSpec, resolve_prompt
from .qa.long_omission import LongOmission, LongOmissionReport, detect_long_omissions
from .qa.token_gap_guard import apply_token_gap_guard
from .resegmentation.artifacts import write_resegmentation_artifacts
from .resegmentation.local_llm import OllamaSegmentationConfig, call_ollama_resegmentation
from .resegmentation.projection import (
    ResegmentationResult,
    ResegmentationSuggestion,
    project_suggestion_to_source,
)
from .resegmentation.quality import (
    ResegmentationQualityConfig,
    assess_resegmentation_quality,
    fallback_segment_source_text,
)
from .retry import (
    RetrySplitDecision,
    build_retry_children,
    clip_speech_segments_absolute,
    plan_retry_split,
)
from .srt import SrtCue, build_final_assembly_report, format_srt, stabilize_cues
from .transcript import Transcript, parse_plain_text_transcript
from .transcript_chunks import (
    TranscriptChunksDocument,
    load_transcript_chunks,
    validate_transcript_chunks,
)
from .vad.base import VadResult, VadSegment
from .vad.firered import FireRedVadBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedChunk:
    chunk: PlannedChunk
    transcript: Transcript
    gemini_call_metadata: dict[str, Any]
    line_spans: list[LineSpan]
    raw_alignment: dict[str, Any]
    qa_report: LongOmissionReport


@dataclass(frozen=True)
class TranscribedChunk:
    chunk: PlannedChunk
    transcription: GeminiTranscriptionResult


@dataclass(frozen=True)
class SegmentedChunk:
    chunk: PlannedChunk
    transcription: GeminiTranscriptionResult
    alignment_lines: list[str]
    segmentation_report: dict[str, Any]


@dataclass
class RetryBudgetState:
    limit: int | None
    generated_children: int = 0
    split_parents: int = 0
    skipped_parents: int = 0

    def reserve_split(self, child_count: int) -> bool:
        if self.limit is not None and self.generated_children + child_count > self.limit:
            self.skipped_parents += 1
            return False
        self.generated_children += child_count
        self.split_parents += 1
        return True


@dataclass(frozen=True)
class RetrySplitRequest:
    chunk: PlannedChunk
    omission: LongOmission
    decision: RetrySplitDecision
    children: tuple[PlannedChunk, ...]


@dataclass(frozen=True)
class RetryUnavailable:
    reason: str


def run_pipeline(config: AppConfig) -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    reporter = ConsoleReporter(verbose=config.verbose, artifacts_dir=config.artifacts_dir)
    mode = "external transcript" if config.transcript_path is not None else "gemini"
    reporter.run_start(
        input_path=config.input_path,
        output_path=config.output_path,
        mode=mode,
        preset=config.preset,
        artifacts_dir=config.artifacts_dir if config.artifacts_explicit else None,
    )
    try:
        if config.transcript_path is not None:
            cue_count = run_pipeline_from_transcript(config, reporter=reporter)
        else:
            cue_count = run_pipeline_with_gemini(config, reporter=reporter)
        write_run_summary_artifact(
            config=config,
            reporter=reporter,
            status="success",
            cue_count=cue_count,
            mode=mode,
        )
        reporter.run_success(
            output_path=config.output_path,
            artifacts_dir=config.artifacts_dir if config.artifacts_explicit else None,
            cue_count=cue_count,
        )
    except Exception as exc:
        failed_stage = reporter.current_stage
        reporter.run_failure(
            error=exc,
            artifacts_dir=config.artifacts_dir if config.artifacts_explicit else None,
            verbose=config.verbose,
        )
        write_error_artifact(
            config=config,
            reporter=reporter,
            error=exc,
            mode=mode,
            failed_stage=failed_stage,
        )
        write_run_summary_artifact(
            config=config,
            reporter=reporter,
            status="failed",
            cue_count=None,
            mode=mode,
            error=exc,
            failed_stage=failed_stage,
        )
        raise


def run_pipeline_with_gemini(config: AppConfig, *, reporter: ConsoleReporter) -> int:
    lifecycle = ModelLifecycleRecorder(config.artifacts_dir / "model_lifecycle.json")
    try:
        reporter.stage_start("Preflight")
        run_preflight_checks(config)
        reporter.stage_end("Preflight")

        normalized_wav = config.artifacts_dir / "source.wav"
        reporter.stage_start("Normalize media")
        normalize_to_wav(config.input_path, normalized_wav)
        source_id = hash_file(normalized_wav)
        duration_sec = probe_duration(normalized_wav)
        write_json(
            config.artifacts_dir / "media.json",
            {
                "schema": "segmentwright.media.v1",
                "input_path": str(config.input_path),
                "normalized_wav": str(normalized_wav),
                "source_id": source_id,
                "duration_sec": duration_sec,
            },
        )
        reporter.stage_end("Normalize media", details=f"{duration_sec:.1f}s source")

        reporter.stage_start("VAD")
        vad_result = run_vad_stage(
            config=config,
            normalized_wav=normalized_wav,
            lifecycle=lifecycle,
        )
        write_json(config.artifacts_dir / "vad.json", serializable(vad_result))
        reporter.stage_end(
            "VAD",
            details=format_count(len(vad_result.speech_segments), "speech segment"),
        )

        reporter.stage_start("Plan chunks")
        chunk_plan = plan_chunks(
            duration_sec=duration_sec,
            speech_segments=vad_result.speech_segments,
            source_id=source_id,
            config=ChunkPlannerConfig(
                target_chunk_sec=config.target_chunk_sec,
                max_chunk_sec=config.max_chunk_sec,
                min_chunk_sec=config.min_retry_split_sec,
            ),
        )
        chunk_audio_paths = split_wav(
            normalized_wav,
            chunk_plan.split_times,
            config.artifacts_dir / "chunks",
        )
        chunks = attach_audio_paths(chunk_plan.chunks, chunk_audio_paths)
        write_json(
            config.artifacts_dir / "chunk_manifest.json",
            {
                "schema": "segmentwright.chunk_manifest.v1",
                "split_times": chunk_plan.split_times,
                "forced_count": chunk_plan.forced_count,
                "candidates": [asdict(candidate) for candidate in chunk_plan.candidates],
                "chunks": [serializable(chunk) for chunk in chunks],
            },
        )
        reporter.stage_end("Plan chunks", details=format_count(len(chunks), "chunk"))

        reporter.stage_start("Configure run")
        prompt_spec = resolve_prompt(
            prompt_preset=config.prompt_preset,
            language=config.language,
            prompt_text=config.prompt_text,
            prompt_file=config.prompt_file,
        )
        write_json(
            config.artifacts_dir / "run_config.json",
            build_run_config_report(
                config=config,
                prompt_spec=prompt_spec,
                ollama_status=None,
            ),
        )
        reporter.stage_end("Configure run")

        cache = TranscriptCache(config.resolved_cache_dir)
        processed_chunks, retry_trace, ollama_status = process_chunk_generations(
            config=config,
            normalized_wav=normalized_wav,
            initial_chunks=chunks,
            speech_segments=vad_result.speech_segments,
            source_id=source_id,
            prompt_spec=prompt_spec,
            cache=cache,
            lifecycle=lifecycle,
            reporter=reporter,
        )
        write_json(
            config.artifacts_dir / "run_config.json",
            build_run_config_report(
                config=config,
                prompt_spec=prompt_spec,
                ollama_status=ollama_status,
            ),
        )
    finally:
        lifecycle.write()
    reporter.stage_start("Finalize")
    write_json(
        config.artifacts_dir / "token_usage.json",
        build_token_usage_report(
            calls=load_gemini_call_metadata(config.artifacts_dir),
            accepted_chunk_ids={processed.chunk.chunk_id for processed in processed_chunks},
        ),
    )
    write_json(
        config.artifacts_dir / "transcripts.json",
        {
            "schema": "segmentwright.transcripts.v1",
            "gemini_model": config.gemini_model,
            "language": config.language,
            "alignment_model": resolve_alignment_model(config),
            "gemini_thinking_level": config.gemini_thinking_level,
            "gemini_temperature": config.gemini_temperature,
            "gemini_response_format": config.response_format,
            "preset": config.preset,
            "prompt_preset": prompt_spec.preset,
            "prompt_source": prompt_spec.source,
            "prompt_language": prompt_spec.language,
            "prompt_sha256": prompt_spec.sha256,
            "segmentation": config.segmentation,
            "chunks": [
                {
                    "chunk_id": processed.chunk.chunk_id,
                    "parent_id": processed.chunk.parent_id,
                    "start_sec": processed.chunk.start_sec,
                    "end_sec": processed.chunk.end_sec,
                    "depth": processed.chunk.depth,
                    "retry_count": processed.chunk.retry_count,
                    "lines": processed.transcript.lines,
                    "raw_json": processed.transcript.raw_json,
                    "empty_transcript": processed.gemini_call_metadata.get(
                        "empty_transcript",
                        len(processed.transcript.lines) == 0,
                    ),
                    "empty_acceptance_reason": processed.gemini_call_metadata.get(
                        "empty_acceptance_reason"
                    ),
                    "post_transcription_action": processed.gemini_call_metadata.get(
                        "post_transcription_action"
                    ),
                    "finish_reason": processed.gemini_call_metadata.get("finish_reason"),
                    "finish_reason_class": processed.gemini_call_metadata.get(
                        "finish_reason_class"
                    ),
                    "gemini_call_metadata_path": str(
                        gemini_call_metadata_path(config.artifacts_dir, processed.chunk)
                    ),
                    "cache_hit": processed.gemini_call_metadata.get("cache_hit"),
                    "token_usage": processed.gemini_call_metadata.get("token_usage"),
                }
                for processed in processed_chunks
            ],
        },
    )

    cues = [
        SrtCue(start_sec=line.start_sec, end_sec=line.end_sec, text=line.text)
        for processed in processed_chunks
        for line in processed.line_spans
    ]
    all_line_spans = [line for processed in processed_chunks for line in processed.line_spans]
    write_json(
        config.artifacts_dir / "alignment.json",
        {
            "schema": "segmentwright.alignment.v1",
            "chunks": [processed.raw_alignment for processed in processed_chunks],
        },
    )
    write_json(
        config.artifacts_dir / "chunk_qa.json",
        {
            "schema": "segmentwright.chunk_qa.v1",
            "chunks": [
                {
                    "chunk_id": processed.chunk.chunk_id,
                    "parent_id": processed.chunk.parent_id,
                    "start_sec": processed.chunk.start_sec,
                    "end_sec": processed.chunk.end_sec,
                    "qa_report": processed.qa_report.to_dict(),
                }
                for processed in processed_chunks
            ],
        },
    )
    qa_report = detect_long_omissions(
        speech_segments=vad_result.speech_segments,
        line_spans=all_line_spans,
        threshold_sec=config.long_omission_sec,
        coverage_gap_merge_sec=config.coverage_gap_merge_sec,
    )
    write_json(config.artifacts_dir / "qa_report.json", qa_report.to_dict())
    if qa_report.has_long_omissions:
        logger.warning(
            "Detected %d long omission candidate(s) after retry processing",
            len(qa_report.long_omissions),
        )
    write_retry_trace_artifact(config.artifacts_dir, retry_trace)
    stable_cues = stabilize_cues(cues)
    write_json(
        config.artifacts_dir / "final_assembly.json",
        build_final_assembly_report(
            input_cues=cues,
            output_cues=stable_cues,
            source_duration_sec=duration_sec,
        ).to_dict(),
    )
    config.output_path.write_text(format_srt(stable_cues), encoding="utf-8")
    reporter.stage_end("Finalize", details=format_count(len(stable_cues), "cue"))
    return len(stable_cues)


def run_pipeline_from_transcript(config: AppConfig, *, reporter: ConsoleReporter) -> int:
    if config.transcript_path is None:
        raise ValueError("--transcript is required for transcript import mode")

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    document = load_transcript_chunks(config.transcript_path)
    effective_language = (
        config.language if config.language_explicit else document.language or config.language
    )
    config = replace(config, language=effective_language, max_retries=0)
    lifecycle = ModelLifecycleRecorder(config.artifacts_dir / "model_lifecycle.json")
    prompt_spec = PromptSpec(
        text="",
        source="external_transcript",
        language=config.language,
        preset=None,
        file=None,
        sha256="",
    )
    ollama_status: OllamaRuntimeStatus | None = None
    retry_trace: list[dict[str, Any]] = []
    try:
        reporter.stage_start("Preflight")
        run_preflight_checks(config)
        reporter.stage_end("Preflight")

        normalized_wav = config.artifacts_dir / "source.wav"
        reporter.stage_start("Normalize media")
        normalize_to_wav(config.input_path, normalized_wav)
        source_id = hash_file(normalized_wav)
        duration_sec = probe_duration(normalized_wav)
        write_json(
            config.artifacts_dir / "media.json",
            {
                "schema": "segmentwright.media.v1",
                "input_path": str(config.input_path),
                "normalized_wav": str(normalized_wav),
                "source_id": source_id,
                "duration_sec": duration_sec,
            },
        )
        reporter.stage_end("Normalize media", details=f"{duration_sec:.1f}s source")

        reporter.stage_start("Import transcript")
        validation = validate_transcript_chunks(
            document,
            duration_sec=duration_sec,
            language=config.language,
            language_explicit=config.language_explicit,
            supported_languages=SUPPORTED_LANGUAGES,
            warn_chunk_sec=config.transcript_chunk_warning_sec,
            strict=config.strict_transcript_chunks,
        )
        write_external_transcript_import_artifacts(
            config=config,
            document=document,
            validation=validation,
        )
        log_transcript_validation_warnings(validation, reporter=reporter)
        reporter.stage_end(
            "Import transcript",
            details=(
                f"{format_count(len(document.chunks), 'chunk')}, "
                f"{format_count(transcript_validation_warning_count(validation), 'warning')}"
            ),
        )

        reporter.stage_start("VAD")
        vad_result = run_vad_stage(
            config=config,
            normalized_wav=normalized_wav,
            lifecycle=lifecycle,
        )
        write_json(config.artifacts_dir / "vad.json", serializable(vad_result))
        reporter.stage_end(
            "VAD",
            details=format_count(len(vad_result.speech_segments), "speech segment"),
        )

        reporter.stage_start("Materialize chunks")
        chunks = materialize_imported_chunks(
            config=config,
            document=document,
            normalized_wav=normalized_wav,
            source_id=source_id,
        )
        write_json(
            config.artifacts_dir / "chunk_manifest.json",
            {
                "schema": "segmentwright.chunk_manifest.v1",
                "source": "external_transcript",
                "split_times": [chunk.end_sec for chunk in chunks[:-1]],
                "forced_count": 0,
                "candidates": [],
                "chunks": [serializable(chunk) for chunk in chunks],
            },
        )
        reporter.stage_end("Materialize chunks", details=format_count(len(chunks), "chunk"))
        reporter.stage_start("Configure run")
        write_json(
            config.artifacts_dir / "run_config.json",
            build_run_config_report(
                config=config,
                prompt_spec=prompt_spec,
                ollama_status=None,
                transcript_document=document,
            ),
        )
        reporter.stage_end("Configure run")

        transcribed = build_imported_transcribed_chunks(
            config=config,
            document=document,
            chunks=chunks,
        )
        segmented, ollama_status = run_segmentation_stage(
            config=config,
            generation=0,
            transcribed=transcribed,
            lifecycle=lifecycle,
            reporter=reporter,
        )
        if ollama_status is not None:
            write_json(config.artifacts_dir / "ollama.json", ollama_status.to_dict())
        processed_chunks = run_alignment_stage(
            config=config,
            generation=0,
            segmented=segmented,
            speech_segments=vad_result.speech_segments,
            lifecycle=lifecycle,
            reporter=reporter,
        )
        write_json(
            config.artifacts_dir / "run_config.json",
            build_run_config_report(
                config=config,
                prompt_spec=prompt_spec,
                ollama_status=ollama_status,
                transcript_document=document,
            ),
        )
    finally:
        lifecycle.write()

    reporter.stage_start("Finalize")
    write_json(
        config.artifacts_dir / "token_usage.json",
        build_external_transcript_token_usage_report(
            accepted_chunk_ids={processed.chunk.chunk_id for processed in processed_chunks},
        ),
    )
    write_transcripts_artifact(
        config=config,
        prompt_spec=prompt_spec,
        processed_chunks=processed_chunks,
        transcription_backend="external_transcript",
    )
    stable_cues = write_alignment_and_srt_artifacts(
        config=config,
        vad_result=vad_result,
        duration_sec=duration_sec,
        processed_chunks=processed_chunks,
        retry_trace=retry_trace,
    )
    reporter.stage_end("Finalize", details=format_count(len(stable_cues), "cue"))
    return len(stable_cues)


def write_external_transcript_import_artifacts(
    *,
    config: AppConfig,
    document: TranscriptChunksDocument,
    validation: Any,
) -> None:
    import_dir = config.artifacts_dir / "external_transcript"
    write_json(import_dir / "input.json", document.raw)
    write_json(
        import_dir / "normalized.json",
        {
            "schema": "segmentwright.external_transcript.v1",
            "source_path": str(document.path),
            "input_schema": document.schema,
            "version": document.version,
            "language": document.language,
            "metadata": document.metadata,
            "chunks": [
                {
                    "id": chunk.chunk_id,
                    "start": chunk.start_sec,
                    "end": chunk.end_sec,
                    "text": chunk.text,
                    "source": chunk.source,
                    "confidence": chunk.confidence,
                    "notes": chunk.notes,
                    "metadata": chunk.metadata,
                    "extra": chunk.extra,
                }
                for chunk in document.chunks
            ],
        },
    )
    write_json(import_dir / "validation.json", validation.to_dict())


def log_transcript_validation_warnings(validation: Any, *, reporter: ConsoleReporter) -> None:
    for warning in validation.warnings:
        logger.warning("Transcript validation warning: %s", warning)
        reporter.warning(f"transcript: {warning}")
    for chunk_id, warnings in validation.chunk_warnings.items():
        for warning in warnings:
            logger.warning("Transcript validation warning for %s: %s", chunk_id, warning)
            reporter.warning(f"transcript {chunk_id}: {warning}")


def transcript_validation_warning_count(validation: Any) -> int:
    return len(validation.warnings) + sum(
        len(warnings) for warnings in validation.chunk_warnings.values()
    )


def materialize_imported_chunks(
    *,
    config: AppConfig,
    document: TranscriptChunksDocument,
    normalized_wav: Path,
    source_id: str,
) -> list[PlannedChunk]:
    chunks: list[PlannedChunk] = []
    for index, imported in enumerate(document.chunks):
        chunk = PlannedChunk(
            chunk_id=imported.chunk_id,
            parent_id=imported.chunk_id,
            source_id=source_id,
            start_sec=imported.start_sec,
            end_sec=imported.end_sec,
            start_sample=int(round(imported.start_sec * 16000)),
            end_sample=int(round(imported.end_sec * 16000)),
            depth=0,
            retry_count=0,
            original_index=index,
            cut_reason="external_transcript",
            cut_quality=None,
            cut_source="external_transcript",
            audio_path=None,
        )
        output_path = config.artifacts_dir / "chunks" / f"audio_{safe_chunk_id(chunk.chunk_id)}.wav"
        extract_wav_segment(
            normalized_wav,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            output_path=output_path,
        )
        chunks.append(replace(chunk, audio_path=output_path))
    return chunks


def build_imported_transcribed_chunks(
    *,
    config: AppConfig,
    document: TranscriptChunksDocument,
    chunks: list[PlannedChunk],
) -> list[TranscribedChunk]:
    if len(document.chunks) != len(chunks):
        raise ValueError(
            f"Transcript/chunk count mismatch: transcripts={len(document.chunks)} "
            f"chunks={len(chunks)}"
        )
    transcribed: list[TranscribedChunk] = []
    for imported, chunk in zip(document.chunks, chunks, strict=True):
        transcript = parse_plain_text_transcript(imported.text, allow_empty=True)
        empty_transcript = len(transcript.lines) == 0
        metadata_path = external_transcript_chunk_metadata_path(config.artifacts_dir, chunk)
        source_metadata = {
            "source": imported.source,
            "confidence": imported.confidence,
            "notes": imported.notes,
            "metadata": imported.metadata,
            "extra": imported.extra,
        }
        metadata = {
            "schema": "segmentwright.external_transcript_chunk.v1",
            "metadata_path": str(metadata_path),
            "status": "success",
            "cache_hit": False,
            "transcription_backend": "external_transcript",
            "backend": imported.source or document.metadata.get("backend"),
            "model": document.metadata.get("model"),
            "language": config.language,
            "segmentation": config.segmentation,
            "chunk": chunk_metadata(chunk),
            "source_metadata": source_metadata,
            "text": imported.text,
            "transcript": {
                "line_count": len(transcript.lines),
                "raw_character_count": len(transcript.raw_json),
                "empty_transcript": empty_transcript,
            },
            "line_count": len(transcript.lines),
            "empty_transcript": empty_transcript,
            "empty_acceptance_reason": (
                "external_transcript_empty_text" if empty_transcript else None
            ),
            "post_transcription_action": (
                "empty_transcript_noop" if empty_transcript else "normal_pipeline"
            ),
            "finish_reason": "EXTERNAL_TRANSCRIPT",
            "finish_reason_class": "external_transcript",
        }
        write_json(metadata_path, metadata)
        transcribed.append(
            TranscribedChunk(
                chunk=chunk,
                transcription=GeminiTranscriptionResult(
                    transcript=transcript,
                    metadata=metadata,
                ),
            )
        )
    return transcribed


def build_external_transcript_token_usage_report(
    *,
    accepted_chunk_ids: set[str],
) -> dict[str, Any]:
    return {
        "schema": "segmentwright.token_usage_report.v1",
        "transcription_backend": "external_transcript",
        "call_count": 0,
        "api_call_count": 0,
        "cache_hit_count": 0,
        "accepted_chunk_ids": sorted(accepted_chunk_ids),
        "all_attempts": summarize_token_calls([]),
        "accepted_chunks": summarize_token_calls([]),
        "api_calls": summarize_token_calls([]),
        "cache_hits": summarize_token_calls([]),
        "calls": [],
        "note": "Transcription token usage is external to Segmentwright in --transcript mode.",
    }


def write_transcripts_artifact(
    *,
    config: AppConfig,
    prompt_spec: PromptSpec,
    processed_chunks: list[ProcessedChunk],
    transcription_backend: str,
) -> None:
    write_json(
        config.artifacts_dir / "transcripts.json",
        {
            "schema": "segmentwright.transcripts.v1",
            "transcription_backend": transcription_backend,
            "gemini_model": config.gemini_model if transcription_backend == "gemini" else None,
            "language": config.language,
            "alignment_model": resolve_alignment_model(config),
            "gemini_thinking_level": (
                config.gemini_thinking_level if transcription_backend == "gemini" else None
            ),
            "gemini_temperature": (
                config.gemini_temperature if transcription_backend == "gemini" else None
            ),
            "gemini_response_format": (
                config.response_format if transcription_backend == "gemini" else None
            ),
            "preset": config.preset,
            "prompt_preset": prompt_spec.preset,
            "prompt_source": prompt_spec.source,
            "prompt_language": prompt_spec.language,
            "prompt_sha256": prompt_spec.sha256,
            "segmentation": config.segmentation,
            "chunks": [
                {
                    "chunk_id": processed.chunk.chunk_id,
                    "parent_id": processed.chunk.parent_id,
                    "start_sec": processed.chunk.start_sec,
                    "end_sec": processed.chunk.end_sec,
                    "depth": processed.chunk.depth,
                    "retry_count": processed.chunk.retry_count,
                    "lines": processed.transcript.lines,
                    "raw_json": processed.transcript.raw_json,
                    "empty_transcript": processed.gemini_call_metadata.get(
                        "empty_transcript",
                        len(processed.transcript.lines) == 0,
                    ),
                    "empty_acceptance_reason": processed.gemini_call_metadata.get(
                        "empty_acceptance_reason"
                    ),
                    "post_transcription_action": processed.gemini_call_metadata.get(
                        "post_transcription_action"
                    ),
                    "finish_reason": processed.gemini_call_metadata.get("finish_reason"),
                    "finish_reason_class": processed.gemini_call_metadata.get(
                        "finish_reason_class"
                    ),
                    "transcription_metadata_path": processed.gemini_call_metadata.get(
                        "metadata_path"
                    ),
                    "gemini_call_metadata_path": (
                        str(gemini_call_metadata_path(config.artifacts_dir, processed.chunk))
                        if transcription_backend == "gemini"
                        else None
                    ),
                    "cache_hit": processed.gemini_call_metadata.get("cache_hit"),
                    "token_usage": processed.gemini_call_metadata.get("token_usage"),
                }
                for processed in processed_chunks
            ],
        },
    )


def write_alignment_and_srt_artifacts(
    *,
    config: AppConfig,
    vad_result: VadResult,
    duration_sec: float,
    processed_chunks: list[ProcessedChunk],
    retry_trace: list[dict[str, Any]],
) -> list[SrtCue]:
    cues = [
        SrtCue(start_sec=line.start_sec, end_sec=line.end_sec, text=line.text)
        for processed in processed_chunks
        for line in processed.line_spans
    ]
    all_line_spans = [line for processed in processed_chunks for line in processed.line_spans]
    write_json(
        config.artifacts_dir / "alignment.json",
        {
            "schema": "segmentwright.alignment.v1",
            "chunks": [processed.raw_alignment for processed in processed_chunks],
        },
    )
    write_json(
        config.artifacts_dir / "chunk_qa.json",
        {
            "schema": "segmentwright.chunk_qa.v1",
            "chunks": [
                {
                    "chunk_id": processed.chunk.chunk_id,
                    "parent_id": processed.chunk.parent_id,
                    "start_sec": processed.chunk.start_sec,
                    "end_sec": processed.chunk.end_sec,
                    "qa_report": processed.qa_report.to_dict(),
                }
                for processed in processed_chunks
            ],
        },
    )
    qa_report = detect_long_omissions(
        speech_segments=vad_result.speech_segments,
        line_spans=all_line_spans,
        threshold_sec=config.long_omission_sec,
        coverage_gap_merge_sec=config.coverage_gap_merge_sec,
    )
    write_json(config.artifacts_dir / "qa_report.json", qa_report.to_dict())
    if qa_report.has_long_omissions:
        logger.warning(
            "Detected %d long omission candidate(s) after transcript import processing",
            len(qa_report.long_omissions),
        )
    write_retry_trace_artifact(config.artifacts_dir, retry_trace)
    stable_cues = stabilize_cues(cues)
    write_json(
        config.artifacts_dir / "final_assembly.json",
        build_final_assembly_report(
            input_cues=cues,
            output_cues=stable_cues,
            source_duration_sec=duration_sec,
        ).to_dict(),
    )
    config.output_path.write_text(format_srt(stable_cues), encoding="utf-8")
    return stable_cues


def write_retry_trace_artifact(
    artifacts_dir: Path,
    retry_trace: list[dict[str, Any]],
) -> None:
    write_json(
        artifacts_dir / "retry_trace.json",
        {
            "schema": "segmentwright.retry_trace.v1",
            "events": retry_trace,
        },
    )


def run_vad_stage(
    *,
    config: AppConfig,
    normalized_wav: Path,
    lifecycle: ModelLifecycleRecorder,
) -> VadResult:
    stage_id = "vad:generation-none"
    backend = FireRedVadBackend(
        model_dir=config.firered_vad_model_dir,
        device=resolve_vad_device(config.device),
    )
    external_log_path = config.artifacts_dir / "external_logs" / "vad.log"
    lifecycle.record(
        stage_id=stage_id,
        generation=None,
        role="vad",
        backend="fireredvad",
        model_name=str(config.firered_vad_model_dir),
        device=resolve_vad_device(config.device),
        event="load_start",
        local=True,
    )
    try:
        with capture_external_output(enabled=not config.verbose, path=external_log_path):
            backend.load()
        lifecycle.record(
            stage_id=stage_id,
            generation=None,
            role="vad",
            backend="fireredvad",
            model_name=str(config.firered_vad_model_dir),
            device=resolve_vad_device(config.device),
            event="load_end",
            local=True,
        )
        with capture_external_output(enabled=not config.verbose, path=external_log_path):
            return backend.detect(normalized_wav)
    except Exception as exc:
        lifecycle.record(
            stage_id=stage_id,
            generation=None,
            role="vad",
            backend="fireredvad",
            model_name=str(config.firered_vad_model_dir),
            device=resolve_vad_device(config.device),
            event="error",
            local=True,
            details={"error": str(exc)},
        )
        raise
    finally:
        lifecycle.record(
            stage_id=stage_id,
            generation=None,
            role="vad",
            backend="fireredvad",
            model_name=str(config.firered_vad_model_dir),
            device=resolve_vad_device(config.device),
            event="unload_start",
            local=True,
        )
        with capture_external_output(enabled=not config.verbose, path=external_log_path):
            backend.close()
        lifecycle.record(
            stage_id=stage_id,
            generation=None,
            role="vad",
            backend="fireredvad",
            model_name=str(config.firered_vad_model_dir),
            device=resolve_vad_device(config.device),
            event="unload_end",
            local=True,
        )


def process_chunk_generations(
    *,
    config: AppConfig,
    normalized_wav: Path,
    initial_chunks: list[PlannedChunk],
    speech_segments: list[VadSegment],
    source_id: str,
    prompt_spec: PromptSpec,
    cache: TranscriptCache,
    lifecycle: ModelLifecycleRecorder,
    reporter: ConsoleReporter,
) -> tuple[list[ProcessedChunk], list[dict[str, Any]], OllamaRuntimeStatus | None]:
    accepted: list[ProcessedChunk] = []
    trace: list[dict[str, Any]] = []
    current_generation = initial_chunks
    generation = 0
    budget_state = RetryBudgetState(limit=config.max_retry_work_items)
    ollama_status: OllamaRuntimeStatus | None = None
    while current_generation:
        transcribed = run_transcription_stage(
            config=config,
            generation=generation,
            chunks=current_generation,
            source_id=source_id,
            prompt_spec=prompt_spec,
            cache=cache,
            lifecycle=lifecycle,
            reporter=reporter,
        )
        segmented, generation_ollama_status = run_segmentation_stage(
            config=config,
            generation=generation,
            transcribed=transcribed,
            lifecycle=lifecycle,
            reporter=reporter,
        )
        if generation_ollama_status is not None:
            ollama_status = generation_ollama_status
            write_json(config.artifacts_dir / "ollama.json", ollama_status.to_dict())
        aligned = run_alignment_stage(
            config=config,
            generation=generation,
            segmented=segmented,
            speech_segments=speech_segments,
            lifecycle=lifecycle,
            reporter=reporter,
        )
        current_generation, generation_trace = plan_retries_for_generation(
            config=config,
            generation=generation,
            normalized_wav=normalized_wav,
            source_id=source_id,
            speech_segments=speech_segments,
            processed=aligned,
            budget_state=budget_state,
            reporter=reporter,
        )
        trace.extend(generation_trace)
        write_retry_trace_artifact(config.artifacts_dir, trace)
        accepted.extend(
            item for item in aligned if not was_superseded_by_retry(item, generation_trace)
        )
        generation += 1
    return (
        sorted(accepted, key=lambda item: (item.chunk.start_sec, item.chunk.end_sec)),
        trace,
        ollama_status,
    )


def pipeline_pass_label(generation: int) -> str:
    if generation == 0:
        return "pass 1"
    return f"retry pass {generation}"


def run_transcription_stage(
    *,
    config: AppConfig,
    generation: int,
    chunks: list[PlannedChunk],
    source_id: str,
    prompt_spec: PromptSpec,
    cache: TranscriptCache,
    lifecycle: ModelLifecycleRecorder,
    reporter: ConsoleReporter,
) -> list[TranscribedChunk]:
    if not chunks:
        return []
    stage_id = f"transcription:generation-{generation}"
    stage_name = f"Transcription {pipeline_pass_label(generation)}"
    reporter.stage_start(stage_name, details=format_count(len(chunks), "chunk"))
    lifecycle.record(
        stage_id=stage_id,
        generation=generation,
        role="transcription",
        backend="gemini",
        model_name=config.gemini_model,
        device=None,
        event="load_start",
        local=False,
    )
    transcriber: GeminiTranscriber | None = None
    results: list[TranscribedChunk] = []
    success = False
    try:
        api_key = load_gemini_api_key(config.env_file)
        transcriber = GeminiTranscriber(
            api_key=api_key,
            model=config.gemini_model,
            prompt=prompt_spec.text,
            response_format=config.response_format,
            thinking_level=config.gemini_thinking_level,
            temperature=config.gemini_temperature,
        )
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="transcription",
            backend="gemini",
            model_name=config.gemini_model,
            device=None,
            event="load_end",
            local=False,
        )
        for index, chunk in enumerate(chunks, start=1):
            transcription = transcribe_one_chunk(
                config,
                chunk,
                source_id,
                transcriber,
                cache,
                prompt_spec=prompt_spec,
            )
            results.append(TranscribedChunk(chunk=chunk, transcription=transcription))
            source = "cache" if transcription.metadata.get("cache_hit") else "api"
            reporter.stage_progress(
                stage_name,
                index,
                len(chunks),
                details=f"id={chunk.chunk_id}, source={source}",
            )
        success = True
    except Exception as exc:
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="transcription",
            backend="gemini",
            model_name=config.gemini_model,
            device=None,
            event="error",
            local=False,
            details={"error": str(exc)},
        )
        raise
    finally:
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="transcription",
            backend="gemini",
            model_name=config.gemini_model,
            device=None,
            event="unload_start",
            local=False,
        )
        if transcriber is not None:
            transcriber.close()
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="transcription",
            backend="gemini",
            model_name=config.gemini_model,
            device=None,
            event="unload_end",
            local=False,
        )
    if success:
        reporter.stage_end(stage_name, details=format_count(len(results), "chunk"))
    return results


def run_segmentation_stage(
    *,
    config: AppConfig,
    generation: int,
    transcribed: list[TranscribedChunk],
    lifecycle: ModelLifecycleRecorder,
    reporter: ConsoleReporter,
) -> tuple[list[SegmentedChunk], OllamaRuntimeStatus | None]:
    if not transcribed:
        return [], None
    stage_id = f"resegmentation:generation-{generation}"
    stage_name = f"Segmentation {pipeline_pass_label(generation)}"
    reporter.stage_start(
        stage_name,
        details=f"{config.segmentation}, {format_count(len(transcribed), 'chunk')}",
    )
    if config.segmentation != "resegmentation":
        segmented = []
        for index, item in enumerate(transcribed, start=1):
            segmented.append(
                SegmentedChunk(
                    chunk=item.chunk,
                    transcription=item.transcription,
                    alignment_lines=item.transcription.transcript.lines,
                    segmentation_report=direct_segmentation_report(
                        config=config,
                        chunk=item.chunk,
                        transcription=item.transcription,
                    ),
                )
            )
            reporter.stage_progress(
                stage_name,
                index,
                len(transcribed),
                details=f"id={item.chunk.chunk_id}",
            )
        reporter.stage_end(stage_name, details=format_count(len(segmented), "direct chunk"))
        return segmented, None

    if all(is_empty_transcript(item.transcription) for item in transcribed):
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="resegmentation",
            backend="ollama",
            model_name=config.resegmentation_model,
            device=None,
            event="skipped",
            local=True,
            details={
                "reason": "all_transcripts_empty",
                "chunk_count": len(transcribed),
            },
        )
        segmented = [
            SegmentedChunk(
                chunk=item.chunk,
                transcription=item.transcription,
                alignment_lines=[],
                segmentation_report=empty_transcript_segmentation_report(
                    config=config,
                    chunk=item.chunk,
                    transcription=item.transcription,
                ),
            )
            for item in transcribed
        ]
        reporter.stage_end(stage_name, status="skipped", details="all transcripts empty")
        return segmented, None

    lifecycle.record(
        stage_id=stage_id,
        generation=generation,
        role="resegmentation",
        backend="ollama",
        model_name=config.resegmentation_model,
        device=None,
        event="load_start",
        local=True,
    )
    manager = ManagedOllamaServer(
        config=build_ollama_runtime_config(config),
        log_dir=config.artifacts_dir / "ollama",
    )
    status: OllamaRuntimeStatus | None = None
    unload_start_recorded = False
    try:
        with manager as status:
            lifecycle.record(
                stage_id=stage_id,
                generation=generation,
                role="resegmentation",
                backend="ollama",
                model_name=config.resegmentation_model,
                device=None,
                event="load_end",
                local=True,
                details={
                    "started_by_segmentwright": status.started_by_segmentwright,
                    "existing_server_used": not status.started_by_segmentwright,
                },
            )
            segmented = []
            for index, item in enumerate(transcribed, start=1):
                lines, report = prepare_alignment_lines(
                    config=config,
                    chunk=item.chunk,
                    transcription=item.transcription,
                )
                segmented.append(
                    SegmentedChunk(
                        chunk=item.chunk,
                        transcription=item.transcription,
                        alignment_lines=lines,
                        segmentation_report=report,
                    )
                )
                reporter.stage_progress(
                    stage_name,
                    index,
                    len(transcribed),
                    details=f"id={item.chunk.chunk_id}",
                )
            lifecycle.record(
                stage_id=stage_id,
                generation=generation,
                role="resegmentation",
                backend="ollama",
                model_name=config.resegmentation_model,
                device=None,
                event="unload_start",
                local=True,
                details={
                    "started_by_segmentwright": status.started_by_segmentwright,
                    "existing_server_used": not status.started_by_segmentwright,
                },
            )
            unload_start_recorded = True
            reporter.stage_end(stage_name, details=format_count(len(segmented), "chunk"))
            return segmented, status
    except Exception as exc:
        effective_status = status or manager.status
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="resegmentation",
            backend="ollama",
            model_name=config.resegmentation_model,
            device=None,
            event="error",
            local=True,
            details={
                "error": str(exc),
                "started_by_segmentwright": effective_status.started_by_segmentwright,
                "existing_server_used": (
                    effective_status.server_available
                    and not effective_status.started_by_segmentwright
                ),
                "model_available": effective_status.model_available,
                "model_unloaded": effective_status.model_unloaded,
            },
        )
        raise
    finally:
        effective_status = status or manager.status
        if not unload_start_recorded:
            lifecycle.record(
                stage_id=stage_id,
                generation=generation,
                role="resegmentation",
                backend="ollama",
                model_name=config.resegmentation_model,
                device=None,
                event="unload_start",
                local=True,
                details={
                    "started_by_segmentwright": effective_status.started_by_segmentwright,
                    "existing_server_used": (
                        effective_status.server_available
                        and not effective_status.started_by_segmentwright
                    ),
                },
            )
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="resegmentation",
            backend="ollama",
            model_name=config.resegmentation_model,
            device=None,
            event="unload_end",
            local=True,
            details={
                "model_unloaded": effective_status.model_unloaded,
                "started_by_segmentwright": effective_status.started_by_segmentwright,
                "existing_server_used": (
                    effective_status.server_available
                    and not effective_status.started_by_segmentwright
                ),
            },
        )


def run_alignment_stage(
    *,
    config: AppConfig,
    generation: int,
    segmented: list[SegmentedChunk],
    speech_segments: list[VadSegment],
    lifecycle: ModelLifecycleRecorder,
    reporter: ConsoleReporter,
) -> list[ProcessedChunk]:
    if not segmented:
        return []
    stage_id = f"alignment:generation-{generation}"
    stage_name = f"Alignment {pipeline_pass_label(generation)}"
    alignment_model = resolve_alignment_model(config)
    reporter.stage_start(
        stage_name,
        details=f"{alignment_model}, {format_count(len(segmented), 'chunk')}",
    )
    if all(is_empty_segmented_chunk(item) for item in segmented):
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="forced_alignment",
            backend="parakeet",
            model_name=alignment_model,
            device=config.device,
            event="skipped",
            local=True,
            details={
                "reason": "all_transcripts_empty",
                "language": config.language,
                "chunk_count": len(segmented),
            },
        )
        processed = [
            align_empty_segmented_chunk(
                config=config,
                segmented=item,
                speech_segments=speech_segments,
            )
            for item in segmented
        ]
        reporter.stage_end(stage_name, status="skipped", details="all transcripts empty")
        return processed

    aligner = ParakeetAlignerBackend(
        device=config.device,
        model_name=alignment_model,
        language=config.language,
    )
    lifecycle.record(
        stage_id=stage_id,
        generation=generation,
        role="forced_alignment",
        backend="parakeet",
        model_name=alignment_model,
        device=config.device,
        event="load_start",
        local=True,
        details={"language": config.language},
    )
    processed: list[ProcessedChunk] = []
    success = False
    external_log_path = config.artifacts_dir / "external_logs" / "alignment.log"
    try:
        with capture_external_output(enabled=not config.verbose, path=external_log_path):
            aligner.load()
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="forced_alignment",
            backend="parakeet",
            model_name=alignment_model,
            device=config.device,
            event="load_end",
            local=True,
            details={"language": config.language},
        )
        for index, item in enumerate(segmented, start=1):
            if is_empty_segmented_chunk(item):
                processed_item = align_empty_segmented_chunk(
                    config=config,
                    segmented=item,
                    speech_segments=speech_segments,
                )
            else:
                with capture_external_output(enabled=not config.verbose, path=external_log_path):
                    processed_item = align_segmented_chunk(
                        config=config,
                        segmented=item,
                        speech_segments=speech_segments,
                        aligner=aligner,
                    )
            processed.append(processed_item)
            reporter.stage_progress(
                stage_name,
                index,
                len(segmented),
                details=f"id={item.chunk.chunk_id}, lines={len(processed_item.line_spans)}",
            )
        success = True
    except Exception as exc:
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="forced_alignment",
            backend="parakeet",
            model_name=alignment_model,
            device=config.device,
            event="error",
            local=True,
            details={"error": str(exc), "language": config.language},
        )
        raise
    finally:
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="forced_alignment",
            backend="parakeet",
            model_name=alignment_model,
            device=config.device,
            event="unload_start",
            local=True,
            details={"language": config.language},
        )
        with capture_external_output(enabled=not config.verbose, path=external_log_path):
            aligner.close()
        lifecycle.record(
            stage_id=stage_id,
            generation=generation,
            role="forced_alignment",
            backend="parakeet",
            model_name=alignment_model,
            device=config.device,
            event="unload_end",
            local=True,
            details={"language": config.language},
        )
    if success:
        reporter.stage_end(stage_name, details=format_count(len(processed), "chunk"))
    return processed


def plan_retries_for_generation(
    *,
    config: AppConfig,
    generation: int,
    normalized_wav: Path,
    source_id: str,
    speech_segments: list[VadSegment],
    processed: list[ProcessedChunk],
    budget_state: RetryBudgetState,
    reporter: ConsoleReporter,
) -> tuple[list[PlannedChunk], list[dict[str, Any]]]:
    next_generation: list[PlannedChunk] = []
    trace: list[dict[str, Any]] = []
    requests: list[RetrySplitRequest] = []
    requested_parent_count = 0
    for aligned in processed:
        if not aligned.qa_report.has_long_omissions:
            continue
        requested_parent_count += 1

        request = maybe_split_for_retry(
            config=config,
            chunk=aligned.chunk,
            speech_segments=speech_segments,
            source_id=source_id,
            omission=aligned.qa_report.long_omissions[0],
        )
        if isinstance(request, RetryUnavailable):
            trace.append(
                {
                    "event": "retry_not_available",
                    "generation": generation,
                    "chunk_id": aligned.chunk.chunk_id,
                    "depth": aligned.chunk.depth,
                    "retry_count": aligned.chunk.retry_count,
                    "reason": request.reason,
                    "long_omissions": [
                        asdict(omission) for omission in aligned.qa_report.long_omissions
                    ],
                }
            )
            continue
        requests.append(request)

    requests.sort(
        key=lambda request: (
            -request.omission.duration_sec,
            request.chunk.start_sec,
            request.chunk.chunk_id,
        )
    )
    split_parent_count = 0
    skipped_parent_count = 0
    generated_child_count = 0
    for request in requests:
        child_count = len(request.children)
        if not budget_state.reserve_split(child_count):
            skipped_parent_count += 1
            trace.append(
                {
                    "event": "retry_skipped_budget",
                    "generation": generation,
                    "chunk_id": request.chunk.chunk_id,
                    "parent_chunk_id": request.chunk.chunk_id,
                    "depth": request.chunk.depth,
                    "retry_count": request.chunk.retry_count,
                    "omission": asdict(request.omission),
                    "required_retry_items": child_count,
                    "retry_budget_used": budget_state.generated_children,
                    "retry_budget_limit": budget_state.limit,
                }
            )
            continue

        children = materialize_retry_children(
            config=config,
            normalized_wav=normalized_wav,
            children=request.children,
        )
        split_parent_count += 1
        generated_child_count += child_count
        trace.append(
            {
                "event": "retry_split",
                "generation": generation,
                "next_generation": generation + 1,
                "parent_chunk_id": request.chunk.chunk_id,
                "split_time_sec": request.decision.split_time_sec,
                "split_source": request.decision.split_source,
                "split_quality": request.decision.split_quality,
                "omission": asdict(request.omission),
                "children": [serializable(child) for child in children],
            }
        )
        next_generation.extend(children)

    if processed and requested_parent_count > len(processed) / 2:
        reporter.warning(
            f"{requested_parent_count} of {len(processed)} chunks in generation {generation} "
            "requested retry; this may indicate a systemic model, prompt, response-format, "
            "or alignment problem."
        )
    if skipped_parent_count:
        details = (
            f" See {config.artifacts_dir / 'retry_trace.json'}."
            if config.artifacts_explicit
            else ""
        )
        reporter.warning(
            f"Retry budget selected {split_parent_count} parent(s) and skipped "
            f"{skipped_parent_count} parent(s) in generation {generation}." + details
        )

    trace.append(
        {
            "event": "generation_summary",
            "generation": generation,
            "processed_items": len(processed),
            "retry_requested_parents": requested_parent_count,
            "retry_split_parents": split_parent_count,
            "retry_generated_children": generated_child_count,
            "retry_skipped_budget_parents": skipped_parent_count,
            "retry_budget_used": budget_state.generated_children,
            "retry_budget_limit": budget_state.limit,
        }
    )
    next_generation.sort(key=lambda chunk: (chunk.start_sec, chunk.end_sec, chunk.chunk_id))
    return next_generation, trace


def was_superseded_by_retry(
    processed: ProcessedChunk,
    retry_trace: list[dict[str, Any]],
) -> bool:
    return any(
        event.get("event") == "retry_split"
        and event.get("parent_chunk_id") == processed.chunk.chunk_id
        for event in retry_trace
    )


def transcribe_one_chunk(
    config: AppConfig,
    chunk: PlannedChunk,
    source_id: str,
    transcriber: GeminiTranscriber,
    cache: TranscriptCache,
    *,
    prompt_spec: PromptSpec,
) -> GeminiTranscriptionResult:
    if chunk.audio_path is None:
        raise ValueError(f"Chunk has no audio path: {chunk.chunk_id}")
    audio_hash = hash_file(chunk.audio_path)
    key = TranscriptCacheKey(
        source_id=source_id,
        chunk_start_sec=chunk.start_sec,
        chunk_end_sec=chunk.end_sec,
        chunk_audio_hash=audio_hash,
        gemini_model=config.gemini_model,
        prompt_preset=prompt_spec.preset,
        prompt_hash=prompt_spec.sha256,
        response_format=config.response_format,
        segmentation=config.segmentation,
        preset=config.preset,
        language=config.language,
        gemini_thinking_level=config.gemini_thinking_level,
        gemini_temperature=config.gemini_temperature,
    )
    cached = None
    if config.cache_enabled and not config.force_gemini:
        cached = cache.load_entry(key)
    if cached is not None:
        logger.info("Using cached transcript for chunk %s", chunk.chunk_id)
        cached_call_metadata = (
            cached.metadata.get("gemini_call_metadata", {})
            if isinstance(cached.metadata.get("gemini_call_metadata"), dict)
            else {}
        )
        metadata = {
            "schema": "segmentwright.gemini_call_metadata.v1",
            "cache_hit": True,
            "status": "success",
            "cache_path": str(cached.path),
            "preset": config.preset,
            "language": config.language,
            "prompt": {
                "source": prompt_spec.source,
                "preset": prompt_spec.preset,
                "language": prompt_spec.language,
                "file": str(prompt_spec.file) if prompt_spec.file else None,
                "sha256": prompt_spec.sha256,
                "character_count": len(prompt_spec.text),
            },
            "segmentation": config.segmentation,
            "chunk": chunk_metadata(chunk),
            "cache_key": asdict(key),
            "cached_metadata": cached.metadata,
            "response": cached_call_metadata.get("response"),
            "finish_reason": cached_call_metadata.get("finish_reason"),
            "finish_reason_class": cached_call_metadata.get("finish_reason_class"),
            "usage_metadata": cached_call_metadata.get("usage_metadata"),
            "token_usage": cached_call_metadata.get("token_usage"),
            "empty_acceptance_reason": cached_call_metadata.get(
                "empty_acceptance_reason"
            ),
            "post_transcription_action": cached_call_metadata.get(
                "post_transcription_action"
            ),
        }
        metadata = enrich_gemini_chunk_metadata(
            metadata=metadata,
            config=config,
            prompt_spec=prompt_spec,
            chunk=chunk,
            cache_key=key,
            transcript=cached.transcript,
        )
        write_json(gemini_call_metadata_path(config.artifacts_dir, chunk), metadata)
        return GeminiTranscriptionResult(transcript=cached.transcript, metadata=metadata)

    logger.info("Calling Gemini for chunk %s", chunk.chunk_id)
    try:
        transcription = transcriber.transcribe(chunk.audio_path)
    except GeminiTranscriptionError as exc:
        metadata = enrich_gemini_chunk_metadata(
            metadata=exc.metadata,
            config=config,
            prompt_spec=prompt_spec,
            chunk=chunk,
            cache_key=key,
            transcript=None,
        )
        write_json(gemini_call_metadata_path(config.artifacts_dir, chunk), metadata)
        raise

    metadata = enrich_gemini_chunk_metadata(
        metadata=transcription.metadata,
        config=config,
        prompt_spec=prompt_spec,
        chunk=chunk,
        cache_key=key,
        transcript=transcription.transcript,
    )
    write_json(gemini_call_metadata_path(config.artifacts_dir, chunk), metadata)
    transcription = GeminiTranscriptionResult(
        transcript=transcription.transcript,
        metadata=metadata,
    )
    if config.cache_enabled:
        cache.save(
            key,
            transcription.transcript,
            {
                "chunk_id": chunk.chunk_id,
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
                "depth": chunk.depth,
                "retry_count": chunk.retry_count,
                "gemini_call_metadata": metadata,
            },
        )
    return transcription


def enrich_gemini_chunk_metadata(
    *,
    metadata: dict[str, Any],
    config: AppConfig,
    prompt_spec: PromptSpec,
    chunk: PlannedChunk,
    cache_key: TranscriptCacheKey,
    transcript: Transcript | None,
) -> dict[str, Any]:
    response = metadata.get("response") if isinstance(metadata.get("response"), dict) else {}
    finish_reasons = response.get("finish_reasons")
    finish_reason = None
    if isinstance(finish_reasons, list) and finish_reasons:
        finish_reason = finish_reasons[0]
    empty_transcript = len(transcript.lines) == 0 if transcript is not None else False
    enriched = {
        **metadata,
        "status": metadata.get("status", "success" if transcript is not None else "failed"),
        "preset": config.preset,
        "language": config.language,
        "prompt": {
            "source": prompt_spec.source,
            "preset": prompt_spec.preset,
            "language": prompt_spec.language,
            "file": str(prompt_spec.file) if prompt_spec.file else None,
            "sha256": prompt_spec.sha256,
            "character_count": len(prompt_spec.text),
        },
        "segmentation": config.segmentation,
        "chunk": chunk_metadata(chunk),
        "cache_key": asdict(cache_key),
        "finish_reason": metadata.get("finish_reason", finish_reason),
        "finish_reason_class": metadata.get(
            "finish_reason_class",
            response.get("finish_reason_class"),
        ),
    }
    if transcript is not None:
        enriched["empty_transcript"] = metadata.get("empty_transcript", empty_transcript)
        enriched["empty_acceptance_reason"] = metadata.get(
            "empty_acceptance_reason",
            "finish_reason_stop_empty_response" if empty_transcript else None,
        )
        enriched["post_transcription_action"] = metadata.get(
            "post_transcription_action",
            "empty_transcript_noop" if empty_transcript else "normal_pipeline",
        )
        existing_transcript = (
            enriched.get("transcript")
            if isinstance(enriched.get("transcript"), dict)
            else {}
        )
        enriched["transcript"] = {
            **existing_transcript,
            "line_count": len(transcript.lines),
            "raw_character_count": len(transcript.raw_json),
            "empty_transcript": empty_transcript,
        }
    return enriched


def align_chunk(
    *,
    config: AppConfig,
    chunk: PlannedChunk,
    transcription: GeminiTranscriptionResult,
    speech_segments: list[VadSegment],
    aligner: ParakeetAlignerBackend,
) -> ProcessedChunk:
    if chunk.audio_path is None:
        raise ValueError(f"Chunk has no audio path: {chunk.chunk_id}")
    chunk_speech = clip_speech_segments_absolute(
        speech_segments=speech_segments,
        chunk_start_sec=chunk.start_sec,
        chunk_end_sec=chunk.end_sec,
    )
    alignment_lines, segmentation_report = prepare_alignment_lines(
        config=config,
        chunk=chunk,
        transcription=transcription,
    )
    if is_empty_transcript(transcription) and not alignment_lines:
        return align_empty_segmented_chunk(
            config=config,
            segmented=SegmentedChunk(
                chunk=chunk,
                transcription=transcription,
                alignment_lines=alignment_lines,
                segmentation_report=segmentation_report,
            ),
            speech_segments=speech_segments,
        )
    result = aligner.align(
        chunk.audio_path,
        alignment_lines,
        chunk.start_sec,
    )
    guarded = apply_token_gap_guard(result.line_spans, speech_segments=chunk_speech)
    qa_report = detect_long_omissions(
        speech_segments=chunk_speech,
        line_spans=guarded.line_spans,
        threshold_sec=config.long_omission_sec,
        coverage_gap_merge_sec=config.coverage_gap_merge_sec,
    )
    raw_alignment = {
        **result.raw,
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "segmentation": segmentation_report,
        "token_gap_guard": guarded.to_dict(),
        "qa_report": qa_report.to_dict(),
    }
    return ProcessedChunk(
        chunk=chunk,
        transcript=transcription.transcript,
        gemini_call_metadata=transcription.metadata,
        line_spans=guarded.line_spans,
        raw_alignment=raw_alignment,
        qa_report=qa_report,
    )


def align_segmented_chunk(
    *,
    config: AppConfig,
    segmented: SegmentedChunk,
    speech_segments: list[VadSegment],
    aligner: ParakeetAlignerBackend,
) -> ProcessedChunk:
    if is_empty_segmented_chunk(segmented):
        return align_empty_segmented_chunk(
            config=config,
            segmented=segmented,
            speech_segments=speech_segments,
        )

    chunk = segmented.chunk
    if chunk.audio_path is None:
        raise ValueError(f"Chunk has no audio path: {chunk.chunk_id}")
    chunk_speech = clip_speech_segments_absolute(
        speech_segments=speech_segments,
        chunk_start_sec=chunk.start_sec,
        chunk_end_sec=chunk.end_sec,
    )
    result = aligner.align(
        chunk.audio_path,
        segmented.alignment_lines,
        chunk.start_sec,
    )
    guarded = apply_token_gap_guard(result.line_spans, speech_segments=chunk_speech)
    qa_report = detect_long_omissions(
        speech_segments=chunk_speech,
        line_spans=guarded.line_spans,
        threshold_sec=config.long_omission_sec,
        coverage_gap_merge_sec=config.coverage_gap_merge_sec,
    )
    raw_alignment = {
        **result.raw,
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "segmentation": segmented.segmentation_report,
        "token_gap_guard": guarded.to_dict(),
        "qa_report": qa_report.to_dict(),
    }
    return ProcessedChunk(
        chunk=chunk,
        transcript=segmented.transcription.transcript,
        gemini_call_metadata=segmented.transcription.metadata,
        line_spans=guarded.line_spans,
        raw_alignment=raw_alignment,
        qa_report=qa_report,
    )


def align_empty_segmented_chunk(
    *,
    config: AppConfig,
    segmented: SegmentedChunk,
    speech_segments: list[VadSegment],
) -> ProcessedChunk:
    chunk = segmented.chunk
    if chunk.audio_path is None:
        raise ValueError(f"Chunk has no audio path: {chunk.chunk_id}")
    chunk_speech = clip_speech_segments_absolute(
        speech_segments=speech_segments,
        chunk_start_sec=chunk.start_sec,
        chunk_end_sec=chunk.end_sec,
    )
    diagnostic_qa_report = detect_long_omissions(
        speech_segments=chunk_speech,
        line_spans=[],
        threshold_sec=config.long_omission_sec,
        coverage_gap_merge_sec=config.coverage_gap_merge_sec,
    )
    # Empty transcripts accepted after normal model completion must not trigger
    # VAD-based retry. VAD disagreement is preserved as diagnostics only.
    qa_report = replace(diagnostic_qa_report, long_omissions=[])
    vad_speech_duration_sec = sum(segment.duration_sec for segment in chunk_speech)
    warnings = (
        ["empty_transcript_with_vad_activity"] if vad_speech_duration_sec > 0 else []
    )
    raw_alignment = {
        "schema": "segmentwright.empty_alignment_noop.v1",
        "model": resolve_alignment_model(config),
        "language": config.language,
        "audio_path": str(chunk.audio_path),
        "offset_sec": chunk.start_sec,
        "line_count": 0,
        "failed": False,
        "skipped": True,
        "skip_reason": "empty_transcript",
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "segmentation": segmented.segmentation_report,
        "empty_transcript": {
            "accepted": True,
            "vad_speech_duration_sec": vad_speech_duration_sec,
            "warnings": warnings,
        },
        "vad_diagnostic_qa_report": diagnostic_qa_report.to_dict(),
        "token_gap_guard": apply_token_gap_guard([], speech_segments=chunk_speech).to_dict(),
        "qa_report": qa_report.to_dict(),
        "qa_suppressed_reason": "empty_transcript_accepted_no_vad_retry",
    }
    return ProcessedChunk(
        chunk=chunk,
        transcript=segmented.transcription.transcript,
        gemini_call_metadata=segmented.transcription.metadata,
        line_spans=[],
        raw_alignment=raw_alignment,
        qa_report=qa_report,
    )


def is_empty_transcript(transcription: GeminiTranscriptionResult) -> bool:
    return len(transcription.transcript.lines) == 0


def is_empty_segmented_chunk(segmented: SegmentedChunk) -> bool:
    return is_empty_transcript(segmented.transcription) and not segmented.alignment_lines


def direct_segmentation_report(
    *,
    config: AppConfig,
    chunk: PlannedChunk,
    transcription: GeminiTranscriptionResult,
) -> dict[str, Any]:
    if is_empty_transcript(transcription):
        return empty_transcript_segmentation_report(
            config=config,
            chunk=chunk,
            transcription=transcription,
        )
    warnings = []
    if config.response_format == "plain_text" and len(transcription.transcript.lines) <= 1:
        warnings.append("plain_text_direct_alignment_has_single_input_line")
        logger.warning(
            "Direct alignment with one plain-text line can produce a coarse subtitle cue "
            "for chunk %s",
            chunk.chunk_id,
        )
    return {
        "schema": "segmentwright.segmentation_report.v1",
        "mode": "direct",
        "input_line_count": len(transcription.transcript.lines),
        "output_line_count": len(transcription.transcript.lines),
        "warnings": warnings,
    }


def empty_transcript_segmentation_report(
    *,
    config: AppConfig,
    chunk: PlannedChunk,
    transcription: GeminiTranscriptionResult,
) -> dict[str, Any]:
    return {
        "schema": "segmentwright.segmentation_report.v1",
        "mode": "empty_transcript_noop",
        "requested_mode": config.segmentation,
        "input_line_count": len(transcription.transcript.lines),
        "output_line_count": 0,
        "warnings": [],
        "empty_transcript": True,
        "empty_acceptance_reason": transcription.metadata.get(
            "empty_acceptance_reason",
            "finish_reason_stop_empty_response",
        ),
        "chunk_id": chunk.chunk_id,
    }


def prepare_alignment_lines(
    *,
    config: AppConfig,
    chunk: PlannedChunk,
    transcription: GeminiTranscriptionResult,
) -> tuple[list[str], dict[str, Any]]:
    if is_empty_transcript(transcription):
        return [], empty_transcript_segmentation_report(
            config=config,
            chunk=chunk,
            transcription=transcription,
        )
    if config.segmentation == "direct":
        return transcription.transcript.lines, direct_segmentation_report(
            config=config,
            chunk=chunk,
            transcription=transcription,
        )
    if config.segmentation != "resegmentation":
        raise ValueError(f"Unknown segmentation mode: {config.segmentation}")

    result = resegment_transcript_for_chunk(
        config=config,
        chunk=chunk,
        transcript=transcription.transcript,
    )
    artifact_dir = resegmentation_artifact_dir(config.artifacts_dir, chunk)
    write_resegmentation_artifacts(output_dir=artifact_dir, result=result)
    return result.lines, {
        "schema": "segmentwright.segmentation_report.v1",
        "mode": "resegmentation",
        "artifact_dir": str(artifact_dir),
        "prompt": config.resegmentation_prompt,
        "preserve_line_breaks": config.resegmentation_preserve_line_breaks,
        "input_line_count": len(transcription.transcript.lines),
        "input_character_count": len(result.source_text),
        "source_text_construction": {
            "source_line_count": len(transcription.transcript.lines),
            "line_joiner": "\\n",
            "collapsed_from_transcript_lines": len(transcription.transcript.lines) > 1,
        },
        "suggested_line_count": len(result.suggestion.lines),
        "projected_line_count": len(result.projected_segments),
        "text_edit_report": result.text_edit_report.to_dict(),
        "projection_report": result.projection_report.to_dict(),
        "quality_gate": (result.suggestion.raw or {}).get("quality_gate"),
    }


def resegment_transcript_for_chunk(
    *,
    config: AppConfig,
    chunk: PlannedChunk,
    transcript: Transcript,
) -> ResegmentationResult:
    source_text = transcript.text.strip()
    quality_config = ResegmentationQualityConfig()
    attempts: list[dict[str, Any]] = []
    last_result: ResegmentationResult | None = None
    ollama_config = OllamaSegmentationConfig(
        model=config.resegmentation_model,
        ollama_url=config.resegmentation_ollama_url,
        prompt_preset=config.resegmentation_prompt,
    )
    max_attempts = config.resegmentation_max_retries + 1
    for attempt_index in range(1, max_attempts + 1):
        suggestion = call_ollama_resegmentation(source_text, config=ollama_config)
        result = project_suggestion_to_source(
            source_text,
            suggestion.lines,
            raw_suggestion=suggestion.raw,
            preserve_line_breaks=config.resegmentation_preserve_line_breaks,
        )
        quality_report = assess_resegmentation_quality(result, config=quality_config)
        attempts.append(
            {
                "attempt": attempt_index,
                "accepted": quality_report.accepted,
                "line_count": len(suggestion.lines),
                "quality_report": quality_report.to_dict(),
            }
        )
        last_result = result
        if quality_report.accepted:
            if quality_report.warnings:
                logger.warning(
                    "Local resegmentation quality gate warning for chunk %s attempt %d/%d: %s",
                    chunk.chunk_id,
                    attempt_index,
                    max_attempts,
                    ", ".join(quality_report.warnings),
                )
            return attach_resegmentation_quality_metadata(
                result,
                attempts=attempts,
                accepted_attempt=attempt_index,
                fallback_used=False,
                max_retries=config.resegmentation_max_retries,
            )

        logger.warning(
            "Local resegmentation quality gate rejected chunk %s attempt %d/%d: %s",
            chunk.chunk_id,
            attempt_index,
            max_attempts,
            ", ".join(quality_report.reasons),
        )

    fallback_lines = fallback_segment_source_text(
        source_text,
        max_non_ws_chars=quality_config.max_projected_segment_non_ws_chars // 2,
    )
    fallback_raw = {
        "schema": "segmentwright.resegmentation_fallback.v1",
        "request": {
            "model": config.resegmentation_model,
            "prompt_preset": config.resegmentation_prompt,
            "fallback_strategy": "source_text_punctuation_and_character_budget",
        },
        "quality_gate": {
            "schema": "segmentwright.resegmentation_quality_gate.v1",
            "max_retries": config.resegmentation_max_retries,
            "attempts": attempts,
            "accepted_attempt": None,
            "fallback_used": True,
            "fallback_line_count": len(fallback_lines),
        },
        "last_failed_result": resegmentation_result_summary(last_result),
    }
    fallback_result = project_suggestion_to_source(
        source_text,
        fallback_lines,
        raw_suggestion=fallback_raw,
        preserve_line_breaks=config.resegmentation_preserve_line_breaks,
    )
    return attach_resegmentation_quality_metadata(
        fallback_result,
        attempts=attempts,
        accepted_attempt=None,
        fallback_used=True,
        max_retries=config.resegmentation_max_retries,
    )


def attach_resegmentation_quality_metadata(
    result: ResegmentationResult,
    *,
    attempts: list[dict[str, Any]],
    accepted_attempt: int | None,
    fallback_used: bool,
    max_retries: int,
) -> ResegmentationResult:
    raw = dict(result.suggestion.raw or {})
    raw["quality_gate"] = {
        "schema": "segmentwright.resegmentation_quality_gate.v1",
        "max_retries": max_retries,
        "attempts": attempts,
        "accepted_attempt": accepted_attempt,
        "fallback_used": fallback_used,
    }
    return ResegmentationResult(
        source_text=result.source_text,
        suggestion=ResegmentationSuggestion(lines=result.suggestion.lines, raw=raw),
        text_edit_report=result.text_edit_report,
        projection_report=result.projection_report,
        projected_segments=result.projected_segments,
    )


def resegmentation_result_summary(result: ResegmentationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "suggested_line_count": len(result.suggestion.lines),
        "projected_line_count": len(result.projected_segments),
        "text_edit_report": result.text_edit_report.to_dict(),
        "projection_report": result.projection_report.to_dict(),
    }


def maybe_split_for_retry(
    *,
    config: AppConfig,
    chunk: PlannedChunk,
    speech_segments: list[VadSegment],
    source_id: str,
    omission: LongOmission,
) -> RetrySplitRequest | RetryUnavailable:
    if chunk.retry_count >= config.max_retries:
        return RetryUnavailable(reason="max_retries_reached")
    if chunk.depth >= config.max_split_depth:
        return RetryUnavailable(reason="max_split_depth_reached")
    decision = plan_retry_split(
        parent=chunk,
        omission=omission,
        speech_segments=speech_segments,
        min_split_sec=config.min_retry_split_sec,
    )
    if decision is None:
        return RetryUnavailable(reason="parent_too_short_to_split")
    children = build_retry_children(
        parent=chunk,
        decision=decision,
        source_id=source_id,
        sample_rate=16000,
    )
    return RetrySplitRequest(
        chunk=chunk,
        omission=omission,
        decision=decision,
        children=tuple(children),
    )


def materialize_retry_children(
    *,
    config: AppConfig,
    normalized_wav: Path,
    children: tuple[PlannedChunk, ...],
) -> list[PlannedChunk]:
    retry_dir = config.artifacts_dir / "retry_chunks"
    materialized_children: list[PlannedChunk] = []
    for child in children:
        output_path = retry_dir / f"audio_{safe_chunk_id(child.chunk_id)}.wav"
        extract_wav_segment(
            normalized_wav,
            start_sec=child.start_sec,
            end_sec=child.end_sec,
            output_path=output_path,
        )
        materialized_children.append(replace(child, audio_path=output_path))
    return materialized_children


def gemini_call_metadata_path(artifacts_dir: Path, chunk: PlannedChunk) -> Path:
    return artifacts_dir / "gemini_calls" / f"chunk_{safe_chunk_id(chunk.chunk_id)}.json"


def external_transcript_chunk_metadata_path(
    artifacts_dir: Path,
    chunk: PlannedChunk,
) -> Path:
    return artifacts_dir / "external_transcript" / f"chunk_{safe_chunk_id(chunk.chunk_id)}.json"


def resegmentation_artifact_dir(artifacts_dir: Path, chunk: PlannedChunk) -> Path:
    return artifacts_dir / "resegmentation" / f"chunk_{safe_chunk_id(chunk.chunk_id)}"


def build_ollama_runtime_config(config: AppConfig) -> OllamaRuntimeConfig:
    return OllamaRuntimeConfig(
        enabled=config.segmentation == "resegmentation",
        base_url=config.resegmentation_ollama_url,
        model=config.resegmentation_model,
        auto_pull=config.ollama_auto_pull,
        start_timeout_sec=config.ollama_start_timeout_sec,
    )


def build_run_config_report(
    *,
    config: AppConfig,
    prompt_spec: PromptSpec,
    ollama_status: OllamaRuntimeStatus | None = None,
    transcript_document: TranscriptChunksDocument | None = None,
) -> dict[str, Any]:
    preset = PRODUCT_PRESETS[config.preset]
    alignment_model = resolve_alignment_model(config)
    transcript_import = None
    if transcript_document is not None:
        transcript_import = {
            "path": str(transcript_document.path),
            "schema": transcript_document.schema,
            "version": transcript_document.version,
            "language": transcript_document.language,
            "metadata": transcript_document.metadata,
            "chunk_count": len(transcript_document.chunks),
        }
    return {
        "schema": "segmentwright.run_config.v1",
        "run_mode": "external_transcript" if transcript_document is not None else "gemini",
        "preset": config.preset,
        "preset_defaults": asdict(preset),
        "language": config.language,
        "language_explicit": config.language_explicit,
        "gemini_model": config.gemini_model,
        "transcript_import": transcript_import,
        "alignment": {
            "model": alignment_model,
            "model_override": config.alignment_model,
            "language": config.language,
        },
        "gemini_thinking_level": config.gemini_thinking_level,
        "gemini_temperature": config.gemini_temperature,
        "prompt": {
            "source": prompt_spec.source,
            "preset": prompt_spec.preset,
            "language": prompt_spec.language,
            "file": str(prompt_spec.file) if prompt_spec.file else None,
            "sha256": prompt_spec.sha256,
            "character_count": len(prompt_spec.text),
        },
        "response_format": config.response_format,
        "segmentation": config.segmentation,
        "resegmentation": {
            "model": config.resegmentation_model,
            "ollama_url": config.resegmentation_ollama_url,
            "prompt": config.resegmentation_prompt,
            "max_retries": config.resegmentation_max_retries,
            "ollama_server": "auto",
            "preserve_line_breaks": config.resegmentation_preserve_line_breaks,
            "ollama_auto_pull": config.ollama_auto_pull,
            "ollama_start_timeout_sec": config.ollama_start_timeout_sec,
            "ollama_status": ollama_status.to_dict() if ollama_status else None,
        },
        "chunking": {
            "target_chunk_sec": config.target_chunk_sec,
            "max_chunk_sec": config.max_chunk_sec,
            "min_retry_split_sec": config.min_retry_split_sec,
        },
        "retry": {
            "long_omission_sec": config.long_omission_sec,
            "coverage_gap_merge_sec": config.coverage_gap_merge_sec,
            "max_retries": config.max_retries,
            "max_split_depth": config.max_split_depth,
            "max_retry_work_items": config.max_retry_work_items,
            "retry_budget_mode": (
                "unlimited" if config.max_retry_work_items is None else "explicit"
            ),
        },
        "cache": {
            "enabled": config.cache_enabled,
            "force_gemini": config.force_gemini,
            "cache_dir": str(config.resolved_cache_dir),
            "key_fields": [
                "source_id",
                "chunk_start_sec",
                "chunk_end_sec",
                "chunk_audio_hash",
                "gemini_model",
                "prompt_preset",
                "prompt_hash",
                "response_format",
                "segmentation",
                "preset",
                "language",
                "gemini_thinking_level",
                "gemini_temperature",
            ],
        },
    }


def chunk_metadata(chunk: PlannedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "start_sec": chunk.start_sec,
        "end_sec": chunk.end_sec,
        "duration_sec": chunk.duration_sec,
        "depth": chunk.depth,
        "retry_count": chunk.retry_count,
        "original_index": chunk.original_index,
        "audio_path": str(chunk.audio_path) if chunk.audio_path else None,
    }


def load_gemini_call_metadata(artifacts_dir: Path) -> list[dict[str, Any]]:
    calls_dir = artifacts_dir / "gemini_calls"
    if not calls_dir.exists():
        return []
    calls = []
    for path in sorted(calls_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            calls.append({"metadata_path": str(path), **data})
    return calls


def build_token_usage_report(
    *,
    calls: list[dict[str, Any]],
    accepted_chunk_ids: set[str],
) -> dict[str, Any]:
    accepted_calls = [
        call
        for call in calls
        if isinstance(call.get("chunk"), dict)
        and call["chunk"].get("chunk_id") in accepted_chunk_ids
    ]
    api_calls = [call for call in calls if not call.get("cache_hit")]
    cache_hits = [call for call in calls if call.get("cache_hit")]
    return {
        "schema": "segmentwright.token_usage_report.v1",
        "call_count": len(calls),
        "api_call_count": len(api_calls),
        "cache_hit_count": len(cache_hits),
        "accepted_chunk_ids": sorted(accepted_chunk_ids),
        "all_attempts": summarize_token_calls(calls),
        "accepted_chunks": summarize_token_calls(accepted_calls),
        "api_calls": summarize_token_calls(api_calls),
        "cache_hits": summarize_token_calls(cache_hits),
        "calls": [
            {
                "metadata_path": call.get("metadata_path"),
                "chunk": call.get("chunk"),
                "cache_hit": call.get("cache_hit"),
                "model": call.get("model"),
                "model_version": call.get("model_version"),
                "response_id": call.get("response_id"),
                "response_format": (call.get("request") or {}).get("response_format")
                if isinstance(call.get("request"), dict)
                else None,
                "preset": call.get("preset"),
                "prompt": call.get("prompt"),
                "segmentation": call.get("segmentation"),
                "token_usage": call.get("token_usage"),
            }
            for call in calls
        ],
    }


def summarize_token_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    details = {
        "prompt_tokens_by_modality": {},
        "cache_tokens_by_modality": {},
        "candidates_tokens_by_modality": {},
        "tool_use_prompt_tokens_by_modality": {},
    }
    audio_seconds = 0.0
    output_chars = 0
    for call in calls:
        chunk = call.get("chunk") if isinstance(call.get("chunk"), dict) else {}
        audio_seconds += float(chunk.get("duration_sec") or 0.0)
        response = call.get("response") if isinstance(call.get("response"), dict) else {}
        output_chars += int(response.get("text_character_count") or 0)
        token_usage = call.get("token_usage") if isinstance(call.get("token_usage"), dict) else {}
        token_counts = (
            token_usage.get("counts") if isinstance(token_usage.get("counts"), dict) else {}
        )
        for key, value in token_counts.items():
            if isinstance(value, int):
                counts[key] = counts.get(key, 0) + value
        token_details = (
            token_usage.get("details") if isinstance(token_usage.get("details"), dict) else {}
        )
        for detail_key in details:
            merge_int_dict(details[detail_key], token_details.get(detail_key))

    prompt_modalities = details["prompt_tokens_by_modality"]
    counts["text_input_token_count"] = prompt_modalities.get("TEXT", 0)
    counts["audio_input_token_count"] = prompt_modalities.get("AUDIO", 0)
    counts["candidate_text_token_count"] = details["candidates_tokens_by_modality"].get(
        "TEXT", 0
    )
    total = counts.get("total_token_count")
    candidates = counts.get("candidates_token_count")
    return {
        "counts": counts,
        "details": details,
        "audio_seconds": round(audio_seconds, 6),
        "output_text_character_count": output_chars,
        "derived": {
            "total_tokens_per_audio_second": round(total / audio_seconds, 6)
            if total is not None and audio_seconds > 0
            else None,
            "candidate_tokens_per_output_character": round(candidates / output_chars, 6)
            if candidates is not None and output_chars > 0
            else None,
        },
    }


def merge_int_dict(target: dict[str, int], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if isinstance(value, int):
            target[str(key)] = target.get(str(key), 0) + value


def safe_chunk_id(chunk_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in chunk_id)


def attach_audio_paths(chunks: list[PlannedChunk], audio_paths: list[Path]) -> list[PlannedChunk]:
    if len(chunks) != len(audio_paths):
        raise ValueError(
            f"Chunk/audio count mismatch: chunks={len(chunks)} audio={len(audio_paths)}"
        )
    return [
        PlannedChunk(
            **{
                **asdict(chunk),
                "audio_path": audio_path,
            }
        )
        for chunk, audio_path in zip(chunks, audio_paths, strict=True)
    ]


def resolve_vad_device(device: str) -> str:
    # FireRed's API accepts a boolean GPU flag, so auto stays CPU until runtime GPU policy is tuned.
    if device == "cuda":
        return "cuda"
    return "cpu"


def theoretical_max_retry_item_count(
    *,
    initial_chunk_count: int,
    max_retries: int,
    max_split_depth: int,
) -> int:
    effective_depth = min(max_retries, max_split_depth)
    return initial_chunk_count * (2 ** (effective_depth + 1) - 2)


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_retry_run_summary(config: AppConfig) -> dict[str, Any]:
    manifest = load_json_object(config.artifacts_dir / "chunk_manifest.json") or {}
    manifest_chunks = manifest.get("chunks")
    initial_chunk_count = len(manifest_chunks) if isinstance(manifest_chunks, list) else None

    retry_trace = load_json_object(config.artifacts_dir / "retry_trace.json") or {}
    raw_events = retry_trace.get("events")
    events = raw_events if isinstance(raw_events, list) else []
    generation_summaries = [
        event
        for event in events
        if isinstance(event, dict) and event.get("event") == "generation_summary"
    ]
    initial_summary = next(
        (event for event in generation_summaries if event.get("generation") == 0),
        None,
    )
    if initial_summary is not None and isinstance(initial_summary.get("processed_items"), int):
        initial_chunk_count = initial_summary["processed_items"]

    processed_retry_item_count = sum(
        event["processed_items"]
        for event in generation_summaries
        if isinstance(event.get("generation"), int)
        and event["generation"] > 0
        and isinstance(event.get("processed_items"), int)
    )
    retry_split_parent_count = sum(
        1 for event in events if isinstance(event, dict) and event.get("event") == "retry_split"
    )
    retry_skipped_budget_parent_count = sum(
        1
        for event in events
        if isinstance(event, dict) and event.get("event") == "retry_skipped_budget"
    )
    if config.transcript_path is not None:
        theoretical_max = 0
    elif initial_chunk_count is None:
        theoretical_max = None
    else:
        theoretical_max = theoretical_max_retry_item_count(
            initial_chunk_count=initial_chunk_count,
            max_retries=config.max_retries,
            max_split_depth=config.max_split_depth,
        )
    return {
        "initial_chunk_count": initial_chunk_count,
        "processed_retry_item_count": processed_retry_item_count,
        "retry_split_parent_count": retry_split_parent_count,
        "retry_skipped_budget_parent_count": retry_skipped_budget_parent_count,
        "retry_budget_limit": config.max_retry_work_items,
        "retry_budget_exhausted": retry_skipped_budget_parent_count > 0,
        "theoretical_max_retry_item_count": theoretical_max,
    }


def write_run_summary_artifact(
    *,
    config: AppConfig,
    reporter: ConsoleReporter,
    status: str,
    cue_count: int | None,
    mode: str,
    error: BaseException | None = None,
    failed_stage: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": "segmentwright.run_summary.v1",
        "status": status,
        "mode": mode,
        "elapsed_sec": round(reporter.elapsed_sec, 3),
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "artifacts_dir": str(config.artifacts_dir),
        "preset": config.preset,
        "language": config.language,
        "segmentation": config.segmentation,
        "gemini_model": config.gemini_model if config.transcript_path is None else None,
        "alignment_model": resolve_alignment_model(config),
        "cue_count": cue_count,
        "warning_count": reporter.warning_count,
        "current_stage": reporter.current_stage,
        "stages": [record.to_dict() for record in reporter.stage_records],
    }
    payload.update(build_retry_run_summary(config))
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "stage": failed_stage or reporter.current_stage,
        }
    write_json(config.artifacts_dir / "run_summary.json", payload)


def write_error_artifact(
    *,
    config: AppConfig,
    reporter: ConsoleReporter,
    error: BaseException,
    mode: str,
    failed_stage: str | None = None,
) -> None:
    write_json(
        config.artifacts_dir / "errors.json",
        {
            "schema": "segmentwright.errors.v1",
            "mode": mode,
            "stage": failed_stage or reporter.current_stage,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exception(error),
            },
        },
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return value
