from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TranscriptChunksError(ValueError):
    """Raised when an imported transcript file is invalid."""


class TranscriptChunkModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    start: float
    end: float
    text: str
    source: str | None = None
    confidence: float | None = None
    notes: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class TranscriptChunksModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_: str | None = Field(default=None, alias="schema")
    version: int
    language: str | None = None
    metadata: dict[str, Any] | None = None
    chunks: list[TranscriptChunkModel]


@dataclass(frozen=True)
class ImportedTranscriptChunk:
    chunk_id: str
    start_sec: float
    end_sec: float
    text: str
    source: str | None
    confidence: float | None
    notes: dict[str, Any] | None
    metadata: dict[str, Any] | None
    extra: dict[str, Any]

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class TranscriptChunksDocument:
    path: Path
    schema: str | None
    version: int
    language: str | None
    metadata: dict[str, Any]
    chunks: list[ImportedTranscriptChunk]
    raw: dict[str, Any]


@dataclass(frozen=True)
class TranscriptChunksValidation:
    warnings: list[str]
    chunk_warnings: dict[str, list[str]]

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings or self.chunk_warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "segmentwright.transcript_chunks_validation.v1",
            "warnings": self.warnings,
            "chunk_warnings": self.chunk_warnings,
            "has_warnings": self.has_warnings,
        }


def load_transcript_chunks(path: Path) -> TranscriptChunksDocument:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptChunksError(f"Transcript file is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise TranscriptChunksError(f"Could not read transcript file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise TranscriptChunksError("Transcript file must contain a JSON object")

    try:
        model = TranscriptChunksModel.model_validate(raw)
    except ValidationError as exc:
        raise TranscriptChunksError(f"Transcript file schema validation failed: {exc}") from exc

    if model.version != 1:
        raise TranscriptChunksError(f"Unsupported transcript version: {model.version!r}")

    chunks = [
        ImportedTranscriptChunk(
            chunk_id=chunk.id or f"chunk_{index:04d}",
            start_sec=chunk.start,
            end_sec=chunk.end,
            text=chunk.text,
            source=chunk.source,
            confidence=chunk.confidence,
            notes=chunk.notes,
            metadata=chunk.metadata,
            extra=dict(chunk.model_extra or {}),
        )
        for index, chunk in enumerate(model.chunks, start=1)
    ]
    return TranscriptChunksDocument(
        path=path,
        schema=model.schema_,
        version=model.version,
        language=model.language,
        metadata=dict(model.metadata or {}),
        chunks=chunks,
        raw=raw,
    )


def validate_transcript_chunks(
    document: TranscriptChunksDocument,
    *,
    duration_sec: float,
    language: str,
    language_explicit: bool,
    supported_languages: set[str],
    warn_chunk_sec: float,
    strict: bool = False,
    epsilon_sec: float = 0.05,
) -> TranscriptChunksValidation:
    errors: list[str] = []
    warnings: list[str] = []
    chunk_warnings: dict[str, list[str]] = {}

    if document.language is None and not language_explicit:
        errors.append("Transcript language is required when --language is not provided")
    if document.language is not None and language_explicit and document.language != language:
        errors.append(
            f"Transcript language {document.language!r} conflicts with CLI language {language!r}"
        )
    effective_language = language if language_explicit else document.language or language
    if effective_language not in supported_languages:
        supported = ", ".join(sorted(supported_languages))
        errors.append(
            f"Unsupported transcript language {effective_language!r}; expected: {supported}"
        )

    if not document.chunks:
        errors.append("Transcript file must contain at least one chunk")

    seen_ids: set[str] = set()
    previous_end: float | None = None
    empty_count = 0
    for chunk in document.chunks:
        chunk_errors: list[str] = []
        local_warnings: list[str] = []
        if chunk.chunk_id in seen_ids:
            chunk_errors.append(f"Duplicate chunk id: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)

        if not math.isfinite(chunk.start_sec) or not math.isfinite(chunk.end_sec):
            chunk_errors.append("Chunk start/end must be finite numbers")
        elif chunk.end_sec <= chunk.start_sec:
            chunk_errors.append("Chunk end must be greater than start")
        else:
            if chunk.start_sec < -epsilon_sec or chunk.end_sec > duration_sec + epsilon_sec:
                chunk_errors.append(
                    f"Chunk range {chunk.start_sec:.3f}-{chunk.end_sec:.3f}s is outside "
                    f"media duration {duration_sec:.3f}s"
                )
            if previous_end is not None and chunk.start_sec < previous_end - epsilon_sec:
                chunk_errors.append(
                    f"Chunk overlaps previous end {previous_end:.3f}s"
                )
            if previous_end is not None and chunk.start_sec > previous_end + epsilon_sec:
                local_warnings.append(
                    f"gap_from_previous_chunk_sec={chunk.start_sec - previous_end:.3f}"
                )
            if chunk.duration_sec > warn_chunk_sec:
                local_warnings.append(
                    f"chunk_duration_sec={chunk.duration_sec:.3f} exceeds warning threshold "
                    f"{warn_chunk_sec:.3f}s"
                )
            if chunk.text.strip() == "":
                empty_count += 1
            elif chunk.duration_sec > 0:
                chars_per_sec = len(chunk.text.strip()) / chunk.duration_sec
                if chars_per_sec > 30:
                    local_warnings.append(
                        f"text_density_chars_per_sec={chars_per_sec:.3f} looks high"
                    )
        previous_end = chunk.end_sec
        if chunk_errors:
            errors.extend(f"{chunk.chunk_id}: {error}" for error in chunk_errors)
        if local_warnings:
            chunk_warnings[chunk.chunk_id] = local_warnings

    if document.chunks and empty_count == len(document.chunks):
        warnings.append("All transcript chunks are empty")

    if strict and (warnings or chunk_warnings):
        errors.extend(warnings)
        for chunk_id, items in chunk_warnings.items():
            errors.extend(f"{chunk_id}: {warning}" for warning in items)

    if errors:
        raise TranscriptChunksError("; ".join(errors))

    return TranscriptChunksValidation(warnings=warnings, chunk_warnings=chunk_warnings)
