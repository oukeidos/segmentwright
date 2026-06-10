from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentwright.transcript_chunks import (
    TranscriptChunksError,
    load_transcript_chunks,
    validate_transcript_chunks,
)


def write_transcript(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_transcript_chunks_minimal_json(tmp_path: Path) -> None:
    path = write_transcript(
        tmp_path / "transcript.json",
        {
            "version": 1,
            "language": "ja",
            "chunks": [
                {"start": 0.0, "end": 2.0, "text": "こんにちは"},
                {"id": "tail", "start": 2.0, "end": 3.0, "text": ""},
            ],
        },
    )

    document = load_transcript_chunks(path)

    assert document.version == 1
    assert document.language == "ja"
    assert [chunk.chunk_id for chunk in document.chunks] == ["chunk_0001", "tail"]
    assert document.chunks[1].text == ""


def test_validate_transcript_chunks_allows_empty_chunk(tmp_path: Path) -> None:
    document = load_transcript_chunks(
        write_transcript(
            tmp_path / "transcript.json",
            {
                "version": 1,
                "language": "en",
                "chunks": [{"start": 0.0, "end": 2.0, "text": ""}],
            },
        )
    )

    validation = validate_transcript_chunks(
        document,
        duration_sec=2.0,
        language="en",
        language_explicit=False,
        supported_languages={"ja", "en"},
        warn_chunk_sec=180.0,
    )

    assert validation.warnings == ["All transcript chunks are empty"]


def test_validate_transcript_chunks_rejects_overlap(tmp_path: Path) -> None:
    document = load_transcript_chunks(
        write_transcript(
            tmp_path / "transcript.json",
            {
                "version": 1,
                "language": "en",
                "chunks": [
                    {"id": "a", "start": 0.0, "end": 2.0, "text": "one"},
                    {"id": "b", "start": 1.5, "end": 3.0, "text": "two"},
                ],
            },
        )
    )

    with pytest.raises(TranscriptChunksError, match="overlaps"):
        validate_transcript_chunks(
            document,
            duration_sec=3.0,
            language="en",
            language_explicit=False,
            supported_languages={"ja", "en"},
            warn_chunk_sec=180.0,
        )


def test_validate_transcript_chunks_warns_on_long_chunk(tmp_path: Path) -> None:
    document = load_transcript_chunks(
        write_transcript(
            tmp_path / "transcript.json",
            {
                "version": 1,
                "language": "en",
                "chunks": [{"id": "long", "start": 0.0, "end": 200.0, "text": "text"}],
            },
        )
    )

    validation = validate_transcript_chunks(
        document,
        duration_sec=200.0,
        language="en",
        language_explicit=False,
        supported_languages={"ja", "en"},
        warn_chunk_sec=180.0,
    )

    assert "exceeds warning threshold" in validation.chunk_warnings["long"][0]


def test_validate_transcript_chunks_strict_promotes_warning(tmp_path: Path) -> None:
    document = load_transcript_chunks(
        write_transcript(
            tmp_path / "transcript.json",
            {
                "version": 1,
                "language": "en",
                "chunks": [{"id": "long", "start": 0.0, "end": 200.0, "text": "text"}],
            },
        )
    )

    with pytest.raises(TranscriptChunksError, match="exceeds warning threshold"):
        validate_transcript_chunks(
            document,
            duration_sec=200.0,
            language="en",
            language_explicit=False,
            supported_languages={"ja", "en"},
            warn_chunk_sec=180.0,
            strict=True,
        )


def test_validate_transcript_chunks_requires_language_without_cli_override(
    tmp_path: Path,
) -> None:
    document = load_transcript_chunks(
        write_transcript(
            tmp_path / "transcript.json",
            {
                "version": 1,
                "chunks": [{"start": 0.0, "end": 2.0, "text": "text"}],
            },
        )
    )

    with pytest.raises(TranscriptChunksError, match="language is required"):
        validate_transcript_chunks(
            document,
            duration_sec=2.0,
            language="ja",
            language_explicit=False,
            supported_languages={"ja", "en"},
            warn_chunk_sec=180.0,
        )
