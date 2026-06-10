from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from segmentwright.cache import TranscriptCache, TranscriptCacheKey
from segmentwright.gemini import (
    build_call_metadata,
    classify_finish_reasons,
    extract_response_text,
    mark_failed_metadata,
    mark_success_metadata,
    parse_raw_transcript,
)
from segmentwright.transcript import (
    ENGLISH_NO_SPEECH_SENTINEL,
    Transcript,
    TranscriptParseError,
)


def test_finish_reason_classification_requires_stop_for_normal_completion() -> None:
    assert classify_finish_reasons(["STOP"]) == "normal_stop"
    assert classify_finish_reasons(["MAX_TOKENS"]) == "incomplete_output"
    assert classify_finish_reasons(["RECITATION"]) == "provider_block"
    assert classify_finish_reasons([]) == "unknown_failure"


def test_empty_plain_text_stop_response_can_be_marked_as_no_speech(tmp_path: Path) -> None:
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")
    response = fake_response(finish_reason="STOP", text="")
    uploaded = SimpleNamespace(
        name="files/1",
        uri="gemini://files/1",
        mime_type="audio/wav",
        state="ACTIVE",
    )

    raw_text = extract_response_text(response)
    metadata = build_call_metadata(
        response=response,
        uploaded=uploaded,
        audio_path=audio_path,
        model="gemini-test",
        mime_type="audio/wav",
        system_instruction="prompt",
        user_instruction="user",
        response_format="plain_text",
        response_mime_type=None,
        response_schema=None,
        thinking_level=None,
        temperature=None,
        raw_response_text=raw_text,
    )
    transcript = parse_raw_transcript(
        raw_text,
        "plain_text",
        allow_empty_plain_text=True,
    )
    metadata = mark_success_metadata(metadata, transcript=transcript)

    assert raw_text == ""
    assert transcript == Transcript(lines=[], raw_json="")
    assert metadata["status"] == "success"
    assert metadata["finish_reason_class"] == "normal_stop"
    assert metadata["empty_transcript"] is True
    assert metadata["empty_acceptance_reason"] == "finish_reason_stop_empty_response"
    assert metadata["post_transcription_action"] == "empty_transcript_noop"


def test_empty_plain_text_recitation_response_is_not_no_speech(tmp_path: Path) -> None:
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")
    response = fake_response(finish_reason="RECITATION", text="")
    uploaded = SimpleNamespace(
        name="files/1",
        uri="gemini://files/1",
        mime_type="audio/wav",
        state="ACTIVE",
    )

    raw_text = extract_response_text(response)
    metadata = build_call_metadata(
        response=response,
        uploaded=uploaded,
        audio_path=audio_path,
        model="gemini-test",
        mime_type="audio/wav",
        system_instruction="prompt",
        user_instruction="user",
        response_format="plain_text",
        response_mime_type=None,
        response_schema=None,
        thinking_level=None,
        temperature=None,
        raw_response_text=raw_text,
    )
    metadata = mark_failed_metadata(
        metadata,
        error_type="finish_reason",
        message="not STOP",
        raw_response_text=raw_text,
    )

    assert metadata["status"] == "failed"
    assert metadata["finish_reason"] == "RECITATION"
    assert metadata["finish_reason_class"] == "provider_block"
    assert metadata["empty_transcript"] is False
    assert metadata["post_transcription_action"] == "failed"


def test_plain_text_no_speech_sentinel_is_marked_as_empty_transcript(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")
    response = fake_response(finish_reason="STOP", text=ENGLISH_NO_SPEECH_SENTINEL)
    uploaded = SimpleNamespace(
        name="files/1",
        uri="gemini://files/1",
        mime_type="audio/wav",
        state="ACTIVE",
    )

    raw_text = extract_response_text(response)
    metadata = build_call_metadata(
        response=response,
        uploaded=uploaded,
        audio_path=audio_path,
        model="gemini-test",
        mime_type="audio/wav",
        system_instruction="prompt",
        user_instruction="user",
        response_format="plain_text",
        response_mime_type=None,
        response_schema=None,
        thinking_level=None,
        temperature=None,
        raw_response_text=raw_text,
    )
    transcript = parse_raw_transcript(raw_text, "plain_text")
    metadata = mark_success_metadata(metadata, transcript=transcript)

    assert transcript == Transcript(lines=[], raw_json=ENGLISH_NO_SPEECH_SENTINEL)
    assert metadata["empty_transcript"] is True
    assert metadata["empty_acceptance_reason"] == "finish_reason_stop_no_speech_sentinel"
    assert metadata["normalized_from_no_speech_sentinel"] is True
    assert metadata["transcript"]["normalized_from_no_speech_sentinel"] is True


def test_empty_plain_text_still_rejects_without_explicit_allow_empty() -> None:
    with pytest.raises(TranscriptParseError):
        parse_raw_transcript("", "plain_text")


def test_plain_text_empty_transcript_cache_round_trips(tmp_path: Path) -> None:
    cache = TranscriptCache(tmp_path / "cache")
    key = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=10.0,
        chunk_audio_hash="audio-hash",
        gemini_model="gemini-test",
        prompt_preset="plain-transcript",
        prompt_hash="prompt-hash",
        response_format="plain_text",
        segmentation="resegmentation",
    )
    transcript = Transcript(lines=[], raw_json="")

    cache.save(
        key,
        transcript,
        {"gemini_call_metadata": {"empty_transcript": True}},
    )
    loaded = cache.load_entry(key)

    assert loaded is not None
    assert loaded.transcript == transcript


def fake_response(*, finish_reason: str, text: str) -> SimpleNamespace:
    content = SimpleNamespace(parts=[SimpleNamespace(text=text)] if text else [])
    candidate = SimpleNamespace(
        index=0,
        finish_reason=finish_reason,
        content=content,
        safety_ratings=[],
    )
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        usage_metadata={
            "promptTokenCount": 10,
            "candidatesTokenCount": 0,
            "totalTokenCount": 10,
        },
        model_version="test-model",
        response_id="response-id",
        prompt_feedback=None,
    )
