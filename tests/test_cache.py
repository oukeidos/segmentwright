from segmentwright.cache import TranscriptCacheKey


def test_transcript_cache_key_is_stable() -> None:
    key = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
    )

    assert key.digest() == key.digest()
    assert len(key.digest()) == 64


def test_transcript_cache_key_changes_with_prompt() -> None:
    base = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
    )
    changed = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="other",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
    )

    assert base.digest() != changed.digest()


def test_transcript_cache_key_changes_with_thinking_level() -> None:
    base = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
        gemini_thinking_level="low",
    )
    changed = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
        gemini_thinking_level="medium",
    )

    assert base.digest() != changed.digest()


def test_transcript_cache_key_changes_with_temperature() -> None:
    base = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
        gemini_temperature=None,
    )
    changed = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
        gemini_temperature=0.0,
    )

    assert base.digest() != changed.digest()


def test_transcript_cache_key_changes_with_response_format() -> None:
    base = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
    )
    changed = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="plain_text",
        segmentation="direct",
        preset="quality-baseline",
    )

    assert base.digest() != changed.digest()


def test_transcript_cache_key_changes_with_segmentation() -> None:
    base = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
    )
    changed = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="resegmentation",
        preset="quality-baseline",
    )

    assert base.digest() != changed.digest()


def test_transcript_cache_key_changes_with_preset() -> None:
    base = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
    )
    changed = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="manual",
    )

    assert base.digest() != changed.digest()


def test_transcript_cache_key_changes_with_language() -> None:
    base = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
        language="ja",
    )
    changed = TranscriptCacheKey(
        source_id="source",
        chunk_start_sec=0.0,
        chunk_end_sec=300.0,
        chunk_audio_hash="audio",
        gemini_model="gemini-3.5-flash",
        prompt_preset="subtitle-lines",
        prompt_hash="hash",
        response_format="json_lines",
        segmentation="direct",
        preset="quality-baseline",
        language="en",
    )

    assert base.digest() != changed.digest()
