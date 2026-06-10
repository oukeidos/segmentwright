from pathlib import Path

import pytest

from segmentwright.gemini import response_contract_for_format
from segmentwright.prompts import (
    ENGLISH_PLAIN_TRANSCRIPT_PROMPT,
    ENGLISH_SUBTITLE_LINES_PROMPT,
    PLAIN_TRANSCRIPT_PROMPT,
    PROMPT_PRESETS,
    SUBTITLE_LINES_PROMPT,
    get_prompt,
    resolve_prompt,
)
from segmentwright.transcript import (
    ENGLISH_NO_SPEECH_SENTINEL,
    JAPANESE_NO_SPEECH_SENTINEL,
    TRANSCRIPT_RESPONSE_SCHEMA,
    Transcript,
    TranscriptParseError,
    parse_plain_text_transcript,
    parse_transcript_lines,
)


def test_product_has_only_two_builtin_prompts() -> None:
    assert set(PROMPT_PRESETS) == {"subtitle-lines", "plain-transcript"}


def test_subtitle_lines_prompt_is_independent_json_lines_contract() -> None:
    prompt = get_prompt("subtitle-lines")

    assert prompt == SUBTITLE_LINES_PROMPT
    assert prompt is not PLAIN_TRANSCRIPT_PROMPT
    assert '"lines" array' in prompt
    assert "Start a new line" in prompt
    assert "Do not add timestamps" in prompt
    assert 'If there is no spoken Japanese, return exactly {"lines":[]}' in prompt


def test_plain_transcript_prompt_is_independent_plain_text_contract() -> None:
    prompt = get_prompt("plain-transcript")

    assert prompt == PLAIN_TRANSCRIPT_PROMPT
    assert prompt is not SUBTITLE_LINES_PROMPT
    assert '"lines" array' not in prompt
    assert "JSON" not in prompt
    assert "Output only the Japanese transcription" in prompt
    assert "Do not add timestamps" in prompt
    assert JAPANESE_NO_SPEECH_SENTINEL in prompt
    assert "Do not output any other text in that no-speech case" in prompt


def test_builtin_prompt_resolution_reports_hash_and_source() -> None:
    spec = resolve_prompt(prompt_preset="plain-transcript")

    assert spec.source == "built_in"
    assert spec.language == "ja"
    assert spec.preset == "plain-transcript"
    assert spec.file is None
    assert spec.text == PLAIN_TRANSCRIPT_PROMPT
    assert len(spec.sha256) == 64


def test_builtin_prompt_resolution_supports_english() -> None:
    subtitle_spec = resolve_prompt(prompt_preset="subtitle-lines", language="en")
    plain_spec = resolve_prompt(prompt_preset="plain-transcript", language="en")

    assert subtitle_spec.text == ENGLISH_SUBTITLE_LINES_PROMPT
    assert plain_spec.text == ENGLISH_PLAIN_TRANSCRIPT_PROMPT
    assert subtitle_spec.language == "en"
    assert plain_spec.language == "en"
    assert subtitle_spec.sha256 != resolve_prompt(prompt_preset="subtitle-lines").sha256
    assert "spoken English" in subtitle_spec.text
    assert "English transcription" in plain_spec.text
    assert 'If there is no spoken English, return exactly {"lines":[]}' in subtitle_spec.text
    assert ENGLISH_NO_SPEECH_SENTINEL in plain_spec.text
    assert "Do not output any other text in that no-speech case" in plain_spec.text


def test_prompt_file_overrides_builtin_prompt(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("custom prompt", encoding="utf-8")

    spec = resolve_prompt(prompt_preset="subtitle-lines", prompt_file=prompt_file)

    assert spec.source == "custom_file"
    assert spec.language == "ja"
    assert spec.preset is None
    assert spec.file == prompt_file
    assert spec.text == "custom prompt"


def test_prompt_text_overrides_builtin_prompt() -> None:
    spec = resolve_prompt(prompt_preset="subtitle-lines", prompt_text="custom prompt")

    assert spec.source == "custom_text"
    assert spec.language == "ja"
    assert spec.preset is None
    assert spec.text == "custom prompt"


def test_prompt_file_and_text_are_mutually_exclusive(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("custom prompt", encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_prompt(
            prompt_preset="subtitle-lines",
            prompt_file=prompt_file,
            prompt_text="custom",
        )


def test_unknown_prompt_preset_fails() -> None:
    with pytest.raises(ValueError, match="Unknown prompt preset"):
        get_prompt("experimental-prompt")


def test_response_contract_for_json_lines_sends_schema() -> None:
    mime_type, schema = response_contract_for_format("json_lines")

    assert mime_type == "application/json"
    assert schema == TRANSCRIPT_RESPONSE_SCHEMA


def test_response_contract_for_plain_text_sends_no_schema() -> None:
    mime_type, schema = response_contract_for_format("plain_text")

    assert mime_type is None
    assert schema is None


def test_parse_plain_text_transcript() -> None:
    transcript = parse_plain_text_transcript("こんにちは\n\nまたね")

    assert transcript.lines == ["こんにちは", "またね"]
    assert transcript.raw_json == "こんにちは\n\nまたね"


def test_parse_plain_text_transcript_rejects_empty_by_default() -> None:
    with pytest.raises(TranscriptParseError, match="did not contain"):
        parse_plain_text_transcript("")


def test_parse_plain_text_transcript_allows_empty_when_explicit() -> None:
    transcript = parse_plain_text_transcript("", allow_empty=True)

    assert transcript == Transcript(lines=[], raw_json="")


def test_parse_plain_text_transcript_normalizes_no_speech_sentinel() -> None:
    transcript = parse_plain_text_transcript(ENGLISH_NO_SPEECH_SENTINEL)

    assert transcript == Transcript(lines=[], raw_json=ENGLISH_NO_SPEECH_SENTINEL)


def test_parse_plain_text_transcript_rejects_mixed_no_speech_sentinel() -> None:
    with pytest.raises(TranscriptParseError, match="mixed a no-speech sentinel"):
        parse_plain_text_transcript(f"{ENGLISH_NO_SPEECH_SENTINEL}\nActual speech")


def test_parse_transcript_lines() -> None:
    raw = '{"lines":[{"text":"  こんにちは  "},{"text":""},{"text":"またね"}]}'

    assert parse_transcript_lines(raw) == ["こんにちは", "またね"]


def test_parse_transcript_allows_empty_json_lines_for_no_speech_chunk() -> None:
    assert parse_transcript_lines('{"lines":[]}') == []


def test_parse_transcript_rejects_missing_lines() -> None:
    with pytest.raises(TranscriptParseError, match="lines"):
        parse_transcript_lines('{"text":"こんにちは"}')
