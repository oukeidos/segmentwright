import json
from pathlib import Path
from typing import Any

import pytest

import segmentwright.pipeline as pipeline
from segmentwright.chunking import PlannedChunk
from segmentwright.config import AppConfig
from segmentwright.gemini import GeminiTranscriptionResult
from segmentwright.resegmentation.local_llm import (
    DEFAULT_RESEGMENTATION_PROMPT_PRESET,
    RESEGMENTATION_PROMPTS,
    OllamaSegmentationConfig,
    build_ollama_resegmentation_payload,
    build_resegmentation_prompt,
    resegmentation_options_for_model,
)
from segmentwright.resegmentation.projection import (
    ResegmentationSuggestion,
    non_whitespace_text,
    project_suggestion_to_source,
)
from segmentwright.resegmentation.quality import (
    assess_resegmentation_quality,
    fallback_segment_source_text,
)
from segmentwright.transcript import Transcript


def test_resegmentation_prompt_titles_and_default_are_stable() -> None:
    assert tuple(RESEGMENTATION_PROMPTS) == ("short", "medium", "long", "full")
    assert DEFAULT_RESEGMENTATION_PROMPT_PRESET == "medium"


def test_resegmentation_medium_prompt_template_is_stable() -> None:
    assert build_resegmentation_prompt("入力テキスト", "medium") == """You are a subtitle editor.

# Core Rules
- Do not rewrite. Keep all words and punctuation; only insert line breaks.
- Break into subtitle-friendly thought groups.
- Prefer punctuation and clear clause boundaries.
- Split after a conjunction only if it starts a new clause.
- Output only the formatted text.

# Task

Input:
入力テキスト

Output:
"""


def test_ollama_payload_disables_thinking_for_all_models() -> None:
    config = OllamaSegmentationConfig(
        model="custom-model",
        ollama_url="http://localhost:11434",
    )

    payload = build_ollama_resegmentation_payload("入力テキスト", config)

    assert payload["model"] == "custom-model"
    assert payload["think"] is False
    assert payload["options"] == {"num_ctx": 4096}
    assert "<|think|>" not in payload["prompt"]


def test_resegmentation_options_keep_temperature_zero_for_tested_gemma3_models() -> None:
    assert resegmentation_options_for_model("gemma3:12b-it-qat") == {
        "num_ctx": 4096,
        "temperature": 0.0,
    }
    assert resegmentation_options_for_model("gemma3:4b-it-qat") == {
        "num_ctx": 4096,
        "temperature": 0.0,
    }
    assert resegmentation_options_for_model("custom-model") == {"num_ctx": 4096}
    assert resegmentation_options_for_model("gemma4:12b") == {"num_ctx": 4096}


def test_ollama_payload_keeps_temperature_zero_and_disables_thinking_for_gemma3() -> None:
    config = OllamaSegmentationConfig(
        model="gemma3:12b-it-qat",
        ollama_url="http://localhost:11434",
    )

    payload = build_ollama_resegmentation_payload("入力テキスト", config)

    assert payload["think"] is False
    assert payload["options"] == {"num_ctx": 4096, "temperature": 0.0}


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r", "\n\n", "\u2028"])
def test_projection_keeps_original_boundaries_and_additional_splits(separator: str) -> None:
    source = f"  첫 문장.{separator}둘째 문장. 마지막 문장.  "
    result = project_suggestion_to_source(source, ["첫 문장. 둘째 문장.", "마지막 문장."])

    assert result.lines == ["첫 문장.", "둘째 문장.", "마지막 문장."]
    assert [segment.index for segment in result.projected_segments] == [0, 1, 2]
    assert result.projection_report.projected_matches_source_non_whitespace
    for segment in result.projected_segments:
        assert source[segment.source_start:segment.source_end].strip() == segment.text


@pytest.mark.parametrize("preserve", [True, False])
def test_resegment_transcript_respects_line_boundary_setting(
    tmp_path: Path, monkeypatch: Any, preserve: bool,
) -> None:
    source = "첫 문장.\n둘째 문장."
    transcript = Transcript(lines=source.splitlines(), raw_json=source)
    config = AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "out.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=tmp_path / ".env",
        segmentation="resegmentation",
        resegmentation_preserve_line_breaks=preserve,
    )
    monkeypatch.setattr(
        pipeline, "call_ollama_resegmentation",
        lambda *args, **kwargs: ResegmentationSuggestion(lines=["첫 문장. 둘째 문장."]),
    )
    result = pipeline.resegment_transcript_for_chunk(
        config=config, chunk=planned_chunk("1", 0.0, 30.0), transcript=transcript,
    )

    assert result.lines == (source.splitlines() if preserve else [source])
    assert (result.suggestion.raw or {})["quality_gate"]["fallback_used"] is False


def test_projection_preserves_original_text_with_clean_suggestion() -> None:
    result = project_suggestion_to_source(
        "今日はいい天気です。また明日。",
        ["今日はいい天気です。", "また明日。"],
    )

    assert result.lines == ["今日はいい天気です。", "また明日。"]
    assert result.text_edit_report.suggested_matches_source_non_whitespace is True
    assert result.projection_report.projected_matches_source_non_whitespace is True


def test_projection_uses_local_llm_boundaries_but_preserves_source_text() -> None:
    source = "高円寺に行きます。また明日。"
    result = project_suggestion_to_source(
        source,
        ["公園寺に行きます。", "また明日。"],
    )

    assert result.lines == ["高円寺に行きます。", "また明日。"]
    assert result.text_edit_report.suggested_matches_source_non_whitespace is False
    assert result.text_edit_report.changed_opcode_count > 0
    assert result.projection_report.projected_matches_source_non_whitespace is True
    assert "local_llm_suggestion_changed_non_whitespace_text" in result.projection_report.warnings
    assert non_whitespace_text("\n".join(result.lines)) == non_whitespace_text(source)


def test_projection_reports_mismatch_when_boundaries_drop_source_text() -> None:
    source = "今日はいい天気です。また明日。最後です。"
    result = project_suggestion_to_source(
        source,
        ["今日はいい天気です。", "また明日。"],
    )

    assert result.text_edit_report.suggested_matches_source_non_whitespace is False
    assert result.projection_report.mismatch is False
    assert result.projection_report.projected_matches_source_non_whitespace is True
    assert non_whitespace_text("\n".join(result.lines)) == non_whitespace_text(source)


def test_resegmentation_quality_rejects_truncated_suggestion() -> None:
    source = "今日はいい天気です。" * 20
    result = project_suggestion_to_source(source, ["今日はいい天気です。"])
    report = assess_resegmentation_quality(result)

    assert report.accepted is False
    assert "local_llm_suggestion_too_short_for_source" in report.reasons
    assert "projected_segment_too_long" in report.warnings


def test_resegmentation_quality_warns_on_long_projected_segment() -> None:
    source = "今日はいい天気です。" * 20
    result = project_suggestion_to_source(source, [source])
    report = assess_resegmentation_quality(result)

    assert report.accepted is True
    assert report.reasons == []
    assert "projected_segment_too_long" in report.warnings


def test_resegmentation_quality_warns_on_too_long_suggestion() -> None:
    source = "今日はいい天気です"
    result = project_suggestion_to_source(source, ["今日はいい天気です。。"])
    report = assess_resegmentation_quality(result)

    assert report.accepted is True
    assert report.reasons == []
    assert "local_llm_suggestion_too_long_for_source" in report.warnings
    assert result.projection_report.projected_matches_source_non_whitespace is True


def test_fallback_segment_source_text_avoids_giant_segments() -> None:
    source = "今日はいい天気です。" * 20
    lines = fallback_segment_source_text(source, max_non_ws_chars=40)

    assert "".join(non_whitespace_text(line) for line in lines) == non_whitespace_text(source)
    assert max(len(non_whitespace_text(line)) for line in lines) <= 40


def test_fallback_segment_source_text_supports_multilingual_sentence_punctuation() -> None:
    source = "مرحبا؟التالي۔फिर।끝!עוד։"
    lines = fallback_segment_source_text(source, max_non_ws_chars=80)

    assert lines == ["مرحبا؟", "التالي۔", "फिर।", "끝!", "עוד։"]
    assert "".join(non_whitespace_text(line) for line in lines) == non_whitespace_text(source)


def test_fallback_segment_source_text_supports_multilingual_soft_punctuation() -> None:
    source = "aaaa،bbbb؛cccc：dddd፣eeee"
    lines = fallback_segment_source_text(source, max_non_ws_chars=10)

    assert lines == ["aaaa،bbbb؛", "cccc：dddd፣", "eeee"]
    assert "".join(non_whitespace_text(line) for line in lines) == non_whitespace_text(source)
    assert max(len(non_whitespace_text(line)) for line in lines) <= 10


def test_resegment_transcript_retries_after_quality_gate_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    transcript = Transcript(
        lines=["今日はいい天気です。" * 20],
        raw_json="今日はいい天気です。" * 20,
    )
    config = AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "out.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=tmp_path / ".env",
        segmentation="resegmentation",
        resegmentation_max_retries=1,
    )
    calls = [
        ResegmentationSuggestion(
            lines=["今日はいい天気です。"],
            raw={"response": {"response": "bad"}},
        ),
        ResegmentationSuggestion(
            lines=["今日はいい天気です。" for _ in range(20)],
            raw={"response": {"response": "good"}},
        ),
    ]

    def fake_call_ollama_resegmentation(*_: Any, **__: Any) -> ResegmentationSuggestion:
        return calls.pop(0)

    monkeypatch.setattr(
        pipeline,
        "call_ollama_resegmentation",
        fake_call_ollama_resegmentation,
    )

    result = pipeline.resegment_transcript_for_chunk(
        config=config,
        chunk=planned_chunk("1", 0.0, 30.0),
        transcript=transcript,
    )

    quality_gate = (result.suggestion.raw or {})["quality_gate"]
    assert quality_gate["accepted_attempt"] == 2
    assert quality_gate["fallback_used"] is False
    assert quality_gate["attempts"][0]["accepted"] is False
    assert quality_gate["attempts"][1]["accepted"] is True
    assert len(result.lines) == 20


def test_resegment_transcript_does_not_retry_too_long_warning(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    transcript = Transcript(
        lines=["今日はいい天気です"],
        raw_json="今日はいい天気です",
    )
    config = AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "out.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=tmp_path / ".env",
        segmentation="resegmentation",
        resegmentation_max_retries=3,
    )
    calls = []

    def fake_call_ollama_resegmentation(*_: Any, **__: Any) -> ResegmentationSuggestion:
        calls.append(1)
        return ResegmentationSuggestion(
            lines=["今日はいい天気です。。"],
            raw={"response": {"response": "warning-only"}},
        )

    monkeypatch.setattr(
        pipeline,
        "call_ollama_resegmentation",
        fake_call_ollama_resegmentation,
    )

    result = pipeline.resegment_transcript_for_chunk(
        config=config,
        chunk=planned_chunk("1", 0.0, 30.0),
        transcript=transcript,
    )

    quality_gate = (result.suggestion.raw or {})["quality_gate"]
    assert len(calls) == 1
    assert quality_gate["accepted_attempt"] == 1
    assert quality_gate["fallback_used"] is False
    assert quality_gate["attempts"][0]["accepted"] is True
    assert quality_gate["attempts"][0]["quality_report"]["reasons"] == []
    assert "local_llm_suggestion_too_long_for_source" in quality_gate["attempts"][0][
        "quality_report"
    ]["warnings"]


def test_resegment_transcript_uses_fallback_after_retry_exhaustion(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    transcript = Transcript(
        lines=["今日はいい天気です。" * 20],
        raw_json="今日はいい天気です。" * 20,
    )
    config = AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "out.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=tmp_path / ".env",
        segmentation="resegmentation",
        resegmentation_max_retries=0,
    )

    monkeypatch.setattr(
        pipeline,
        "call_ollama_resegmentation",
        lambda *_args, **_kwargs: ResegmentationSuggestion(
            lines=["今日はいい天気です。"],
            raw={"response": {"response": "bad"}},
        ),
    )

    result = pipeline.resegment_transcript_for_chunk(
        config=config,
        chunk=planned_chunk("1", 0.0, 30.0),
        transcript=transcript,
    )

    quality_gate = (result.suggestion.raw or {})["quality_gate"]
    assert quality_gate["accepted_attempt"] is None
    assert quality_gate["fallback_used"] is True
    assert len(result.lines) > 1
    assert max(len(non_whitespace_text(line)) for line in result.lines) <= 60


def test_prepare_alignment_lines_writes_resegmentation_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    chunk = planned_chunk("1", 0.0, 30.0)
    config = AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "out.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=tmp_path / ".env",
        segmentation="resegmentation",
        response_format="plain_text",
        prompt_preset="plain-transcript",
    )
    transcript = Transcript(
        lines=["高円寺に行きます。また明日。"],
        raw_json="高円寺に行きます。また明日。",
    )
    result = project_suggestion_to_source(
        transcript.text,
        ["公園寺に行きます。", "また明日。"],
        raw_suggestion={"request": {"prompt": "PROMPT BODY"}, "response": {}},
    )

    monkeypatch.setattr(
        pipeline,
        "resegment_transcript_for_chunk",
        lambda **_: result,
    )
    lines, report = pipeline.prepare_alignment_lines(
        config=config,
        chunk=chunk,
        transcription=GeminiTranscriptionResult(transcript=transcript, metadata={}),
    )

    artifact_dir = tmp_path / "artifacts" / "resegmentation" / "chunk_1"
    assert lines == ["高円寺に行きます。", "また明日。"]
    assert report["mode"] == "resegmentation"
    assert report["prompt"] == "medium"
    assert (artifact_dir / "input.txt").read_text(encoding="utf-8") == transcript.text
    assert (artifact_dir / "prompt.txt").read_text(encoding="utf-8") == "PROMPT BODY"
    assert (artifact_dir / "suggestions.json").exists()
    assert (artifact_dir / "text_edit_report.json").exists()
    assert (artifact_dir / "projection.json").exists()
    assert (artifact_dir / "projected_segments.json").exists()
    projection = json.loads((artifact_dir / "projection.json").read_text(encoding="utf-8"))
    assert projection["projected_matches_source_non_whitespace"] is True
    assert "local_llm_suggestion_changed_non_whitespace_text" in projection["warnings"]
    assert report["source_text_construction"] == {
        "source_line_count": 1,
        "line_joiner": "\\n",
        "collapsed_from_transcript_lines": False,
    }


def test_prepare_alignment_lines_skips_resegmentation_for_empty_transcript(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    chunk = planned_chunk("1", 0.0, 30.0)
    config = AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "out.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=tmp_path / ".env",
        segmentation="resegmentation",
        response_format="plain_text",
        prompt_preset="plain-transcript",
    )
    transcript = Transcript(lines=[], raw_json="")

    def fail_resegmentation(*_: Any, **__: Any) -> None:
        raise AssertionError("empty transcript must not call local resegmentation")

    monkeypatch.setattr(pipeline, "resegment_transcript_for_chunk", fail_resegmentation)
    lines, report = pipeline.prepare_alignment_lines(
        config=config,
        chunk=chunk,
        transcription=GeminiTranscriptionResult(
            transcript=transcript,
            metadata={
                "empty_transcript": True,
                "empty_acceptance_reason": "finish_reason_stop_empty_response",
            },
        ),
    )

    assert lines == []
    assert report["mode"] == "empty_transcript_noop"
    assert report["requested_mode"] == "resegmentation"
    assert report["empty_transcript"] is True


def planned_chunk(chunk_id: str, start_sec: float, end_sec: float) -> PlannedChunk:
    return PlannedChunk(
        chunk_id=chunk_id,
        parent_id=chunk_id,
        source_id="source",
        start_sec=start_sec,
        end_sec=end_sec,
        start_sample=round(start_sec * 16000),
        end_sample=round(end_sec * 16000),
        depth=0,
        retry_count=0,
        original_index=0,
        cut_reason="test",
        cut_quality=None,
        cut_source="test",
        audio_path=None,
    )
