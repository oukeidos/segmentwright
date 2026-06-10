from __future__ import annotations

from pathlib import Path
from typing import Any

import segmentwright.pipeline as pipeline
from segmentwright.aligners.base import AlignmentResult, LineSpan, TokenSpan
from segmentwright.cache import TranscriptCache
from segmentwright.chunking import PlannedChunk
from segmentwright.config import AppConfig
from segmentwright.gemini import GeminiTranscriptionResult
from segmentwright.qa.long_omission import LongOmission
from segmentwright.retry import (
    build_retry_children,
    clip_speech_segments_to_chunk,
    plan_retry_split,
)
from segmentwright.transcript import Transcript
from segmentwright.vad.base import VadSegment


def test_plan_retry_split_uses_nearest_valid_silence_gap() -> None:
    parent = planned_chunk("1", 0.0, 120.0)
    decision = plan_retry_split(
        parent=parent,
        omission=LongOmission(0.0, 120.0, 120.0),
        speech_segments=[VadSegment(0.0, 40.0), VadSegment(50.0, 120.0)],
        min_split_sec=30.0,
    )

    assert decision is not None
    assert decision.split_time_sec == 45.0
    assert decision.split_source == "vad_silence_gap"


def test_plan_retry_split_rejects_parent_too_short_to_split() -> None:
    decision = plan_retry_split(
        parent=planned_chunk("1", 0.0, 40.0),
        omission=LongOmission(0.0, 40.0, 40.0),
        speech_segments=[VadSegment(0.0, 40.0)],
        min_split_sec=30.0,
    )

    assert decision is None


def test_clip_speech_segments_to_chunk_returns_local_times() -> None:
    clipped = clip_speech_segments_to_chunk(
        speech_segments=[VadSegment(5.0, 15.0), VadSegment(18.0, 30.0)],
        chunk_start_sec=10.0,
        chunk_end_sec=20.0,
    )

    assert clipped == [
        VadSegment(0.0, 5.0),
        VadSegment(8.0, 10.0),
    ]


def test_build_retry_children_preserves_parent_metadata() -> None:
    parent = planned_chunk("1", 0.0, 120.0)
    decision = plan_retry_split(
        parent=parent,
        omission=LongOmission(0.0, 120.0, 120.0),
        speech_segments=[VadSegment(0.0, 120.0)],
        min_split_sec=30.0,
    )
    assert decision is not None

    children = build_retry_children(
        parent=parent,
        decision=decision,
        source_id="source",
        sample_rate=16000,
    )

    assert [child.chunk_id for child in children] == ["1.1", "1.2"]
    assert [child.parent_id for child in children] == ["1", "1"]
    assert [child.retry_count for child in children] == [1, 1]
    assert [child.depth for child in children] == [1, 1]


def test_process_work_queue_retries_long_omission_by_splitting(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    parent_audio = tmp_path / "parent.wav"
    parent_audio.write_bytes(b"parent")
    normalized_wav = tmp_path / "source.wav"
    normalized_wav.write_bytes(b"source")
    config = AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "out.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=tmp_path / ".env",
        max_retries=1,
        max_split_depth=2,
        max_work_items=4,
        min_retry_split_sec=30.0,
        long_omission_sec=50.0,
        cache_enabled=False,
    )
    config.input_path.write_bytes(b"input")
    config.artifacts_dir.mkdir()
    parent = planned_chunk("1", 0.0, 120.0, audio_path=parent_audio)

    def fake_extract(
        input_wav: Path,
        *,
        start_sec: float,
        end_sec: float,
        output_path: Path,
    ) -> None:
        del input_wav, start_sec, end_sec
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"child")

    monkeypatch.setattr(pipeline, "extract_wav_segment", fake_extract)
    events: list[str] = []
    transcriber = FakeTranscriber(events=events)
    processed, trace = pipeline.process_work_queue(
        config=config,
        normalized_wav=normalized_wav,
        initial_chunks=[parent],
        speech_segments=[VadSegment(0.0, 120.0)],
        source_id="source",
        transcriber=transcriber,
        cache=TranscriptCache(tmp_path / "cache"),
        aligner=FakeAligner(parent_audio=parent_audio, events=events),
    )

    assert len(processed) == 2
    assert [item.chunk.chunk_id for item in processed] == ["1.1", "1.2"]
    assert len(trace) == 1
    assert trace[0]["event"] == "retry_split"
    assert transcriber.calls == 3
    assert events == [
        "transcribe:parent.wav",
        "align:parent.wav",
        "transcribe:audio_1_1.wav",
        "transcribe:audio_1_2.wav",
        "align:audio_1_1.wav",
        "align:audio_1_2.wav",
    ]


def test_process_work_queue_does_not_retry_accepted_empty_transcript(
    tmp_path: Path,
) -> None:
    parent_audio = tmp_path / "parent.wav"
    parent_audio.write_bytes(b"parent")
    normalized_wav = tmp_path / "source.wav"
    normalized_wav.write_bytes(b"source")
    config = AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "out.srt",
        artifacts_dir=tmp_path / "artifacts",
        env_file=tmp_path / ".env",
        max_retries=1,
        max_split_depth=2,
        max_work_items=4,
        min_retry_split_sec=30.0,
        long_omission_sec=50.0,
        cache_enabled=False,
    )
    config.input_path.write_bytes(b"input")
    config.artifacts_dir.mkdir()
    parent = planned_chunk("1", 0.0, 120.0, audio_path=parent_audio)
    events: list[str] = []
    transcriber = EmptyTranscriber(events=events)
    processed, trace = pipeline.process_work_queue(
        config=config,
        normalized_wav=normalized_wav,
        initial_chunks=[parent],
        speech_segments=[VadSegment(0.0, 120.0)],
        source_id="source",
        transcriber=transcriber,
        cache=TranscriptCache(tmp_path / "cache"),
        aligner=FailingAligner(),
    )

    assert len(processed) == 1
    assert processed[0].chunk.chunk_id == "1"
    assert processed[0].line_spans == []
    assert processed[0].qa_report.has_long_omissions is False
    assert processed[0].raw_alignment["skipped"] is True
    assert processed[0].raw_alignment["empty_transcript"]["warnings"] == [
        "empty_transcript_with_vad_activity"
    ]
    assert trace == []
    assert transcriber.calls == 1
    assert events == ["transcribe:parent.wav"]


class FakeTranscriber:
    def __init__(self, *, events: list[str]) -> None:
        self.events = events
        self.calls = 0

    def transcribe(self, audio_path: Path) -> GeminiTranscriptionResult:
        self.events.append(f"transcribe:{audio_path.name}")
        self.calls += 1
        return GeminiTranscriptionResult(
            transcript=Transcript(
                lines=["line"],
                raw_json='{"lines":[{"text":"line"}]}',
            ),
            metadata={
                "schema": "segmentwright.gemini_call_metadata.v1",
                "cache_hit": False,
                "usage_metadata": {},
                "token_usage": {},
            },
        )


class EmptyTranscriber:
    def __init__(self, *, events: list[str]) -> None:
        self.events = events
        self.calls = 0

    def transcribe(self, audio_path: Path) -> GeminiTranscriptionResult:
        self.events.append(f"transcribe:{audio_path.name}")
        self.calls += 1
        return GeminiTranscriptionResult(
            transcript=Transcript(lines=[], raw_json=""),
            metadata={
                "schema": "segmentwright.gemini_call_metadata.v1",
                "cache_hit": False,
                "status": "success",
                "empty_transcript": True,
                "empty_acceptance_reason": "finish_reason_stop_empty_response",
                "post_transcription_action": "empty_transcript_noop",
                "usage_metadata": {},
                "token_usage": {},
            },
        )


class FakeAligner:
    def __init__(self, *, parent_audio: Path, events: list[str]) -> None:
        self.parent_audio = parent_audio
        self.events = events

    def align(self, audio_path: Path, lines: list[str], offset: float) -> AlignmentResult:
        del lines
        self.events.append(f"align:{audio_path.name}")
        if audio_path == self.parent_audio:
            line_spans: list[LineSpan] = []
        else:
            line_spans = [
                LineSpan(
                    text="line",
                    start_sec=offset,
                    end_sec=offset + 60.0,
                    token_spans=[TokenSpan("line", offset, offset + 60.0)],
                )
            ]
        return AlignmentResult(
            line_spans=line_spans,
            raw={"schema": "test.alignment", "audio_path": str(audio_path)},
        )


class FailingAligner:
    def align(self, audio_path: Path, lines: list[str], offset: float) -> AlignmentResult:
        del audio_path, lines, offset
        raise AssertionError("empty transcript must not call forced alignment")


def planned_chunk(
    chunk_id: str,
    start_sec: float,
    end_sec: float,
    *,
    audio_path: Path | None = None,
) -> PlannedChunk:
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
        audio_path=audio_path,
    )
