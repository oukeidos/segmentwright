from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest

import segmentwright.pipeline as pipeline
from segmentwright.cache import TranscriptCache
from segmentwright.chunking import PlannedChunk
from segmentwright.config import AppConfig
from segmentwright.gemini import GeminiTranscriptionResult
from segmentwright.lifecycle import ModelLifecycleRecorder
from segmentwright.logging_utils import ConsoleReporter
from segmentwright.qa.long_omission import LongOmission, LongOmissionReport
from segmentwright.retry import (
    build_retry_children,
    clip_speech_segments_to_chunk,
    plan_retry_split,
)
from segmentwright.transcript import Transcript
from segmentwright.vad.base import VadSegment


def planned_chunk(
    chunk_id: str,
    start_sec: float,
    end_sec: float,
    *,
    audio_path: Path | None = None,
    original_index: int = 0,
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
        original_index=original_index,
        cut_reason="test",
        cut_quality=None,
        cut_source="test",
        audio_path=audio_path,
    )


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

    assert clipped == [VadSegment(0.0, 5.0), VadSegment(8.0, 10.0)]


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


def test_production_generation_loop_retries_long_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, trace, _, generation_sizes = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=1,
        retry_duration=lambda chunk: 100.0 if chunk.retry_count == 0 else None,
    )

    assert [item.chunk.chunk_id for item in accepted] == ["1.1", "1.2"]
    assert generation_sizes == [(0, 1), (1, 2)]
    assert [event["event"] for event in trace] == [
        "retry_split",
        "generation_summary",
        "generation_summary",
    ]


def test_production_generation_loop_accepts_empty_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, trace, _, generation_sizes = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=1,
        retry_duration=lambda chunk: None,
        empty_transcript=True,
    )

    assert len(accepted) == 1
    assert accepted[0].chunk.chunk_id == "1"
    assert accepted[0].line_spans == []
    assert accepted[0].qa_report.has_long_omissions is False
    assert accepted[0].raw_alignment["skipped"] is True
    assert accepted[0].raw_alignment["empty_transcript"]["warnings"] == [
        "empty_transcript_with_vad_activity"
    ]
    assert [event["event"] for event in trace] == ["generation_summary"]
    assert generation_sizes == [(0, 1)]


def test_36_initial_chunks_complete_without_retry_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, trace, config, generation_sizes = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=36,
        retry_duration=lambda chunk: None,
    )

    assert config.max_retry_work_items is None
    assert len(accepted) == 36
    assert generation_sizes == [(0, 36)]
    assert generation_summaries(trace) == [
        {
            "event": "generation_summary",
            "generation": 0,
            "processed_items": 36,
            "retry_requested_parents": 0,
            "retry_split_parents": 0,
            "retry_generated_children": 0,
            "retry_skipped_budget_parents": 0,
            "retry_budget_used": 0,
            "retry_budget_limit": None,
        }
    ]


def test_initial_chunks_do_not_consume_explicit_retry_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, trace, _, _ = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=36,
        max_retry_work_items=1,
        retry_duration=lambda chunk: None,
    )

    assert len(accepted) == 36
    assert generation_summaries(trace)[0]["retry_budget_used"] == 0


def test_max_retries_zero_disables_retry_splitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, trace, config, generation_sizes = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=1,
        max_retries=0,
        retry_duration=lambda chunk: 100.0,
    )

    assert [item.chunk.chunk_id for item in accepted] == ["1"]
    assert generation_sizes == [(0, 1)]
    unavailable = events_named(trace, "retry_not_available")
    assert [event["reason"] for event in unavailable] == ["max_retries_reached"]
    assert not (config.artifacts_dir / "retry_chunks").exists()


def test_unlimited_budget_allows_more_than_sixteen_retry_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, trace, _, generation_sizes = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=9,
        retry_duration=lambda chunk: 100.0 if chunk.retry_count == 0 else None,
    )

    assert len(accepted) == 18
    assert generation_sizes == [(0, 9), (1, 18)]
    assert len(events_named(trace, "retry_split")) == 9
    assert generation_summaries(trace)[0]["retry_budget_used"] == 18


def test_explicit_budget_permits_only_atomic_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, trace, _, generation_sizes = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=9,
        max_retry_work_items=16,
        retry_duration=lambda chunk: 100.0 if chunk.retry_count == 0 else None,
    )

    assert generation_sizes == [(0, 9), (1, 16)]
    assert len(events_named(trace, "retry_split")) == 8
    skipped = events_named(trace, "retry_skipped_budget")
    assert [event["parent_chunk_id"] for event in skipped] == ["9"]
    assert any(item.chunk.chunk_id == "9" for item in accepted)
    assert not any(item.chunk.chunk_id.startswith("9.") for item in accepted)
    assert len(accepted) == 17


def test_budget_of_one_materializes_no_retry_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, trace, config, generation_sizes = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=1,
        max_retry_work_items=1,
        retry_duration=lambda chunk: 100.0 if chunk.retry_count == 0 else None,
    )

    assert [item.chunk.chunk_id for item in accepted] == ["1"]
    assert generation_sizes == [(0, 1)]
    assert len(events_named(trace, "retry_skipped_budget")) == 1
    assert not (config.artifacts_dir / "retry_chunks").exists()


def test_budget_allocation_prioritizes_longer_omissions_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    durations = {"1": 70.0, "2": 110.0, "3": 90.0}
    _, trace, _, _ = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=3,
        max_retry_work_items=4,
        retry_duration=lambda chunk: durations.get(chunk.chunk_id),
    )

    assert [event["parent_chunk_id"] for event in events_named(trace, "retry_split")] == [
        "2",
        "3",
    ]
    assert [event["parent_chunk_id"] for event in events_named(trace, "retry_skipped_budget")] == [
        "1"
    ]


def test_cache_hits_and_local_segmentation_do_not_consume_retry_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, trace, _, _ = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=1,
        max_retry_work_items=2,
        retry_duration=lambda chunk: 100.0 if chunk.retry_count == 0 else None,
        cache_hit=True,
        segmentation_attempts=3,
    )

    summaries = generation_summaries(trace)
    assert summaries[0]["retry_budget_used"] == 2
    assert summaries[1]["retry_budget_used"] == 2


def test_retry_warnings_report_systemic_rate_and_budget_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=3,
        max_retry_work_items=2,
        retry_duration=lambda chunk: 100.0 if chunk.retry_count == 0 else None,
        artifacts_explicit=True,
        reporter_stream=stream,
    )

    output = stream.getvalue()
    assert "may indicate a systemic" in output
    assert "selected 1 parent(s) and skipped 2 parent(s)" in output
    assert "retry_trace.json" in output


@pytest.mark.parametrize(
    ("chunk", "config_changes", "reason"),
    [
        (
            replace(planned_chunk("1", 0.0, 120.0), retry_count=1),
            {"max_retries": 1, "max_split_depth": 2},
            "max_retries_reached",
        ),
        (
            replace(planned_chunk("1", 0.0, 120.0), depth=1),
            {"max_retries": 2, "max_split_depth": 1},
            "max_split_depth_reached",
        ),
        (
            planned_chunk("1", 0.0, 40.0),
            {"max_retries": 1, "max_split_depth": 2},
            "parent_too_short_to_split",
        ),
    ],
)
def test_retry_unavailable_reason_is_structured(
    tmp_path: Path,
    chunk: PlannedChunk,
    config_changes: dict[str, int],
    reason: str,
) -> None:
    config = replace(make_config(tmp_path), **config_changes)
    result = pipeline.maybe_split_for_retry(
        config=config,
        chunk=chunk,
        speech_segments=[VadSegment(chunk.start_sec, chunk.end_sec)],
        source_id="source",
        omission=LongOmission(chunk.start_sec, chunk.end_sec, chunk.duration_sec),
    )

    assert isinstance(result, pipeline.RetryUnavailable)
    assert result.reason == reason


@pytest.mark.parametrize(
    ("initial_count", "max_retries", "max_split_depth", "expected"),
    [(36, 1, 2, 72), (3, 2, 2, 18), (5, 0, 4, 0), (2, 4, 1, 4)],
)
def test_theoretical_retry_maximum(
    initial_count: int,
    max_retries: int,
    max_split_depth: int,
    expected: int,
) -> None:
    assert (
        pipeline.theoretical_max_retry_item_count(
            initial_chunk_count=initial_count,
            max_retries=max_retries,
            max_split_depth=max_split_depth,
        )
        == expected
    )


def test_run_config_reports_retry_budget_modes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    prompt_spec = pipeline.resolve_prompt(
        prompt_preset=config.prompt_preset,
        language=config.language,
        prompt_text=None,
        prompt_file=None,
    )

    unlimited = pipeline.build_run_config_report(config=config, prompt_spec=prompt_spec)
    explicit = pipeline.build_run_config_report(
        config=replace(config, max_retry_work_items=16),
        prompt_spec=prompt_spec,
    )

    assert unlimited["retry"]["max_retry_work_items"] is None
    assert unlimited["retry"]["retry_budget_mode"] == "unlimited"
    assert explicit["retry"]["max_retry_work_items"] == 16
    assert explicit["retry"]["retry_budget_mode"] == "explicit"
    assert "max_work_items" not in explicit["retry"]


def test_run_summary_aggregates_retry_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, config, _ = run_generation_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        initial_count=9,
        max_retry_work_items=16,
        retry_duration=lambda chunk: 100.0 if chunk.retry_count == 0 else None,
    )

    assert pipeline.build_retry_run_summary(config) == {
        "initial_chunk_count": 9,
        "processed_retry_item_count": 16,
        "retry_split_parent_count": 8,
        "retry_skipped_budget_parent_count": 1,
        "retry_budget_limit": 16,
        "retry_budget_exhausted": True,
        "theoretical_max_retry_item_count": 18,
    }


def test_legacy_work_queue_is_removed() -> None:
    assert not hasattr(pipeline, "process_work_queue")


def run_generation_fixture(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_count: int,
    retry_duration: Callable[[PlannedChunk], float | None],
    max_retry_work_items: int | None = None,
    max_retries: int = 1,
    empty_transcript: bool = False,
    cache_hit: bool = False,
    segmentation_attempts: int = 1,
    artifacts_explicit: bool = False,
    reporter_stream: StringIO | None = None,
) -> tuple[
    list[pipeline.ProcessedChunk],
    list[dict[str, object]],
    AppConfig,
    list[tuple[int, int]],
]:
    config = replace(
        make_config(tmp_path),
        max_retry_work_items=max_retry_work_items,
        max_retries=max_retries,
        artifacts_explicit=artifacts_explicit,
    )
    normalized_wav = tmp_path / "source.wav"
    normalized_wav.write_bytes(b"source")
    chunks: list[PlannedChunk] = []
    for index in range(initial_count):
        audio_path = tmp_path / f"parent-{index + 1}.wav"
        audio_path.write_bytes(b"parent")
        chunks.append(
            planned_chunk(
                str(index + 1),
                index * 120.0,
                (index + 1) * 120.0,
                audio_path=audio_path,
                original_index=index,
            )
        )
    pipeline.write_json(
        config.artifacts_dir / "chunk_manifest.json",
        {"schema": "test.chunk_manifest", "chunks": [pipeline.chunk_metadata(c) for c in chunks]},
    )
    speech_segments = [VadSegment(0.0, initial_count * 120.0)] if initial_count else []
    generation_sizes: list[tuple[int, int]] = []

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

    def fake_transcription_stage(
        *,
        generation: int,
        chunks: list[PlannedChunk],
        **_kwargs: object,
    ) -> list[pipeline.TranscribedChunk]:
        generation_sizes.append((generation, len(chunks)))
        lines = [] if empty_transcript else ["line"]
        return [
            pipeline.TranscribedChunk(
                chunk=chunk,
                transcription=GeminiTranscriptionResult(
                    transcript=Transcript(lines=lines, raw_json="" if empty_transcript else "{}"),
                    metadata={
                        "cache_hit": cache_hit,
                        "empty_transcript": empty_transcript,
                        "empty_acceptance_reason": (
                            "finish_reason_stop_empty_response" if empty_transcript else None
                        ),
                        "post_transcription_action": (
                            "empty_transcript_noop" if empty_transcript else None
                        ),
                    },
                ),
            )
            for chunk in chunks
        ]

    def fake_segmentation_stage(
        *,
        transcribed: list[pipeline.TranscribedChunk],
        **_kwargs: object,
    ) -> tuple[list[pipeline.SegmentedChunk], None]:
        return (
            [
                pipeline.SegmentedChunk(
                    chunk=item.chunk,
                    transcription=item.transcription,
                    alignment_lines=item.transcription.transcript.lines,
                    segmentation_report={"attempts": segmentation_attempts},
                )
                for item in transcribed
            ],
            None,
        )

    def fake_alignment_stage(
        *,
        segmented: list[pipeline.SegmentedChunk],
        speech_segments: list[VadSegment],
        **_kwargs: object,
    ) -> list[pipeline.ProcessedChunk]:
        if empty_transcript:
            return [
                pipeline.align_segmented_chunk(
                    config=config,
                    segmented=item,
                    speech_segments=speech_segments,
                    aligner=FailingAligner(),
                )
                for item in segmented
            ]
        processed: list[pipeline.ProcessedChunk] = []
        for item in segmented:
            duration = retry_duration(item.chunk)
            processed.append(
                pipeline.ProcessedChunk(
                    chunk=item.chunk,
                    transcript=item.transcription.transcript,
                    gemini_call_metadata=item.transcription.metadata,
                    line_spans=[],
                    raw_alignment={"chunk_id": item.chunk.chunk_id},
                    qa_report=omission_report(item.chunk, duration),
                )
            )
        return processed

    monkeypatch.setattr(pipeline, "extract_wav_segment", fake_extract)
    monkeypatch.setattr(pipeline, "run_transcription_stage", fake_transcription_stage)
    monkeypatch.setattr(pipeline, "run_segmentation_stage", fake_segmentation_stage)
    monkeypatch.setattr(pipeline, "run_alignment_stage", fake_alignment_stage)
    prompt_spec = pipeline.resolve_prompt(
        prompt_preset=config.prompt_preset,
        language=config.language,
        prompt_text=None,
        prompt_file=None,
    )
    reporter = ConsoleReporter(
        verbose=False,
        artifacts_dir=config.artifacts_dir,
        stream=reporter_stream if reporter_stream is not None else StringIO(),
    )
    accepted, trace, _ = pipeline.process_chunk_generations(
        config=config,
        normalized_wav=normalized_wav,
        initial_chunks=chunks,
        speech_segments=speech_segments,
        source_id="source",
        prompt_spec=prompt_spec,
        cache=TranscriptCache(tmp_path / "cache"),
        lifecycle=ModelLifecycleRecorder(config.artifacts_dir / "model_lifecycle.json"),
        reporter=reporter,
    )
    return accepted, trace, config, generation_sizes


def make_config(tmp_path: Path) -> AppConfig:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return AppConfig(
        input_path=input_path,
        output_path=tmp_path / "out.srt",
        artifacts_dir=artifacts_dir,
        env_file=tmp_path / ".env",
        max_retries=1,
        max_split_depth=2,
        min_retry_split_sec=30.0,
        long_omission_sec=50.0,
        cache_enabled=False,
    )


def omission_report(chunk: PlannedChunk, duration: float | None) -> LongOmissionReport:
    omissions = []
    if duration is not None:
        omissions = [
            LongOmission(
                start_sec=chunk.start_sec,
                end_sec=chunk.start_sec + duration,
                duration_sec=duration,
            )
        ]
    return LongOmissionReport(
        threshold_sec=50.0,
        coverage_gap_merge_sec=2.0,
        speech_intervals=[],
        subtitle_intervals=[],
        uncovered_intervals=[],
        long_omissions=omissions,
    )


def events_named(trace: list[dict[str, object]], name: str) -> list[dict[str, object]]:
    return [event for event in trace if event.get("event") == name]


def generation_summaries(trace: list[dict[str, object]]) -> list[dict[str, object]]:
    return events_named(trace, "generation_summary")


class FailingAligner:
    def align(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("empty transcript must not call forced alignment")
