from __future__ import annotations

from dataclasses import dataclass

from .chunking import PlannedChunk, silence_gap_candidates
from .qa.long_omission import LongOmission
from .vad.base import VadSegment


@dataclass(frozen=True)
class RetrySplitDecision:
    split_time_sec: float
    split_source: str
    split_quality: float | None


def plan_retry_split(
    *,
    parent: PlannedChunk,
    omission: LongOmission,
    speech_segments: list[VadSegment],
    min_split_sec: float,
) -> RetrySplitDecision | None:
    min_abs = parent.start_sec + min_split_sec
    max_abs = parent.end_sec - min_split_sec
    if max_abs <= min_abs:
        return None

    desired_abs = (omission.start_sec + omission.end_sec) / 2.0
    desired_abs = max(min_abs, min(max_abs, desired_abs))
    local_speech = clip_speech_segments_to_chunk(
        speech_segments=speech_segments,
        chunk_start_sec=parent.start_sec,
        chunk_end_sec=parent.end_sec,
    )
    candidates = silence_gap_candidates(
        duration_sec=parent.duration_sec,
        speech_segments=local_speech,
        min_silence_sec=0.08,
    )
    valid_candidates = [
        candidate
        for candidate in candidates
        if min_split_sec <= candidate.time_sec <= parent.duration_sec - min_split_sec
    ]
    if valid_candidates:
        local_desired = desired_abs - parent.start_sec
        selected = min(
            valid_candidates,
            key=lambda candidate: (
                abs(candidate.time_sec - local_desired),
                -candidate.quality,
            ),
        )
        return RetrySplitDecision(
            split_time_sec=parent.start_sec + selected.time_sec,
            split_source=selected.source,
            split_quality=selected.quality,
        )
    return RetrySplitDecision(
        split_time_sec=desired_abs,
        split_source="retry_forced_midpoint",
        split_quality=None,
    )


def build_retry_children(
    *,
    parent: PlannedChunk,
    decision: RetrySplitDecision,
    source_id: str,
    sample_rate: int,
) -> list[PlannedChunk]:
    split = decision.split_time_sec
    return [
        build_retry_child(
            parent=parent,
            child_index=1,
            start_sec=parent.start_sec,
            end_sec=split,
            source_id=source_id,
            sample_rate=sample_rate,
            decision=decision,
        ),
        build_retry_child(
            parent=parent,
            child_index=2,
            start_sec=split,
            end_sec=parent.end_sec,
            source_id=source_id,
            sample_rate=sample_rate,
            decision=decision,
        ),
    ]


def build_retry_child(
    *,
    parent: PlannedChunk,
    child_index: int,
    start_sec: float,
    end_sec: float,
    source_id: str,
    sample_rate: int,
    decision: RetrySplitDecision,
) -> PlannedChunk:
    child_id = f"{parent.chunk_id}.{child_index}"
    return PlannedChunk(
        chunk_id=child_id,
        parent_id=parent.chunk_id,
        source_id=source_id,
        start_sec=start_sec,
        end_sec=end_sec,
        start_sample=round(start_sec * sample_rate),
        end_sample=round(end_sec * sample_rate),
        depth=parent.depth + 1,
        retry_count=parent.retry_count + 1,
        original_index=parent.original_index,
        cut_reason="retry_long_omission_split",
        cut_quality=decision.split_quality,
        cut_source=decision.split_source,
    )


def clip_speech_segments_to_chunk(
    *,
    speech_segments: list[VadSegment],
    chunk_start_sec: float,
    chunk_end_sec: float,
) -> list[VadSegment]:
    clipped: list[VadSegment] = []
    for segment in speech_segments:
        start = max(chunk_start_sec, segment.start_sec)
        end = min(chunk_end_sec, segment.end_sec)
        if end > start:
            clipped.append(
                VadSegment(
                    start_sec=start - chunk_start_sec,
                    end_sec=end - chunk_start_sec,
                    confidence=segment.confidence,
                )
            )
    return clipped


def clip_speech_segments_absolute(
    *,
    speech_segments: list[VadSegment],
    chunk_start_sec: float,
    chunk_end_sec: float,
) -> list[VadSegment]:
    clipped: list[VadSegment] = []
    for segment in speech_segments:
        start = max(chunk_start_sec, segment.start_sec)
        end = min(chunk_end_sec, segment.end_sec)
        if end > start:
            clipped.append(
                VadSegment(
                    start_sec=start,
                    end_sec=end,
                    confidence=segment.confidence,
                )
            )
    return clipped
