from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_MAX_CHUNK_SEC,
    DEFAULT_MIN_RETRY_SPLIT_SEC,
    DEFAULT_TARGET_CHUNK_SEC,
)
from .vad.base import VadSegment


@dataclass(frozen=True)
class SplitCandidate:
    time_sec: float
    silence_start_sec: float
    silence_end_sec: float
    silence_duration_sec: float
    left_speech_end_sec: float | None
    right_speech_start_sec: float | None
    quality: float
    source: str
    forced: bool = False


@dataclass(frozen=True)
class PlannedChunk:
    chunk_id: str
    parent_id: str
    source_id: str
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int
    depth: int
    retry_count: int
    original_index: int
    cut_reason: str
    cut_quality: float | None
    cut_source: str
    audio_path: Path | None = None

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class ChunkPlan:
    candidates: list[SplitCandidate]
    split_times: list[float]
    chunks: list[PlannedChunk]
    forced_count: int


@dataclass(frozen=True)
class ChunkPlannerConfig:
    target_chunk_sec: float = DEFAULT_TARGET_CHUNK_SEC
    max_chunk_sec: float = DEFAULT_MAX_CHUNK_SEC
    min_chunk_sec: float = DEFAULT_MIN_RETRY_SPLIT_SEC
    sample_rate: int = 16000
    min_silence_sec: float = 0.08
    boundary_epsilon_sec: float = 0.05
    forced_penalty: float = 6.0
    quality_weight: float = 0.75
    length_weight: float = 2.0


def plan_chunks(
    *,
    duration_sec: float,
    speech_segments: list[VadSegment],
    source_id: str,
    config: ChunkPlannerConfig,
) -> ChunkPlan:
    if duration_sec <= 0:
        return ChunkPlan(candidates=[], split_times=[], chunks=[], forced_count=0)

    merged_speech = merge_speech_segments(speech_segments, duration_sec)
    candidates = silence_gap_candidates(
        duration_sec=duration_sec,
        speech_segments=merged_speech,
        min_silence_sec=config.min_silence_sec,
    )
    split_times, forced_count = plan_split_times(
        duration_sec=duration_sec,
        candidates=candidates,
        config=config,
    )
    split_times = normalize_split_times(duration_sec, split_times, config.boundary_epsilon_sec)
    chunks = build_chunks(
        duration_sec=duration_sec,
        split_times=split_times,
        source_id=source_id,
        candidates=candidates,
        sample_rate=config.sample_rate,
    )
    return ChunkPlan(
        candidates=candidates,
        split_times=split_times,
        chunks=chunks,
        forced_count=forced_count,
    )


def merge_speech_segments(
    segments: list[VadSegment],
    duration_sec: float,
    gap_sec: float = 0.05,
) -> list[VadSegment]:
    valid = [
        VadSegment(
            start_sec=max(0.0, min(duration_sec, segment.start_sec)),
            end_sec=max(0.0, min(duration_sec, segment.end_sec)),
            confidence=segment.confidence,
        )
        for segment in segments
        if segment.end_sec > segment.start_sec
    ]
    valid.sort(key=lambda item: item.start_sec)
    merged: list[VadSegment] = []
    for segment in valid:
        if not merged or segment.start_sec - merged[-1].end_sec > gap_sec:
            merged.append(segment)
            continue
        previous = merged[-1]
        merged[-1] = VadSegment(
            start_sec=previous.start_sec,
            end_sec=max(previous.end_sec, segment.end_sec),
            confidence=max_confidence(previous.confidence, segment.confidence),
        )
    return merged


def silence_gap_candidates(
    *,
    duration_sec: float,
    speech_segments: list[VadSegment],
    min_silence_sec: float,
) -> list[SplitCandidate]:
    candidates: list[SplitCandidate] = []
    previous_end = 0.0
    previous_speech_end: float | None = None
    for segment in speech_segments:
        add_gap_candidate(
            candidates,
            start=previous_end,
            end=segment.start_sec,
            duration_sec=duration_sec,
            min_silence_sec=min_silence_sec,
            left_speech_end=previous_speech_end,
            right_speech_start=segment.start_sec,
        )
        previous_end = segment.end_sec
        previous_speech_end = segment.end_sec
    add_gap_candidate(
        candidates,
        start=previous_end,
        end=duration_sec,
        duration_sec=duration_sec,
        min_silence_sec=min_silence_sec,
        left_speech_end=previous_speech_end,
        right_speech_start=None,
    )
    return candidates


def add_gap_candidate(
    candidates: list[SplitCandidate],
    *,
    start: float,
    end: float,
    duration_sec: float,
    min_silence_sec: float,
    left_speech_end: float | None,
    right_speech_start: float | None,
) -> None:
    gap = end - start
    if gap < min_silence_sec:
        return
    # Edge silence is useful for coverage artifacts but unsafe as a split point.
    if start <= 0.0 or end >= duration_sec:
        return
    midpoint = start + gap / 2.0
    quality = candidate_quality(gap)
    candidates.append(
        SplitCandidate(
            time_sec=midpoint,
            silence_start_sec=start,
            silence_end_sec=end,
            silence_duration_sec=gap,
            left_speech_end_sec=left_speech_end,
            right_speech_start_sec=right_speech_start,
            quality=quality,
            source="vad_silence_gap",
        )
    )


def candidate_quality(silence_duration_sec: float) -> float:
    return max(0.05, min(1.0, silence_duration_sec / 2.0))


def plan_split_times(
    *,
    duration_sec: float,
    candidates: list[SplitCandidate],
    config: ChunkPlannerConfig,
) -> tuple[list[float], int]:
    if duration_sec <= config.max_chunk_sec + 1e-6:
        return [], 0

    nodes = build_nodes(duration_sec, candidates, config.max_chunk_sec)
    end_index = len(nodes) - 1
    inf = math.inf
    dp = [inf] * len(nodes)
    prev = [-1] * len(nodes)
    dp[0] = 0.0

    for idx, node in enumerate(nodes):
        if math.isinf(dp[idx]):
            continue
        for next_idx in range(idx + 1, len(nodes)):
            next_node = nodes[next_idx]
            segment_len = next_node.time_sec - node.time_sec
            if segment_len > config.max_chunk_sec + 1e-6:
                break
            if segment_len <= 0:
                continue
            cost = edge_cost(segment_len, next_node, next_idx == end_index, config)
            if dp[idx] + cost < dp[next_idx]:
                dp[next_idx] = dp[idx] + cost
                prev[next_idx] = idx

    if prev[end_index] == -1:
        forced_splits = forced_fallback(duration_sec, config.max_chunk_sec)
        return forced_splits, int(duration_sec // config.max_chunk_sec)

    split_times: list[float] = []
    forced_count = 0
    cursor = end_index
    while cursor > 0:
        parent = prev[cursor]
        if parent < 0:
            break
        node = nodes[cursor]
        if cursor != end_index:
            split_times.append(node.time_sec)
            if node.forced:
                forced_count += 1
        cursor = parent
    split_times.reverse()
    return split_times, forced_count


def build_nodes(
    duration_sec: float, candidates: list[SplitCandidate], max_chunk_sec: float
) -> list[SplitCandidate]:
    nodes: list[SplitCandidate] = [
        SplitCandidate(0.0, 0.0, 0.0, 0.0, None, None, 1.0, "source_start")
    ]
    nodes.extend(candidates)
    forced = max_chunk_sec
    while forced < duration_sec - 1e-6:
        nodes.append(
            SplitCandidate(
                time_sec=forced,
                silence_start_sec=forced,
                silence_end_sec=forced,
                silence_duration_sec=0.0,
                left_speech_end_sec=None,
                right_speech_start_sec=None,
                quality=0.0,
                source="forced_anchor",
                forced=True,
            )
        )
        forced += max_chunk_sec
    nodes.append(
        SplitCandidate(duration_sec, duration_sec, duration_sec, 0.0, None, None, 1.0, "source_end")
    )
    by_ms: dict[int, SplitCandidate] = {}
    for node in nodes:
        key = round(node.time_sec * 1000)
        existing = by_ms.get(key)
        if (
            existing is None
            or (existing.forced and not node.forced)
            or node.quality > existing.quality
        ):
            by_ms[key] = node
    return sorted(by_ms.values(), key=lambda item: item.time_sec)


def edge_cost(
    segment_len: float,
    destination: SplitCandidate,
    is_end: bool,
    config: ChunkPlannerConfig,
) -> float:
    target = config.target_chunk_sec
    length_penalty = ((segment_len - target) / max(target, 1.0)) ** 2
    quality_penalty = 1.0 - max(0.0, min(1.0, destination.quality))
    cost = config.length_weight * length_penalty + config.quality_weight * quality_penalty
    if destination.forced and not is_end:
        cost += config.forced_penalty
    return cost


def forced_fallback(duration_sec: float, max_chunk_sec: float) -> list[float]:
    splits: list[float] = []
    cursor = max_chunk_sec
    while cursor < duration_sec - 1e-6:
        splits.append(cursor)
        cursor += max_chunk_sec
    return splits


def normalize_split_times(
    duration_sec: float,
    split_times: list[float],
    epsilon_sec: float,
) -> list[float]:
    output: list[float] = []
    seen_ms: set[int] = set()
    for value in sorted(split_times):
        if value <= epsilon_sec or value >= duration_sec - epsilon_sec:
            continue
        ms = round(value * 1000)
        if ms in seen_ms:
            continue
        seen_ms.add(ms)
        output.append(ms / 1000.0)
    return output


def build_chunks(
    *,
    duration_sec: float,
    split_times: list[float],
    source_id: str,
    candidates: list[SplitCandidate],
    sample_rate: int,
) -> list[PlannedChunk]:
    boundaries = [0.0, *split_times, duration_sec]
    candidate_by_ms = {round(candidate.time_sec * 1000): candidate for candidate in candidates}
    chunks: list[PlannedChunk] = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        split_candidate = candidate_by_ms.get(round(start * 1000))
        chunks.append(
            PlannedChunk(
                chunk_id=str(index + 1),
                parent_id=str(index + 1),
                source_id=source_id,
                start_sec=start,
                end_sec=end,
                start_sample=round(start * sample_rate),
                end_sample=round(end * sample_rate),
                depth=0,
                retry_count=0,
                original_index=index,
                cut_reason="source_start" if index == 0 else "planned_split",
                cut_quality=None if split_candidate is None else split_candidate.quality,
                cut_source=(
                    "source_start"
                    if index == 0
                    else (split_candidate.source if split_candidate else "unknown")
                ),
            )
        )
    return chunks


def max_confidence(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None
