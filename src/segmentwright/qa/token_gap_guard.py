from __future__ import annotations

from dataclasses import asdict, dataclass

from segmentwright.aligners.base import LineSpan
from segmentwright.vad.base import VadSegment


@dataclass(frozen=True)
class TokenGapGuardConfig:
    min_edge_gap_sec: float = 0.7
    edge_pad_sec: float = 0.16
    min_adjustment_sec: float = 0.2
    vad_expand_gap_sec: float = 0.3
    min_candidate_speech_ratio: float = 0.35
    min_coverage_gain: float = 0.12


@dataclass(frozen=True)
class TokenGapCorrection:
    line_index: int
    text: str
    side: str
    before_start_sec: float
    before_end_sec: float
    after_start_sec: float
    after_end_sec: float
    removed_leading_tokens: int
    removed_trailing_tokens: int
    leading_gap_sec: float | None
    trailing_gap_sec: float | None
    original_speech_ratio: float | None
    candidate_speech_ratio: float | None


@dataclass(frozen=True)
class TokenGapGuardResult:
    line_spans: list[LineSpan]
    corrections: list[TokenGapCorrection]

    def to_dict(self) -> dict:
        return {
            "schema": "segmentwright.token_gap_guard.v1",
            "corrections": [asdict(correction) for correction in self.corrections],
        }


def apply_token_gap_guard(
    line_spans: list[LineSpan],
    *,
    speech_segments: list[VadSegment] | None = None,
    config: TokenGapGuardConfig | None = None,
) -> TokenGapGuardResult:
    config = config or TokenGapGuardConfig()
    speech_segments = sorted(speech_segments or [], key=lambda segment: segment.start_sec)
    output: list[LineSpan] = []
    corrections: list[TokenGapCorrection] = []
    for index, line_span in enumerate(line_spans):
        corrected, correction = maybe_correct_line_span(
            line_span=line_span,
            line_index=index,
            speech_segments=speech_segments,
            config=config,
        )
        output.append(corrected)
        if correction is not None:
            corrections.append(correction)
    return TokenGapGuardResult(line_spans=output, corrections=corrections)


def maybe_correct_line_span(
    *,
    line_span: LineSpan,
    line_index: int,
    speech_segments: list[VadSegment],
    config: TokenGapGuardConfig,
) -> tuple[LineSpan, TokenGapCorrection | None]:
    tokens = line_span.token_spans
    if len(tokens) < 2:
        return line_span, None

    start_index = 0
    end_index = len(tokens) - 1
    leading_gap = None
    trailing_gap = None

    while start_index < end_index:
        gap = tokens[start_index + 1].start_sec - tokens[start_index].end_sec
        if gap < config.min_edge_gap_sec:
            break
        leading_gap = gap if leading_gap is None else max(leading_gap, gap)
        start_index += 1

    while end_index > start_index:
        gap = tokens[end_index].start_sec - tokens[end_index - 1].end_sec
        if gap < config.min_edge_gap_sec:
            break
        trailing_gap = gap if trailing_gap is None else max(trailing_gap, gap)
        end_index -= 1

    if start_index == 0 and end_index == len(tokens) - 1:
        return line_span, None

    candidate_start = (
        max(0.0, tokens[start_index].start_sec - config.edge_pad_sec)
        if start_index > 0
        else line_span.start_sec
    )
    candidate_end = (
        tokens[end_index].end_sec + config.edge_pad_sec
        if end_index < len(tokens) - 1
        else line_span.end_sec
    )
    candidate_start, candidate_end = expand_to_nearby_speech(
        start_sec=candidate_start,
        end_sec=candidate_end,
        speech_segments=speech_segments,
        max_gap_sec=config.vad_expand_gap_sec,
    )
    # VAD expansion may fill the retained token cluster, but it must not move a
    # boundary that was not identified as an edge outlier.
    if start_index == 0:
        candidate_start = line_span.start_sec
    if end_index == len(tokens) - 1:
        candidate_end = line_span.end_sec
    candidate_start = max(candidate_start, line_span.start_sec)
    candidate_end = min(candidate_end, line_span.end_sec)

    if candidate_end <= candidate_start:
        return line_span, None

    before_duration = line_span.end_sec - line_span.start_sec
    after_duration = candidate_end - candidate_start
    if before_duration - after_duration < config.min_adjustment_sec:
        return line_span, None

    original_speech_ratio = None
    candidate_speech_ratio = None
    if speech_segments:
        original_speech_ratio = overlap_ratio(
            line_span.start_sec,
            line_span.end_sec,
            speech_segments,
        )
        candidate_speech_ratio = overlap_ratio(candidate_start, candidate_end, speech_segments)
        # Edge-gap trimming is accepted only when the remaining token cluster is more
        # consistent with VAD speech coverage. This keeps real multi-phrase pauses intact.
        if candidate_speech_ratio < config.min_candidate_speech_ratio:
            return line_span, None
        if candidate_speech_ratio < original_speech_ratio + config.min_coverage_gain:
            return line_span, None

    side = correction_side(start_index, end_index, len(tokens))
    correction = TokenGapCorrection(
        line_index=line_index,
        text=line_span.text,
        side=side,
        before_start_sec=line_span.start_sec,
        before_end_sec=line_span.end_sec,
        after_start_sec=candidate_start,
        after_end_sec=candidate_end,
        removed_leading_tokens=start_index,
        removed_trailing_tokens=len(tokens) - 1 - end_index,
        leading_gap_sec=leading_gap,
        trailing_gap_sec=trailing_gap,
        original_speech_ratio=original_speech_ratio,
        candidate_speech_ratio=candidate_speech_ratio,
    )
    return (
        LineSpan(
            text=line_span.text,
            start_sec=candidate_start,
            end_sec=candidate_end,
            token_spans=line_span.token_spans,
        ),
        correction,
    )


def correction_side(start_index: int, end_index: int, token_count: int) -> str:
    leading = start_index > 0
    trailing = end_index < token_count - 1
    if leading and trailing:
        return "both"
    if leading:
        return "leading"
    return "trailing"


def expand_to_nearby_speech(
    *,
    start_sec: float,
    end_sec: float,
    speech_segments: list[VadSegment],
    max_gap_sec: float,
) -> tuple[float, float]:
    if not speech_segments:
        return start_sec, end_sec

    changed = True
    expanded_start = start_sec
    expanded_end = end_sec
    while changed:
        changed = False
        for segment in speech_segments:
            if segment.end_sec < expanded_start - max_gap_sec:
                continue
            if segment.start_sec > expanded_end + max_gap_sec:
                break
            new_start = min(expanded_start, segment.start_sec)
            new_end = max(expanded_end, segment.end_sec)
            if new_start != expanded_start or new_end != expanded_end:
                expanded_start = new_start
                expanded_end = new_end
                changed = True
    return expanded_start, expanded_end


def overlap_ratio(
    start_sec: float,
    end_sec: float,
    speech_segments: list[VadSegment],
) -> float:
    duration = end_sec - start_sec
    if duration <= 0:
        return 0.0
    return overlap_duration(start_sec, end_sec, speech_segments) / duration


def overlap_duration(
    start_sec: float,
    end_sec: float,
    speech_segments: list[VadSegment],
) -> float:
    total = 0.0
    for segment in speech_segments:
        if segment.end_sec <= start_sec:
            continue
        if segment.start_sec >= end_sec:
            break
        overlap_start = max(start_sec, segment.start_sec)
        overlap_end = min(end_sec, segment.end_sec)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total
