from __future__ import annotations

import pytest

from segmentwright.aligners.base import LineSpan, TokenSpan
from segmentwright.qa.intervals import TimeInterval, subtract_intervals
from segmentwright.qa.long_omission import detect_long_omissions
from segmentwright.qa.token_gap_guard import apply_token_gap_guard
from segmentwright.vad.base import VadSegment


def test_subtract_intervals_returns_uncovered_ranges() -> None:
    uncovered = subtract_intervals(
        [TimeInterval(0.0, 10.0)],
        [TimeInterval(1.0, 2.0), TimeInterval(4.0, 8.0)],
    )

    assert uncovered == [
        TimeInterval(0.0, 1.0),
        TimeInterval(2.0, 4.0),
        TimeInterval(8.0, 10.0),
    ]


def test_detect_long_omissions_merges_subtitle_coverage_gaps() -> None:
    report = detect_long_omissions(
        speech_segments=[VadSegment(0.0, 120.0)],
        line_spans=[
            line_span("a", 0.0, 20.0),
            line_span("b", 21.5, 40.0),
            line_span("c", 100.0, 120.0),
        ],
        threshold_sec=55.0,
        coverage_gap_merge_sec=2.0,
    )

    assert report.has_long_omissions
    assert len(report.long_omissions) == 1
    assert report.long_omissions[0].start_sec == 40.0
    assert report.long_omissions[0].end_sec == 100.0


def test_token_gap_guard_shifts_isolated_first_token_start() -> None:
    line = LineSpan(
        text="うまい",
        start_sec=41.52,
        end_sec=43.52,
        token_spans=[
            TokenSpan("う", 41.52, 41.60, 0.7),
            TokenSpan("ま", 43.20, 43.28, 0.8),
            TokenSpan("い", 43.44, 43.52, 0.9),
        ],
    )

    result = apply_token_gap_guard(
        [line],
        speech_segments=[VadSegment(43.1, 43.6)],
    )

    assert len(result.corrections) == 1
    assert result.corrections[0].side == "leading"
    assert result.line_spans[0].start_sec == pytest.approx(43.04)
    assert result.line_spans[0].end_sec == 43.52


def test_token_gap_guard_shifts_isolated_last_token_end() -> None:
    line = LineSpan(
        text="書け、書け、書くんだ、ゆい！",
        start_sec=797.2,
        end_sec=819.6,
        token_spans=[
            TokenSpan("書", 797.2, 797.3),
            TokenSpan("け", 797.5, 797.6),
            TokenSpan("書", 798.4, 798.5),
            TokenSpan("け", 798.7, 798.8),
            TokenSpan("書", 799.8, 799.9),
            TokenSpan("く", 800.0, 800.1),
            TokenSpan("んだ", 800.2, 800.3),
            TokenSpan("ゆ", 800.4, 800.5),
            TokenSpan("い", 819.5, 819.6),
        ],
    )

    result = apply_token_gap_guard(
        [line],
        speech_segments=[
            VadSegment(797.5, 798.2),
            VadSegment(798.7, 799.4),
            VadSegment(799.9, 802.0),
        ],
    )

    assert len(result.corrections) == 1
    assert result.corrections[0].side == "trailing"
    assert result.corrections[0].removed_trailing_tokens == 1
    assert result.line_spans[0].start_sec == pytest.approx(797.2)
    assert result.line_spans[0].end_sec == pytest.approx(802.0)


def test_token_gap_guard_keeps_real_trailing_phrase_cluster() -> None:
    line = LineSpan(
        text="これはかなり長い発話です",
        start_sec=10.0,
        end_sec=16.0,
        token_spans=[
            TokenSpan("こ", 10.0, 10.1),
            TokenSpan("れ", 10.2, 10.3),
            TokenSpan("は", 10.4, 10.5),
            TokenSpan("長", 10.6, 10.7),
            TokenSpan("い", 10.8, 10.9),
            TokenSpan("発", 15.0, 15.1),
            TokenSpan("話", 15.2, 15.3),
        ],
    )

    result = apply_token_gap_guard(
        [line],
        speech_segments=[
            VadSegment(10.0, 11.8),
            VadSegment(15.0, 15.4),
        ],
    )

    assert not result.corrections
    assert result.line_spans[0] == line


def line_span(text: str, start: float, end: float) -> LineSpan:
    return LineSpan(
        text=text,
        start_sec=start,
        end_sec=end,
        token_spans=[TokenSpan(text, start, end)],
    )
