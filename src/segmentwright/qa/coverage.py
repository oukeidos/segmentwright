from __future__ import annotations

from dataclasses import asdict, dataclass

from segmentwright.aligners.base import LineSpan
from segmentwright.vad.base import VadSegment

from .intervals import TimeInterval, merge_intervals, subtract_intervals


@dataclass(frozen=True)
class CoverageReport:
    coverage_gap_merge_sec: float
    speech_intervals: list[TimeInterval]
    subtitle_intervals: list[TimeInterval]
    uncovered_intervals: list[TimeInterval]

    def to_dict(self) -> dict:
        return {
            "schema": "segmentwright.coverage_report.v1",
            "coverage_gap_merge_sec": self.coverage_gap_merge_sec,
            "speech_intervals": [asdict(interval) for interval in self.speech_intervals],
            "subtitle_intervals": [asdict(interval) for interval in self.subtitle_intervals],
            "uncovered_intervals": [asdict(interval) for interval in self.uncovered_intervals],
        }


def compute_coverage(
    *,
    speech_segments: list[VadSegment],
    line_spans: list[LineSpan],
    coverage_gap_merge_sec: float,
) -> CoverageReport:
    speech_intervals = merge_intervals(
        [TimeInterval(segment.start_sec, segment.end_sec) for segment in speech_segments]
    )
    # Ordinary subtitle cue spacing should not look like missing speech, so coverage
    # intervals are merged before subtracting them from VAD speech regions.
    subtitle_intervals = merge_intervals(
        [TimeInterval(line.start_sec, line.end_sec) for line in line_spans],
        merge_gap_sec=coverage_gap_merge_sec,
    )
    return CoverageReport(
        coverage_gap_merge_sec=coverage_gap_merge_sec,
        speech_intervals=speech_intervals,
        subtitle_intervals=subtitle_intervals,
        uncovered_intervals=subtract_intervals(speech_intervals, subtitle_intervals),
    )
