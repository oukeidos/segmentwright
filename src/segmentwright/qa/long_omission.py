from __future__ import annotations

from dataclasses import asdict, dataclass

from segmentwright.aligners.base import LineSpan
from segmentwright.vad.base import VadSegment

from .coverage import compute_coverage
from .intervals import TimeInterval


@dataclass(frozen=True)
class LongOmission:
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass(frozen=True)
class LongOmissionReport:
    threshold_sec: float
    coverage_gap_merge_sec: float
    speech_intervals: list[TimeInterval]
    subtitle_intervals: list[TimeInterval]
    uncovered_intervals: list[TimeInterval]
    long_omissions: list[LongOmission]

    @property
    def has_long_omissions(self) -> bool:
        return bool(self.long_omissions)

    def to_dict(self) -> dict:
        return {
            "schema": "segmentwright.long_omission_report.v1",
            "threshold_sec": self.threshold_sec,
            "coverage_gap_merge_sec": self.coverage_gap_merge_sec,
            "speech_intervals": [asdict(interval) for interval in self.speech_intervals],
            "subtitle_intervals": [asdict(interval) for interval in self.subtitle_intervals],
            "uncovered_intervals": [asdict(interval) for interval in self.uncovered_intervals],
            "long_omissions": [asdict(omission) for omission in self.long_omissions],
            "has_long_omissions": self.has_long_omissions,
        }


def detect_long_omissions(
    *,
    speech_segments: list[VadSegment],
    line_spans: list[LineSpan],
    threshold_sec: float,
    coverage_gap_merge_sec: float,
) -> LongOmissionReport:
    coverage = compute_coverage(
        speech_segments=speech_segments,
        line_spans=line_spans,
        coverage_gap_merge_sec=coverage_gap_merge_sec,
    )
    long_omissions = [
        LongOmission(
            start_sec=interval.start_sec,
            end_sec=interval.end_sec,
            duration_sec=interval.duration_sec,
        )
        for interval in coverage.uncovered_intervals
        if interval.duration_sec >= threshold_sec
    ]
    return LongOmissionReport(
        threshold_sec=threshold_sec,
        coverage_gap_merge_sec=coverage_gap_merge_sec,
        speech_intervals=coverage.speech_intervals,
        subtitle_intervals=coverage.subtitle_intervals,
        uncovered_intervals=coverage.uncovered_intervals,
        long_omissions=long_omissions,
    )
