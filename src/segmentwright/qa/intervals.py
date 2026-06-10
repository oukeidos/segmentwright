from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeInterval:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def merge_intervals(
    intervals: list[TimeInterval],
    *,
    merge_gap_sec: float = 0.0,
) -> list[TimeInterval]:
    valid = [
        TimeInterval(interval.start_sec, interval.end_sec)
        for interval in intervals
        if interval.end_sec > interval.start_sec
    ]
    valid.sort(key=lambda interval: (interval.start_sec, interval.end_sec))
    merged: list[TimeInterval] = []
    for interval in valid:
        if not merged or interval.start_sec - merged[-1].end_sec > merge_gap_sec:
            merged.append(interval)
            continue
        previous = merged[-1]
        merged[-1] = TimeInterval(
            start_sec=previous.start_sec,
            end_sec=max(previous.end_sec, interval.end_sec),
        )
    return merged


def subtract_intervals(
    source_intervals: list[TimeInterval],
    covered_intervals: list[TimeInterval],
) -> list[TimeInterval]:
    uncovered: list[TimeInterval] = []
    covered = merge_intervals(covered_intervals)
    for source in merge_intervals(source_intervals):
        cursor = source.start_sec
        for cover in covered:
            if cover.end_sec <= cursor:
                continue
            if cover.start_sec >= source.end_sec:
                break
            if cover.start_sec > cursor:
                uncovered.append(TimeInterval(cursor, min(cover.start_sec, source.end_sec)))
            cursor = max(cursor, cover.end_sec)
            if cursor >= source.end_sec:
                break
        if cursor < source.end_sec:
            uncovered.append(TimeInterval(cursor, source.end_sec))
    return uncovered
