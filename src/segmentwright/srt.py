from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SrtCue:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class FinalAssemblyReport:
    source_duration_sec: float
    input_cues: int
    output_cues: int
    input_non_positive_duration_cues: int
    output_non_positive_duration_cues: int
    output_order_errors: int
    output_overlap_errors: int
    output_out_of_range_cues: int
    max_output_end_sec: float | None

    @property
    def valid(self) -> bool:
        return (
            self.output_non_positive_duration_cues == 0
            and self.output_order_errors == 0
            and self.output_overlap_errors == 0
            and self.output_out_of_range_cues == 0
        )

    def to_dict(self) -> dict:
        return {
            "schema": "segmentwright.final_assembly_report.v1",
            **asdict(self),
            "valid": self.valid,
        }


def format_srt(cues: list[SrtCue]) -> str:
    lines: list[str] = []
    sorted_cues = sorted(cues, key=lambda item: (item.start_sec, item.end_sec))
    for index, cue in enumerate(sorted_cues, start=1):
        if cue.end_sec <= cue.start_sec:
            continue
        lines.append(str(index))
        lines.append(f"{format_timestamp(cue.start_sec)} --> {format_timestamp(cue.end_sec)}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def stabilize_cues(cues: list[SrtCue], min_duration_sec: float = 0.05) -> list[SrtCue]:
    if min_duration_sec <= 0:
        raise ValueError("min_duration_sec must be greater than 0")
    stable: list[SrtCue] = []
    cursor = 0.0
    for cue in sorted(cues, key=lambda item: (item.start_sec, item.end_sec)):
        if not cue.text.strip():
            continue
        if not math.isfinite(cue.start_sec) or not math.isfinite(cue.end_sec):
            raise ValueError(f"Non-finite cue timestamp: {cue}")
        start = max(0.0, cue.start_sec, cursor)
        end = max(cue.end_sec, start + min_duration_sec)
        stable.append(SrtCue(start_sec=start, end_sec=end, text=cue.text))
        cursor = end
    return stable


def build_final_assembly_report(
    *,
    input_cues: list[SrtCue],
    output_cues: list[SrtCue],
    source_duration_sec: float,
) -> FinalAssemblyReport:
    previous_start = -math.inf
    previous_end = -math.inf
    order_errors = 0
    overlap_errors = 0
    non_positive_output = 0
    out_of_range = 0
    max_end: float | None = None
    for cue in output_cues:
        if cue.end_sec <= cue.start_sec:
            non_positive_output += 1
        if cue.start_sec < previous_start:
            order_errors += 1
        if cue.start_sec < previous_end:
            overlap_errors += 1
        if cue.start_sec < 0 or cue.end_sec > source_duration_sec:
            out_of_range += 1
        previous_start = cue.start_sec
        previous_end = cue.end_sec
        max_end = cue.end_sec if max_end is None else max(max_end, cue.end_sec)
    return FinalAssemblyReport(
        source_duration_sec=source_duration_sec,
        input_cues=len(input_cues),
        output_cues=len(output_cues),
        input_non_positive_duration_cues=sum(
            1 for cue in input_cues if cue.end_sec <= cue.start_sec
        ),
        output_non_positive_duration_cues=non_positive_output,
        output_order_errors=order_errors,
        output_overlap_errors=overlap_errors,
        output_out_of_range_cues=out_of_range,
        max_output_end_sec=max_end,
    )


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_ms = round(seconds * 1000)
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
