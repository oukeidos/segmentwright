import pytest

from segmentwright.srt import (
    SrtCue,
    build_final_assembly_report,
    format_srt,
    format_timestamp,
    stabilize_cues,
)


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(65.4321) == "00:01:05,432"


def test_format_srt_skips_invalid_cues() -> None:
    text = format_srt(
        [
            SrtCue(1.0, 0.5, "bad"),
            SrtCue(0.0, 1.0, "hello"),
        ]
    )

    assert "bad" not in text
    assert "1\n00:00:00,000 --> 00:00:01,000\nhello\n" == text


def test_stabilize_cues_enforces_monotonic_positive_durations() -> None:
    stable = stabilize_cues(
        [
            SrtCue(1.0, 2.0, "a"),
            SrtCue(1.5, 1.6, "b"),
            SrtCue(3.0, 3.0, "c"),
        ]
    )

    assert stable[0] == SrtCue(1.0, 2.0, "a")
    assert stable[1].start_sec == 2.0
    assert stable[1].end_sec == pytest.approx(2.05)
    assert stable[2].start_sec == 3.0
    assert stable[2].end_sec == pytest.approx(3.05)


def test_build_final_assembly_report_detects_output_errors() -> None:
    report = build_final_assembly_report(
        input_cues=[SrtCue(1.0, 0.5, "bad")],
        output_cues=[
            SrtCue(0.0, 1.0, "a"),
            SrtCue(0.5, 0.4, "b"),
            SrtCue(3.0, 6.0, "c"),
        ],
        source_duration_sec=5.0,
    )

    assert report.input_non_positive_duration_cues == 1
    assert report.output_non_positive_duration_cues == 1
    assert report.output_overlap_errors == 1
    assert report.output_out_of_range_cues == 1
    assert not report.valid
