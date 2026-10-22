from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ResegmentationSuggestion:
    lines: list[str]
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "segmentwright.resegmentation_suggestion.v1",
            "line_count": len(self.lines),
            "lines": self.lines,
            "raw": self.raw,
        }


@dataclass(frozen=True)
class TextEditReport:
    source_non_whitespace_character_count: int
    suggested_non_whitespace_character_count: int
    suggested_matches_source_non_whitespace: bool
    opcode_count: int
    changed_opcode_count: int
    changed_spans: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "segmentwright.resegmentation_text_edit_report.v1", **asdict(self)}


@dataclass(frozen=True)
class ProjectedSegment:
    index: int
    text: str
    source_start: int
    source_end: int
    normalized_start: int
    normalized_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectionReport:
    source_non_whitespace_character_count: int
    suggested_non_whitespace_character_count: int
    projected_non_whitespace_character_count: int
    projected_matches_source_non_whitespace: bool
    used_source_text_projection: bool
    projection_strategy: str
    mismatch: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "segmentwright.resegmentation_projection_report.v1", **asdict(self)}


@dataclass(frozen=True)
class ResegmentationResult:
    source_text: str
    suggestion: ResegmentationSuggestion
    text_edit_report: TextEditReport
    projection_report: ProjectionReport
    projected_segments: list[ProjectedSegment]

    @property
    def lines(self) -> list[str]:
        return [segment.text for segment in self.projected_segments]


def project_suggestion_to_source(
    source_text: str,
    suggested_lines: list[str],
    *,
    raw_suggestion: dict[str, Any] | None = None,
    preserve_line_breaks: bool = True,
) -> ResegmentationResult:
    cleaned_suggestions = [str(line).strip() for line in suggested_lines if str(line).strip()]
    if not source_text.strip():
        raise ValueError("Cannot resegment empty transcript text")
    if not cleaned_suggestions:
        raise ValueError("Local resegmentation did not return any segment suggestions")

    source_normalized = non_whitespace_text(source_text)
    suggested_normalized = non_whitespace_text("\n".join(cleaned_suggestions))
    text_edit_report = build_text_edit_report(source_normalized, suggested_normalized)
    line_lengths = [len(non_whitespace_text(line)) for line in cleaned_suggestions]
    suggested_to_source = map_suggested_boundary_to_source_indices(
        source_normalized,
        suggested_normalized,
    )

    source_boundaries = [0]
    suggested_cursor = 0
    for line_length in line_lengths[:-1]:
        suggested_cursor += line_length
        source_boundaries.append(suggested_to_source[suggested_cursor])
    source_boundaries.append(len(source_normalized))

    if preserve_line_breaks:
        # Original line boundaries are mandatory; the LLM may only add boundaries.
        source_cursor = 0
        for source_line in source_text.splitlines(keepends=True):
            source_cursor += len(non_whitespace_text(source_line))
            source_boundaries.append(source_cursor)
        source_boundaries = sorted(set(source_boundaries))

    segments: list[ProjectedSegment] = []
    for start_normalized, end_normalized in zip(
        source_boundaries, source_boundaries[1:], strict=False
    ):
        start = source_text_position_for_normalized_index(source_text, start_normalized)
        end = source_text_position_for_normalized_index(source_text, end_normalized)
        text = source_text[start:end].strip()
        if not text:
            continue
        segments.append(
            ProjectedSegment(
                index=len(segments),
                text=text,
                source_start=start,
                source_end=end,
                normalized_start=start_normalized,
                normalized_end=end_normalized,
            )
        )

    if not segments:
        raise ValueError("Projected resegmentation produced no non-empty source segments")

    projected_normalized = non_whitespace_text("\n".join(segment.text for segment in segments))
    warnings = []
    if suggested_normalized != source_normalized:
        warnings.append("local_llm_suggestion_changed_non_whitespace_text")
    if projected_normalized != source_normalized:
        warnings.append("projected_segments_do_not_preserve_source_non_whitespace_text")
    projection_report = ProjectionReport(
        source_non_whitespace_character_count=len(source_normalized),
        suggested_non_whitespace_character_count=len(suggested_normalized),
        projected_non_whitespace_character_count=len(projected_normalized),
        projected_matches_source_non_whitespace=projected_normalized == source_normalized,
        used_source_text_projection=True,
        projection_strategy="sequence_matcher_boundary_mapping",
        mismatch=projected_normalized != source_normalized,
        warnings=warnings,
    )
    return ResegmentationResult(
        source_text=source_text,
        suggestion=ResegmentationSuggestion(lines=cleaned_suggestions, raw=raw_suggestion),
        text_edit_report=text_edit_report,
        projection_report=projection_report,
        projected_segments=segments,
    )


def build_text_edit_report(source_normalized: str, suggested_normalized: str) -> TextEditReport:
    matcher = difflib.SequenceMatcher(None, source_normalized, suggested_normalized, autojunk=False)
    opcodes = matcher.get_opcodes()
    changed = []
    for tag, source_start, source_end, suggested_start, suggested_end in opcodes:
        if tag == "equal":
            continue
        changed.append(
            {
                "tag": tag,
                "source_start": source_start,
                "source_end": source_end,
                "suggested_start": suggested_start,
                "suggested_end": suggested_end,
                "source_text": source_normalized[source_start:source_end],
                "suggested_text": suggested_normalized[suggested_start:suggested_end],
            }
        )
    return TextEditReport(
        source_non_whitespace_character_count=len(source_normalized),
        suggested_non_whitespace_character_count=len(suggested_normalized),
        suggested_matches_source_non_whitespace=source_normalized == suggested_normalized,
        opcode_count=len(opcodes),
        changed_opcode_count=len(changed),
        changed_spans=changed[:50],
    )


def non_whitespace_text(text: str) -> str:
    return WHITESPACE_RE.sub("", text)


def map_suggested_boundary_to_source_indices(source: str, suggested: str) -> list[int]:
    mapping = [0] * (len(suggested) + 1)
    matcher = difflib.SequenceMatcher(None, source, suggested, autojunk=False)
    for tag, source_start, source_end, suggested_start, suggested_end in matcher.get_opcodes():
        suggested_span = suggested_end - suggested_start
        source_span = source_end - source_start
        if suggested_span == 0:
            continue
        for offset in range(suggested_span + 1):
            suggested_index = suggested_start + offset
            if suggested_index > len(suggested):
                continue
            if tag == "equal":
                source_index = source_start + min(offset, source_span)
            else:
                ratio = offset / suggested_span
                source_index = round(source_start + ratio * source_span)
            mapping[suggested_index] = max(0, min(len(source), source_index))

    cursor = 0
    for index, value in enumerate(mapping):
        cursor = max(cursor, value)
        mapping[index] = cursor
    mapping[0] = 0
    mapping[-1] = len(source)
    return mapping


def source_text_position_for_normalized_index(source_text: str, normalized_index: int) -> int:
    if normalized_index <= 0:
        return 0
    seen = 0
    for position, character in enumerate(source_text):
        if character.isspace():
            continue
        seen += 1
        if seen >= normalized_index:
            return position + 1
    return len(source_text)
