from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .projection import ResegmentationResult, non_whitespace_text

SENTENCE_BOUNDARY_CHARS = "。．｡.!！？?؟۔।॥။៕។།༎።፧፨։…⋯‥"
SOFT_BOUNDARY_CHARS = "、､,，;；:：،؛၊៖፡፣፤፥՝"
SENTENCE_BOUNDARY_RE = re.compile(f"[{re.escape(SENTENCE_BOUNDARY_CHARS)}]+")
SOFT_BOUNDARY_RE = re.compile(f"(?<=[{re.escape(SOFT_BOUNDARY_CHARS)}])|\\s+")


@dataclass(frozen=True)
class ResegmentationQualityConfig:
    min_suggested_source_ratio: float = 0.9
    max_suggested_source_ratio: float = 1.1
    max_projected_segment_non_ws_chars: int = 120


@dataclass(frozen=True)
class ResegmentationQualityReport:
    accepted: bool
    reasons: list[str]
    warnings: list[str]
    source_non_whitespace_character_count: int
    suggested_non_whitespace_character_count: int
    suggested_source_ratio: float | None
    projected_segment_count: int
    max_projected_segment_non_whitespace_character_count: int
    max_projected_segment_character_count: int
    config: ResegmentationQualityConfig

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "segmentwright.resegmentation_quality_report.v1", **asdict(self)}


def assess_resegmentation_quality(
    result: ResegmentationResult,
    *,
    config: ResegmentationQualityConfig | None = None,
) -> ResegmentationQualityReport:
    config = config or ResegmentationQualityConfig()
    source_count = result.text_edit_report.source_non_whitespace_character_count
    suggested_count = result.text_edit_report.suggested_non_whitespace_character_count
    ratio = suggested_count / source_count if source_count else None
    projected_counts = [
        len(non_whitespace_text(segment.text)) for segment in result.projected_segments
    ]
    projected_char_counts = [len(segment.text) for segment in result.projected_segments]
    max_projected_non_ws = max(projected_counts, default=0)
    max_projected_chars = max(projected_char_counts, default=0)

    reasons: list[str] = []
    warnings: list[str] = []
    if not result.projection_report.projected_matches_source_non_whitespace:
        reasons.append("projected_segments_do_not_preserve_source_non_whitespace_text")
    if ratio is not None and ratio < config.min_suggested_source_ratio:
        reasons.append("local_llm_suggestion_too_short_for_source")
    if ratio is not None and ratio > config.max_suggested_source_ratio:
        warnings.append("local_llm_suggestion_too_long_for_source")
    if max_projected_non_ws > config.max_projected_segment_non_ws_chars:
        warnings.append("projected_segment_too_long")

    return ResegmentationQualityReport(
        accepted=not reasons,
        reasons=reasons,
        warnings=warnings,
        source_non_whitespace_character_count=source_count,
        suggested_non_whitespace_character_count=suggested_count,
        suggested_source_ratio=ratio,
        projected_segment_count=len(result.projected_segments),
        max_projected_segment_non_whitespace_character_count=max_projected_non_ws,
        max_projected_segment_character_count=max_projected_chars,
        config=config,
    )


def fallback_segment_source_text(
    source_text: str,
    *,
    max_non_ws_chars: int = 40,
) -> list[str]:
    if max_non_ws_chars <= 0:
        raise ValueError("max_non_ws_chars must be greater than 0")

    output: list[str] = []
    for line in [item.strip() for item in source_text.splitlines() if item.strip()]:
        for sentence in split_by_sentence_boundary(line):
            output.extend(split_long_segment(sentence, max_non_ws_chars=max_non_ws_chars))
    return output or [source_text.strip()]


def split_by_sentence_boundary(text: str) -> list[str]:
    parts: list[str] = []
    cursor = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end()
        part = text[cursor:end].strip()
        if part:
            parts.append(part)
        cursor = end
    tail = text[cursor:].strip()
    if tail:
        parts.append(tail)
    return parts or [text.strip()]


def split_long_segment(text: str, *, max_non_ws_chars: int) -> list[str]:
    if len(non_whitespace_text(text)) <= max_non_ws_chars:
        return [text.strip()]

    parts = [part.strip() for part in SOFT_BOUNDARY_RE.split(text) if part.strip()]
    if len(parts) <= 1:
        return hard_split_by_non_ws(text, max_non_ws_chars=max_non_ws_chars)

    output: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + part).strip() if current else part
        if current and len(non_whitespace_text(candidate)) > max_non_ws_chars:
            output.extend(hard_split_by_non_ws(current, max_non_ws_chars=max_non_ws_chars))
            current = part
        else:
            current = candidate
    if current:
        output.extend(hard_split_by_non_ws(current, max_non_ws_chars=max_non_ws_chars))
    return output


def hard_split_by_non_ws(text: str, *, max_non_ws_chars: int) -> list[str]:
    output: list[str] = []
    start = 0
    seen = 0
    for index, character in enumerate(text):
        if character.isspace():
            continue
        seen += 1
        if seen >= max_non_ws_chars:
            part = text[start : index + 1].strip()
            if part:
                output.append(part)
            start = index + 1
            seen = 0
    tail = text[start:].strip()
    if tail:
        output.append(tail)
    return output
