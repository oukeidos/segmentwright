from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TokenSpan:
    token: str
    start_sec: float
    end_sec: float
    score: float | None = None


@dataclass(frozen=True)
class LineSpan:
    text: str
    start_sec: float
    end_sec: float
    token_spans: list[TokenSpan]


@dataclass(frozen=True)
class AlignmentResult:
    line_spans: list[LineSpan]
    raw: dict


class AlignerBackend(Protocol):
    def align(self, audio_path: Path, lines: list[str], offset: float) -> AlignmentResult: ...
