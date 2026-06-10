from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class VadSegment:
    start_sec: float
    end_sec: float
    confidence: float | None = None

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class VadResult:
    duration_sec: float
    speech_segments: list[VadSegment]
    raw: dict


class VadBackend(Protocol):
    def detect(self, audio_path: Path) -> VadResult: ...
