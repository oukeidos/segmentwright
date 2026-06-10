from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

JAPANESE_NO_SPEECH_SENTINEL = "((NO_SPOKEN_JAPANESE_TO_TRANSCRIBE))"
ENGLISH_NO_SPEECH_SENTINEL = "((NO_SPOKEN_ENGLISH_TO_TRANSCRIBE))"
NO_SPEECH_SENTINELS = frozenset(
    {
        JAPANESE_NO_SPEECH_SENTINEL,
        ENGLISH_NO_SPEECH_SENTINEL,
    }
)


class TranscriptParseError(ValueError):
    """Raised when Gemini transcript JSON does not match the expected shape."""


@dataclass(frozen=True)
class Transcript:
    lines: list[str]
    raw_json: str

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class TranscriptLineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class TranscriptResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lines: list[TranscriptLineModel]


TRANSCRIPT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
    "required": ["lines"],
}


def parse_transcript_lines(raw: str) -> list[str]:
    data = _load_json_object(raw)
    try:
        response = TranscriptResponseModel.model_validate(data)
    except ValidationError as exc:
        raise TranscriptParseError(f"Gemini response schema validation failed: {exc}") from exc

    parsed: list[str] = []
    for line in response.lines:
        text = line.text.strip()
        if text:
            parsed.append(text)

    return parsed


def parse_transcript(raw: str) -> Transcript:
    return Transcript(
        lines=parse_transcript_lines(raw),
        raw_json=_strip_markdown_fence(raw.strip()),
    )


def parse_plain_text_transcript(raw: str, *, allow_empty: bool = False) -> Transcript:
    text = _strip_markdown_fence((raw or "").strip())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentinel_lines = [line for line in lines if is_no_speech_sentinel_line(line)]
    if sentinel_lines:
        if len(lines) == 1:
            return Transcript(lines=[], raw_json=text)
        raise TranscriptParseError(
            "Gemini response mixed a no-speech sentinel with transcript text"
        )
    if not lines and text:
        lines = [text]
    if not lines:
        if allow_empty:
            return Transcript(lines=[], raw_json="")
        raise TranscriptParseError("Gemini response did not contain any transcript text")
    return Transcript(lines=lines, raw_json=text)


def is_no_speech_sentinel_line(text: str) -> bool:
    return text.strip() in NO_SPEECH_SENTINELS


def is_no_speech_sentinel_text(text: str) -> bool:
    stripped = _strip_markdown_fence((text or "").strip())
    return stripped in NO_SPEECH_SENTINELS


def _load_json_object(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise TranscriptParseError("Gemini response was empty")
    raw = _strip_markdown_fence(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TranscriptParseError(f"Gemini response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TranscriptParseError("Gemini response must be a JSON object")
    return data


def _strip_markdown_fence(raw: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    return match.group(1).strip() if match else raw
