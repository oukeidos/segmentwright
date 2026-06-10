from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .transcript import Transcript, parse_plain_text_transcript, parse_transcript


@dataclass(frozen=True)
class TranscriptCacheKey:
    source_id: str
    chunk_start_sec: float
    chunk_end_sec: float
    chunk_audio_hash: str
    gemini_model: str
    prompt_preset: str | None
    prompt_hash: str
    response_format: str
    segmentation: str
    preset: str | None = None
    language: str = "ja"
    gemini_thinking_level: str | None = None
    gemini_temperature: float | None = None

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedTranscript:
    transcript: Transcript
    metadata: dict[str, Any]
    path: Path
    raw_payload: dict[str, Any]


class TranscriptCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def path_for(self, key: TranscriptCacheKey) -> Path:
        return self.cache_dir / "gemini_transcripts" / f"{key.digest()}.json"

    def load(self, key: TranscriptCacheKey) -> Transcript | None:
        entry = self.load_entry(key)
        return entry.transcript if entry is not None else None

    def load_entry(self, key: TranscriptCacheKey) -> CachedTranscript | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_json = data.get("raw_json")
        if not isinstance(raw_json, str):
            return None
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        response_format = data.get("response_format")
        transcript = (
            parse_plain_text_transcript(raw_json, allow_empty=True)
            if response_format == "plain_text"
            else parse_transcript(raw_json)
        )
        return CachedTranscript(
            transcript=transcript,
            metadata=metadata,
            path=path,
            raw_payload=data,
        )

    def save(
        self,
        key: TranscriptCacheKey,
        transcript: Transcript,
        metadata: dict[str, Any],
    ) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "segmentwright.gemini_transcript_cache.v1",
            "key": asdict(key),
            "metadata": metadata,
            "response_format": key.response_format,
            "lines": transcript.lines,
            "raw_json": transcript.raw_json,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
