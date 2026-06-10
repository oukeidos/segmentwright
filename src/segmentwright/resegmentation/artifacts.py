from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .projection import ResegmentationResult


def write_resegmentation_artifacts(
    *,
    output_dir: Path,
    result: ResegmentationResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input.txt").write_text(result.source_text, encoding="utf-8")
    prompt = extract_prompt(result.suggestion.raw)
    if prompt is not None:
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    write_json(output_dir / "suggestions.json", result.suggestion.to_dict())
    write_json(output_dir / "text_edit_report.json", result.text_edit_report.to_dict())
    write_json(output_dir / "projection.json", result.projection_report.to_dict())
    write_json(
        output_dir / "projected_segments.json",
        {
            "schema": "segmentwright.resegmentation_projected_segments.v1",
            "segment_count": len(result.projected_segments),
            "segments": [segment.to_dict() for segment in result.projected_segments],
        },
    )


def write_json(path: Path, payload: Any) -> None:
    text = json.dumps(serializable(payload), ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


def extract_prompt(raw: dict[str, Any] | None) -> str | None:
    if not isinstance(raw, dict):
        return None
    request = raw.get("request")
    if not isinstance(request, dict):
        return None
    prompt = request.get("prompt")
    return prompt if isinstance(prompt, str) else None


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return value
