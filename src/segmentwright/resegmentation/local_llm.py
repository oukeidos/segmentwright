from __future__ import annotations

import json

# ruff: noqa: E501
from dataclasses import dataclass
from typing import Any
from urllib import request

from .projection import ResegmentationSuggestion

RESEGMENTATION_PROMPTS = {
    "short": """You are a professional subtitle editor.

# Core Rules
- Divide text into the smallest possible grammatical units.
- **MANDATORY: Insert a line break after conjunctive particles.**
- **MANDATORY: Even if the sentence is very short, you MUST split it at the conjunction.**
- Each line should contain only ONE clause.
- Output ONLY the formatted text.

# Task

Input:
{text}

Output:
""",
    "medium": """You are a subtitle editor.

# Core Rules
- Do not rewrite. Keep all words and punctuation; only insert line breaks.
- Break into subtitle-friendly thought groups.
- Prefer punctuation and clear clause boundaries.
- Split after a conjunction only if it starts a new clause.
- Output only the formatted text.

# Task

Input:
{text}

Output:
""",
    "long": """You are a subtitle segmenter.

# Core Rules
1. **Strict Fidelity**: Output input text exactly. Only add line breaks.
2. **Group Short Items**: Keep short phrases or lists on the same line, even if they have commas (e.g., "A, B, and C").
3. **Split by Length**: Only split commas if the line is becoming too long or marks a major transition.
4. **Visual Balance**: Aim for lines that are neither too short (1-2 words) nor too long.

# Example

Input:
He checked the map, packed his bag, and put on his boots, but he realized the storm was too strong to leave the shelter safely.

Output:
He checked the map, packed his bag, and put on his boots,
but he realized the storm was too strong
to leave the shelter safely.

# Task

Input:
{text}

Output:
""",
    "full": """You are a subtitle segmenter.

# Core Rules
1. **Strict Fidelity**: Output input text exactly. Do not change words or punctuation.
2. **Split ONLY at Sentence Endings**: Insert a line break **ONLY** after a full stop, question mark, or exclamation mark.
3. **No Internal Splitting**: Do **NOT** split lines at commas, conjunctions, or clauses. Even very long sentences must remain on a single line.
4. **One Sentence = One Segment**: Each segment must contain exactly one full sentence.

# Task

Input:
{text}

Output:
"""
}
RESEGMENTATION_PROMPT_PRESETS = frozenset(RESEGMENTATION_PROMPTS)
DEFAULT_RESEGMENTATION_PROMPT_PRESET = "medium"
DEFAULT_RESEGMENTATION_NUM_CTX = 4096
TEMPERATURE_ZERO_RESEGMENTATION_MODELS = frozenset(
    {
        "gemma3:4b-it-qat",
        "gemma3:12b-it-qat",
    }
)


@dataclass(frozen=True)
class OllamaSegmentationConfig:
    model: str
    ollama_url: str
    prompt_preset: str = DEFAULT_RESEGMENTATION_PROMPT_PRESET
    timeout_sec: float = 300.0


def call_ollama_resegmentation(
    text: str,
    *,
    config: OllamaSegmentationConfig,
) -> ResegmentationSuggestion:
    payload = build_ollama_resegmentation_payload(text, config)
    url = config.ollama_url.rstrip("/") + "/api/generate"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=config.timeout_sec) as response:  # noqa: S310
        raw_payload = json.loads(response.read().decode("utf-8"))
    response_text = raw_payload.get("response")
    if not isinstance(response_text, str):
        raise RuntimeError("Ollama resegmentation response did not contain a text response")
    lines = parse_segmented_lines(response_text)
    return ResegmentationSuggestion(
        lines=lines,
        raw={
            "request": {
                "model": config.model,
                "prompt_preset": config.prompt_preset,
                "prompt": payload["prompt"],
                "stream": payload["stream"],
                "think": payload.get("think"),
                "keep_alive": payload["keep_alive"],
                "options": payload["options"],
            },
            "response": raw_payload,
        },
    )


def build_ollama_resegmentation_payload(
    text: str,
    config: OllamaSegmentationConfig,
) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "prompt": build_resegmentation_prompt(text, config.prompt_preset),
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": resegmentation_options_for_model(config.model),
    }
    return payload


def resegmentation_options_for_model(model: str) -> dict[str, int | float]:
    options: dict[str, int | float] = {"num_ctx": DEFAULT_RESEGMENTATION_NUM_CTX}
    if model.lower() in TEMPERATURE_ZERO_RESEGMENTATION_MODELS:
        options["temperature"] = 0.0
    return options


def build_resegmentation_prompt(
    text: str,
    prompt_preset: str = DEFAULT_RESEGMENTATION_PROMPT_PRESET,
) -> str:
    try:
        template = RESEGMENTATION_PROMPTS[prompt_preset]
    except KeyError as exc:
        supported = ", ".join(sorted(RESEGMENTATION_PROMPTS))
        raise ValueError(
            f"Unknown resegmentation prompt preset {prompt_preset!r}; expected one of: {supported}"
        ) from exc
    return template.format(text=text)


def parse_segmented_lines(response_text: str) -> list[str]:
    text = strip_markdown_fence(response_text.strip())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Ollama resegmentation returned no segment lines")
    return lines


def strip_markdown_fence(raw: str) -> str:
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return raw


def suggestion_to_jsonable(suggestion: ResegmentationSuggestion) -> dict[str, Any]:
    return suggestion.to_dict()
