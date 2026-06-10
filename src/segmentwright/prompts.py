from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .transcript import ENGLISH_NO_SPEECH_SENTINEL, JAPANESE_NO_SPEECH_SENTINEL

JAPANESE_SUBTITLE_LINES_PROMPT = (
    "Transcribe all spoken Japanese into subtitle-style transcript lines, including brief "
    "narration, voice-over, and bridge lines between dialogue turns. Start a new line when the "
    "speaker changes or may have changed. Also split long utterances more aggressively at "
    "Japanese punctuation and natural clause boundaries, especially after 。, 、, ？, ！, or clear "
    "phrase endings. Keep each line short enough for video subtitles while preserving the spoken "
    'wording. Return only JSON with a "lines" array; each item must contain exactly one '
    'transcript line in "text". Do not add timestamps, translations, explanations, markdown, '
    'speaker labels, or extra fields. If there is no spoken Japanese, return exactly {"lines":[]}.'
)

JAPANESE_PLAIN_TRANSCRIPT_PROMPT = (
    "Transcribe all spoken Japanese, including brief narration, voice-over, and bridge speech "
    "between dialogue turns. Output only the Japanese transcription. Do not add timestamps, "
    "translations, explanations, markdown, speaker labels, or labels. If there is no spoken "
    f"Japanese to transcribe, output exactly this single line: {JAPANESE_NO_SPEECH_SENTINEL}. "
    "Do not output any other text in that no-speech case."
)

ENGLISH_SUBTITLE_LINES_PROMPT = (
    "Transcribe all spoken English into subtitle-style transcript lines, including brief "
    "narration, voice-over, and bridge speech between dialogue turns. Start a new line when the "
    "speaker changes or may have changed. Also split long utterances at English punctuation and "
    "natural clause boundaries, especially after periods, commas, question marks, exclamation "
    "marks, or clear phrase endings. Keep each line short enough for video subtitles while "
    'preserving the spoken wording. Return only JSON with a "lines" array; each item must contain '
    'exactly one transcript line in "text". Do not add timestamps, translations, explanations, '
    "markdown, speaker labels, or extra fields. If there is no spoken English, return exactly "
    '{"lines":[]}.'
)

ENGLISH_PLAIN_TRANSCRIPT_PROMPT = (
    "Transcribe all spoken English, including brief narration, voice-over, and bridge speech "
    "between dialogue turns. Output only the English transcription. Do not add timestamps, "
    "translations, explanations, markdown, speaker labels, or labels. If there is no spoken "
    f"English to transcribe, output exactly this single line: {ENGLISH_NO_SPEECH_SENTINEL}. "
    "Do not output any other text in that no-speech case."
)

SUBTITLE_LINES_PROMPT = JAPANESE_SUBTITLE_LINES_PROMPT
PLAIN_TRANSCRIPT_PROMPT = JAPANESE_PLAIN_TRANSCRIPT_PROMPT
SUBTITLE_LINES_PROMPT_PRESET = "subtitle-lines"
PLAIN_TRANSCRIPT_PROMPT_PRESET = "plain-transcript"
PROMPT_PRESETS = {
    SUBTITLE_LINES_PROMPT_PRESET: SUBTITLE_LINES_PROMPT,
    PLAIN_TRANSCRIPT_PROMPT_PRESET: PLAIN_TRANSCRIPT_PROMPT,
}
LANGUAGE_PROMPT_PRESETS = {
    "ja": {
        SUBTITLE_LINES_PROMPT_PRESET: JAPANESE_SUBTITLE_LINES_PROMPT,
        PLAIN_TRANSCRIPT_PROMPT_PRESET: JAPANESE_PLAIN_TRANSCRIPT_PROMPT,
    },
    "en": {
        SUBTITLE_LINES_PROMPT_PRESET: ENGLISH_SUBTITLE_LINES_PROMPT,
        PLAIN_TRANSCRIPT_PROMPT_PRESET: ENGLISH_PLAIN_TRANSCRIPT_PROMPT,
    },
}


@dataclass(frozen=True)
class PromptSpec:
    text: str
    source: str
    language: str
    preset: str | None
    file: Path | None
    sha256: str


def get_prompt(preset: str, *, language: str = "ja") -> str:
    try:
        language_prompts = LANGUAGE_PROMPT_PRESETS[language]
    except KeyError as exc:
        available = ", ".join(sorted(LANGUAGE_PROMPT_PRESETS))
        raise ValueError(f"Unknown prompt language {language!r}; available: {available}") from exc
    try:
        return language_prompts[preset]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_PRESETS))
        raise ValueError(f"Unknown prompt preset {preset!r}; available: {available}") from exc


def resolve_prompt(
    *,
    prompt_preset: str,
    language: str = "ja",
    prompt_text: str | None = None,
    prompt_file: Path | None = None,
) -> PromptSpec:
    if prompt_text is not None and prompt_file is not None:
        raise ValueError("--prompt-file and --prompt-text are mutually exclusive")
    if prompt_text is not None:
        text = prompt_text.strip()
        if not text:
            raise ValueError("--prompt-text must not be empty")
        return PromptSpec(
            text=text,
            source="custom_text",
            language=language,
            preset=None,
            file=None,
            sha256=hash_prompt(text),
        )
    if prompt_file is not None:
        text = prompt_file.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file is empty: {prompt_file}")
        return PromptSpec(
            text=text,
            source="custom_file",
            language=language,
            preset=None,
            file=prompt_file,
            sha256=hash_prompt(text),
        )
    text = get_prompt(prompt_preset, language=language)
    return PromptSpec(
        text=text,
        source="built_in",
        language=language,
        preset=prompt_preset,
        file=None,
        sha256=hash_prompt(text),
    )


def hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
