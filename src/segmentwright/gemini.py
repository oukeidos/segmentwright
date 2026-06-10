from __future__ import annotations

import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transcript import (
    TRANSCRIPT_RESPONSE_SCHEMA,
    Transcript,
    TranscriptParseError,
    is_no_speech_sentinel_text,
    parse_plain_text_transcript,
    parse_transcript,
)


@dataclass(frozen=True)
class GeminiTranscriptionResult:
    transcript: Transcript
    metadata: dict[str, Any]


class GeminiTranscriptionError(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


class GeminiTranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        response_format: str = "json_lines",
        thinking_level: str | None = None,
        temperature: float | None = None,
        file_processing_timeout_sec: float = 300.0,
        file_poll_interval_sec: float = 5.0,
    ) -> None:
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.prompt = prompt
        self.response_format = response_format
        self.thinking_level = thinking_level
        self.temperature = temperature
        self.file_processing_timeout_sec = file_processing_timeout_sec
        self.file_poll_interval_sec = file_poll_interval_sec

    def transcribe(self, audio_path: Path) -> GeminiTranscriptionResult:
        from google.genai import types

        uploaded = self.client.files.upload(file=str(audio_path))
        try:
            uploaded = self._wait_until_active(uploaded)
            mime_type = getattr(uploaded, "mime_type", None) or guess_mime_type(audio_path)
            generate_config_kwargs: dict[str, Any] = {
                "system_instruction": self.prompt,
                # Leave sampling and output-limit parameters unset. Current Gemini
                # models are expected to use provider defaults unless the product
                # has a model-specific reason to override them.
            }
            response_mime_type, response_schema = response_contract_for_format(
                self.response_format
            )
            if response_mime_type is not None:
                generate_config_kwargs["response_mime_type"] = response_mime_type
                generate_config_kwargs["response_schema"] = response_schema
            if self.thinking_level is not None:
                generate_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=self.thinking_level
                )
            if self.temperature is not None:
                generate_config_kwargs["temperature"] = self.temperature

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Strictly follow the transcription instruction."},
                            {
                                "file_data": {
                                    "file_uri": uploaded.uri,
                                    "mime_type": mime_type,
                                }
                            },
                        ],
                    }
                ],
                config=types.GenerateContentConfig(**generate_config_kwargs),
            )
            raw_text = extract_response_text(response)
            metadata = build_call_metadata(
                response=response,
                uploaded=uploaded,
                audio_path=audio_path,
                model=self.model,
                mime_type=mime_type,
                system_instruction=self.prompt,
                user_instruction="Strictly follow the transcription instruction.",
                response_format=self.response_format,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
                thinking_level=self.thinking_level,
                temperature=self.temperature,
                raw_response_text=raw_text,
            )
            finish_reason_class = metadata["response"]["finish_reason_class"]
            if finish_reason_class != "normal_stop":
                failure = mark_failed_metadata(
                    metadata,
                    error_type="finish_reason",
                    message=(
                        "Gemini did not finish with STOP; refusing to use transcript "
                        f"with finish class {finish_reason_class!r}"
                    ),
                    raw_response_text=raw_text,
                )
                raise GeminiTranscriptionError(
                    "Gemini transcription did not finish normally",
                    metadata=failure,
                )
            try:
                transcript = parse_raw_transcript(
                    raw_text,
                    self.response_format,
                    allow_empty_plain_text=True,
                )
            except TranscriptParseError as exc:
                failure = mark_failed_metadata(
                    metadata,
                    error_type="transcript_parse",
                    message=str(exc),
                    raw_response_text=raw_text,
                )
                raise GeminiTranscriptionError(
                    "Gemini transcription response could not be parsed",
                    metadata=failure,
                ) from exc

            metadata = mark_success_metadata(metadata, transcript=transcript)
            return GeminiTranscriptionResult(transcript=transcript, metadata=metadata)
        finally:
            self._delete_uploaded_file(uploaded)

    def _wait_until_active(self, uploaded: Any) -> Any:
        name = getattr(uploaded, "name", None)
        if not name:
            return uploaded
        deadline = time.monotonic() + self.file_processing_timeout_sec
        current = uploaded
        while time.monotonic() < deadline:
            state = file_state_name(current)
            if state in {"", "ACTIVE", "FILE_STATE_ACTIVE", "STATE_UNSPECIFIED"}:
                return current
            if state in {"FAILED", "FILE_STATE_FAILED"}:
                raise RuntimeError(f"Gemini file processing failed for {name}")
            time.sleep(self.file_poll_interval_sec)
            current = self.client.files.get(name=name)
        raise TimeoutError(
            f"Gemini file processing timeout after {self.file_processing_timeout_sec}s"
        )

    def _delete_uploaded_file(self, uploaded: Any) -> None:
        name = getattr(uploaded, "name", None)
        if name:
            self.client.files.delete(name=name)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    if path.suffix.lower() == ".m4a":
        return "audio/mp4"
    if path.suffix.lower() == ".wav":
        return "audio/wav"
    return "application/octet-stream"


def parse_raw_transcript(
    raw_text: str,
    response_format: str,
    *,
    allow_empty_plain_text: bool = False,
) -> Transcript:
    if response_format == "json_lines":
        return parse_transcript(raw_text)
    if response_format == "plain_text":
        return parse_plain_text_transcript(raw_text, allow_empty=allow_empty_plain_text)
    raise ValueError(f"Unknown Gemini response format: {response_format}")


def response_contract_for_format(
    response_format: str,
) -> tuple[str | None, dict[str, Any] | None]:
    if response_format == "json_lines":
        return "application/json", TRANSCRIPT_RESPONSE_SCHEMA
    if response_format == "plain_text":
        return None, None
    raise ValueError(f"Unknown Gemini response format: {response_format}")


def file_state_name(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return ""
    return str(getattr(state, "name", state)).upper()


def extract_response_text(response: Any) -> str:
    try:
        text = getattr(response, "text", None)
    except Exception:
        text = None
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text)
    if chunks:
        return "".join(chunks)

    return ""


def safe_response_summary(response: Any) -> dict[str, Any]:
    return {
        "finish_reason": response_finish_reasons(response),
        "usage_metadata": str(getattr(response, "usage_metadata", "")),
    }


NORMAL_STOP_FINISH_REASONS = {"STOP"}
INCOMPLETE_OUTPUT_FINISH_REASONS = {"MAX_TOKENS"}
PROVIDER_BLOCK_FINISH_REASONS = {
    "BLOCKLIST",
    "MODEL_ARMOR",
    "PROHIBITED_CONTENT",
    "RECITATION",
    "SAFETY",
    "SPII",
}


def response_finish_reasons(response: Any) -> list[str]:
    return [
        finish_reason_name(getattr(candidate, "finish_reason", ""))
        for candidate in (getattr(response, "candidates", None) or [])
    ]


def finish_reason_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def classify_finish_reasons(finish_reasons: list[str]) -> str:
    reasons = [reason for reason in finish_reasons if reason]
    if reasons and all(reason in NORMAL_STOP_FINISH_REASONS for reason in reasons):
        return "normal_stop"
    if any(reason in INCOMPLETE_OUTPUT_FINISH_REASONS for reason in reasons):
        return "incomplete_output"
    if any(reason in PROVIDER_BLOCK_FINISH_REASONS for reason in reasons):
        return "provider_block"
    return "unknown_failure"


def build_call_metadata(
    *,
    response: Any,
    uploaded: Any,
    audio_path: Path,
    model: str,
    mime_type: str,
    system_instruction: str,
    user_instruction: str,
    response_format: str,
    response_mime_type: str | None,
    response_schema: dict[str, Any] | None,
    thinking_level: str | None,
    temperature: float | None,
    raw_response_text: str,
) -> dict[str, Any]:
    usage_metadata = jsonable(getattr(response, "usage_metadata", None))
    candidates = list(getattr(response, "candidates", None) or [])
    finish_reasons = response_finish_reasons(response)
    finish_reason_class = classify_finish_reasons(finish_reasons)
    return {
        "schema": "segmentwright.gemini_call_metadata.v1",
        "cache_hit": False,
        "status": "received",
        "model": model,
        "model_version": scalar_attr(response, "model_version"),
        "response_id": scalar_attr(response, "response_id"),
        "mime_type": mime_type,
        "audio_file": {
            "path": str(audio_path),
            "bytes": audio_path.stat().st_size if audio_path.exists() else None,
        },
        "uploaded_file": {
            "name": scalar_attr(uploaded, "name"),
            "uri": scalar_attr(uploaded, "uri"),
            "mime_type": scalar_attr(uploaded, "mime_type"),
            "state": file_state_name(uploaded),
        },
        "request": {
            "system_instruction_character_count": len(system_instruction),
            "user_instruction_character_count": len(user_instruction),
            "response_format": response_format,
            "thinking_level": thinking_level,
            "thinking_config": {"thinking_level": thinking_level}
            if thinking_level is not None
            else None,
            "temperature": temperature,
            "response_mime_type": response_mime_type,
            "response_schema": response_schema,
        },
        "response": {
            "text_character_count": len(raw_response_text),
            "empty_text_response": not raw_response_text.strip(),
            "candidate_count": len(candidates),
            "finish_reasons": finish_reasons,
            "finish_reason_class": finish_reason_class,
            "candidates": [candidate_summary(candidate) for candidate in candidates],
            "prompt_feedback": jsonable(getattr(response, "prompt_feedback", None)),
        },
        "usage_metadata": usage_metadata,
        "token_usage": normalize_usage_metadata(usage_metadata),
    }


def mark_success_metadata(
    metadata: dict[str, Any],
    *,
    transcript: Transcript,
) -> dict[str, Any]:
    empty_transcript = len(transcript.lines) == 0
    sentinel_empty = empty_transcript and is_no_speech_sentinel_text(transcript.raw_json)
    response = metadata.get("response") if isinstance(metadata.get("response"), dict) else {}
    finish_reasons = response.get("finish_reasons")
    finish_reason = (
        finish_reasons[0] if isinstance(finish_reasons, list) and finish_reasons else None
    )
    return {
        **metadata,
        "status": "success",
        "finish_reason": finish_reason,
        "finish_reason_class": response.get("finish_reason_class"),
        "empty_transcript": empty_transcript,
        "empty_acceptance_reason": (
            empty_acceptance_reason(sentinel_empty=sentinel_empty)
            if empty_transcript
            else None
        ),
        "normalized_from_no_speech_sentinel": sentinel_empty,
        "post_transcription_action": (
            "empty_transcript_noop" if empty_transcript else "normal_pipeline"
        ),
        "transcript": {
            "line_count": len(transcript.lines),
            "raw_character_count": len(transcript.raw_json),
            "empty_transcript": empty_transcript,
            "normalized_from_no_speech_sentinel": sentinel_empty,
        },
    }


def empty_acceptance_reason(*, sentinel_empty: bool) -> str:
    if sentinel_empty:
        return "finish_reason_stop_no_speech_sentinel"
    return "finish_reason_stop_empty_response"


def mark_failed_metadata(
    metadata: dict[str, Any],
    *,
    error_type: str,
    message: str,
    raw_response_text: str,
) -> dict[str, Any]:
    response = metadata.get("response") if isinstance(metadata.get("response"), dict) else {}
    finish_reasons = response.get("finish_reasons")
    finish_reason = (
        finish_reasons[0] if isinstance(finish_reasons, list) and finish_reasons else None
    )
    return {
        **metadata,
        "status": "failed",
        "finish_reason": finish_reason,
        "finish_reason_class": response.get("finish_reason_class"),
        "empty_transcript": False,
        "empty_acceptance_reason": None,
        "post_transcription_action": "failed",
        "failure": {
            "type": error_type,
            "message": message,
            "raw_response_text": raw_response_text,
            "safe_response_summary": safe_response_summary_from_metadata(metadata),
        },
    }


def safe_response_summary_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    response = metadata.get("response") if isinstance(metadata.get("response"), dict) else {}
    return {
        "finish_reason": response.get("finish_reasons", []),
        "finish_reason_class": response.get("finish_reason_class"),
        "usage_metadata": metadata.get("usage_metadata"),
    }


def candidate_summary(candidate: Any) -> dict[str, Any]:
    return {
        "index": scalar_attr(candidate, "index"),
        "finish_reason": finish_reason_name(getattr(candidate, "finish_reason", "")),
        "finish_message": scalar_attr(candidate, "finish_message"),
        "token_count": scalar_attr(candidate, "token_count"),
        "avg_logprobs": scalar_attr(candidate, "avg_logprobs"),
        "safety_ratings": jsonable(getattr(candidate, "safety_ratings", None)),
        "citation_metadata": jsonable(getattr(candidate, "citation_metadata", None)),
    }


def normalize_usage_metadata(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    counts = {
        "prompt_token_count": int_or_none(get_any(data, "promptTokenCount", "prompt_token_count")),
        "cached_content_token_count": int_or_none(
            get_any(data, "cachedContentTokenCount", "cached_content_token_count")
        ),
        "candidates_token_count": int_or_none(
            get_any(data, "candidatesTokenCount", "candidates_token_count")
        ),
        "tool_use_prompt_token_count": int_or_none(
            get_any(data, "toolUsePromptTokenCount", "tool_use_prompt_token_count")
        ),
        "thoughts_token_count": int_or_none(
            get_any(data, "thoughtsTokenCount", "thoughts_token_count")
        ),
        "total_token_count": int_or_none(get_any(data, "totalTokenCount", "total_token_count")),
    }
    prompt = counts["prompt_token_count"]
    cached = counts["cached_content_token_count"]
    counts["non_cached_prompt_token_count"] = (
        prompt - cached if prompt is not None and cached is not None else prompt
    )
    return {
        "schema": "segmentwright.gemini_token_usage.v1",
        "counts": counts,
        "details": {
            "prompt_tokens_by_modality": modality_counts(
                get_any(data, "promptTokensDetails", "prompt_tokens_details")
            ),
            "cache_tokens_by_modality": modality_counts(
                get_any(data, "cacheTokensDetails", "cache_tokens_details")
            ),
            "candidates_tokens_by_modality": modality_counts(
                get_any(data, "candidatesTokensDetails", "candidates_tokens_details")
            ),
            "tool_use_prompt_tokens_by_modality": modality_counts(
                get_any(data, "toolUsePromptTokensDetails", "tool_use_prompt_tokens_details")
            ),
        },
        "service_tier": get_any(data, "serviceTier", "service_tier"),
        "raw": data,
    }


def modality_counts(items: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        modality = get_any(item, "modality")
        token_count = int_or_none(get_any(item, "tokenCount", "token_count"))
        if modality is None or token_count is None:
            continue
        name = str(modality).upper()
        counts[name] = counts.get(name, 0) + token_count
    return counts


def get_any(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def scalar_attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump(mode="json", by_alias=True, exclude_none=True))
        except TypeError:
            return jsonable(value.model_dump())
    if hasattr(value, "to_json_dict"):
        return jsonable(value.to_json_dict())
    if hasattr(value, "__dict__"):
        return {
            key: jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)
