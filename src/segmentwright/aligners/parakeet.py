from __future__ import annotations

import gc
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .base import AlignerBackend, AlignmentResult, LineSpan, TokenSpan

DEFAULT_JAPANESE_PARAKEET_MODEL = "nvidia/parakeet-tdt_ctc-0.6b-ja"
DEFAULT_ENGLISH_PARAKEET_MODEL = "nvidia/parakeet-ctc-0.6b"
DEFAULT_PARAKEET_MODEL = DEFAULT_JAPANESE_PARAKEET_MODEL
DEFAULT_SAMPLE_RATE = 16000
SUPPORTED_ALIGNMENT_LANGUAGES = {"ja", "en"}


class AlignmentError(RuntimeError):
    """Raised when CTC forced alignment cannot produce usable spans."""


@dataclass(frozen=True)
class CtcTokenSpan:
    token_id: int
    token: str
    start_frame: int
    end_frame: int
    score: float | None = None

    @property
    def frame_duration(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class ParakeetRuntime:
    model: Any
    decoder: Any
    decoder_name: str
    torch: Any
    torchaudio: Any
    forced_align: Any
    merge_tokens: Any
    torch_device: Any
    runtime_device: str
    blank_id: int


class ParakeetAlignerBackend(AlignerBackend):
    def __init__(
        self,
        *,
        device: str = "auto",
        model_name: str = DEFAULT_PARAKEET_MODEL,
        language: str = "ja",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate
        self._runtime: ParakeetRuntime | None = None

    def align(self, audio_path: Path, lines: list[str], offset: float) -> AlignmentResult:
        clean_lines = [line.strip() for line in lines if line.strip()]
        if not clean_lines:
            return AlignmentResult(
                line_spans=[],
                raw={
                    "schema": "segmentwright.parakeet_alignment.v1",
                    "model": self.model_name,
                    "language": self.language,
                    "normalizer": normalizer_name_for_language(self.language),
                    "audio_path": str(audio_path),
                    "offset_sec": offset,
                    "line_count": 0,
                    "failed": False,
                },
            )

        runtime = self._load_runtime()
        tokenized_lines = [
            tokenize_for_nemo_bpe_ctc(runtime.model.tokenizer, line, language=self.language)
            for line in clean_lines
        ]
        target_ids = [token_id for token_ids in tokenized_lines for token_id in token_ids]
        if not target_ids:
            raise AlignmentError("Parakeet alignment target is empty after CTC normalization")

        waveform, valid_frames, emissions, aligned_tokens, scores = self._run_ctc_alignment(
            audio_path=audio_path,
            target_ids=target_ids,
            runtime=runtime,
        )
        spans = runtime.merge_tokens(
            aligned_tokens[0].detach().cpu(),
            scores[0].detach().cpu(),
            blank=runtime.blank_id,
        )
        if len(spans) < len(target_ids):
            raise AlignmentError(
                "Parakeet produced fewer aligned spans than target tokens: "
                f"{len(spans)} < {len(target_ids)}"
            )

        seconds_per_frame = waveform.shape[0] / self.sample_rate / max(valid_frames, 1)
        ctc_spans = convert_torchaudio_spans(
            spans=spans,
            token_ids=target_ids,
            tokenizer=runtime.model.tokenizer,
        )
        line_spans = project_token_spans_to_lines(
            lines=clean_lines,
            tokenized_lines=tokenized_lines,
            ctc_spans=ctc_spans,
            seconds_per_frame=seconds_per_frame,
            offset_sec=offset,
        )
        return AlignmentResult(
            line_spans=line_spans,
            raw=build_raw_alignment(
                model_name=self.model_name,
                language=self.language,
                normalizer=normalizer_name_for_language(self.language),
                decoder_name=runtime.decoder_name,
                audio_path=audio_path,
                offset=offset,
                runtime_device=runtime.runtime_device,
                sample_rate=self.sample_rate,
                clean_lines=clean_lines,
                tokenized_lines=tokenized_lines,
                target_ids=target_ids,
                ctc_spans=ctc_spans,
                line_spans=line_spans,
                valid_frames=valid_frames,
                emissions_frames=int(emissions.shape[1]),
                seconds_per_frame=seconds_per_frame,
            ),
        )

    def load(self) -> None:
        self._load_runtime()

    def close(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is None:
            return
        torch = runtime.torch
        del runtime
        gc.collect()
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
            cuda.empty_cache()

    def _load_runtime(self) -> ParakeetRuntime:
        if self._runtime is not None:
            return self._runtime
        try:
            import nemo.collections.asr as nemo_asr
            import torch
            import torchaudio
            import torchaudio.functional as functional
        except ImportError as exc:
            raise RuntimeError(
                "Parakeet alignment dependencies are not installed. "
                "Install the product ML dependencies before running alignment."
            ) from exc

        runtime_device = resolve_torch_device(self.device, torch)
        torch_device = torch.device(runtime_device)
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name,
            map_location="cpu",
        )
        model = model.to(torch_device)
        model.eval()
        decoder, decoder_name = resolve_ctc_decoder(model)
        blank_id = resolve_ctc_blank_id(decoder)
        self._runtime = ParakeetRuntime(
            model=model,
            decoder=decoder,
            decoder_name=decoder_name,
            torch=torch,
            torchaudio=torchaudio,
            forced_align=functional.forced_align,
            merge_tokens=functional.merge_tokens,
            torch_device=torch_device,
            runtime_device=runtime_device,
            blank_id=blank_id,
        )
        return self._runtime

    def _run_ctc_alignment(
        self,
        *,
        audio_path: Path,
        target_ids: list[int],
        runtime: ParakeetRuntime,
    ) -> tuple[Any, int, Any, Any, Any]:
        waveform, sample_rate = runtime.torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            waveform = runtime.torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.sample_rate,
            )
        waveform = waveform.to(runtime.torch_device)
        input_length = runtime.torch.tensor(
            [waveform.shape[0]],
            dtype=runtime.torch.long,
            device=runtime.torch_device,
        )
        targets = runtime.torch.tensor(
            [target_ids],
            dtype=runtime.torch.int64,
            device=runtime.torch_device,
        )

        with runtime.torch.inference_mode():
            forward_output = runtime.model.forward(
                input_signal=waveform.unsqueeze(0),
                input_signal_length=input_length,
            )
            logits, encoded_len = resolve_ctc_forward_logits(
                forward_output=forward_output,
                decoder=runtime.decoder,
            )
            valid_frames = int(encoded_len[0].detach().cpu())
            logits = logits[:, :valid_frames, :]
            emissions = runtime.torch.log_softmax(logits, dim=-1)
            aligned_tokens, scores = runtime.forced_align(
                emissions,
                targets,
                blank=runtime.blank_id,
            )
        return waveform, valid_frames, emissions, aligned_tokens, scores


def resolve_torch_device(device: str, torch: Any) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return "cuda"
    if device == "cpu":
        return "cpu"
    if device != "auto":
        raise ValueError(f"Invalid alignment device: {device}")
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_ctc_decoder(model: Any) -> tuple[Any, str]:
    for attribute in ("ctc_decoder", "decoder"):
        decoder = getattr(model, attribute, None)
        if decoder is not None:
            return decoder, attribute
    raise RuntimeError(
        "Parakeet model does not expose a CTC decoder. Expected ctc_decoder or decoder."
    )


def resolve_ctc_blank_id(decoder: Any) -> int:
    num_classes_with_blank = getattr(decoder, "num_classes_with_blank", None)
    if num_classes_with_blank is not None:
        return int(num_classes_with_blank) - 1
    vocabulary = getattr(decoder, "vocabulary", None)
    if vocabulary is not None:
        return len(vocabulary)
    num_classes = getattr(decoder, "num_classes", None)
    if num_classes is not None:
        return int(num_classes) - 1
    raise RuntimeError("Unable to resolve CTC blank id from Parakeet decoder")


def call_ctc_decoder(decoder: Any, encoded: Any) -> Any:
    try:
        return decoder(encoder_output=encoded)
    except TypeError as keyword_error:
        try:
            return decoder(encoded)
        except TypeError as positional_error:
            raise keyword_error from positional_error


def resolve_ctc_forward_logits(*, forward_output: Any, decoder: Any) -> tuple[Any, Any]:
    if not isinstance(forward_output, tuple):
        raise RuntimeError("Unexpected Parakeet forward output: expected tuple")
    if len(forward_output) == 2:
        encoded, encoded_len = forward_output
        logits = call_ctc_decoder(decoder, encoded)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits, encoded_len
    if len(forward_output) >= 3:
        logits_or_log_probs, encoded_len = forward_output[0], forward_output[1]
        return logits_or_log_probs, encoded_len
    raise RuntimeError(
        "Unexpected Parakeet forward output: expected encoder output or CTC logits"
    )


def tokenize_for_nemo_bpe_ctc(tokenizer: Any, text: str, *, language: str = "ja") -> list[int]:
    normalized = normalize_for_sentencepiece_ctc(text, language=language)
    token_ids = tokenizer.text_to_ids(normalized) if normalized else []
    output: list[int] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == 0:
            continue
        piece_text = tokenizer.ids_to_text([token_id])
        if not piece_text.strip():
            continue
        output.append(token_id)
    return output


def normalize_for_sentencepiece_ctc(text: str, *, language: str = "ja") -> str:
    if language == "ja":
        return normalize_japanese_for_sentencepiece_ctc(text)
    if language == "en":
        return normalize_english_for_sentencepiece_ctc(text)
    supported = ", ".join(sorted(SUPPORTED_ALIGNMENT_LANGUAGES))
    raise ValueError(f"Invalid alignment language {language!r}; expected one of: {supported}")


def normalize_japanese_for_sentencepiece_ctc(text: str) -> str:
    chars: list[str] = []
    for char in text:
        if char.isspace():
            continue
        if char in '、。，．！？!?「」『』（）()[]【】…・:/\\"“”’‘':
            continue
        chars.append(char)
    return "".join(chars)


def normalize_english_for_sentencepiece_ctc(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    chars: list[str] = []
    for char in normalized:
        if unicodedata.combining(char):
            continue
        char = char.lower()
        if char in {"'", "’", "‘", "`"}:
            continue
        if char in {"-", "‐", "‑", "‒", "–", "—", "―", "/"}:
            chars.append(" ")
            continue
        if char.isspace():
            chars.append(" ")
            continue
        if char.isalnum():
            chars.append(char)
            continue
        if unicodedata.category(char).startswith("P"):
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def normalizer_name_for_language(language: str) -> str:
    if language == "ja":
        return "japanese_remove_whitespace_and_subtitle_punctuation"
    if language == "en":
        return "english_lowercase_preserve_word_boundaries"
    return f"unknown:{language}"


def convert_torchaudio_spans(
    *,
    spans: list[Any],
    token_ids: list[int],
    tokenizer: Any,
) -> list[CtcTokenSpan]:
    output: list[CtcTokenSpan] = []
    for index, span in enumerate(spans[: len(token_ids)]):
        token_id = int(token_ids[index])
        output.append(
            CtcTokenSpan(
                token_id=token_id,
                token=tokenizer.ids_to_text([token_id]),
                start_frame=int(span.start),
                end_frame=int(span.end),
                score=float(span.score) if getattr(span, "score", None) is not None else None,
            )
        )
    return output


def project_token_spans_to_lines(
    *,
    lines: list[str],
    tokenized_lines: list[list[int]],
    ctc_spans: list[CtcTokenSpan],
    seconds_per_frame: float,
    offset_sec: float,
) -> list[LineSpan]:
    line_spans: list[LineSpan] = []
    span_index = 0
    for line, token_ids in zip(lines, tokenized_lines, strict=True):
        count = len(token_ids)
        if count == 0:
            continue
        span_slice = ctc_spans[span_index : span_index + count]
        span_index += count
        if not span_slice:
            continue
        token_spans = [
            TokenSpan(
                token=span.token,
                start_sec=offset_sec + span.start_frame * seconds_per_frame,
                end_sec=offset_sec + span.end_frame * seconds_per_frame,
                score=span.score,
            )
            for span in span_slice
        ]
        line_spans.append(
            LineSpan(
                text=line,
                start_sec=token_spans[0].start_sec,
                end_sec=token_spans[-1].end_sec,
                token_spans=token_spans,
            )
        )
    return line_spans


def build_raw_alignment(
    *,
    model_name: str,
    language: str,
    normalizer: str,
    decoder_name: str,
    audio_path: Path,
    offset: float,
    runtime_device: str,
    sample_rate: int,
    clean_lines: list[str],
    tokenized_lines: list[list[int]],
    target_ids: list[int],
    ctc_spans: list[CtcTokenSpan],
    line_spans: list[LineSpan],
    valid_frames: int,
    emissions_frames: int,
    seconds_per_frame: float,
) -> dict[str, Any]:
    zero_or_negative = sum(1 for span in ctc_spans if span.end_frame <= span.start_frame)
    short_lt_50ms = sum(
        1 for span in ctc_spans if 0 < span.frame_duration * seconds_per_frame < 0.05
    )
    return {
        "schema": "segmentwright.parakeet_alignment.v1",
        "model": model_name,
        "language": language,
        "normalizer": normalizer,
        "decoder": decoder_name,
        "audio_path": str(audio_path),
        "offset_sec": offset,
        "device": runtime_device,
        "sample_rate": sample_rate,
        "line_count": len(clean_lines),
        "aligned_lines": len(line_spans),
        "skipped_empty_target_lines": sum(1 for token_ids in tokenized_lines if not token_ids),
        "target_units": len(target_ids),
        "aligned_units": len(ctc_spans),
        "zero_or_negative_units": zero_or_negative,
        "short_units_lt_50ms": short_lt_50ms,
        "unit_short_ratio": short_lt_50ms / len(ctc_spans) if ctc_spans else 0.0,
        "valid_frames": valid_frames,
        "emissions_frames": emissions_frames,
        "seconds_per_frame": seconds_per_frame,
        "failed": False,
        "line_spans": [line_span_to_dict(line_span) for line_span in line_spans],
    }


def line_span_to_dict(line_span: LineSpan) -> dict[str, Any]:
    return {
        "text": line_span.text,
        "start_sec": line_span.start_sec,
        "end_sec": line_span.end_sec,
        "token_spans": [asdict(token_span) for token_span in line_span.token_spans],
    }
