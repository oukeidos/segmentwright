from __future__ import annotations

import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

from segmentwright.aligners.base import LineSpan
from segmentwright.aligners.parakeet import (
    CtcTokenSpan,
    call_ctc_decoder,
    normalize_for_sentencepiece_ctc,
    project_token_spans_to_lines,
    resolve_ctc_blank_id,
    resolve_ctc_decoder,
    resolve_ctc_forward_logits,
    tokenize_for_nemo_bpe_ctc,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.char_to_id = {
            "あ": 1,
            "い": 2,
            "る": 3,
            "美": 4,
            "味": 5,
            "し": 6,
            "い2": 7,
        }
        self.id_to_text = {
            0: "",
            1: "あ",
            2: "い",
            3: "る",
            4: "美",
            5: "味",
            6: "し",
            7: "い",
        }

    def text_to_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for char in text:
            if char == "い" and 7 not in ids and "美" in text:
                ids.append(7)
                continue
            ids.append(self.char_to_id.get(char, 0))
        return ids

    def ids_to_text(self, ids: list[int]) -> str:
        return "".join(self.id_to_text[token_id] for token_id in ids)


class FakeEnglishTokenizer:
    def __init__(self) -> None:
        self.pieces = {
            "hello": 1,
            "world": 2,
            "dont": 3,
            "stop": 4,
            "co": 5,
            "operate": 6,
        }
        self.id_to_piece = {token_id: piece for piece, token_id in self.pieces.items()}

    def text_to_ids(self, text: str) -> list[int]:
        return [self.pieces.get(piece, 0) for piece in text.split()]

    def ids_to_text(self, ids: list[int]) -> str:
        return " ".join(self.id_to_piece.get(token_id, "") for token_id in ids)


class FakeKeywordDecoder:
    num_classes_with_blank = 11

    def __call__(self, *, encoder_output: object) -> tuple[object]:
        return (encoder_output,)


class FakePositionalDecoder:
    num_classes_with_blank = 13

    def __call__(self, encoder_output: object) -> object:
        return encoder_output


class FakeCtcModel:
    ctc_decoder = FakeKeywordDecoder()


class FakeDecoderModel:
    decoder = FakePositionalDecoder()


def test_torchcodec_decodes_alignment_wav(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    pcm_samples = [0, 16_384, -16_384, 32_767] * 200
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(struct.pack(f"<{len(pcm_samples)}h", *pcm_samples))

    script = """
import json
import sys
from pathlib import Path
from torchcodec.decoders import AudioDecoder
from segmentwright.aligners.parakeet import decode_audio_for_alignment

waveform = decode_audio_for_alignment(
    audio_path=Path(sys.argv[1]),
    sample_rate=16_000,
    audio_decoder=AudioDecoder,
)
print(json.dumps({
    "dtype": str(waveform.dtype),
    "sample_count": waveform.numel(),
    "all_finite": waveform.isfinite().all().item(),
    "peak": waveform.abs().max().item(),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-Wd", "-c", script, str(audio_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["dtype"] == "torch.float32"
    assert result["sample_count"] == 1_600
    assert result["all_finite"] is True
    assert result["peak"] > 0.5
    assert "DeprecationWarning" not in completed.stderr


def test_torchaudio_forced_alignment_runtime_contract() -> None:
    script = """
import json
import torch
import torchaudio.functional as functional

emissions = torch.tensor([[[4., 1., 0.], [1., 4., 0.], [0., 4., 1.],
                           [0., 1., 4.], [0., 1., 4.]]]).log_softmax(dim=-1)
targets = torch.tensor([[1, 2]], dtype=torch.int64)
aligned_tokens, scores = functional.forced_align(emissions, targets, blank=0)
spans = functional.merge_tokens(aligned_tokens[0], scores[0], blank=0)
print(json.dumps({
    "tokens": aligned_tokens.tolist(),
    "spans": [[span.token, span.start, span.end] for span in spans],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-Wd", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["tokens"] == [[0, 1, 1, 2, 2]]
    assert result["spans"] == [[1, 1, 3], [2, 3, 5]]
    assert "DeprecationWarning" not in completed.stderr


def test_normalize_for_sentencepiece_ctc_removes_subtitle_punctuation() -> None:
    assert normalize_for_sentencepiece_ctc(" あ、いる！？ ") == "あいる"
    assert normalize_for_sentencepiece_ctc("「美味しい」") == "美味しい"


def test_english_normalize_for_sentencepiece_ctc_preserves_word_boundaries() -> None:
    assert (
        normalize_for_sentencepiece_ctc("Hello, WORLD! Don't stop.", language="en")
        == "hello world dont stop"
    )
    assert (
        normalize_for_sentencepiece_ctc("Co-operate / re-enter", language="en")
        == "co operate re enter"
    )


def test_tokenize_for_nemo_bpe_ctc_filters_empty_control_ids() -> None:
    tokenizer = FakeTokenizer()

    assert tokenize_for_nemo_bpe_ctc(tokenizer, "あ、Xいる") == [1, 2, 3]


def test_tokenize_for_nemo_bpe_ctc_uses_language_specific_normalization() -> None:
    tokenizer = FakeEnglishTokenizer()

    assert tokenize_for_nemo_bpe_ctc(tokenizer, "Hello, WORLD! Don't stop.", language="en") == [
        1,
        2,
        3,
        4,
    ]


def test_resolve_ctc_decoder_supports_hybrid_and_pure_ctc_models() -> None:
    ctc_decoder, ctc_name = resolve_ctc_decoder(FakeCtcModel())
    decoder, decoder_name = resolve_ctc_decoder(FakeDecoderModel())

    assert ctc_name == "ctc_decoder"
    assert ctc_decoder is FakeCtcModel.ctc_decoder
    assert resolve_ctc_blank_id(ctc_decoder) == 10
    assert decoder_name == "decoder"
    assert decoder is FakeDecoderModel.decoder
    assert resolve_ctc_blank_id(decoder) == 12


def test_call_ctc_decoder_supports_keyword_and_positional_call_shapes() -> None:
    encoded = object()

    assert call_ctc_decoder(FakeKeywordDecoder(), encoded) == (encoded,)
    assert call_ctc_decoder(FakePositionalDecoder(), encoded) is encoded


def test_resolve_ctc_forward_logits_supports_encoder_and_ctc_outputs() -> None:
    encoded = object()
    encoded_len = object()
    logits = object()

    assert resolve_ctc_forward_logits(
        forward_output=(encoded, encoded_len),
        decoder=FakePositionalDecoder(),
    ) == (encoded, encoded_len)
    assert resolve_ctc_forward_logits(
        forward_output=(logits, encoded_len, object()),
        decoder=FakeKeywordDecoder(),
    ) == (logits, encoded_len)


def test_project_token_spans_to_lines_uses_token_counts_and_offset() -> None:
    spans = [
        CtcTokenSpan(1, "あ", 0, 2, 0.9),
        CtcTokenSpan(2, "い", 3, 5, 0.8),
        CtcTokenSpan(3, "る", 5, 9, 0.7),
        CtcTokenSpan(4, "美", 20, 23, 0.6),
        CtcTokenSpan(5, "味", 23, 25, 0.5),
    ]

    line_spans = project_token_spans_to_lines(
        lines=["あ、いる", "。", "美味"],
        tokenized_lines=[[1, 2, 3], [], [4, 5]],
        ctc_spans=spans,
        seconds_per_frame=0.02,
        offset_sec=100.0,
    )

    assert [span.text for span in line_spans] == ["あ、いる", "美味"]
    assert_line_span(line_spans[0], start=100.0, end=100.18, token_count=3)
    assert_line_span(line_spans[1], start=100.4, end=100.5, token_count=2)


def assert_line_span(
    line_span: LineSpan,
    *,
    start: float,
    end: float,
    token_count: int,
) -> None:
    assert line_span.start_sec == start
    assert line_span.end_sec == end
    assert len(line_span.token_spans) == token_count
