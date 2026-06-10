from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from .base import VadResult, VadSegment


class FireRedVadBackend:
    def __init__(
        self,
        *,
        model_dir: Path,
        device: str = "auto",
        speech_threshold: float = 0.4,
    ) -> None:
        self.model_dir = model_dir.expanduser()
        self.device = device
        self.speech_threshold = speech_threshold
        self._vad: Any | None = None

    def detect(self, audio_path: Path) -> VadResult:
        vad = self._load()
        result, probs = vad.detect(str(audio_path))
        timestamps = result.get("timestamps", [])
        segments = [
            VadSegment(start_sec=float(start), end_sec=float(end))
            for start, end in timestamps
            if float(end) > float(start)
        ]
        return VadResult(
            duration_sec=float(result.get("dur", 0.0)),
            speech_segments=segments,
            raw={
                "engine": "fireredvad",
                "model_dir": str(self.model_dir),
                "device": self.device,
                "result": result,
                # Probabilities can be large; preserve only a string summary in the first MVP pass.
                "probabilities_summary": str(type(probs)),
            },
        )

    def load(self) -> None:
        self._load()

    def close(self) -> None:
        self._vad = None
        gc.collect()

    def _load(self) -> Any:
        if self._vad is not None:
            return self._vad
        if not self.model_dir.exists():
            raise FileNotFoundError(
                "FireRedVAD model directory not found after preflight preparation: "
                f"{self.model_dir}"
            )
        try:
            from fireredvad import FireRedVad, FireRedVadConfig
        except ImportError as exc:
            raise RuntimeError(
                "fireredvad is not installed. Install product ML dependencies before running VAD."
            ) from exc

        use_gpu = self.device == "cuda"
        config = FireRedVadConfig(
            use_gpu=use_gpu,
            smooth_window_size=5,
            speech_threshold=self.speech_threshold,
            min_speech_frame=20,
            max_speech_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        )
        self._vad = FireRedVad.from_pretrained(str(self.model_dir), config)
        return self._vad
