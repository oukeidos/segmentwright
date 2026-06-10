from __future__ import annotations

from segmentwright.qa.token_gap_guard import (
    TokenGapCorrection as ShortCueGapCorrection,
)
from segmentwright.qa.token_gap_guard import (
    TokenGapGuardConfig as ShortCueGapGuardConfig,
)
from segmentwright.qa.token_gap_guard import TokenGapGuardResult as ShortCueGapGuardResult
from segmentwright.qa.token_gap_guard import apply_token_gap_guard

__all__ = [
    "ShortCueGapCorrection",
    "ShortCueGapGuardConfig",
    "ShortCueGapGuardResult",
    "apply_short_cue_gap_guard",
    "is_short_japanese_utterance",
]


def apply_short_cue_gap_guard(
    *args,
    **kwargs,
) -> ShortCueGapGuardResult:
    return apply_token_gap_guard(*args, **kwargs)


def is_short_japanese_utterance(text: str) -> bool:
    compact = "".join(char for char in text if not char.isspace())
    compact = compact.strip('、。，．！？!?「」『』（）()[]【】…・:/\\"“”’‘')
    return 0 < len(compact) <= 4
