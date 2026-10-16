"""
Token estimation that calibrates itself against real provider usage, so context-window decisions
are accurate without shipping a tokenizer for every model.

- TokenEstimator(chars_per_token=3.6): adaptive characters-per-token ratio.
  - .estimate(chars) -> int: estimated tokens for a character count.
  - .observe(chars, tokens) -> None: updates the ratio (EMA, clamped to 1.5-6.0) from a real measurement.
"""

from __future__ import annotations


class TokenEstimator:
    def __init__(self, chars_per_token: float = 3.6) -> None:
        self.ratio = chars_per_token

    def estimate(self, chars: int) -> int:
        return int(chars / self.ratio)

    def observe(self, chars: int, tokens: int) -> None:
        if chars < 2000 or tokens < 500:  # too small to be a reliable sample
            return
        sample = chars / tokens
        self.ratio = min(6.0, max(1.5, 0.7 * self.ratio + 0.3 * sample))
