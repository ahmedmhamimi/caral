"""
Detects an agent stuck repeating the same tool calls with the same results, which burns API
spend without progress. Warns first (a hint injected for the model), then stops the run.

- LoopGuard(warn_after=3, stop_after=5): .observe(calls, results) -> "ok" | "warn" | "stop".
  A repeat means an identical set of (tool, args) producing identical results as the previous step.
- LOOP_HINT: text appended for the model when a loop is suspected.
"""

from __future__ import annotations

import hashlib
import json

LOOP_HINT = (
    "Note: you have made this exact set of tool calls several times in a row with the same "
    "results. Stop repeating it. Change your approach, or explain to the user what is blocking you."
)


class LoopGuard:
    def __init__(self, warn_after: int = 3, stop_after: int = 5) -> None:
        self.warn_after = warn_after
        self.stop_after = stop_after
        self._last: str | None = None
        self._count = 0

    def observe(self, calls, results: list[dict]) -> str:
        payload = json.dumps(
            [[c.name, c.input] for c in calls] + [[r.get("content", ""), bool(r.get("is_error"))] for r in results],
            sort_keys=True, default=str,
        )
        sig = hashlib.sha1(payload.encode()).hexdigest()
        self._count = self._count + 1 if sig == self._last else 1
        self._last = sig
        if self._count >= self.stop_after:
            return "stop"
        if self._count >= self.warn_after:
            return "warn"
        return "ok"

    def reset(self) -> None:
        self._last, self._count = None, 0
