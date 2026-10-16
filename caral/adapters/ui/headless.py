"""
Non-interactive renderers for scripts and CI: plain streamed text, or machine-readable JSONL.

- PlainRenderer(out, err): streams assistant text to `out`; notices, retries and errors go to `err`.
- JsonlRenderer(out): writes every event as one JSON line (the full event vocabulary).
"""

from __future__ import annotations

import json
import sys

from ...domain.events import Event


class PlainRenderer:
    def __init__(self, out=None, err=None) -> None:
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self._need_newline = False

    def __call__(self, event: Event) -> None:
        name = type(event).__name__
        if name == "TextDelta":
            self.out.write(event.text)
            self.out.flush()
            self._need_newline = not event.text.endswith("\n")
        elif name == "ModelResponded" and self._need_newline:
            self.out.write("\n")
            self._need_newline = False
        elif name == "Notice":
            self.err.write(f"caral: {event.level}: {event.message}\n")
        elif name == "RetryScheduled":
            self.err.write(f"caral: {event.reason}; retrying in {event.delay_s:.0f}s\n")
        elif name == "RunFinished" and event.status != "ok":
            self.err.write(f"caral: {event.status}: {event.error}\n")


class JsonlRenderer:
    def __init__(self, out=None) -> None:
        self.out = out or sys.stdout

    def __call__(self, event: Event) -> None:
        self.out.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")
        self.out.flush()
