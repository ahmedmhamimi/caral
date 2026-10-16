"""
Activity indicator. Gives continuous system-status feedback during waits (nothing printed for the
first 300 ms, so fast steps do not flicker), shows elapsed time, and reveals the Ctrl-C hint only
once the wait is long enough to need it. Silent when output is not a terminal.

- Spinner(terminal, theme): .start(label) -> None; .update(label) -> None; .stop() -> None;
  .running -> bool.
"""

from __future__ import annotations

import threading
import time

from ...domain.textutil import human_duration
from .term import Terminal
from .theme import Theme, visible_len


class Spinner:
    def __init__(self, terminal: Terminal, theme: Theme) -> None:
        self.t = terminal
        self.theme = theme
        self._label = ""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self, label: str) -> None:
        self._label = label
        if self._thread is not None or not self.t.interactive:
            return
        self._stop.clear()
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, label: str) -> None:
        self._label = label

    def stop(self) -> None:
        th = self._thread
        if th is None:
            return
        self._stop.set()
        th.join(timeout=1.0)
        self._thread = None
        self.t.erase_line()

    def _loop(self) -> None:
        frames = self.theme.g("spinner")
        i = 0
        while not self._stop.wait(0.1):
            elapsed = time.monotonic() - self._started
            if elapsed < 0.3:
                continue
            parts = [f"{frames[i % len(frames)]} {self._label}"]
            if elapsed >= 2:
                parts.append(human_duration(elapsed))
            if elapsed >= 6:
                parts.append("Ctrl-C to interrupt")
            text = self.theme.paint("muted", " · ".join(parts))
            width = self.t.width - 1
            if visible_len(text) > width:
                text = text[: width]
            with self.t.lock:
                if self._stop.is_set():
                    break
                self.t.write("\r\033[2K" + text)
            i += 1
