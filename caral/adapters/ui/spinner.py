"""
Activity indicator. Gives continuous system-status feedback during waits (nothing printed for the
first 300 ms, so fast steps do not flicker), shows elapsed time, a live extra segment (the running
token and cost counters) and reveals the Ctrl-C hint only once the wait is long enough to need it.
Each part has its own color so the line is scannable at a glance. Silent when output is not a terminal.

- Spinner(terminal, theme, extra_fn=None): .start(label) -> None; .update(label) -> None; .stop() -> None;
  .running -> bool; .extra_fn: Callable[[], str] | None, called every frame and returning an already
  colored segment ("" to hide it); .compose(frame, label, elapsed, width) -> str builds one frame and
  drops the least important parts (hint, then extra, then time) until it fits the terminal width.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from ...domain.textutil import human_duration
from .term import Terminal
from .theme import Theme, visible_len


class Spinner:
    def __init__(self, terminal: Terminal, theme: Theme, extra_fn: Callable[[], str] | None = None) -> None:
        self.t = terminal
        self.theme = theme
        self.extra_fn = extra_fn
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

    def compose(self, frame: str, label: str, elapsed: float, width: int) -> str:
        p = self.theme.paint
        extra = ""
        if self.extra_fn is not None:
            try:
                extra = self.extra_fn()
            except Exception:  # noqa: BLE001 - a display bug must never kill the spinner thread
                extra = ""
        optional = []  # (colored text), most important first
        if elapsed >= 2:
            optional.append(p("time", human_duration(elapsed)))
        if extra:
            optional.append(extra)
        if elapsed >= 6:
            optional.append(p("label", "Ctrl-C to interrupt"))
        sep = p("sep", " · ")
        head = p("accent", frame) + " " + p("bold", label)
        while True:
            line = sep.join([head] + optional)
            if visible_len(line) <= width or not optional:
                return line
            optional.pop()

    def _loop(self) -> None:
        frames = self.theme.g("spinner")
        i = 0
        while not self._stop.wait(0.1):
            elapsed = time.monotonic() - self._started
            if elapsed < 0.3:
                continue
            text = self.compose(frames[i % len(frames)], self._label, elapsed, self.t.width - 1)
            with self.t.lock:
                if self._stop.is_set():
                    break
                self.t.write("\r\033[2K" + text)
            i += 1
