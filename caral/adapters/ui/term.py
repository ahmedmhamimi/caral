"""
Terminal handle: one place that owns the output stream, its lock, width and interactivity so that
the spinner thread, parallel tool threads and the renderer never interleave writes.

- Terminal(stream=None): .write(text), .writeln(text="") , .width -> int, .interactive -> bool,
  .erase_line() -> None, .lock (RLock).
"""

from __future__ import annotations

import shutil
import sys
import threading


class Terminal:
    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self.lock = threading.RLock()

    @property
    def interactive(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    @property
    def width(self) -> int:
        return max(40, min(shutil.get_terminal_size((100, 24)).columns, 160))

    def write(self, text: str) -> None:
        with self.lock:
            try:
                self.stream.write(text)
                self.stream.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass

    def writeln(self, text: str = "") -> None:
        self.write(text + "\n")

    def erase_line(self) -> None:
        if self.interactive:
            self.write("\r\033[2K")
