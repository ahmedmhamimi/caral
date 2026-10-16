"""
Single-keypress input for confirmations (fewer actions than type-then-Enter; Fitts's law). Falls
back to line input when stdin is not a terminal.

- read_key() -> str: one key, lowercased. Returns "esc" for Escape and "ctrl-c" for Ctrl-C.
- read_choice(valid, default=None) -> str: loops until a key in `valid` (or esc/ctrl-c -> "esc").
"""

from __future__ import annotations

import os
import select
import sys


def read_key() -> str:
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip().lower()[:1] if line else "esc"
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        ch = msvcrt.getwch()
        return "ctrl-c" if ch == "\x03" else "esc" if ch == "\x1b" else ch.lower()
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        data = os.read(fd, 1)
        if data == b"\x1b":  # swallow the rest of an escape sequence (arrow keys etc.)
            while select.select([fd], [], [], 0.02)[0]:
                os.read(fd, 8)
            return "esc"
        if data == b"\x03":
            return "ctrl-c"
        if data == b"\x04":
            return "esc"
        return data.decode("utf-8", errors="ignore").lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_choice(valid: str, default: str | None = None) -> str:
    while True:
        k = read_key()
        if k in ("esc", "ctrl-c"):
            return "esc"
        if k in ("\r", "\n", "") and default:
            return default
        if k in valid:
            return k
