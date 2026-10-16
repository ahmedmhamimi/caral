"""
Semantic theme. UI code asks for roles ("error", "muted") and glyph names ("ok", "tool"), never raw
escape codes. Uses only the 16 base ANSI colors plus bold/dim so it inherits the user's palette
(works on light and dark terminals), never encodes meaning by color alone (every status also has
a symbol and a word), and degrades to plain ASCII without color (NO_COLOR, TERM=dumb, pipes).

- Theme(color, unicode): .paint(role, text) -> str; .sgr(role) -> str; .g(name) -> str; .RESET.
  Roles: accent ok warn error add del hunk code muted bold, combinable: "bold accent".
- detect_theme(stream=None, environ=None) -> Theme: honors NO_COLOR, CLICOLOR_FORCE, TERM=dumb,
  CARAL_ASCII, and the stream encoding.
- strip_ansi(text) -> str / visible_len(text) -> int / wrap_nonprinting(text) -> str (readline-safe prompts).
- GLYPHS_UNICODE / GLYPHS_ASCII: glyph tables.
"""

from __future__ import annotations

import os
import re
import sys

RESET = "\033[0m"
_ROLES = {
    "accent": "36", "ok": "32", "warn": "33", "error": "31", "add": "32", "del": "31", "hunk": "36",
    "code": "36", "muted": "2", "bold": "1", "italic": "3",
}
GLYPHS_UNICODE = {
    "tool": "▸", "ok": "✓", "fail": "✗", "warn": "!", "info": "·", "bar": "│", "ellipsis": "…",
    "prompt": "❯", "bullet": "•", "arrow": "→", "sub": "↳", "stop": "■", "ask": "?", "open": "┌",
    "close": "└", "todo_pending": "○", "todo_active": "◐", "todo_done": "●",
    "spinner": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
}
GLYPHS_ASCII = {
    "tool": ">", "ok": "+", "fail": "x", "warn": "!", "info": "-", "bar": "|", "ellipsis": "...",
    "prompt": ">", "bullet": "*", "arrow": "->", "sub": "`-", "stop": "#", "ask": "?", "open": "+",
    "close": "+", "todo_pending": "[ ]", "todo_active": "[~]", "todo_done": "[x]",
    "spinner": "|/-\\",
}
_ANSI = re.compile(r"\033\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def wrap_nonprinting(text: str) -> str:
    """Mark escape sequences as zero-width for readline so line editing measures the prompt correctly."""
    return _ANSI.sub(lambda m: "\001" + m.group(0) + "\002", text)


class Theme:
    RESET = RESET

    def __init__(self, color: bool, unicode: bool) -> None:
        self.color = color
        self.unicode = unicode
        self._glyphs = GLYPHS_UNICODE if unicode else GLYPHS_ASCII

    def sgr(self, role: str) -> str:
        if not self.color:
            return ""
        codes = [_ROLES[r] for r in role.split() if r in _ROLES]
        return "".join(f"\033[{c}m" for c in codes)

    def paint(self, role: str, text: str) -> str:
        seq = self.sgr(role)
        return f"{seq}{text}{RESET}" if seq else text

    def g(self, name: str) -> str:
        return self._glyphs[name]


def detect_theme(stream=None, environ=None) -> Theme:
    env = os.environ if environ is None else environ
    stream = stream or sys.stdout
    isatty = bool(getattr(stream, "isatty", lambda: False)())
    if env.get("NO_COLOR"):
        color = False
    elif env.get("CLICOLOR_FORCE") not in (None, "", "0"):
        color = True
    else:
        color = isatty and env.get("TERM") != "dumb"
    enc = (getattr(stream, "encoding", "") or "").lower()
    unicode = "utf" in enc and not env.get("CARAL_ASCII") and env.get("TERM") != "linux"
    return Theme(color, unicode)
