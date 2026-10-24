"""
Semantic theme. UI code asks for roles ("error", "money_lo") and glyph names ("ok", "tool"), never raw
escape codes. Two palettes: "basic" uses only the 16 ANSI colors plus bold/dim so it inherits the
user's terminal palette; "256" gives every role its own distinct hue (tool categories, token kinds,
money levels, headings, diffs) and is picked automatically when the terminal advertises 256 colors
(TERM=*256color or COLORTERM set). Meaning is never carried by color alone (every status also has a
symbol and a word) and everything degrades to plain ASCII without color (NO_COLOR, TERM=dumb, pipes).

- Theme(color, unicode, palette="basic"): .paint(role, text) -> str; .sgr(role) -> str; .g(name) -> str;
  .RESET; .palette. Roles combine: "bold accent". Unknown roles are ignored.
  - .role_for_category(category) -> str: role name for a tool category (read/edit/exec/net/sub).
  - .money_role(usd, warn, hot) -> str: green below warn, amber below hot, red from hot upward.
- detect_theme(stream=None, environ=None) -> Theme: honors NO_COLOR, CLICOLOR_FORCE, TERM=dumb,
  CARAL_ASCII, CARAL_COLORS=basic|256, and the stream encoding.
- strip_ansi(text) -> str / visible_len(text) -> int / wrap_nonprinting(text) -> str (readline-safe prompts).
- ROLES: role -> (basic SGR, 256-color SGR). GLYPHS_UNICODE / GLYPHS_ASCII: glyph tables.
"""

from __future__ import annotations

import os
import re
import sys

RESET = "\033[0m"
# role: (basic 16-color SGR, 256-color SGR). Hues are mid-tone so they read on dark and light terminals.
ROLES: dict[str, tuple[str, str]] = {
    "accent": ("36", "38;5;45"), "ok": ("32", "38;5;78"), "warn": ("33", "38;5;214"), "error": ("31", "38;5;203"),
    "add": ("32", "38;5;114"), "del": ("31", "38;5;210"), "hunk": ("35", "38;5;177"),
    "code": ("36", "38;5;80"), "muted": ("2", "38;5;245"), "bold": ("1", "1"), "italic": ("3", "3"),
    "heading": ("1;35", "1;38;5;177"), "bullet": ("33", "38;5;221"), "quote": ("34", "38;5;111"),
    "user": ("1;95", "1;38;5;213"), "sep": ("90", "38;5;240"), "label": ("90", "38;5;246"),
    "time": ("94", "38;5;111"), "path": ("96", "38;5;116"),
    # tool categories
    "cat_read": ("34", "38;5;75"), "cat_edit": ("33", "38;5;221"), "cat_exec": ("35", "38;5;207"),
    "cat_net": ("91", "38;5;209"), "cat_sub": ("95", "38;5;141"), "cat_other": ("37", "38;5;250"),
    # token kinds
    "tok_in": ("34", "38;5;75"), "tok_cache": ("36", "38;5;80"), "tok_out": ("35", "38;5;177"),
    # money levels
    "money_lo": ("32", "38;5;78"), "money_mid": ("33", "38;5;220"), "money_hi": ("31", "38;5;203"),
    "money_unknown": ("93", "38;5;215"),
}
GLYPHS_UNICODE = {
    "tool": "▸", "ok": "✓", "fail": "✗", "warn": "!", "info": "·", "bar": "│", "ellipsis": "…",
    "prompt": "❯", "bullet": "•", "arrow": "→", "sub": "↳", "stop": "■", "ask": "?", "open": "┌",
    "close": "└", "todo_pending": "○", "todo_active": "◐", "todo_done": "●",
    "spinner": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "step": "◆",
}
GLYPHS_ASCII = {
    "tool": ">", "ok": "+", "fail": "x", "warn": "!", "info": "-", "bar": "|", "ellipsis": "...",
    "prompt": ">", "bullet": "*", "arrow": "->", "sub": "`-", "stop": "#", "ask": "?", "open": "+",
    "close": "+", "todo_pending": "[ ]", "todo_active": "[~]", "todo_done": "[x]",
    "spinner": "|/-\\", "step": "*",
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

    def __init__(self, color: bool, unicode: bool, palette: str = "basic") -> None:
        self.color = color
        self.unicode = unicode
        self.palette = palette if palette in ("basic", "256") else "basic"
        self._glyphs = GLYPHS_UNICODE if unicode else GLYPHS_ASCII

    def sgr(self, role: str) -> str:
        if not self.color:
            return ""
        idx = 1 if self.palette == "256" else 0
        codes = [ROLES[r][idx] for r in role.split() if r in ROLES]
        return "".join(f"\033[{c}m" for c in codes)

    def paint(self, role: str, text: str) -> str:
        seq = self.sgr(role)
        return f"{seq}{text}{RESET}" if seq else text

    def g(self, name: str) -> str:
        return self._glyphs[name]

    @staticmethod
    def role_for_category(category: str) -> str:
        return {"read": "cat_read", "edit": "cat_edit", "exec": "cat_exec", "net": "cat_net",
                "sub": "cat_sub"}.get(category, "cat_other")

    @staticmethod
    def money_role(usd: float, warn: float, hot: float) -> str:
        return "money_lo" if usd < warn else "money_mid" if usd < hot else "money_hi"


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
    forced = (env.get("CARAL_COLORS") or "").lower()
    if forced in ("basic", "256"):
        palette = forced
    elif "256color" in (env.get("TERM") or "") or env.get("COLORTERM"):
        palette = "256"
    else:
        palette = "basic"
    return Theme(color, unicode, palette)
