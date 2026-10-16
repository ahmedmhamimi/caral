"""
Streaming Markdown renderer for terminals. Consumes text deltas as they arrive and renders
immediately (no waiting for a full line or paragraph), handling headings, bullets, quotes,
fenced code blocks (with a gutter, never styled inline), **bold** and `code`. Inline state resets
at every newline so an unbalanced marker can never bleed into later output. Without color it is a
transparent pass-through (Markdown source is already readable as plain text).

- StreamingMarkdown(write, theme): .feed(text) -> None; .flush() -> None; .at_line_start -> bool.
"""

from __future__ import annotations

from typing import Callable

from .theme import Theme


class StreamingMarkdown:
    def __init__(self, write: Callable[[str], None], theme: Theme) -> None:
        self.write = write
        self.t = theme
        self.enabled = theme.color
        self.in_fence = False
        self._reset_line()
        self.at_line_start = True

    def _reset_line(self) -> None:
        self.mode = "start"  # start | normal | heading | quote | code | fence_open | fence_close
        self._buf = ""
        self._lang = ""
        self.bold = False
        self.code = False
        self._star = False
        self._skip_space = False

    # -- public ----------------------------------------------------------------------------
    def feed(self, text: str) -> None:
        if not self.enabled:
            if text:
                self.write(text)
                self.at_line_start = text.endswith("\n")
            return
        for ch in text:
            self._char(ch)

    def flush(self) -> None:
        if not self.enabled:
            if not self.at_line_start:
                self.write("\n")
                self.at_line_start = True
            return
        if self.mode != "start" or self._buf or not self.at_line_start:
            self._end_line()

    # -- state machine ---------------------------------------------------------------------
    def _out(self, s: str) -> None:
        if s:
            self.write(s)
            self.at_line_start = s.endswith("\n")

    def _style(self) -> None:
        seq = self.t.RESET
        if self.bold:
            seq += self.t.sgr("bold")
        if self.code:
            seq += self.t.sgr("code")
        self._out(seq)

    def _char(self, ch: str) -> None:
        if ch == "\r":
            return
        if ch == "\n":
            self._end_line()
            return
        m = self.mode
        if m == "start":
            self._buf += ch
            self._decide()
        elif m == "normal":
            self._inline(ch)
        elif m == "heading":
            self._out(ch)
        elif m == "quote":
            if self._skip_space and ch == " ":
                self._skip_space = False
                return
            self._skip_space = False
            self._out(ch)
        elif m == "code":
            self._out(ch)
        elif m in ("fence_open", "fence_close"):
            self._lang += ch

    def _decide(self) -> None:
        s = self._buf.lstrip(" ")
        indent = self._buf[: len(self._buf) - len(s)]
        if s == "":
            if len(indent) > 12:
                self._to_normal()
            return
        if self.in_fence:
            if "```".startswith(s) and len(s) < 3:
                return
            if s.startswith("```"):
                self.mode, self._lang = "fence_close", ""
                return
            self._start_code()
            return
        if "```".startswith(s) and len(s) < 3:
            return
        if s.startswith("```"):
            self.mode, self._lang = "fence_open", s[3:]
            return
        c = s[0]
        if c == "#":
            hashes = len(s) - len(s.lstrip("#"))
            if hashes == len(s) and hashes <= 6:
                return  # still reading the #'s
            if hashes <= 6 and s[hashes] == " ":
                self.mode = "heading"
                self._out(self.t.sgr("bold accent") + indent + s[hashes + 1 :])
                return
        elif c in "-*+":
            if len(s) == 1:
                return
            if s[1] == " ":
                self.mode = "normal"
                self._out(indent + self.t.paint("accent", self.t.g("bullet")) + " ")
                self._buf = ""
                self._inline_text(s[2:])
                return
        elif c == ">":
            self.mode = "quote"
            self._skip_space = True
            self._out(self.t.sgr("muted") + self.t.g("bar") + " ")
            self._buf = ""
            rest = s[1:]
            if rest.startswith(" "):
                rest, self._skip_space = rest[1:], False
            self._out(rest)
            return
        self._to_normal()

    def _inline_text(self, text: str) -> None:
        for ch in text:
            self._inline(ch)

    def _to_normal(self) -> None:
        buf, self._buf, self.mode = self._buf, "", "normal"
        self._inline_text(buf)

    def _start_code(self) -> None:
        buf, self._buf, self.mode = self._buf, "", "code"
        self._out(self.t.paint("muted", "  " + self.t.g("bar") + " ") + self.t.sgr("code") + buf)

    def _inline(self, ch: str) -> None:
        if self.code:
            if ch == "`":
                self.code = False
                self._style()
            else:
                self._out(ch)
            return
        if self._star:
            self._star = False
            if ch == "*":
                self.bold = not self.bold
                self._style()
                return
            self._out("*")
        if ch == "*":
            self._star = True
        elif ch == "`":
            self.code = True
            self._style()
        else:
            self._out(ch)

    def _end_line(self) -> None:
        m = self.mode
        if m == "start" and self._buf:
            self._to_normal()
            m = "normal"
        if m == "normal" and self._star:
            self._out("*")
        if m == "fence_open":
            self.in_fence = True
            lang = self._lang.strip()
            self._out(self.t.paint("muted", "  " + self.t.g("open") + (f" {lang}" if lang else "")))
        elif m == "fence_close":
            self.in_fence = False
            self._out(self.t.paint("muted", "  " + self.t.g("close")))
        elif m == "start" and self.in_fence:
            self._out(self.t.paint("muted", "  " + self.t.g("bar")))
        self._out(self.t.RESET + "\n")
        self._reset_line()
