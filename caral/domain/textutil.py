"""
Small pure text helpers shared by tools and the UI.

- truncate_middle(text, limit) -> str: keeps head and tail of long text with a marker between.
- make_diff(old, new, path, context=3, max_lines=120) -> str: unified diff, capped.
- clip(text, n) -> str: single-line truncation with an ellipsis.
- human_tokens(n) -> str: 1234 -> "1.2k".
- human_duration(seconds) -> str: 0.42 -> "420ms", 75 -> "1m15s".
"""

from __future__ import annotations

import difflib


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n\n... [{cut} characters truncated] ...\n\n{text[-tail:]}"


def make_diff(old: str, new: str, path: str, context: int = 3, max_lines: int = 120) -> str:
    lines = list(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), fromfile=f"a/{path.lstrip('/')}", tofile=f"b/{path.lstrip('/')}",
            lineterm="", n=context,
        )
    )
    if len(lines) > max_lines:
        extra = len(lines) - max_lines
        lines = lines[:max_lines] + [f"... ({extra} more diff lines)"]
    return "\n".join(lines)


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: max(0, n - 1)] + "…"


def human_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 100_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1000:.0f}k"


def human_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"
