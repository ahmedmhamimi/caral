"""
Secret redaction for traces, logs and diagnostics. Conservative patterns for common credential
shapes; applied to everything written to disk for debugging.

- redact(text) -> str: replaces credential-like substrings with [REDACTED].
- redact_obj(obj) -> Any: recursively redacts strings inside dicts/lists.
"""

from __future__ import annotations

import re
from typing import Any

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|passw(or)?d)(\"?\s*[:=]\s*\"?)([^\s\",']{8,})"),
]


def redact(text: str) -> str:
    for p in _PATTERNS[:-1]:
        text = p.sub("[REDACTED]", text)
    return _PATTERNS[-1].sub(lambda m: f"{m.group(1)}{m.group(3)}[REDACTED]", text)


def redact_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
