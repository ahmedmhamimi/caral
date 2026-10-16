"""
Structured trace: every engine event of a session is appended as one redacted JSON line, giving a
complete, greppable audit trail (what was asked, which tools ran, what was approved, tokens, cache
hits, timings, errors). The trace is just another EventBus subscriber.

- TraceWriter(path_or_fn, max_bytes=20_000_000): callable subscriber that follows the active session
  (path may be a callable); .close(); .path. Redacts secrets, caps size.
- trace_path(session_id) -> Path: <data_dir>/traces/<id>.jsonl.
- summarize_trace(path) -> dict: aggregate stats (runs, steps, tool counts/errors, tokens, cache hit ratio, time).
- format_summary(stats) -> str: human-readable report for `caral --trace` and /trace.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from ...config import data_dir
from ...domain.events import Event
from ...domain.redact import redact_obj


def trace_path(session_id: str) -> Path:
    d = data_dir() / "traces"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{session_id}.jsonl"


class TraceWriter:
    def __init__(self, path, max_bytes: int = 20_000_000) -> None:
        self._path_fn: Callable[[], Path] = path if callable(path) else (lambda: path)
        self.max_bytes = max_bytes
        self._f = None
        self._cur: Path | None = None
        self._size = 0

    @property
    def path(self) -> Path | None:
        return self._cur

    def _ensure(self) -> bool:
        target = self._path_fn()
        if target != self._cur:  # session changed (/clear, /resume): follow it
            self.close()
            self._f = open(target, "a", encoding="utf-8")
            self._cur = target
            self._size = target.stat().st_size
        return self._f is not None

    def __call__(self, event: Event) -> None:
        d = event.to_dict()
        if d["type"] == "TextDelta":
            return  # deltas are noise; the final text is in the session log
        if not self._ensure() or self._size >= self.max_bytes:
            return
        if isinstance(d.get("content"), str):
            d["content"] = d["content"][:500]
        d.pop("display", None)
        line = json.dumps(redact_obj(d), ensure_ascii=False, default=str) + "\n"
        self._f.write(line)
        self._f.flush()
        self._size += len(line)

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
        self._f, self._cur = None, None


def summarize_trace(path: Path) -> dict:
    s = {
        "runs": 0, "steps": 0, "tool_calls": Counter(), "tool_errors": Counter(), "denied": 0,
        "tool_time": 0.0, "model_time": 0.0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
        "compactions": 0, "prunes": 0, "retries": 0, "statuses": Counter(), "errors": [],
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = e.get("type")
            if t == "RunFinished":
                s["runs"] += 1
                s["statuses"][e["status"]] += 1
                if e.get("error"):
                    s["errors"].append(e["error"][:160])
            elif t == "ModelResponded":
                s["steps"] += 1
                s["model_time"] += e["duration_s"]
                u = e["usage"]
                s["input"] += u["input_tokens"]
                s["output"] += u["output_tokens"]
                s["cache_read"] += u["cache_read"]
                s["cache_write"] += u["cache_write"]
            elif t == "ToolFinished":
                s["tool_calls"][e["name"]] += 1
                s["tool_time"] += e["duration_s"]
                if e["is_error"]:
                    s["tool_errors"][e["name"]] += 1
            elif t == "PermissionResolved" and e["decision"] == "deny":
                s["denied"] += 1
            elif t == "ContextCompacted":
                s["compactions"] += 1
            elif t == "ContextPruned":
                s["prunes"] += 1
            elif t == "RetryScheduled":
                s["retries"] += 1
    return s


def format_summary(s: dict) -> str:
    total_prompt = s["input"] + s["cache_read"] + s["cache_write"]
    hit = (s["cache_read"] / total_prompt * 100) if total_prompt else 0
    lines = [
        f"runs {s['runs']} · model calls {s['steps']} · tool calls {sum(s['tool_calls'].values())}",
        f"time: model {s['model_time']:.1f}s · tools {s['tool_time']:.1f}s",
        f"tokens: {s['input']} fresh in · {s['cache_read']} cached in ({hit:.0f}% cache hit) · "
        f"{s['cache_write']} cache writes · {s['output']} out",
        f"context: {s['prunes']} prunes · {s['compactions']} compactions · {s['retries']} retries · {s['denied']} denied actions",
    ]
    if s["tool_calls"]:
        lines.append("tools: " + ", ".join(
            f"{n}×{c}" + (f" ({s['tool_errors'][n]} failed)" if s["tool_errors"][n] else "")
            for n, c in s["tool_calls"].most_common()))
    if s["errors"]:
        lines.append("errors: " + " | ".join(s["errors"][-3:]))
    return "\n".join(lines)
