"""
Conversation invariants. Providers reject histories where a tool_use has no matching tool_result
(or vice versa), which happens after crashes, kills and interrupted sessions. These functions
detect and repair such histories so a resumed session can never wedge itself.

- validate_conversation(messages) -> list[str]: human-readable problems, empty when valid.
- repair_conversation(messages) -> (list[dict], list[str]): returns a valid copy plus notes on fixes.
- ensure_unique_ids(blocks) -> list[dict]: renames duplicate/empty tool_use ids within one response.
- SYNTHETIC_RESULT: text used for results fabricated during repair.
"""

from __future__ import annotations

import copy

from .messages import text_block, tool_result_block

SYNTHETIC_RESULT = "Tool call did not complete (the session was interrupted before it finished)."


def _blocks(msg: dict) -> list[dict]:
    c = msg.get("content")
    return c if isinstance(c, list) else ([text_block(c)] if isinstance(c, str) and c else [])


def _tool_use_ids(msg: dict) -> list[str]:
    if msg.get("role") != "assistant":
        return []
    return [b["id"] for b in _blocks(msg) if b.get("type") == "tool_use"]


def _result_ids(msg: dict) -> list[str]:
    if msg.get("role") != "user":
        return []
    return [b["tool_use_id"] for b in _blocks(msg) if b.get("type") == "tool_result"]


def validate_conversation(messages: list[dict]) -> list[str]:
    problems: list[str] = []
    seen_ids: set[str] = set()
    for i, m in enumerate(messages):
        if not _blocks(m):
            problems.append(f"message {i} ({m.get('role')}) is empty")
        for tid in _tool_use_ids(m):
            if tid in seen_ids:
                problems.append(f"message {i}: duplicate tool_use id {tid}")
            seen_ids.add(tid)
        prev_ids = set(_tool_use_ids(messages[i - 1])) if i else set()
        for rid in _result_ids(m):
            if rid not in prev_ids:
                problems.append(f"message {i}: tool_result {rid} has no matching tool_use before it")
        if m.get("role") == "assistant":
            want = set(_tool_use_ids(m))
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            have = set(_result_ids(nxt)) if nxt else set()
            for tid in want - have:
                problems.append(f"message {i}: tool_use {tid} has no tool_result")
    if messages and messages[0].get("role") != "user":
        problems.append("conversation does not start with a user message")
    return problems


def repair_conversation(messages: list[dict]) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    msgs = [m for m in messages if _blocks(m)]
    if len(msgs) != len(messages):
        notes.append(f"dropped {len(messages) - len(msgs)} empty message(s)")

    # Pass 1: drop tool_results that do not answer the immediately preceding assistant message.
    cleaned: list[dict] = []
    open_ids: set[str] = set()
    for m in msgs:
        if m.get("role") == "user" and _result_ids(m):
            keep = [b for b in _blocks(m) if b.get("type") != "tool_result" or b["tool_use_id"] in open_ids]
            if len(keep) != len(_blocks(m)):
                notes.append("dropped orphan tool_result block(s)")
            if not keep:
                open_ids = set()
                continue
            m = {**m, "content": keep}
        cleaned.append(m)
        open_ids = set(_tool_use_ids(m))

    # Pass 2: ensure every tool_use is answered, results first in the following user message.
    out: list[dict] = []
    i = 0
    while i < len(cleaned):
        m = cleaned[i]
        out.append(m)
        ids = _tool_use_ids(m)
        nxt = cleaned[i + 1] if i + 1 < len(cleaned) else None
        if ids:
            have = set(_result_ids(nxt)) if nxt else set()
            missing = [t for t in ids if t not in have]
            if missing:
                notes.append(f"added {len(missing)} synthetic tool_result(s)")
                synth = [tool_result_block(t, SYNTHETIC_RESULT, True) for t in missing]
                if nxt is not None and nxt.get("role") == "user":
                    merged = synth + _blocks(nxt)
                    results = [b for b in merged if b.get("type") == "tool_result"]
                    others = [b for b in merged if b.get("type") != "tool_result"]
                    out.append({**nxt, "content": results + others})
                    i += 2
                    continue
                out.append({"role": "user", "content": synth})
        i += 1

    if out and out[0].get("role") != "user":
        out.insert(0, {"role": "user", "content": [text_block("[Earlier conversation omitted.]")]})
        notes.append("conversation did not start with a user message; inserted a placeholder")
    return copy.deepcopy(out) if notes else messages, notes


def ensure_unique_ids(blocks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for n, b in enumerate(blocks):
        if b.get("type") == "tool_use":
            tid = b.get("id") or f"call_{n}"
            base, k = tid, 1
            while tid in seen:
                k += 1
                tid = f"{base}_{k}"
            seen.add(tid)
            if tid != b.get("id"):
                b = {**b, "id": tid}
        out.append(b)
    return out
