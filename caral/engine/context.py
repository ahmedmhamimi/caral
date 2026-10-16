"""
Context-window economics. Three escalating tools, cheapest first: (1) prune stale bulky tool
output in one batch;
(2) summarize older history with the small model; (3) emergency-drop old messages if even that
fails. Never splits a tool_use from its tool_result. Pruning and compaction rewrite the cached
prompt prefix (a cache write on the whole context), so they run only when the window is
actually filling (defaults 70% / 85%), never eagerly.

- ContextManager(llm, config): .plan(used_tokens) -> "none" | "prune" | "compact".
  - .prune(messages, keep_recent=6) -> list[str]: shortens stale large tool results in place; returns pruned tool_use ids.
  - .compact(messages, focus=None) -> (list[dict], str): summarizes old history; returns (messages, method).
  - .emergency_drop(messages) -> list[dict]: drops oldest messages at a safe boundary with a note.
- find_split_index(messages, keep) -> int: first safe index to keep verbatim (0 = nothing to cut).
- SUMMARY_PROMPT / PRUNED_NOTE / DROPPED_NOTE: static text.
"""

from __future__ import annotations

from ..domain.messages import has_tool_results, render_transcript, text_block, user_text

SUMMARY_PROMPT = """\
Summarize the coding session below so work can continue from your summary alone. Be specific and \
factual. Use these sections:

1. Goal: what the user is trying to achieve (quote key requirements).
2. Progress: what has been done, in order.
3. Files: paths created or modified, and what changed in each.
4. Findings: important facts learned about the codebase, commands that work, errors seen.
5. Open items: what remains, including the immediate next step.

Keep it under 700 words. Do not include greetings or commentary."""

PRUNED_NOTE = "\n[... output pruned to save context; re-run the tool if you need it ...]"
DROPPED_NOTE = "[Earlier conversation was dropped to fit the context window. Re-inspect files if needed.]"


def find_split_index(messages: list[dict], keep: int) -> int:
    if len(messages) <= keep:
        return 0
    i = len(messages) - keep
    while i < len(messages):
        m = messages[i]
        if m["role"] == "assistant" or (m["role"] == "user" and not has_tool_results(m)):
            return i
        i += 1
    return 0


class ContextManager:
    def __init__(self, llm, config) -> None:
        self.llm = llm
        self.config = config

    def plan(self, used_tokens: int) -> str:
        window = self.config.context_window
        if used_tokens >= int(window * self.config.compact_threshold):
            return "compact"
        if used_tokens >= int(window * self.config.prune_threshold):
            return "prune"
        return "none"

    def prune(self, messages: list[dict], keep_recent: int = 6) -> list[str]:
        """Shorten stale bulky tool results in place; returns the tool_use ids that were pruned."""
        cutoff = max(0, len(messages) - keep_recent)
        pruned: list[str] = []
        for m in messages[:cutoff]:
            if m["role"] != "user" or not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if b.get("type") != "tool_result":
                    continue
                c = b.get("content")
                if isinstance(c, str) and len(c) > 600 and not c.endswith(PRUNED_NOTE):
                    b["content"] = c[:300] + PRUNED_NOTE
                    pruned.append(b.get("tool_use_id", ""))
        return pruned

    def compact(self, messages: list[dict], focus: str | None = None) -> tuple[list[dict], str]:
        split = find_split_index(messages, keep=6) or find_split_index(messages, keep=2)
        if split <= 0:
            return messages, "none"
        old, recent = messages[:split], messages[split:]
        transcript = render_transcript(old)
        if len(transcript) > 120_000:
            transcript = "[earlier history omitted]\n" + transcript[-120_000:]
        prompt = SUMMARY_PROMPT + "\n\n"
        if focus:
            prompt += f"Pay special attention to: {focus}\n\n"
        prompt += f"<session>\n{transcript}\n</session>"
        try:
            summary = self.llm.complete(
                "You write precise technical summaries of coding sessions.", prompt,
                max_tokens=2000, model=self.config.small_model or None,
            ).strip()
        except Exception:  # noqa: BLE001 - fall back rather than wedge the session
            summary = ""
        if not summary:
            return self.emergency_drop(messages), "drop"
        note = f"[Conversation summary — earlier messages were compacted]\n\n{summary}"
        return self._prepend(recent, note), "summary"

    def emergency_drop(self, messages: list[dict]) -> list[dict]:
        split = find_split_index(messages, keep=4) or find_split_index(messages, keep=2)
        if split <= 0:
            return messages
        return self._prepend(messages[split:], DROPPED_NOTE)

    @staticmethod
    def _prepend(recent: list[dict], note: str) -> list[dict]:
        if recent and recent[0]["role"] == "user":
            first = dict(recent[0])
            first["content"] = [text_block(note)] + list(first["content"])
            return [first] + recent[1:]
        return [user_text(note)] + recent
