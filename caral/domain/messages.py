"""
Message and block types shared by every layer. Messages use an Anthropic-style block format
internally; providers translate to and from their own wire formats.

- Usage: token counters (input_tokens uncached, output_tokens, cache_read, cache_write).
  - .add(other) -> None; .to_dict() -> dict; Usage.from_dict(d) -> Usage.
  - .context_tokens -> int: tokens occupying the context window (input + cache_read + cache_write).
  - .billable -> int: input + cache_write + output (everything except cheap cache reads).
  - .cache_hit_ratio -> float: share of prompt tokens served from cache.
- ToolCall: one requested invocation (id, name, input).
- Turn: a finished model response. .text -> str, .tool_calls -> list[ToolCall].
- text_block / tool_use_block / tool_result_block -> dict: block constructors.
- user_text(text) / user_blocks(blocks) / assistant_message(blocks) -> dict: message constructors.
- message_text(msg) -> str: concatenated text of a message.
- has_tool_results(msg) -> bool: True if a user message carries tool_result blocks.
- count_chars(messages, system="") -> int: character size of a conversation.
- render_transcript(messages, result_limit=600) -> str: compact plain-text transcript for summarization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read += other.cache_read
        self.cache_write += other.cache_write

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cache_read": self.cache_read, "cache_write": self.cache_write,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Usage":
        d = d or {}
        return cls(
            int(d.get("input_tokens", 0)), int(d.get("output_tokens", 0)),
            int(d.get("cache_read", 0)), int(d.get("cache_write", 0)),
        )

    @property
    def context_tokens(self) -> int:
        return self.input_tokens + self.cache_read + self.cache_write

    @property
    def billable(self) -> int:
        return self.input_tokens + self.cache_write + self.output_tokens

    @property
    def cache_hit_ratio(self) -> float:
        total = self.context_tokens
        return self.cache_read / total if total else 0.0


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    blocks: list[dict] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.blocks if b.get("type") == "text")

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(b["id"], b["name"], b.get("input") if isinstance(b.get("input"), dict) else {})
            for b in self.blocks
            if b.get("type") == "tool_use"
        ]


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_use_block(id: str, name: str, input: dict) -> dict:
    return {"type": "tool_use", "id": id, "name": name, "input": input}


def tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def user_text(text: str) -> dict:
    return {"role": "user", "content": [text_block(text)]}


def user_blocks(blocks: list[dict]) -> dict:
    return {"role": "user", "content": blocks}


def assistant_message(blocks: list[dict]) -> dict:
    return {"role": "assistant", "content": blocks}


def message_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def has_tool_results(msg: dict) -> bool:
    content = msg.get("content")
    return isinstance(content, list) and any(b.get("type") == "tool_result" for b in content)


def _block_chars(block: dict) -> int:
    t = block.get("type")
    if t == "text":
        return len(block.get("text", ""))
    if t == "tool_use":
        return len(block.get("name", "")) + len(json.dumps(block.get("input", {})))
    if t == "tool_result":
        c = block.get("content", "")
        return len(c) if isinstance(c, str) else len(json.dumps(c))
    return len(json.dumps(block))


def count_chars(messages: list[dict], system: str = "") -> int:
    chars = len(system)
    for m in messages:
        content = m.get("content")
        chars += len(content) if isinstance(content, str) else sum(_block_chars(b) for b in content or [])
        chars += 16
    return chars


def render_transcript(messages: list[dict], result_limit: int = 600) -> str:
    out: list[str] = []
    for m in messages:
        role = m.get("role", "?").upper()
        content = m.get("content")
        if isinstance(content, str):
            out.append(f"{role}: {content}")
            continue
        for b in content or []:
            t = b.get("type")
            if t == "text":
                out.append(f"{role}: {b.get('text', '')}")
            elif t == "tool_use":
                args = json.dumps(b.get("input", {}), ensure_ascii=False)
                out.append(f"ASSISTANT called {b.get('name')}({args[:400]}{'...' if len(args) > 400 else ''})")
            elif t == "tool_result":
                c: Any = b.get("content", "")
                c = c if isinstance(c, str) else json.dumps(c)
                if len(c) > result_limit:
                    half = result_limit // 2
                    c = c[:half] + " ...[snip]... " + c[-half:]
                out.append(f"TOOL {'ERROR' if b.get('is_error') else 'RESULT'}: {c}")
    return "\n".join(out)
