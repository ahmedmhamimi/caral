"""
OpenAI-compatible chat-completions adapter (OpenAI, OpenRouter, Groq, DeepSeek, Together, Ollama,
LM Studio, vLLM, ...). Adapts to server quirks at runtime: drops stream_options or swaps the
max-token parameter name when a server rejects them.

Prompt caching: OpenAI and DeepSeek cache prefixes automatically, but Claude models behind an
OpenAI-style gateway (OpenRouter) only cache where the request marks breakpoints. For those the
adapter marks the system message, the latest user message (a stable anchor) and the newest message
(so tool results inside a long run are cached too). If a server rejects the markers the adapter
steps down (all -> system+user only -> none) and remembers it for the session.

- OpenAIProvider: Provider for /chat/completions.
  - .stream(...) -> Turn; ._payload(...) -> dict; ._headers() -> dict.
- convert_messages(system, messages) -> list[dict]: internal blocks -> OpenAI chat messages.
- convert_tools(tools) -> list[dict]: internal schemas -> OpenAI function tools.
- wants_cache_markers(cfg, model) -> bool: True for Claude models on OpenRouter (cache_markers="auto"),
  always with "on", never with "off" or prompt_cache=False.
- add_cache_markers(messages, mode="full") -> list[dict]: copy of converted messages with
  {"type": "ephemeral"} cache_control on text parts; mode "basic" skips the newest-message breakpoint.
- parse_usage(u) -> Usage: normalizes OpenAI/DeepSeek/OpenRouter usage (cache reads and writes split out).
"""

from __future__ import annotations

import json
from typing import Callable
from urllib.parse import urlparse

from ...domain.messages import Turn, Usage
from ...errors import ProviderError
from .base import Provider, open_stream, sse_events

_STOP_MAP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens", "function_call": "tool_use"}


def convert_tools(tools: list[dict]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in tools
    ]


CACHE_CONTROL = {"type": "ephemeral"}


def wants_cache_markers(cfg, model: str) -> bool:
    mode = (getattr(cfg, "cache_markers", "auto") or "auto").lower()
    if mode == "off" or not getattr(cfg, "prompt_cache", True):
        return False
    if mode == "on":
        return True
    host = (urlparse(cfg.base_url or "").hostname or "").lower()
    m = (model or "").lower()
    return host.endswith("openrouter.ai") and (m.startswith("anthropic/") or "claude" in m)


def _mark(msg: dict) -> None:
    c = msg.get("content")
    if isinstance(c, str) and c:
        msg["content"] = [{"type": "text", "text": c, "cache_control": dict(CACHE_CONTROL)}]
    elif isinstance(c, list) and c and isinstance(c[-1], dict) and c[-1].get("type") == "text" and c[-1].get("text"):
        msg["content"] = c[:-1] + [{**c[-1], "cache_control": dict(CACHE_CONTROL)}]


def add_cache_markers(messages: list[dict], mode: str = "full") -> list[dict]:
    """Breakpoints (at most 3 of the 4 Anthropic allows): system, latest user message, newest message."""
    msgs = [dict(m) for m in messages]
    targets: set[int] = set()
    if msgs and msgs[0].get("role") == "system":
        targets.add(0)
    users = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    if users:
        targets.add(users[-1])
    if mode == "full" and msgs:
        targets.add(len(msgs) - 1)
    for i in targets:
        _mark(msgs[i])
    return msgs


def convert_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            if content:
                out.append({"role": m["role"], "content": content})
            continue
        content = content or []
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        if m["role"] == "user":
            for b in content:  # tool results must directly follow the assistant's tool_calls
                if b.get("type") == "tool_result":
                    c = b.get("content", "")
                    c = c if isinstance(c, str) else json.dumps(c)
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": ("ERROR: " + c) if b.get("is_error") else c})
            if text:
                out.append({"role": "user", "content": text})
        else:
            calls = [
                {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
                for b in content if b.get("type") == "tool_use"
            ]
            if text or calls:
                msg: dict = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
    return out


def parse_usage(u: dict) -> Usage:
    if "prompt_cache_hit_tokens" in u:  # DeepSeek style
        hit = u.get("prompt_cache_hit_tokens", 0)
        return Usage(u.get("prompt_cache_miss_tokens", 0), u.get("completion_tokens", 0), hit, 0)
    prompt = u.get("prompt_tokens", 0)
    details = u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0) or 0
    written = details.get("cache_write_tokens", 0) or 0   # reported by OpenRouter for Anthropic models
    return Usage(max(0, prompt - cached - written), u.get("completion_tokens", 0), cached, written)


class OpenAIProvider(Provider):
    kind = "openai"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._stream_options = True
        self._cache_mode = "full"   # full -> basic -> off, stepped down if a server rejects the markers
        self._token_param = cfg.max_tokens_param or (
            "max_completion_tokens" if "api.openai.com" in cfg.base_url else "max_tokens"
        )

    def _headers(self) -> dict:
        h = {"content-type": "application/json"}
        if self.cfg.api_key:
            h["authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _payload(self, system, messages, tools, model=None, max_tokens=None) -> dict:
        msgs = convert_messages(system, messages)
        if self._cache_mode != "off" and wants_cache_markers(self.cfg, model or self.cfg.model):
            msgs = add_cache_markers(msgs, self._cache_mode)
        payload: dict = {
            "model": model or self.cfg.model, "messages": msgs, "stream": True,
            self._token_param: max_tokens or self.cfg.max_tokens,
        }
        if self._stream_options:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = convert_tools(tools)
        if self.cfg.temperature is not None:
            payload["temperature"] = self.cfg.temperature
        return payload

    def _open(self, system, messages, tools, model, max_tokens, on_retry):
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        for _ in range(5):  # a few capability fallbacks (stream options, token param name, cache markers)
            try:
                return open_stream(self.client, url, self._headers(),
                                   self._payload(system, messages, tools, model, max_tokens),
                                   max_retries=self.cfg.max_retries, on_retry=on_retry)
            except ProviderError as e:
                msg = str(e).lower()
                if e.status in (400, 422) and "stream_options" in msg and self._stream_options:
                    self._stream_options = False
                elif e.status in (400, 422) and "max_completion_tokens" in msg and self._token_param != "max_tokens":
                    self._token_param = "max_tokens"
                elif e.status in (400, 422) and "max_tokens" in msg and self._token_param == "max_tokens":
                    self._token_param = "max_completion_tokens"
                elif (e.status in (400, 422) and self._cache_mode != "off" and wants_cache_markers(self.cfg, model or self.cfg.model)
                      and ("cache" in msg or "content" in msg)):
                    self._cache_mode = "basic" if self._cache_mode == "full" else "off"
                else:
                    raise
        raise ProviderError("Server rejected the request parameters repeatedly.", kind="bad_request")

    def stream(
        self, system, messages, tools, on_text: Callable[[str], None] | None = None,
        on_retry=None, model: str | None = None, max_tokens: int | None = None,
    ) -> Turn:
        resp = self._open(system, messages, tools, model, max_tokens, on_retry)
        text_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage = Usage()
        finish = None
        try:
            for _event, data in sse_events(resp):
                if data.strip() == "[DONE]":
                    break
                try:
                    d = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict) and "error" in d and not d.get("choices"):
                    err = d["error"]
                    raise ProviderError(f"Provider error: {err.get('message') if isinstance(err, dict) else err}", kind="server")
                if d.get("usage"):
                    usage = parse_usage(d["usage"])
                for choice in d.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text_parts.append(delta["content"])
                        if on_text:
                            on_text(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"Stream interrupted: {e}", retryable=True, kind="network") from e
        finally:
            resp.close()

        blocks: list[dict] = []
        text = "".join(text_parts)
        if text:
            blocks.append({"type": "text", "text": text})
        for i, idx in enumerate(sorted(calls)):
            c = calls[idx]
            try:
                args = json.loads(c["args"]) if c["args"].strip() else {}
                if not isinstance(args, dict):
                    args = {"_parse_error": c["args"][:200]}
            except json.JSONDecodeError:
                args = {"_parse_error": c["args"][:200]}
            blocks.append({"type": "tool_use", "id": c["id"] or f"call_{idx}_{i}", "name": c["name"], "input": args})
        stop = _STOP_MAP.get(finish or "stop", "end_turn")
        if calls and stop == "end_turn":
            stop = "tool_use"
        return Turn(blocks=blocks, stop_reason=stop, usage=usage)
