"""
Anthropic Messages API adapter with SSE streaming and prompt caching (system prompt, tool list and
the newest message carry cache breakpoints so each step only pays full price for new content).

- AnthropicProvider: Provider for /v1/messages.
  - .stream(...) -> Turn; ._payload(system, messages, tools, model, max_tokens) -> dict; ._headers() -> dict;
    .models_url() -> str.
- clean_messages(messages) -> list[dict]: strips private keys and empty text blocks.
"""

from __future__ import annotations

import json
from typing import Callable

from ...domain.messages import Turn, Usage
from ...errors import ProviderError
from .base import Provider, open_stream, sse_events


def clean_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            if content:
                out.append({"role": m["role"], "content": content})
            continue
        blocks = [
            {k: v for k, v in b.items() if not k.startswith("_")}
            for b in content or []
            if not (b.get("type") == "text" and not b.get("text"))
        ]
        if blocks:
            out.append({"role": m["role"], "content": blocks})
    return out


class AnthropicProvider(Provider):
    kind = "anthropic"

    def models_url(self) -> str:
        return self.cfg.base_url.rstrip("/") + "/v1/models?limit=100"

    def _headers(self) -> dict:
        return {"x-api-key": self.cfg.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def _payload(self, system, messages, tools, model=None, max_tokens=None) -> dict:
        cache = self.cfg.prompt_cache
        msgs = clean_messages(messages)
        payload: dict = {
            "model": model or self.cfg.model, "max_tokens": max_tokens or self.cfg.max_tokens,
            "stream": True, "messages": msgs,
        }
        if system:
            block = {"type": "text", "text": system}
            if cache:
                block["cache_control"] = {"type": "ephemeral"}
            payload["system"] = [block]
        if tools:
            tool_list = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in tools]
            if cache:
                tool_list[-1] = {**tool_list[-1], "cache_control": {"type": "ephemeral"}}
            payload["tools"] = tool_list
        if cache and msgs and isinstance(msgs[-1]["content"], list):
            last = msgs[-1]["content"]
            last[-1] = {**last[-1], "cache_control": {"type": "ephemeral"}}
        if self.cfg.temperature is not None:
            payload["temperature"] = self.cfg.temperature
        return payload

    def stream(
        self, system, messages, tools, on_text: Callable[[str], None] | None = None,
        on_retry=None, model: str | None = None, max_tokens: int | None = None,
    ) -> Turn:
        url = self.cfg.base_url.rstrip("/") + "/v1/messages"
        resp = open_stream(
            self.client, url, self._headers(), self._payload(system, messages, tools, model, max_tokens),
            max_retries=self.cfg.max_retries, on_retry=on_retry,
        )
        blocks: dict[int, dict] = {}
        usage = Usage()
        stop_reason = "end_turn"
        try:
            for _event, data in sse_events(resp):
                try:
                    d = json.loads(data)
                except json.JSONDecodeError:
                    continue
                t = d.get("type")
                if t == "message_start":
                    u = d.get("message", {}).get("usage", {})
                    usage.input_tokens = u.get("input_tokens", 0)
                    usage.cache_read = u.get("cache_read_input_tokens", 0)
                    usage.cache_write = u.get("cache_creation_input_tokens", 0)
                elif t == "content_block_start":
                    cb, idx = d["content_block"], d["index"]
                    if cb["type"] == "text":
                        blocks[idx] = {"type": "text", "text": cb.get("text", "")}
                    elif cb["type"] == "tool_use":
                        blocks[idx] = {"type": "tool_use", "id": cb["id"], "name": cb["name"], "input": {}, "_json": ""}
                elif t == "content_block_delta":
                    blk, delta = blocks.get(d["index"]), d["delta"]
                    if blk is None:
                        continue
                    if delta["type"] == "text_delta":
                        blk["text"] += delta["text"]
                        if on_text:
                            on_text(delta["text"])
                    elif delta["type"] == "input_json_delta":
                        blk["_json"] += delta.get("partial_json", "")
                elif t == "message_delta":
                    stop_reason = d.get("delta", {}).get("stop_reason") or stop_reason
                    usage.output_tokens = d.get("usage", {}).get("output_tokens", usage.output_tokens)
                elif t == "error":
                    err = d.get("error", {})
                    etype = err.get("type", "error")
                    raise ProviderError(
                        f"{etype}: {err.get('message', data)}",
                        retryable=etype in ("overloaded_error", "api_error"), kind="overloaded" if etype == "overloaded_error" else "server",
                    )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001 - connection dropped mid-stream
            raise ProviderError(f"Stream interrupted: {e}", retryable=True, kind="network") from e
        finally:
            resp.close()

        final: list[dict] = []
        for idx in sorted(blocks):
            b = blocks[idx]
            if b["type"] == "tool_use":
                raw = b.pop("_json", "")
                try:
                    parsed = json.loads(raw) if raw.strip() else {}
                    b["input"] = parsed if isinstance(parsed, dict) else {"_parse_error": raw[:200]}
                except json.JSONDecodeError:
                    b["input"] = {"_parse_error": raw[:200]}
                final.append(b)
            elif b["text"]:
                final.append(b)
        return Turn(blocks=final, stop_reason=stop_reason, usage=usage)
