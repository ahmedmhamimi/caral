"""
Provider plumbing shared by concrete LLM adapters: streaming POST with retry/backoff and an SSE
parser. Concrete providers only translate wire formats.

- Provider: abstract base implementing the LLM port.
  - .stream(system, messages, tools, on_text=None, on_retry=None, model=None, max_tokens=None) -> Turn.
  - .complete(system, prompt, max_tokens=2048, model=None) -> str: tool-less single-shot completion.
  - .list_models() -> list[str]: model ids the key can use; .models_url() -> str.
  - .close() -> None.
- make_client() -> httpx.Client: streaming-friendly timeouts, no env proxies surprises.
- open_stream(client, url, headers, payload, max_retries=4, on_retry=None) -> httpx.Response:
  retries 408/409/429/5xx and transient network errors with backoff (honors Retry-After).
- sse_events(response) -> Iterator[(event | None, data)]: parses server-sent events.
- backoff_delay(attempt, retry_after=None) -> float.
- error_message(body) -> str: extracts a readable message from an error body.
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, Iterator

import httpx

from ...domain.messages import Turn, user_text
from ...errors import ProviderError, classify_http_error


def make_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=30.0), follow_redirects=True
    )


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None and retry_after > 0:
        return min(retry_after, 60.0)
    return min(2.0**attempt, 30.0) * (0.75 + random.random() * 0.5)


def error_message(body: str) -> str:
    try:
        data = json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else data
        return (err.get("message") or json.dumps(err)) if isinstance(err, dict) else str(err)
    except (json.JSONDecodeError, AttributeError):
        return body.strip()[:500] or "(empty response body)"


def open_stream(
    client: httpx.Client, url: str, headers: dict, payload: dict, max_retries: int = 4,
    on_retry: Callable[[int, float, str], None] | None = None,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            resp = client.send(client.build_request("POST", url, headers=headers, json=payload), stream=True)
        except httpx.HTTPError as e:
            if attempt >= max_retries:
                raise ProviderError(
                    f"Network error: {e}", retryable=True, kind="network",
                    hint="Check your connection and the base URL.",
                ) from e
            delay = backoff_delay(attempt)
            if on_retry:
                on_retry(attempt + 1, delay, f"network error ({type(e).__name__})")
            time.sleep(delay)
            attempt += 1
            continue
        if resp.status_code < 400:
            return resp
        body = resp.read().decode("utf-8", "replace")
        status = resp.status_code
        retry_after_raw = resp.headers.get("retry-after")
        resp.close()
        err = classify_http_error(status, error_message(body))
        if err.retryable and attempt < max_retries:
            try:
                retry_after = float(retry_after_raw) if retry_after_raw else None
            except ValueError:
                retry_after = None
            delay = backoff_delay(attempt, retry_after)
            if on_retry:
                on_retry(attempt + 1, delay, f"HTTP {status}")
            time.sleep(delay)
            attempt += 1
            continue
        raise err


def sse_events(response: httpx.Response) -> Iterator[tuple[str | None, str]]:
    event: str | None = None
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield event, "\n".join(data_lines)


class Provider(ABC):
    kind = "base"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.client = make_client()

    @abstractmethod
    def stream(
        self, system: str, messages: list[dict], tools: list[dict],
        on_text: Callable[[str], None] | None = None,
        on_retry: Callable[[int, float, str], None] | None = None,
        model: str | None = None, max_tokens: int | None = None,
    ) -> Turn:
        """Run one model call, streaming text to on_text, and return the finished Turn."""

    def complete(self, system: str, prompt: str, max_tokens: int = 2048, model: str | None = None) -> str:
        return self.stream(system, [user_text(prompt)], [], model=model, max_tokens=max_tokens).text

    def models_url(self) -> str:
        return self.cfg.base_url.rstrip("/") + "/models"

    def list_models(self) -> list[str]:
        """Ask the provider which model ids this key can use (OpenAI-style GET /models)."""
        try:
            r = self.client.get(self.models_url(), headers=self._headers(), timeout=20.0)
        except httpx.HTTPError as e:
            raise ProviderError(f"Network error: {e}", retryable=True, kind="network",
                                hint="Check your connection and the base URL.") from e
        if r.status_code >= 400:
            raise classify_http_error(r.status_code, error_message(r.text))
        try:
            data = r.json()
        except ValueError as e:
            raise ProviderError("The /models endpoint did not return JSON.", kind="bad_response") from e
        items = data.get("data", []) if isinstance(data, dict) else data
        return sorted({str(i.get("id") or i.get("name")) for i in items if isinstance(i, dict) and (i.get("id") or i.get("name"))})

    def close(self) -> None:
        self.client.close()
