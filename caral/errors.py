"""
Typed error taxonomy. Every failure the user can see maps to one of these, carrying a message
(what happened) and a hint (how to fix it) so the UI never shows a bare traceback.

- CaralError(message, hint=""): base class. .user_message() -> str renders "what\n  -> how to fix".
- ConfigError: bad or missing configuration (exit code 2).
- ToolError: a tool failed in a way the model cannot recover from.
- Interrupted: user pressed Ctrl-C (exit code 130).
- ProviderError(message, status=None, retryable=False, kind="unknown", hint=""): LLM API failure.
- ContextLengthError: the prompt exceeded the model's context window (triggers auto-compaction).
- classify_http_error(status, message) -> ProviderError: maps an HTTP failure to the right subclass/hint.
"""

from __future__ import annotations

import re


class CaralError(Exception):
    exit_code = 1

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def user_message(self) -> str:
        return str(self) + (f"\n  -> {self.hint}" if self.hint else "")


class ConfigError(CaralError):
    exit_code = 2


class ToolError(CaralError):
    pass


class Interrupted(CaralError):
    exit_code = 130


class ProviderError(CaralError):
    def __init__(
        self, message: str, status: int | None = None, retryable: bool = False,
        kind: str = "unknown", hint: str = "",
    ) -> None:
        super().__init__(message, hint=hint)
        self.status = status
        self.retryable = retryable
        self.kind = kind


class ContextLengthError(ProviderError):
    pass


_CTX = re.compile(
    r"prompt is too long|context[_ ]length|maximum context|too many tokens|context window|"
    r"reduce the length|input is too long|exceeds the (model|context)",
    re.I,
)


def classify_http_error(status: int, message: str) -> ProviderError:
    text = f"HTTP {status}: {message}"
    if status in (400, 413, 422) and _CTX.search(message):
        return ContextLengthError(
            text, status, kind="context_length", hint="Caral will compact the conversation and retry."
        )
    if status in (401, 403):
        return ProviderError(text, status, kind="auth", hint="Check the API key for this provider (caral --doctor).")
    if status == 404:
        return ProviderError(text, status, kind="not_found", hint="Check the model name and base URL. `caral --models` lists what your key can use.")
    if status == 429:
        return ProviderError(text, status, True, "rate_limit", "Rate limited; Caral retries with backoff.")
    if status in (408, 409, 500, 502, 503, 504, 529):
        return ProviderError(text, status, True, "server", "The provider is having trouble; retrying may help.")
    if status in (400, 422):
        return ProviderError(text, status, kind="bad_request", hint="The request was rejected; check model and parameters.")
    return ProviderError(text, status)
