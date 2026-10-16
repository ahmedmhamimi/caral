"""
Ports: the narrow interfaces the engine depends on. Adapters implement them; tests fake them.
The engine imports nothing from caral.adapters.

- LLM (Protocol): .stream(system, messages, tools, on_text=None, on_retry=None, model=None,
  max_tokens=None) -> Turn; .complete(system, prompt, max_tokens=2048, model=None) -> str; .close().
- Approver (Protocol): .ask(request: ApprovalRequest) -> Approval.
- Store (Protocol): .append(session, messages), .snapshot(session), .save_meta(session).
"""

from __future__ import annotations

from typing import Callable, Protocol

from .domain.events import Approval, ApprovalRequest
from .domain.messages import Turn


class LLM(Protocol):
    def stream(
        self, system: str, messages: list[dict], tools: list[dict],
        on_text: Callable[[str], None] | None = None,
        on_retry: Callable[[int, float, str], None] | None = None,
        model: str | None = None, max_tokens: int | None = None,
    ) -> Turn: ...

    def complete(self, system: str, prompt: str, max_tokens: int = 2048, model: str | None = None) -> str: ...

    def close(self) -> None: ...


class Approver(Protocol):
    def ask(self, request: ApprovalRequest) -> Approval: ...


class Store(Protocol):
    def append(self, session, messages: list[dict]) -> None: ...

    def snapshot(self, session) -> None: ...

    def save_meta(self, session) -> None: ...
