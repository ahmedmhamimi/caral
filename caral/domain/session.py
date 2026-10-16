"""
Session value object: the durable state of one conversation. Persistence lives in
adapters.store; this module is pure data.

- Session: id, cwd, model, created, updated, title, messages, usage (Usage), todos, persisted (count of
  messages already written to the store).
  - Session.new(cwd, model) -> Session: fresh session with a unique, sortable id.
  - .derive_title() -> None: sets title from the first real user message.
  - .meta() -> dict: JSON-safe metadata (no messages).
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from .messages import Usage, message_text


@dataclass
class Session:
    id: str
    cwd: str
    model: str
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    title: str = ""
    messages: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    todos: list = field(default_factory=list)
    persisted: int = 0

    @staticmethod
    def new(cwd, model: str) -> "Session":
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        return Session(id=sid, cwd=str(cwd), model=model)

    def derive_title(self) -> None:
        if self.title:
            return
        for m in self.messages:
            if m["role"] == "user":
                text = message_text(m).strip().replace("\n", " ")
                if text and not text.startswith("[Conversation summary") and not text.startswith("[Earlier"):
                    self.title = text[:80]
                    return

    def meta(self) -> dict:
        return {
            "id": self.id, "cwd": self.cwd, "model": self.model, "created": self.created,
            "updated": self.updated, "title": self.title, "usage": self.usage.to_dict(),
            "todos": self.todos, "turns": len(self.messages),
        }
