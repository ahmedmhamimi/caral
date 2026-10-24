"""
Typed event vocabulary. The engine emits these; the terminal UI, JSONL trace, metrics and tests
are interchangeable subscribers. Adding a new consumer never touches the engine.

- Event: base (kw-only dataclass) with ts and .to_dict() -> dict (JSON-safe, includes "type").
- RunStarted / RunFinished: one user message processed (status, steps, tool_calls, usage, duration).
- StepStarted / ModelResponded: one model round-trip.
- TextDelta: streamed assistant text.
- ToolRequested / PermissionResolved / ToolStarted / ToolFinished: lifecycle of one tool call.
- TodosUpdated: todo list replaced.
- ContextPruned / ContextCompacted: context shrinking.
- RetryScheduled: transient failure, retry pending.
- SubagentStarted / SubagentFinished: delegated task boundaries (Finished carries the model and usage billed).
- AuxUsage(kind, model, usage): tokens spent outside the main loop (context summaries) so cost meters see them.
- Notice(level, message): free-form user-visible note.
- ApprovalRequest / Approval: request/response pair for the Approver port.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any

from .messages import Usage


@dataclass(kw_only=True)
class Event:
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": type(self).__name__}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = v.to_dict() if hasattr(v, "to_dict") else v
        return d


@dataclass(kw_only=True)
class RunStarted(Event):
    run_id: str
    prompt_chars: int
    model: str


@dataclass(kw_only=True)
class StepStarted(Event):
    run_id: str
    step: int


@dataclass(kw_only=True)
class TextDelta(Event):
    text: str


@dataclass(kw_only=True)
class ModelResponded(Event):
    run_id: str
    step: int
    model: str
    stop_reason: str
    usage: Usage
    duration_s: float
    tool_calls: int


@dataclass(kw_only=True)
class ToolRequested(Event):
    call_id: str
    name: str
    summary: str
    category: str
    step: int


@dataclass(kw_only=True)
class PermissionResolved(Event):
    call_id: str
    name: str
    decision: str  # allow | deny
    source: str  # policy | user | session-rule
    reason: str = ""


@dataclass(kw_only=True)
class ToolStarted(Event):
    call_id: str
    name: str
    summary: str
    category: str


@dataclass(kw_only=True)
class ToolFinished(Event):
    call_id: str
    name: str
    is_error: bool
    duration_s: float
    content: str
    chars: int
    display: str | None = None


@dataclass(kw_only=True)
class TodosUpdated(Event):
    todos: list


@dataclass(kw_only=True)
class ContextPruned(Event):
    count: int


@dataclass(kw_only=True)
class ContextCompacted(Event):
    before: int
    after: int
    method: str  # summary | drop
    model: str = ""


@dataclass(kw_only=True)
class RetryScheduled(Event):
    attempt: int
    delay_s: float
    reason: str


@dataclass(kw_only=True)
class SubagentStarted(Event):
    description: str


@dataclass(kw_only=True)
class SubagentFinished(Event):
    description: str
    duration_s: float
    usage: Usage
    status: str
    model: str = ""


@dataclass(kw_only=True)
class AuxUsage(Event):
    kind: str  # compaction
    model: str
    usage: Usage


@dataclass(kw_only=True)
class Notice(Event):
    level: str  # info | warn | error
    message: str


@dataclass(kw_only=True)
class RunFinished(Event):
    run_id: str
    status: str  # ok | interrupted | error | step_limit | budget | loop
    steps: int
    tool_calls: int
    duration_s: float
    usage: Usage
    error: str = ""


@dataclass
class ApprovalRequest:
    call_id: str
    tool: str
    summary: str
    category: str
    reason: str
    diff: str | None = None


@dataclass
class Approval:
    choice: str  # yes | no | always
    feedback: str = ""
