"""
The agent loop. Pure orchestration: asks the LLM port for a turn, runs tools through the
executor, appends results, repeats. Everything observable is published as events; everything
external (LLM, approval, storage) is injected through ports.

Cost controls built in: batched context pruning, small-model compaction, read dedupe (via tools),
loop detection, per-run token budget, automatic recovery from context-length errors.

- RunResult: text, status (ok|interrupted|error|step_limit|budget|loop), steps, tool_calls, duration_s,
  usage, error. .ok -> bool, .exit_code -> int.
- Agent(config, llm, registry, policy, approver, bus, session=None, store=None, cwd=None, subagent=False):
  - .run(text) -> RunResult: processes one user message through the full tool loop.
  - .reset() -> None / .resume(session) -> None: start a fresh or load an existing session.
  - .compact(focus=None) -> str: force summarization; returns a status line.
  - .undo() -> list[str]: restore files changed in the last turn.
  - .context_usage() -> (used_tokens, window): for status displays.
  - .system_prompt() -> str: cached system prompt (rebuilt by .refresh_system()).
  - ._call_model(run_id, step) -> Turn: one LLM call with retry and context-length recovery.
  - ._maybe_compact() -> None: prune then compact according to thresholds.
  - ._forget_pruned_views(ids) -> None: drops read-dedupe entries only for files whose output was pruned.
  - ._spawn(description, prompt) -> str: runs a read-only subagent on the small model.
  - ._persist() -> None / ._repair() -> None / ._close_interrupted() -> None: durability helpers.
- READ_ONLY_TOOLS: tool names a subagent may use.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.conversation import ensure_unique_ids, repair_conversation
from ..domain.events import (
    ContextCompacted, ContextPruned, ModelResponded, Notice, RetryScheduled, RunFinished,
    RunStarted, StepStarted, SubagentFinished, SubagentStarted, TextDelta,
)
from ..domain.messages import (
    Turn, Usage, assistant_message, count_chars, text_block, user_blocks, user_text,
)
from ..domain.session import Session
from ..domain.tokens import TokenEstimator
from ..domain.tools import ToolContext, resolve_path
from ..errors import CaralError, ContextLengthError, ProviderError
from .approvers import StaticApprover
from .bus import EventBus
from .context import ContextManager
from .executor import ToolExecutor
from .loopguard import LOOP_HINT, LoopGuard
from .permissions import PermissionManager
from .prompt import build_system_prompt

log = logging.getLogger("caral.agent")

READ_ONLY_TOOLS = ["read_file", "list_dir", "glob", "grep", "web_fetch"]
SUBAGENT_MAX_STEPS = 30
_EXIT = {"ok": 0, "error": 1, "step_limit": 3, "budget": 3, "loop": 3, "interrupted": 130}


@dataclass
class RunResult:
    text: str = ""
    status: str = "ok"
    steps: int = 0
    tool_calls: int = 0
    duration_s: float = 0.0
    usage: Usage = field(default_factory=Usage)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def exit_code(self) -> int:
        return _EXIT.get(self.status, 1)


class Agent:
    def __init__(
        self, config, llm, registry, policy: PermissionManager, approver, bus: EventBus,
        session: Session | None = None, store=None, cwd=None, subagent: bool = False,
        estimator: TokenEstimator | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.registry = registry
        self.policy = policy
        self.bus = bus
        self.store = store
        self.subagent = subagent
        self.estimator = estimator or TokenEstimator()
        cwd = Path(cwd or Path.cwd()).resolve()
        self.session = session or Session.new(cwd, config.model)
        self.ctx = ToolContext(cwd=cwd, config=config)
        self.ctx.todos = self.session.todos
        if not subagent and registry.get("task") is not None:
            self.ctx.spawn_subagent = self._spawn
        self.executor = ToolExecutor(registry, policy, approver, bus, self.ctx, config)
        self.approver = approver
        self.context = ContextManager(llm, config)
        self.loop = LoopGuard()
        self.max_steps = SUBAGENT_MAX_STEPS if subagent else config.max_steps
        self.model = (config.small_model or config.model) if subagent else config.model
        self._system: str | None = None
        self._anchor_tokens = 0
        self._anchor_len = 0
        self._sub_usage = Usage()
        self._lock = threading.Lock()
        self._repair()

    # -- state -----------------------------------------------------------------------------
    @property
    def messages(self) -> list[dict]:
        return self.session.messages

    def system_prompt(self) -> str:
        if self._system is None:
            self._system = build_system_prompt(self.config, self.ctx.cwd, subagent=self.subagent)
        return self._system

    def refresh_system(self) -> None:
        self._system = None

    def reset(self) -> None:
        self.session = Session.new(self.ctx.cwd, self.config.model)
        self.ctx.todos = self.session.todos
        self.ctx.backups.clear()
        self.ctx.read_state.clear()
        self.ctx.invalidate_views()
        self._reset_anchor()
        self._system = None

    def resume(self, session: Session) -> None:
        self.session = session
        self.ctx.todos = session.todos
        self.ctx.read_state.clear()
        self.ctx.invalidate_views()
        self._reset_anchor()
        self._repair()
        session.persisted = len(session.messages)

    def _reset_anchor(self) -> None:
        self._anchor_tokens = 0
        self._anchor_len = 0
        self.loop.reset()

    def _tools_chars(self) -> int:
        return len(json.dumps(self.registry.schemas()))

    def context_usage(self) -> tuple[int, int]:
        msgs = self.messages
        if self._anchor_tokens and self._anchor_len <= len(msgs):
            extra = count_chars(msgs[self._anchor_len:])
            used = self._anchor_tokens + self.estimator.estimate(extra)
        else:
            used = self.estimator.estimate(count_chars(msgs, self.system_prompt()) + self._tools_chars())
        return used, self.config.context_window

    # -- durability ------------------------------------------------------------------------
    def _persist(self) -> None:
        if self.store is None or self.subagent:
            return
        try:
            self.session.todos = self.ctx.todos
            self.session.derive_title()
            self.store.append(self.session, self.messages[self.session.persisted:])
            self.session.persisted = len(self.messages)
            self.store.save_meta(self.session)
        except OSError as e:
            self.bus.emit(Notice(level="warn", message=f"could not save session: {e}"))

    def _repair(self) -> None:
        fixed, notes = repair_conversation(self.session.messages)
        if notes:
            self.session.messages = fixed
            self.session.persisted = min(self.session.persisted, len(fixed))
            if not self.subagent:
                self.bus.emit(Notice(level="info", message="repaired conversation: " + "; ".join(sorted(set(notes)))))

    def _close_interrupted(self) -> None:
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.append(assistant_message([text_block("[Interrupted by the user.]")]))

    # -- context management ----------------------------------------------------------------
    def _maybe_compact(self) -> None:
        used, _ = self.context_usage()
        if self.context.plan(used) == "none":
            return
        pruned_ids = self.context.prune(self.messages)
        if pruned_ids:
            self.bus.emit(ContextPruned(count=len(pruned_ids)))
            self._forget_pruned_views(set(pruned_ids))
            self._reset_anchor()
            used, _ = self.context_usage()
        if self.context.plan(used) == "compact":
            self._compact(None)

    def _forget_pruned_views(self, pruned_ids: set[str]) -> None:
        """Only files whose read output was pruned lose their dedupe entry; recent reads stay deduped."""
        for m in self.messages:
            if m["role"] != "assistant":
                continue
            for b in m["content"]:
                if b.get("type") == "tool_use" and b["id"] in pruned_ids and b["name"] == "read_file":
                    raw = (b.get("input") or {}).get("path")
                    if isinstance(raw, str):
                        self.ctx.forget_view(resolve_path(self.ctx, raw))

    def _compact(self, focus: str | None) -> bool:
        before = len(self.messages)
        new, method = self.context.compact(self.messages, focus)
        if method == "none":
            return False
        self.session.messages = new
        self.ctx.invalidate_views()
        self._reset_anchor()
        if self.store is not None and not self.subagent:
            try:
                self.store.snapshot(self.session)
                self.session.persisted = len(new)
            except OSError:
                pass
        self.bus.emit(ContextCompacted(before=before, after=len(new), method=method,
                                       model=self.config.small_model or self.config.model))
        return True

    def compact(self, focus: str | None = None) -> str:
        before = len(self.messages)
        if not self._compact(focus):
            return "Nothing to compact yet."
        return f"Compacted {before} messages into {len(self.messages)}."

    def undo(self) -> list[str]:
        return self.ctx.undo_last_turn()

    # -- model call ------------------------------------------------------------------------
    def _call_model(self, run_id: str, step: int) -> Turn:
        ctx_retries = 0
        attempts = 0
        while True:
            streamed = [False]

            def on_text(t: str) -> None:
                streamed[0] = True
                self.bus.emit(TextDelta(text=t))

            def on_retry(n: int, delay: float, reason: str) -> None:
                self.bus.emit(RetryScheduled(attempt=n, delay_s=delay, reason=reason))

            sent = list(self.messages)
            t0 = time.monotonic()
            try:
                turn = self.llm.stream(
                    self.system_prompt(), sent, self.registry.schemas(),
                    on_text=on_text, on_retry=on_retry, model=self.model,
                )
            except ContextLengthError:
                if ctx_retries >= 1:
                    raise
                ctx_retries += 1
                self.bus.emit(Notice(level="warn", message="context window exceeded; compacting and retrying"))
                if not self._compact(None):
                    self.session.messages = self.context.emergency_drop(self.messages)
                    self._reset_anchor()
                continue
            except ProviderError as e:
                if e.retryable and not streamed[0] and attempts < 2:
                    attempts += 1
                    delay = min(2.0 ** attempts, 10.0)
                    self.bus.emit(RetryScheduled(attempt=attempts, delay_s=delay, reason=e.kind))
                    time.sleep(delay)
                    continue
                raise
            turn.blocks = ensure_unique_ids(turn.blocks)
            self.bus.emit(ModelResponded(
                run_id=run_id, step=step, model=self.model, stop_reason=turn.stop_reason, usage=turn.usage,
                duration_s=time.monotonic() - t0, tool_calls=len(turn.tool_calls),
            ))
            if turn.usage.context_tokens:
                self._anchor_tokens = turn.usage.context_tokens
                self._anchor_len = len(sent)
                self.estimator.observe(
                    count_chars(sent, self.system_prompt()) + self._tools_chars(), turn.usage.context_tokens
                )
            return turn

    # -- main loop -------------------------------------------------------------------------
    def run(self, text: str) -> RunResult:
        run_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        usage = Usage()
        steps = tool_calls = 0
        status, error, final = "ok", "", ""
        self._sub_usage = Usage()
        self.ctx.begin_turn()
        self.loop.reset()
        self.messages.append(user_text(text))
        self.bus.emit(RunStarted(run_id=run_id, prompt_chars=len(text), model=self.model))
        try:
            for step in range(1, self.max_steps + 1):
                steps = step
                self._maybe_compact()
                self.bus.emit(StepStarted(run_id=run_id, step=step))
                turn = self._call_model(run_id, step)
                usage.add(turn.usage)
                if turn.text:
                    final = turn.text
                if not turn.blocks:
                    self.bus.emit(Notice(level="warn", message="the model returned an empty response"))
                    break
                self.messages.append(assistant_message(turn.blocks))
                calls = turn.tool_calls
                if not calls:
                    if turn.stop_reason == "max_tokens":
                        self.bus.emit(Notice(level="warn", message="response hit max_tokens; asking the model to continue"))
                        self.messages.append(user_text("Your response was cut off. Continue from where you stopped."))
                        continue
                    break
                tool_calls += len(calls)
                results, interrupted = self.executor.execute(calls, step)
                verdict = self.loop.observe(calls, results)
                blocks = results + ([text_block(LOOP_HINT)] if verdict == "warn" else [])
                self.messages.append(user_blocks(blocks))
                self._persist()
                if interrupted:
                    status = "interrupted"
                    break
                if verdict == "stop":
                    status, error = "loop", "Stopped: the same tool calls were repeated with identical results."
                    break
                spent = usage.billable + self._sub_usage.billable
                if self.config.token_budget and spent >= self.config.token_budget:
                    status = "budget"
                    error = f"Stopped: token budget reached ({spent} >= {self.config.token_budget})."
                    break
            else:
                status = "step_limit"
                error = f"Stopped after {self.max_steps} steps. Send another message to continue."
        except KeyboardInterrupt:
            status = "interrupted"
            self._close_interrupted()
        except CaralError as e:
            status, error = "error", e.user_message()
            self._repair()
        except Exception as e:  # noqa: BLE001 - last line of defense: never kill the REPL
            log.exception("unexpected error in agent loop")
            status, error = "error", f"Internal error: {type(e).__name__}: {e}"
            self._repair()
        finally:
            usage.add(self._sub_usage)
            self.session.usage.add(usage)
            self._persist()
        duration = time.monotonic() - started
        self.bus.emit(RunFinished(run_id=run_id, status=status, steps=steps, tool_calls=tool_calls,
                                  duration_s=duration, usage=usage, error=error))
        return RunResult(final, status, steps, tool_calls, duration, usage, error)

    # -- subagents -------------------------------------------------------------------------
    def _spawn(self, description: str, prompt: str) -> str:
        self.bus.emit(SubagentStarted(description=description))
        perms = PermissionManager("ask", self.ctx.root, self.config)
        perms.state.hosts = self.policy.state.hosts
        child = Agent(
            self.config, self.llm, self.registry.subset(READ_ONLY_TOOLS), perms, StaticApprover("no"),
            EventBus(), Session(id="sub", cwd=str(self.ctx.cwd), model=self.config.model), None,
            self.ctx.cwd, subagent=True, estimator=self.estimator,
        )
        result = child.run(prompt)
        with self._lock:
            self._sub_usage.add(result.usage)
        self.bus.emit(SubagentFinished(description=description, duration_s=result.duration_s,
                                       usage=result.usage, status=result.status))
        if result.status == "interrupted":
            raise KeyboardInterrupt
        if result.status == "error":
            raise CaralError(result.error)
        return result.text or result.error
