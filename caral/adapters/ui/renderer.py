"""
Terminal front-end: subscribes to engine events and renders them; also implements the Approver
port. Design rules (each maps to an established usability principle):
  - Visibility of system status (Nielsen #1): spinner with elapsed time, per-tool start/finish lines,
    and a one-line run summary with steps, tools, time, tokens, cache-hit and context usage.
  - Progressive disclosure: results are collapsed to a one-line summary; failures and shell output
    expand automatically; /expand shows the last full result.
  - Error prevention and recoverability: edits show a diff before approval; /undo reverts a turn.
  - Recognition over recall: approval prompt lists every key; prompt shows mode and context use.
  - Redundant coding: every status has a symbol and a word, never color alone.
  - Fitts's/Hick's law: approval is one keypress among at most four choices.

- TerminalRenderer(terminal, theme, status_fn=None): .__call__(event); .ask(request) -> Approval
  (a diff shown in the approval prompt is not printed a second time after the tool runs);
  .prompt_text() -> str; .last_result -> (name, content) | None; .print_help_hint().
  - .print_summary(event) formats RunFinished; ._print_diff(diff, limit) caps long diffs.
- format_result_line(event) -> str: "ok 0.4s · 120 lines" style summary of a ToolFinished.
"""

from __future__ import annotations

from ...domain.events import Approval, ApprovalRequest, Event
from ...domain.textutil import clip, human_duration, human_tokens
from .keys import read_choice
from .markdown import StreamingMarkdown
from .spinner import Spinner
from .term import Terminal
from .theme import Theme, wrap_nonprinting

_CATEGORY_LABEL = {"edit": "Edit file", "exec": "Run command", "net": "Fetch URL"}
_PREVIEW_TOOLS = {"bash", "grep", "glob", "list_dir"}


def format_result_line(content: str, is_error: bool, duration_s: float) -> str:
    lines = content.count("\n") + 1 if content.strip() else 0
    unit = "line" if lines == 1 else "lines"
    return f"{human_duration(duration_s)} · {lines} {unit}" if not is_error else f"failed in {human_duration(duration_s)}"


class TerminalRenderer:
    def __init__(self, terminal: Terminal, theme: Theme, status_fn=None, max_preview: int = 6) -> None:
        self.t = terminal
        self.theme = theme
        self.status_fn = status_fn
        self.max_preview = max_preview
        self.spinner = Spinner(terminal, theme)
        self.md = StreamingMarkdown(terminal.write, theme)
        self.last_result: tuple[str, str] | None = None
        self._running: dict[str, str] = {}
        self._previewed: set[str] = set()

    # -- helpers ---------------------------------------------------------------------------
    def _p(self, role: str, text: str) -> str:
        return self.theme.paint(role, text)

    def _line(self, text: str = "") -> None:
        self.t.writeln(text)

    def _quiet(self) -> None:
        self.spinner.stop()
        self.md.flush()

    def _print_diff(self, diff: str, limit: int = 40, indent: str = "    ") -> None:
        lines = diff.split("\n")
        for ln in lines[:limit]:
            role = "bold" if ln.startswith(("+++", "---")) else "hunk" if ln.startswith("@@") else \
                "add" if ln.startswith("+") else "del" if ln.startswith("-") else "muted"
            self._line(indent + self._p(role, clip_keep(ln, self.t.width - len(indent) - 1)))
        if len(lines) > limit:
            self._line(indent + self._p("muted", f"{self.theme.g('ellipsis')} {len(lines) - limit} more diff lines"))

    # -- event dispatch --------------------------------------------------------------------
    def __call__(self, event: Event) -> None:
        handler = getattr(self, "on_" + type(event).__name__, None)
        if handler:
            handler(event)

    def on_StepStarted(self, e) -> None:
        self.md.flush()
        self.spinner.start("Thinking")

    def on_TextDelta(self, e) -> None:
        self.spinner.stop()
        self.md.feed(e.text)

    def on_ModelResponded(self, e) -> None:
        self.md.flush()

    def on_ToolStarted(self, e) -> None:
        self._quiet()
        self._running[e.call_id] = e.name
        head = self._p("accent", self.theme.g("tool")) + " " + self._p("bold", e.name)
        room = self.t.width - len(e.name) - 6
        self._line("  " + head + "  " + self._p("muted", clip(e.summary, max(20, room))))
        self.spinner.start(f"Running {e.name}" if len(self._running) == 1 else f"Running {len(self._running)} tools")

    def on_ToolFinished(self, e) -> None:
        self._running.pop(e.call_id, None)
        if not self._running:
            self.spinner.stop()
        else:
            self.spinner.update(f"Running {len(self._running)} tools")
        self.last_result = (e.name, e.content)
        t = self.theme
        if e.is_error:
            self._line("    " + self._p("error", f"{t.g('fail')} error") + self._p("muted", f" · {human_duration(e.duration_s)}"))
            for ln in e.content.strip().split("\n")[: self.max_preview + 2]:
                self._line("    " + self._p("error", t.g("bar")) + " " + clip(ln, self.t.width - 8))
        else:
            self._line("    " + self._p("ok", f"{t.g('ok')} ok") + self._p("muted", " · " + format_result_line(e.content, False, e.duration_s)))
            if e.name in _PREVIEW_TOOLS and e.content.strip():
                rows = e.content.strip().split("\n")
                for ln in rows[: self.max_preview]:
                    self._line("    " + self._p("muted", t.g("bar") + " " + clip(ln, self.t.width - 8)))
                if len(rows) > self.max_preview:
                    self._line("    " + self._p("muted", f"{t.g('ellipsis')} {len(rows) - self.max_preview} more lines (/expand to show all)"))
        if e.display and e.name != "todo_write" and e.call_id not in self._previewed:
            self._print_diff(e.display, limit=30)   # skipped when the user already saw this diff in the prompt
        self._previewed.discard(e.call_id)
        if self._running:
            self.spinner.start("Running")

    def on_TodosUpdated(self, e) -> None:
        self._quiet()
        t = self.theme
        for item in e.todos:
            sym = {"pending": t.g("todo_pending"), "in_progress": t.g("todo_active"), "completed": t.g("todo_done")}[item["status"]]
            role = {"pending": "", "in_progress": "bold", "completed": "muted"}[item["status"]]
            self._line("    " + (self._p(role, f"{sym} {item['content']}") if role else f"{sym} {item['content']}"))

    def on_ContextPruned(self, e) -> None:
        self._line(self._p("muted", f"  {self.theme.g('info')} trimmed {e.count} old tool outputs to save context"))

    def on_ContextCompacted(self, e) -> None:
        self._quiet()
        how = "summarized" if e.method == "summary" else "dropped"
        self._line(self._p("muted", f"  {self.theme.g('info')} context {how}: {e.before} → {e.after} messages"))

    def on_RetryScheduled(self, e) -> None:
        self._quiet()
        self._line(self._p("warn", f"  {self.theme.g('warn')} {e.reason}; retrying in {e.delay_s:.0f}s (attempt {e.attempt})"))
        self.spinner.start("Waiting to retry")

    def on_Notice(self, e) -> None:
        self._quiet()
        role = {"warn": "warn", "error": "error"}.get(e.level, "muted")
        sym = {"warn": self.theme.g("warn"), "error": self.theme.g("fail")}.get(e.level, self.theme.g("info"))
        self._line(self._p(role, f"  {sym} {e.message}"))

    def on_SubagentStarted(self, e) -> None:
        self._quiet()
        self._line("  " + self._p("muted", f"{self.theme.g('sub')} subagent: {e.description}"))
        self.spinner.start(f"Subagent: {clip(e.description, 30)}")

    def on_SubagentFinished(self, e) -> None:
        self._line("  " + self._p("muted", f"{self.theme.g('sub')} subagent {e.status} in {human_duration(e.duration_s)}"))

    def on_RunFinished(self, e) -> None:
        self._quiet()
        t = self.theme
        stats = [f"{e.steps} step{'s' if e.steps != 1 else ''}"]
        if e.tool_calls:
            stats.append(f"{e.tool_calls} tool call{'s' if e.tool_calls != 1 else ''}")
        stats.append(human_duration(e.duration_s))
        stats.append(f"{human_tokens(e.usage.billable)} tokens")
        if e.usage.cache_read:
            stats.append(f"{e.usage.cache_hit_ratio:.0%} cached")
        if self.status_fn:
            used, window = self.status_fn()
            stats.append(f"context {used / window:.0%}")
        line = " · ".join(stats)
        if e.status == "ok":
            self._line(self._p("muted", f"{t.g('ok')} {line}"))
        elif e.status == "interrupted":
            self._line(self._p("warn", f"{t.g('stop')} interrupted") + self._p("muted", f" · {line} · conversation kept; /undo reverts file changes"))
        else:
            label = {"error": "error", "step_limit": "step limit", "budget": "budget reached", "loop": "stopped (loop)"}.get(e.status, e.status)
            self._line(self._p("error" if e.status == "error" else "warn", f"{t.g('fail')} {label}") + self._p("muted", f" · {line}"))
            for ln in (e.error or "").split("\n"):
                self._line("  " + ln)

    # -- approval --------------------------------------------------------------------------
    def ask(self, req: ApprovalRequest) -> Approval:
        self._quiet()
        t = self.theme
        label = _CATEGORY_LABEL.get(req.category, req.tool)
        kind = {"edit": "edit", "exec": "command", "net": "network"}.get(req.category, req.category)
        tag = f"[{kind}]"
        self._line()
        self._line(self._p("warn bold", f"{t.g('ask')} {label}") + "  " + self._p("muted", tag))
        self._line("  " + clip(req.summary, self.t.width - 4))
        if req.reason:
            self._line("  " + self._p("muted", f"why: {req.reason}"))
        if req.diff:
            self._print_diff(req.diff, limit=40, indent="  ")
            self._previewed.add(req.call_id)
        hint = "[y] yes   [n] no   [a] always (this session)   [f] no, tell Caral why"
        self.t.write("  " + self._p("warn", "Allow?") + " " + self._p("muted", hint) + " ")
        key = read_choice("ynaf")
        self._line()
        if key == "f":
            try:
                fb = input("  Feedback for the model: ").strip()
            except EOFError:
                fb = ""
            self._line("  " + self._p("muted", f"{t.g('arrow')} denied with feedback"))
            return Approval("no", fb)
        outcome = {"y": ("yes", "allowed", "ok"), "a": ("always", "allowed for this session", "ok")}.get(key, ("no", "denied", "warn"))
        self._line("  " + self._p(outcome[2], f"{t.g('arrow')} {outcome[1]}"))
        return Approval(outcome[0])

    # -- prompt ----------------------------------------------------------------------------
    def prompt_text(self, mode: str) -> str:
        pct = ""
        if self.status_fn:
            used, window = self.status_fn()
            pct = f" · {used / window:.0%}"
        status = self._p("muted", f"{mode}{pct}")
        text = f"{status} {self._p('accent bold', self.theme.g('prompt'))} "
        return wrap_nonprinting(text)

    def print_help_hint(self) -> None:
        self._line(self._p("muted", "  /help for commands · Ctrl-C interrupts · end a line with \\ for multi-line"))


def clip_keep(text: str, width: int) -> str:
    """Truncate without collapsing whitespace (diffs must keep their indentation)."""
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"
