"""
Terminal front-end: subscribes to engine events and renders them; also implements the Approver
port. Design rules (each maps to an established usability principle):
  - Visibility of system status (Nielsen #1): spinner with elapsed time and live token/cost counters,
    per-tool start/finish lines, a meter line after every model call, and a one-line run summary with
    steps, tools, time, tokens, cost, cache-hit and context usage.
  - Progressive disclosure: results are collapsed to a one-line summary; failures and shell output
    expand automatically; /expand shows the last full result.
  - Error prevention and recoverability: edits show a diff before approval; /undo reverts a turn.
  - Recognition over recall: approval prompt lists every key; prompt shows mode, context use and spend.
  - Redundant coding: every status has a symbol and a word, never color alone; each kind of thing
    (tool category, token kind, money level, diff side) also has its own hue so output is scannable.
  - Fitts's/Hick's law: approval is one keypress among at most four choices.

- TerminalRenderer(terminal, theme, status_fn=None, max_preview=6, meter=None): .__call__(event);
  .ask(request) -> Approval (a diff shown in the approval prompt is not printed a second time after the
  tool runs); .prompt_text(mode) -> str; .last_result -> (name, content) | None; .print_help_hint().
  - .step_meter: bool, print the token/cost line after every model call (toggled by /meter).
  - .print_summary(event) formats RunFinished; ._print_diff(diff, limit) caps long diffs.
  - .meter_line(call) -> str: the colored per-call counters line; .live_extra() -> str: spinner segment.
- format_result_line(content, is_error, duration_s) -> str: "0.4s · 120 lines" style summary of a ToolFinished.
- clip_keep(text, width) -> str: truncation that preserves whitespace (for diffs).
- STEP_COST / RUN_COST / TOTAL_COST: (amber, red) dollar thresholds used to color amounts.
- MODE_ROLE: permission mode -> theme role (plan blue, ask amber, auto-edit teal, yolo bold red).
"""

from __future__ import annotations

from ...domain.events import Approval, ApprovalRequest, Event
from ...domain.pricing import format_money
from ...domain.textutil import clip, human_duration, human_tokens
from .keys import read_choice
from .markdown import StreamingMarkdown
from .spinner import Spinner
from .term import Terminal
from .theme import Theme, wrap_nonprinting

_CATEGORY_LABEL = {"edit": "Edit file", "exec": "Run command", "net": "Fetch URL"}
_PREVIEW_TOOLS = {"bash", "grep", "glob", "list_dir"}
MODE_ROLE = {"plan": "cat_read", "ask": "warn", "auto-edit": "accent", "yolo": "bold error"}
STEP_COST = (0.02, 0.10)    # a single model call: amber from $0.02, red from $0.10
RUN_COST = (0.10, 0.50)     # one prompt (all its steps)
TOTAL_COST = (1.00, 5.00)   # since launch


def format_result_line(content: str, is_error: bool, duration_s: float) -> str:
    lines = content.count("\n") + 1 if content.strip() else 0
    unit = "line" if lines == 1 else "lines"
    return f"{human_duration(duration_s)} · {lines} {unit}" if not is_error else f"failed in {human_duration(duration_s)}"


class TerminalRenderer:
    def __init__(self, terminal: Terminal, theme: Theme, status_fn=None, max_preview: int = 6, meter=None) -> None:
        self.t = terminal
        self.theme = theme
        self.status_fn = status_fn
        self.max_preview = max_preview
        self.meter = meter
        self.step_meter = True
        self.spinner = Spinner(terminal, theme, extra_fn=self.live_extra)
        self.md = StreamingMarkdown(terminal.write, theme)
        self.last_result: tuple[str, str] | None = None
        self._running: dict[str, str] = {}
        self._previewed: set[str] = set()
        self._warned_unpriced: set[str] = set()

    # -- helpers ---------------------------------------------------------------------------
    def _p(self, role: str, text: str) -> str:
        return self.theme.paint(role, text)

    def _line(self, text: str = "") -> None:
        self.t.writeln(text)

    def _quiet(self) -> None:
        self.spinner.stop()
        self.md.flush()

    def _sep(self, text: str = " · ") -> str:
        return self._p("sep", text)

    def _money(self, usd: float, thresholds: tuple[float, float], partial: bool = False) -> str:
        m = self.meter
        if m is not None and m.free:
            return self._p("money_lo", "local $0")
        if partial and usd == 0:
            return self._p("money_unknown", "cost n/a")
        role = self.theme.money_role(usd, *thresholds)
        return self._p(role, format_money(usd) + ("+" if partial else ""))

    def _pct(self, frac: float) -> str:
        role = "ok" if frac < 0.5 else "warn" if frac < 0.8 else "error"
        return self._p(role, f"{frac:.0%}")

    def _print_diff(self, diff: str, limit: int = 40, indent: str = "    ") -> None:
        lines = diff.split("\n")
        for ln in lines[:limit]:
            role = "bold path" if ln.startswith(("+++", "---")) else "hunk" if ln.startswith("@@") else \
                "add" if ln.startswith("+") else "del" if ln.startswith("-") else "muted"
            self._line(indent + self._p(role, clip_keep(ln, self.t.width - len(indent) - 1)))
        if len(lines) > limit:
            self._line(indent + self._p("label", f"{self.theme.g('ellipsis')} {len(lines) - limit} more diff lines"))

    # -- live counters ---------------------------------------------------------------------
    def _partial(self) -> bool:
        return bool(self.meter is not None and self.meter.unpriced)

    def live_extra(self) -> str:
        """Spinner segment: running spend and tokens for the current prompt (empty until a call has billed)."""
        m = self.meter
        if m is None or m.run_usage.billable == 0:
            return ""
        return self._money(m.run_cost, RUN_COST, self._partial()) + self._sep() + \
            self._p("tok_in", f"{human_tokens(m.run_usage.billable)} tok")

    def meter_line(self, call) -> str:
        """Per-call counters: in / cache / out tokens, this call's cost, then run and launch totals."""
        u, m = call.usage, self.meter
        sep = self._p("sep", f" {self.theme.g('bar')} ")
        parts = [
            self._p("label", "in ") + self._p("tok_in", human_tokens(u.input_tokens + u.cache_write)),
            self._p("label", "cache ") + self._p("tok_cache", human_tokens(u.cache_read)),
            self._p("label", "out ") + self._p("tok_out", human_tokens(u.output_tokens)),
        ]
        if call.cost is None:
            cost = self._p("money_unknown", "cost n/a")
        else:
            cost = self._money(call.cost, STEP_COST)
        totals = self._p("label", "run ") + self._money(m.run_cost, RUN_COST, self._partial()) + self._sep() + \
            self._p("label", "total ") + self._money(m.total_cost, TOTAL_COST, self._partial())
        return "  " + self._p("accent", self.theme.g("step")) + " " + sep.join(parts + [cost]) + sep + totals

    def _warn_unpriced(self) -> None:
        m = self.meter
        for model in sorted(m.unpriced - self._warned_unpriced):
            self._warned_unpriced.add(model)
            self._line(self._p("money_unknown", f"  {self.theme.g('warn')} no price known for '{model}': tokens are counted, "
                               "cost is not. Add it with /prices or ~/.config/caral/prices.json"))

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
        self._quiet()
        m = self.meter
        if self.step_meter and m is not None and m.last_call is not None and e.usage.context_tokens + e.usage.output_tokens:
            self._line(self.meter_line(m.last_call))
            self._warn_unpriced()

    def on_ToolStarted(self, e) -> None:
        self._quiet()
        self._running[e.call_id] = e.name
        role = self.theme.role_for_category(e.category)
        head = self._p(role, self.theme.g("tool")) + " " + self._p(f"bold {role}", e.name)
        room = self.t.width - len(e.name) - 6
        self._line("  " + head + "  " + clip(e.summary, max(20, room)))
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
            self._line("    " + self._p("bold error", f"{t.g('fail')} error") + self._p("time", f" · {human_duration(e.duration_s)}"))
            for ln in e.content.strip().split("\n")[: self.max_preview + 2]:
                self._line("    " + self._p("error", t.g("bar")) + " " + self._p("del", clip(ln, self.t.width - 8)))
        else:
            self._line("    " + self._p("ok", f"{t.g('ok')} ok") + self._sep() + self._p("time", human_duration(e.duration_s))
                       + self._sep() + self._p("label", format_result_line(e.content, False, e.duration_s).split(" · ", 1)[1]))
            if e.name in _PREVIEW_TOOLS and e.content.strip():
                rows = e.content.strip().split("\n")
                bar = self._p("sep", t.g("bar"))
                for ln in rows[: self.max_preview]:
                    self._line("    " + bar + " " + clip(ln, self.t.width - 8))
                if len(rows) > self.max_preview:
                    self._line("    " + self._p("label", f"{t.g('ellipsis')} {len(rows) - self.max_preview} more lines (/expand to show all)"))
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
            gl_role = {"pending": "label", "in_progress": "warn", "completed": "ok"}[item["status"]]
            txt_role = {"pending": "", "in_progress": "bold warn", "completed": "muted"}[item["status"]]
            text = self._p(txt_role, item["content"]) if txt_role else item["content"]
            self._line("    " + self._p(gl_role, sym) + " " + text)

    def on_ContextPruned(self, e) -> None:
        self._line(self._p("quote", f"  {self.theme.g('info')} trimmed {e.count} old tool outputs to save context"))

    def on_ContextCompacted(self, e) -> None:
        self._quiet()
        how = "summarized" if e.method == "summary" else "dropped"
        self._line(self._p("quote", f"  {self.theme.g('info')} context {how}: {e.before} → {e.after} messages"))

    def on_RetryScheduled(self, e) -> None:
        self._quiet()
        self._line(self._p("warn", f"  {self.theme.g('warn')} {e.reason}; retrying in {e.delay_s:.0f}s (attempt {e.attempt})"))
        self.spinner.start("Waiting to retry")

    def on_Notice(self, e) -> None:
        self._quiet()
        role = {"warn": "warn", "error": "error"}.get(e.level, "quote")
        sym = {"warn": self.theme.g("warn"), "error": self.theme.g("fail")}.get(e.level, self.theme.g("info"))
        self._line(self._p(role, f"  {sym} {e.message}"))

    def on_SubagentStarted(self, e) -> None:
        self._quiet()
        self._line("  " + self._p("cat_sub", f"{self.theme.g('sub')} subagent: {e.description}"))
        self.spinner.start(f"Subagent: {clip(e.description, 30)}")

    def on_SubagentFinished(self, e) -> None:
        self._line("  " + self._p("cat_sub", f"{self.theme.g('sub')} subagent {e.status}") + self._sep() + self._p("time", human_duration(e.duration_s)))

    def on_RunFinished(self, e) -> None:
        self._quiet()
        t, m = self.theme, self.meter
        stats = [self._p("bold", f"{e.steps} step{'s' if e.steps != 1 else ''}")]
        if e.tool_calls:
            stats.append(self._p("cat_exec", f"{e.tool_calls} tool call{'s' if e.tool_calls != 1 else ''}"))
        stats.append(self._p("time", human_duration(e.duration_s)))
        stats.append(self._p("tok_in", f"{human_tokens(e.usage.billable)} tokens"))
        if m is not None:
            stats.append(self._money(m.run_cost, RUN_COST, self._partial()))
        if e.usage.cache_read:
            stats.append(self._p("tok_cache", f"{e.usage.cache_hit_ratio:.0%} cached"))
        if self.status_fn:
            used, window = self.status_fn()
            stats.append(self._p("label", "context ") + self._pct(used / window))
        line = self._sep().join(stats)
        if e.status == "ok":
            self._line(self._p("ok", t.g("ok")) + " " + line)
        elif e.status == "interrupted":
            self._line(self._p("warn", f"{t.g('stop')} interrupted") + self._sep() + line + self._p("label", " · conversation kept; /undo reverts file changes"))
        else:
            label = {"error": "error", "step_limit": "step limit", "budget": "budget reached", "loop": "stopped (loop)"}.get(e.status, e.status)
            self._line(self._p("bold error" if e.status == "error" else "bold warn", f"{t.g('fail')} {label}") + self._sep() + line)
            for ln in (e.error or "").split("\n"):
                self._line("  " + ln)

    # -- approval --------------------------------------------------------------------------
    def ask(self, req: ApprovalRequest) -> Approval:
        self._quiet()
        t = self.theme
        label = _CATEGORY_LABEL.get(req.category, req.tool)
        kind = {"edit": "edit", "exec": "command", "net": "network"}.get(req.category, req.category)
        role = t.role_for_category(req.category)
        self._line()
        self._line(self._p(f"bold {role}", f"{t.g('ask')} {label}") + "  " + self._p("label", f"[{kind}]"))
        self._line("  " + clip(req.summary, self.t.width - 4))
        if req.reason:
            self._line("  " + self._p("label", f"why: {req.reason}"))
        if req.diff:
            self._print_diff(req.diff, limit=40, indent="  ")
            self._previewed.add(req.call_id)
        hint = (self._p("ok", "[y]") + self._p("label", " yes   ") + self._p("error", "[n]") + self._p("label", " no   ")
                + self._p("accent", "[a]") + self._p("label", " always (this session)   ") + self._p("warn", "[f]")
                + self._p("label", " no, tell Caral why"))
        self.t.write("  " + self._p("bold warn", "Allow?") + " " + hint + " ")
        key = read_choice("ynaf")
        self._line()
        if key == "f":
            try:
                fb = input("  Feedback for the model: ").strip()
            except EOFError:
                fb = ""
            self._line("  " + self._p("warn", f"{t.g('arrow')} denied with feedback"))
            return Approval("no", fb)
        outcome = {"y": ("yes", "allowed", "ok"), "a": ("always", "allowed for this session", "ok")}.get(key, ("no", "denied", "warn"))
        self._line("  " + self._p(outcome[2], f"{t.g('arrow')} {outcome[1]}"))
        return Approval(outcome[0])

    # -- prompt ----------------------------------------------------------------------------
    def prompt_text(self, mode: str) -> str:
        parts = [self._p(MODE_ROLE.get(mode, "label"), mode)]
        if self.status_fn:
            used, window = self.status_fn()
            parts.append(self._pct(used / window))
        if self.meter is not None and (self.meter.total_usage.billable or self.meter.free):
            parts.append(self._money(self.meter.total_cost, TOTAL_COST, self._partial()))
        status = self._sep().join(parts)
        return wrap_nonprinting(f"{status} {self._p('user', self.theme.g('prompt'))} ")

    def print_help_hint(self) -> None:
        self._line(self._p("label", "  /help for commands · /usage for spend · Ctrl-C interrupts · end a line with \\ for multi-line"))


def clip_keep(text: str, width: int) -> str:
    """Truncate without collapsing whitespace (diffs must keep their indentation)."""
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"
