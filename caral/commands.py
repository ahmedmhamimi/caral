"""
Slash commands as a registry (add a command = register one function; nothing else changes).

- SlashCommand(name, help, handler, group="session", aliases=()): one command.
- CommandRegistry: .register(cmd), .get(name), .names() -> list[str], .dispatch(line, app) -> bool,
  .suggest(name) -> list[str] (typo help), .help_text() -> str (grouped).
- build_registry() -> CommandRegistry: the built-in commands (help, exit, clear, compact, undo, mode,
  model, models, todos, usage, prices, meter, sessions, resume, tools, config, trace, expand, init).
  /usage shows tokens, estimated spend and a per-model breakdown; /prices shows, lists or refreshes the
  price table; /meter toggles the per-call counter line.
- INIT_PROMPT: prompt behind /init.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from typing import Callable

from .adapters.store.trace import format_summary, summarize_trace, trace_path
from .config import PERMISSION_MODES
from .domain.pricing import count_models, format_money
from .domain.textutil import human_tokens

INIT_PROMPT = (
    "Explore this project and write a concise CARAL.md in the project root for future agent sessions: "
    "how to build, test and lint, the layout of the main directories, coding conventions you can "
    "observe, and any gotchas. Keep it under 80 lines and base it only on what you verified."
)


@dataclass(eq=False)
class SlashCommand:
    name: str
    help: str
    handler: Callable
    group: str = "session"
    aliases: tuple = field(default_factory=tuple)


class CommandRegistry:
    def __init__(self) -> None:
        self._cmds: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        self._cmds[cmd.name] = cmd
        for a in cmd.aliases:
            self._cmds[a] = cmd

    def get(self, name: str) -> SlashCommand | None:
        return self._cmds.get(name)

    def names(self) -> list[str]:
        return sorted(self._cmds)

    def suggest(self, name: str) -> list[str]:
        return difflib.get_close_matches(name, self._cmds, n=2, cutoff=0.5)

    def dispatch(self, line: str, app) -> bool:
        if not line.startswith("/"):
            return False
        name, _, rest = line[1:].partition(" ")
        cmd = self.get(name)
        if cmd is None:
            sugg = self.suggest(name)
            app.say(f"Unknown command /{name}." + (f" Did you mean {' or '.join('/' + s for s in sugg)}?" if sugg else " Try /help."), "error")
            return True
        cmd.handler(app, rest.strip())
        return True

    def help_text(self) -> str:
        groups: dict[str, list[SlashCommand]] = {}
        for c in {id(c): c for c in self._cmds.values()}.values():
            groups.setdefault(c.group, []).append(c)
        lines = []
        for g in ("session", "safety", "inspect"):
            if g in groups:
                lines.append(g.capitalize())
                for c in sorted(groups[g], key=lambda c: c.name):
                    names = "/" + c.name + "".join(f", /{a}" for a in c.aliases)
                    lines.append(f"  {names:<18} {c.help}")
        lines.append("")
        lines.append("Tips: end a line with \\ to continue it · Ctrl-C interrupts a running turn")
        return "\n".join(lines)


# -- handlers ------------------------------------------------------------------------------
def _help(app, _a): app.say(app.commands.help_text())


def _exit(app, _a): raise SystemExit(0)


def _clear(app, _a):
    app.agent.reset()
    app.say("Started a fresh conversation.")


def _compact(app, a):
    try:
        app.say(app.agent.compact(a or None))
    except Exception as e:  # noqa: BLE001
        app.say(f"compact failed: {e}", "error")


def _undo(app, _a):
    restored = app.agent.undo()
    app.say(("Restored:\n  " + "\n  ".join(restored)) if restored else "Nothing to undo.")


def _mode(app, a):
    pol = app.agent.policy
    if not a:
        desc = {"plan": "read-only", "ask": "approve edits and commands", "auto-edit": "project edits automatic", "yolo": "everything automatic"}
        app.say(f"mode: {pol.mode} ({desc[pol.mode]})\noptions: " + ", ".join(PERMISSION_MODES))
        return
    try:
        pol.set_mode(a)
        app.say(f"mode set to {a}")
    except ValueError as e:
        app.say(str(e), "error")


def _model(app, a):
    if a:
        app.config.model = a
        app.agent.model = a
        app.agent.refresh_system()
    app.say(f"model: {app.agent.model}")


def _todos(app, _a):
    todos = app.agent.ctx.todos
    if not todos:
        app.say("No todos.")
        return
    from .domain.events import TodosUpdated
    app.bus.emit(TodosUpdated(todos=list(todos)))


def _usage(app, _a):
    u = app.agent.session.usage
    used, window = app.agent.context_usage()
    p = app.paint
    app.say(
        f"{p('label', 'session:')} {p('tok_in', human_tokens(u.billable))} billable tokens "
        f"({p('tok_in', human_tokens(u.input_tokens))} fresh in, {p('tok_in', human_tokens(u.cache_write))} cache writes, "
        f"{p('tok_out', human_tokens(u.output_tokens))} out) · {p('tok_cache', human_tokens(u.cache_read))} cached reads "
        f"({p('tok_cache', f'{u.cache_hit_ratio:.0%}')} hit)\n"
        f"{p('label', 'context:')} {human_tokens(used)} / {human_tokens(window)} ({used / window:.0%})"
    )
    m = app.meter
    if m is None or not m.total_usage.context_tokens + m.total_usage.output_tokens:
        return
    if m.free:
        app.say(f"{p('label', 'spend')} {p('money_lo', 'local model: no API cost')}")
        return
    partial = bool(m.unpriced)
    app.say(
        f"{p('label', 'spend since launch')} {p(app.paint_money_role(m.total_cost), format_money(m.total_cost) + ('+' if partial else ''))}"
        f" (estimate from list prices) · {p('label', 'last prompt')} {format_money(m.run_cost)}"
    )
    for model, t in sorted(m.by_model().items(), key=lambda kv: -kv[1].cost):
        cost = p("money_unknown", "no price") if not t.priced and t.cost == 0 else format_money(t.cost) + ("+" if not t.priced else "")
        app.say(
            f"  {p('bold', model)}  {t.calls} call{'s' if t.calls != 1 else ''} · in {human_tokens(t.usage.input_tokens + t.usage.cache_write)}"
            f" · cache {human_tokens(t.usage.cache_read)} · out {human_tokens(t.usage.output_tokens)} · {cost}"
        )
    if partial:
        app.say(p("money_unknown", "  + = some models have no known price (see /prices)"))


def _price_row(app, model: str) -> str:
    p = app.paint
    price = app.prices.lookup(model)
    if price is None:
        return f"  {p('bold', model)}  {p('money_unknown', 'no price known')} {p('label', '(add one in ~/.config/caral/prices.json, or /prices update)')}"
    r = price
    cache = f"cache read ${r.cache_read_rate:g} / write ${r.cache_write_rate:g}"
    origin = app.prices.layer_of(model)
    out = (f"  {p('bold', model)}  in {p('tok_in', f'${r.input:g}')} · out {p('tok_out', f'${r.output:g}')} · "
           f"{p('tok_cache', cache)} per 1M tokens {p('label', f'[{origin}]')}")
    if r.note:
        out += "\n    " + p("label", r.note)
    return out


def _prices(app, a):
    from .adapters.providers import fetch_openrouter_prices
    from .adapters.store import fetched_age_days, overrides_path, save_fetched
    from .domain.pricing import BUNDLED_AS_OF
    from .errors import CaralError
    if app.prices is None:
        app.say("No price book loaded.", "error")
        return
    arg = a.strip()
    if arg == "update":
        try:
            fresh = fetch_openrouter_prices()
        except CaralError as e:
            app.say(f"Could not update prices: {e}" + (f"\n  {e.hint}" if getattr(e, "hint", "") else ""), "error")
            return
        path = save_fetched(fresh)
        app.prices.fetched = fresh
        app.say(f"Fetched {count_models(fresh)} model prices from OpenRouter ({path}). Built-in and override prices still win; "
                "these fill gaps. OpenRouter prices can differ from a vendor's direct API.")
        return
    if arg == "list":
        for k in app.prices.known():
            price = app.prices.lookup(k)
            app.say(f"  {k:<34} in ${price.input:g} · out ${price.output:g}  [{app.prices.layer_of(k)}]")
        return
    if arg:
        app.say(_price_row(app, arg))
        return
    models = [app.config.model] + ([app.config.small_model] if app.config.small_model else [])
    for mdl in models:
        app.say(_price_row(app, mdl))
    age = fetched_age_days()
    app.say(app.paint("label", f"  built-in table checked {BUNDLED_AS_OF} · {len(app.prices.known())} models known · "
                      f"overrides: {overrides_path()}" + (f" · fetched cache {age:.0f}d old" if age is not None else "")))
    app.say(app.paint("label", "  /prices list · /prices <model> · /prices update (fetch OpenRouter's public list)"))


def _meter(app, a):
    r = app.renderer
    if not hasattr(r, "step_meter"):
        app.say("The per-call meter is only available in the interactive terminal.")
        return
    if a in ("on", "off"):
        r.step_meter = a == "on"
    elif a:
        app.say("Usage: /meter [on|off]", "error")
        return
    app.say(f"per-call token/cost line: {'on' if r.step_meter else 'off'}  (the prompt, spinner and run summary always show spend)")


def _sessions(app, _a):
    rows = app.store.list(15)
    if not rows:
        app.say("No saved sessions.")
    for m in rows:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.get("updated", 0)))
        app.say(f"{m['id']}  {when}  {m.get('title', '')[:60]}")


def _resume(app, a):
    if not a:
        app.say("usage: /resume <id or prefix>  (see /sessions)", "error")
        return
    try:
        app.agent.resume(app.store.load(a))
        app.say(f"Resumed {app.agent.session.id} ({len(app.agent.messages)} messages)")
    except FileNotFoundError as e:
        app.say(str(e), "error")


def _models(app, _a):
    from .errors import CaralError

    try:
        ids = app.llm.list_models()
    except CaralError as e:
        app.say(e.user_message(), "error")
        return
    for i in ids:
        app.say(("* " if i == app.agent.model else "  ") + i)
    app.say("Switch with /model <id>.")


def _tools(app, _a):
    for t in app.agent.registry.names():
        app.say(f"  {t}")


def _config(app, _a): app.say(app.config.describe())


def _trace(app, _a):
    p = trace_path(app.agent.session.id)
    if not p.exists():
        app.say("No trace yet for this session.")
        return
    app.say(format_summary(summarize_trace(p)) + f"\n{p}")


def _expand(app, _a):
    last = app.last_result()
    if not last:
        app.say("Nothing to expand yet.")
        return
    app.say(f"{last[0]} output:\n{last[1]}")


def _init(app, _a): app.run_prompt(INIT_PROMPT)


def build_registry() -> CommandRegistry:
    r = CommandRegistry()
    for c in (
        SlashCommand("help", "show this help", _help, "session", ("?",)),
        SlashCommand("exit", "leave Caral", _exit, "session", ("quit",)),
        SlashCommand("clear", "start a fresh conversation", _clear),
        SlashCommand("compact", "summarize history to free context [focus]", _compact),
        SlashCommand("resume", "load a saved session <id>", _resume),
        SlashCommand("sessions", "list recent sessions", _sessions),
        SlashCommand("init", "have Caral write CARAL.md for this project", _init),
        SlashCommand("mode", "show/set permission mode (plan|ask|auto-edit|yolo)", _mode, "safety"),
        SlashCommand("undo", "revert file changes from the last turn", _undo, "safety"),
        SlashCommand("model", "show or switch the model", _model, "inspect"),
        SlashCommand("models", "list models your key can use", _models, "inspect"),
        SlashCommand("todos", "show the todo list", _todos, "inspect"),
        SlashCommand("usage", "tokens, spend and context use", _usage, "inspect", ("cost",)),
        SlashCommand("prices", "show, list or refresh model prices [list|<model>|update]", _prices, "inspect"),
        SlashCommand("meter", "per-call token/cost line on|off", _meter, "inspect"),
        SlashCommand("trace", "summarize this session's trace", _trace, "inspect"),
        SlashCommand("expand", "show the last tool output in full", _expand, "inspect"),
        SlashCommand("tools", "list available tools", _tools, "inspect"),
        SlashCommand("config", "show effective configuration", _config, "inspect"),
    ):
        r.register(c)
    return r
