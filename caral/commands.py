"""
Slash commands as a registry (add a command = register one function; nothing else changes).

- SlashCommand(name, help, handler, group="session", aliases=()): one command.
- CommandRegistry: .register(cmd), .get(name), .names() -> list[str], .dispatch(line, app) -> bool,
  .suggest(name) -> list[str] (typo help), .help_text() -> str (grouped).
- build_registry() -> CommandRegistry: the built-in commands (help, exit, clear, compact, undo, mode,
  model, models, todos, usage, sessions, resume, tools, config, trace, expand, init).
- INIT_PROMPT: prompt behind /init.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from typing import Callable

from .adapters.store.trace import format_summary, summarize_trace, trace_path
from .config import PERMISSION_MODES
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
    app.say(
        f"session: {human_tokens(u.billable)} billable tokens ({human_tokens(u.input_tokens)} fresh in, "
        f"{human_tokens(u.cache_write)} cache writes, {human_tokens(u.output_tokens)} out) · "
        f"{human_tokens(u.cache_read)} cached reads ({u.cache_hit_ratio:.0%} hit)\n"
        f"context: {human_tokens(used)} / {human_tokens(window)} ({used / window:.0%})"
    )


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
        SlashCommand("usage", "tokens, cache hits and context use", _usage, "inspect"),
        SlashCommand("trace", "summarize this session's trace", _trace, "inspect"),
        SlashCommand("expand", "show the last tool output in full", _expand, "inspect"),
        SlashCommand("tools", "list available tools", _tools, "inspect"),
        SlashCommand("config", "show effective configuration", _config, "inspect"),
    ):
        r.register(c)
    return r
