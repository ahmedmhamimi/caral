"""
Composition root: the only module that knows every concrete class. Builds the object graph
(provider, tools, policy, bus, store, renderer, agent) from a Config. Everything else receives
its dependencies, which keeps each layer independently testable and replaceable.

- App: wired application. .agent, .config, .bus, .store, .terminal, .renderer, .commands, .meter
  (CostMeter), .prices (PriceBook), .price_problems (list[str] found while loading price files).
  - .say(text, level="info") -> None: print a message through the active front-end.
  - .paint(role, text) -> str: theme-colored text in the TUI, unchanged otherwise (for multi-colored command output).
  - .paint_money_role(usd) -> str: role name for a total-spend amount (green / amber / red).
  - .run_prompt(text) -> RunResult: run one prompt. .last_result() -> (name, content) | None.
  - .prompt_text() -> str: interactive prompt string. .close() -> None: flush trace, close HTTP client.
- build_app(config, cwd, output="tui", resume=None, continue_session=False, trace=True, debug=False) -> App.
  output: "tui" (interactive), "plain" (streamed text), "jsonl" (events), "none" (silent, for tests).
- is_local_provider(config) -> bool: True for the built-in local runners (ollama, lmstudio), whose usage is free.
  A custom localhost gateway is not assumed free; price its model at 0 in prices.json if it is.
- LOCAL_PROVIDERS: names treated as free.
- refresh_prices_if_needed(config, prices, fetch=None) -> bool: on OpenRouter, when the model (or small model) has no
  known price, fetch OpenRouter's public list once (4 s timeout), cache it and fill the gap; silent on failure.
- setup_logging(debug) -> None: file logging under the data directory when debug is on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .adapters.providers import create_provider
from .adapters.providers import fetch_openrouter_prices
from .adapters.store import SessionStore, TraceWriter, load_price_book, save_fetched, trace_path
from .adapters.tools import default_registry, load_plugin_tools
from .adapters.ui import JsonlRenderer, PlainRenderer, Terminal, TerminalRenderer, detect_theme
from .commands import CommandRegistry, build_registry
from .config import Config, data_dir
from .domain.events import Notice
from .domain.pricing import PriceBook
from .engine.agent import Agent, RunResult
from .engine.approvers import StaticApprover
from .engine.bus import EventBus
from .engine.meter import CostMeter
from .engine.permissions import PermissionManager


def setup_logging(debug: bool) -> None:
    if not debug:
        logging.getLogger("caral").addHandler(logging.NullHandler())
        return
    data_dir().mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(data_dir() / "caral.log"), level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


LOCAL_PROVIDERS = ("ollama", "lmstudio")


def is_local_provider(config: Config) -> bool:
    """Only the built-in local runners are free by definition. A custom provider on localhost may be a
    gateway to a paid API, so it is not assumed free; give its model a zero price in prices.json instead."""
    return config.provider in LOCAL_PROVIDERS


def refresh_prices_if_needed(config: Config, prices: PriceBook, fetch=None) -> bool:
    host = (urlparse(config.base_url or "").hostname or "").lower()
    if not host.endswith("openrouter.ai"):
        return False
    if all(prices.lookup(m) for m in (config.model, config.small_model) if m):
        return False
    try:
        fresh = (fetch or fetch_openrouter_prices)(timeout=4.0)
    except Exception:  # noqa: BLE001 - pricing is a nicety: never block startup (cost shows n/a and /prices explains)
        return False
    prices.fetched = fresh
    try:
        save_fetched(fresh)
    except OSError:
        pass
    return True


@dataclass
class App:
    config: Config
    agent: Agent
    bus: EventBus
    store: SessionStore
    terminal: Terminal | None
    renderer: object | None
    commands: CommandRegistry
    trace: TraceWriter | None
    llm: object
    meter: CostMeter | None = None
    prices: PriceBook | None = None
    price_problems: list | None = None

    def say(self, text: str, level: str = "info") -> None:
        if isinstance(self.renderer, TerminalRenderer):
            self.renderer.md.flush()
            role = {"error": "error", "warn": "warn"}.get(level, "")
            for line in text.split("\n"):
                self.terminal.writeln(self.renderer.theme.paint(role, line) if role else line)  # type: ignore[union-attr]
        else:
            self.bus.emit(Notice(level=level, message=text))

    def paint(self, role: str, text: str) -> str:
        if isinstance(self.renderer, TerminalRenderer):
            return self.renderer.theme.paint(role, text)
        return text

    def paint_money_role(self, usd: float) -> str:
        return "money_lo" if usd < 1.0 else "money_mid" if usd < 5.0 else "money_hi"

    def run_prompt(self, text: str) -> RunResult:
        return self.agent.run(text)

    def last_result(self):
        return getattr(self.renderer, "last_result", None)

    def prompt_text(self) -> str:
        if isinstance(self.renderer, TerminalRenderer):
            return self.renderer.prompt_text(self.agent.policy.mode)
        return "> "

    def close(self) -> None:
        if self.trace:
            self.trace.close()
        try:
            self.llm.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def build_app(
    config: Config, cwd: Path, output: str = "tui", resume: str | None = None,
    continue_session: bool = False, trace: bool = True, debug: bool = False,
) -> App:
    setup_logging(debug)
    cwd = Path(cwd).resolve()
    llm = create_provider(config)
    registry = default_registry()
    load_plugin_tools(registry)
    policy = PermissionManager(config.permission_mode, cwd, config)
    bus = EventBus()
    prices, price_problems = load_price_book()
    refresh_prices_if_needed(config, prices)
    meter = CostMeter(prices, free=is_local_provider(config))
    bus.subscribe(meter, name="meter")  # first, so every renderer reads up-to-date totals
    store = SessionStore()
    store.prune(config.session_retention_days)

    session = None
    if resume:
        session = store.load(resume)
    elif continue_session:
        session = store.latest(cwd)

    terminal = theme = renderer = None
    approver = StaticApprover("no")
    agent_ref: dict = {}
    if output == "tui":
        terminal = Terminal()
        theme = detect_theme(terminal.stream)
        renderer = TerminalRenderer(terminal, theme, status_fn=lambda: agent_ref["a"].context_usage(), meter=meter)
        approver = renderer
        bus.subscribe(renderer, name="terminal")
    elif output == "plain":
        renderer = PlainRenderer()
        bus.subscribe(renderer, name="plain")
    elif output == "jsonl":
        renderer = JsonlRenderer()
        bus.subscribe(renderer, name="jsonl")

    agent = Agent(config, llm, registry, policy, approver, bus, session=session, store=store, cwd=cwd)
    agent_ref["a"] = agent
    writer = None
    if trace:
        try:
            writer = TraceWriter(lambda: trace_path(agent.session.id))
            bus.subscribe(writer, name="trace")
        except OSError:
            writer = None
    return App(config, agent, bus, store, terminal, renderer, build_registry(), writer, llm, meter, prices, price_problems)
