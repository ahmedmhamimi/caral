"""
Command-line entry point: argument parsing, the interactive REPL, one-shot (-p) mode, and the
diagnostic commands (--doctor, --list-sessions, --trace). All wiring is delegated to app.build_app.

Exit codes: 0 ok, 1 error, 2 configuration/usage error, 3 step/budget/loop limit, 130 interrupted.

- main(argv=None) -> int: entry point.
- build_parser() -> argparse.ArgumentParser.
- run_oneshot(args, config, cwd) -> int: -p mode (text, json or jsonl output).
- run_repl(app, initial="") -> int: interactive loop (history, completion, multi-line, double Ctrl-C to exit).
- update_prices() -> int: fetches OpenRouter's public prices into the local cache (--update-prices).
- doctor(args) -> int: environment and configuration diagnostics (--online also checks key and model).
- list_models_cmd(args) -> int: prints the model ids the key can use (--models).
- read_input(app) -> str | None: one logical input (backslash continues a line); None on EOF.
- setup_readline(app) -> None: history file and slash-command tab completion.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .adapters.store import SessionStore, format_summary, summarize_trace, trace_path
from .app import App, build_app
from .domain.pricing import count_models
from .config import PERMISSION_MODES, PRESETS, config_dir, data_dir, load_config
from .errors import CaralError, ConfigError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="caral", description="Caral — a provider-agnostic terminal coding agent.",
        epilog="Examples:\n  caral\n  caral \"fix the failing test\"\n  caral -p \"summarize this repo\" --mode plan\n"
               "  caral --provider openai --model gpt-4.1\n  caral --provider ollama --model qwen2.5-coder:14b\n  caral --doctor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", nargs="*", help="initial prompt (starts the REPL unless -p is given)")
    p.add_argument("-p", "--print", dest="print_mode", action="store_true", help="run once, print the answer, exit")
    p.add_argument("--output-format", choices=("text", "json", "jsonl"), default="text", help="format for -p (default text)")
    p.add_argument("--provider", help=f"provider preset ({', '.join(PRESETS)}) or a custom name from your config")
    p.add_argument("--model", help="model name")
    p.add_argument("--small-model", help="cheaper model for compaction and subagents")
    p.add_argument("--base-url", help="override the API base URL (any OpenAI-compatible server)")
    p.add_argument("--mode", choices=PERMISSION_MODES, help="permission mode")
    p.add_argument("--max-steps", type=int, help="max model calls per message")
    p.add_argument("--budget", type=int, help="max billable tokens per message (0 = unlimited)")
    p.add_argument("--cwd", help="working directory")
    p.add_argument("-c", "--continue", dest="cont", action="store_true", help="continue the latest session in this directory")
    p.add_argument("--resume", metavar="ID", help="resume a session by id or prefix")
    p.add_argument("--no-trace", action="store_true", help="do not write the structured trace")
    p.add_argument("--debug", action="store_true", help="write debug logs to the data directory")
    p.add_argument("--ascii", action="store_true", help="ASCII-only symbols")
    p.add_argument("--doctor", action="store_true", help="diagnose installation and configuration, then exit")
    p.add_argument("--online", action="store_true", help="with --doctor: also verify the key and model against the provider")
    p.add_argument("--models", action="store_true", help="list the models your API key can use, then exit")
    p.add_argument("--update-prices", action="store_true", help="fetch current model prices from OpenRouter's public list, then exit")
    p.add_argument("--list-sessions", action="store_true", help="list saved sessions, then exit")
    p.add_argument("--trace", nargs="?", const="latest", metavar="ID", help="summarize a session trace, then exit")
    p.add_argument("--version", action="version", version=f"caral {__version__}")
    return p


def _overrides(args) -> dict:
    return {
        "provider": args.provider, "model": args.model, "small_model": args.small_model,
        "base_url": args.base_url, "permission_mode": args.mode, "max_steps": args.max_steps,
        "token_budget": args.budget,
    }


def update_prices() -> int:
    from .adapters.providers import fetch_openrouter_prices
    from .adapters.store import save_fetched
    try:
        fresh = fetch_openrouter_prices()
    except CaralError as e:
        print(f"caral: could not update prices: {e}", file=sys.stderr)
        return 1
    print(f"Saved {count_models(fresh)} model prices to {save_fetched(fresh)}")
    print("Built-in and override prices still take precedence; fetched prices only fill gaps.")
    return 0


def doctor(args) -> int:
    ok = True

    def line(good: bool | None, label: str, detail: str = "") -> None:
        nonlocal ok
        mark = {True: "✓", False: "✗", None: "·"}[good]
        if good is False:
            ok = False
        print(f" {mark} {label}" + (f"  {detail}" if detail else ""))

    print(f"Caral {__version__} diagnostics")
    line(sys.version_info >= (3, 10), "Python >= 3.10", sys.version.split()[0])
    try:
        import httpx

        line(True, "httpx installed", httpx.__version__)
    except ImportError:
        line(False, "httpx installed", "pip install httpx")
    line(None, "ripgrep (optional, faster grep)", shutil.which("rg") or "not found; using built-in search")
    line(None, "git (optional)", shutil.which("git") or "not found")
    try:
        data_dir().mkdir(parents=True, exist_ok=True)
        probe = data_dir() / ".probe"
        probe.write_text("x")
        probe.unlink()
        line(True, "data directory writable", str(data_dir()))
    except OSError as e:
        line(False, "data directory writable", str(e))
    line(None, "config file", str(config_dir() / "config.json") + ("" if (config_dir() / "config.json").exists() else " (none; defaults)"))
    try:
        cfg = load_config(Path(args.cwd or Path.cwd()).resolve(), _overrides(args))
        how = f"auto-detected from {cfg.api_key_env}" if cfg.provider_auto else "selected explicitly"
        line(True, f"provider {cfg.provider} ({cfg.kind})", f"model {cfg.model} · {how}")
        if len(cfg.keys_found) > 1:
            line(None, "other keys found", ", ".join(n for n in cfg.keys_found if n != cfg.provider) + "  (use --provider to switch)")
        line(True, "API key", "found" if cfg.api_key else "not required")
        line(None, "endpoint", cfg.base_url)
        if args.online:
            from .adapters.providers import create_provider
            from .errors import ProviderError

            llm = create_provider(cfg)
            try:
                ids = llm.list_models()
                line(True, "provider reachable, key accepted", f"{len(ids)} models")
                if cfg.model in ids:
                    line(True, f"model {cfg.model} available")
                else:
                    sample = ", ".join(ids[:6]) + (" ..." if len(ids) > 6 else "")
                    line(False, f"model {cfg.model} NOT offered to this key", f"try: {sample}  (caral --models)")
            except ProviderError as e:
                line(False, "provider check", e.user_message().replace("\n", " "))
            finally:
                llm.close()
    except ConfigError as e:
        line(False, "configuration", e.user_message().replace("\n", " "))
    from .adapters.ui import detect_theme

    th = detect_theme()
    line(None, "terminal", f"color={'on' if th.color else 'off'} unicode={'on' if th.unicode else 'off'} tty={sys.stdout.isatty()}")
    print("\nAll good." if ok else "\nFix the items marked ✗ and run again.")
    return 0 if ok else 2


def list_models_cmd(args) -> int:
    from .adapters.providers import create_provider

    cwd = Path(args.cwd or Path.cwd()).resolve()
    try:
        cfg = load_config(cwd, _overrides(args))
        llm = create_provider(cfg)
    except CaralError as e:
        print(f"caral: {e.user_message()}", file=sys.stderr)
        return e.exit_code
    try:
        ids = llm.list_models()
    except CaralError as e:
        print(f"caral: {e.user_message()}", file=sys.stderr)
        return e.exit_code
    finally:
        llm.close()
    for i in ids:
        print(("* " if i == cfg.model else "  ") + i)
    if cfg.model not in ids:
        print(f"caral: current model '{cfg.model}' is not offered to this key; pass --model <id> from the list above.", file=sys.stderr)
        return 1
    return 0


def _show_trace(key: str) -> int:
    store = SessionStore()
    try:
        sid = store.latest().id if key == "latest" and store.latest() else store.load(key).id
    except (FileNotFoundError, AttributeError):
        print("caral: no such session", file=sys.stderr)
        return 2
    p = trace_path(sid)
    if not p.exists():
        print(f"caral: no trace for {sid}", file=sys.stderr)
        return 2
    print(f"trace {sid}\n" + format_summary(summarize_trace(p)) + f"\n{p}")
    return 0


def run_oneshot(args, config, cwd: Path) -> int:
    prompt = " ".join(args.prompt).strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        prompt = f"{prompt}\n\n{piped}" if prompt and piped else (prompt or piped)
    if not prompt:
        print("caral: -p needs a prompt (argument or stdin)", file=sys.stderr)
        return 2
    output = {"text": "plain", "json": "none", "jsonl": "jsonl"}[args.output_format]
    app = build_app(config, cwd, output=output, resume=args.resume, continue_session=args.cont,
                    trace=not args.no_trace, debug=args.debug)
    try:
        result = app.run_prompt(prompt)
    finally:
        app.close()
    if args.output_format == "json":
        print(json.dumps({
            "status": result.status, "text": result.text, "error": result.error, "steps": result.steps,
            "tool_calls": result.tool_calls, "duration_s": round(result.duration_s, 3), "usage": result.usage.to_dict(),
            "cost_usd": round(app.meter.total_cost, 6), "unpriced_models": sorted(app.meter.unpriced),
        }, ensure_ascii=False))
    return result.exit_code


def setup_readline(app: App) -> None:
    try:
        import atexit
        import readline
    except ImportError:
        return
    hist = config_dir() / "history"
    try:
        hist.parent.mkdir(parents=True, exist_ok=True)
        readline.read_history_file(hist)
    except OSError:
        pass
    readline.set_history_length(1000)
    atexit.register(lambda: _save_history(readline, hist))
    names = ["/" + n for n in app.commands.names()]

    def complete(text: str, state: int):
        if not text.startswith("/"):
            return None
        matches = [n for n in names if n.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def _save_history(readline, path: Path) -> None:
    try:
        readline.write_history_file(path)
    except OSError:
        pass


def read_input(app: App) -> str | None:
    lines: list[str] = []
    while True:
        prompt = app.prompt_text() if not lines else "… "
        try:
            line = input(prompt)
        except EOFError:
            return None if not lines else "\n".join(lines)
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def run_repl(app: App, initial: str = "") -> int:
    from .adapters.ui import TerminalRenderer

    setup_readline(app)
    cfg = app.config
    if isinstance(app.renderer, TerminalRenderer):
        r = app.renderer
        from .adapters.ui.renderer import MODE_ROLE
        sep = r._sep()
        price = app.prices.lookup(app.agent.model) if app.prices is not None else None
        if app.meter is not None and app.meter.free:
            price_txt = r._p("money_lo", "local: free")
        elif price is None:
            price_txt = r._p("money_unknown", "price unknown (/prices)")
        else:
            price_txt = r._p("tok_in", f"${price.input:g}") + r._p("label", " in / ") + r._p("tok_out", f"${price.output:g}") + r._p("label", " out per 1M")
        r._line(r._p("bold accent", "caral") + " " + r._p("label", __version__) + sep + r._p("cat_read", cfg.provider) + r._p("sep", "/")
                + r._p("bold path", app.agent.model) + sep + r._p(MODE_ROLE.get(app.agent.policy.mode, "label"), f"{app.agent.policy.mode} mode")
                + sep + price_txt)
        r.print_help_hint()
        for problem in app.price_problems or []:
            app.say(f"prices: {problem}", "warn")
    pending, last_interrupt = initial, 0.0
    while True:
        if pending:
            line, pending = pending, ""
        else:
            try:
                line = read_input(app)
            except KeyboardInterrupt:
                print()
                if time.monotonic() - last_interrupt < 2.0:
                    return 0
                last_interrupt = time.monotonic()
                app.say("Press Ctrl-C again (or type /exit) to quit.", "warn")
                continue
            if line is None:
                print()
                return 0
        if not line:
            continue
        try:
            if app.commands.dispatch(line, app):
                continue
        except SystemExit:
            return 0
        except Exception as e:  # noqa: BLE001 - a broken command must never end the session
            app.say(f"command failed: {type(e).__name__}: {e}", "error")
            continue
        try:
            app.run_prompt(line)
        except KeyboardInterrupt:
            app.say("interrupted", "warn")
        print()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ascii:
        os.environ["CARAL_ASCII"] = "1"
    if args.doctor:
        return doctor(args)
    if args.update_prices:
        return update_prices()
    if args.models:
        return list_models_cmd(args)
    if args.list_sessions:
        for m in SessionStore().list(50):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.get("updated", 0)))
            print(f"{m['id']}  {when}  {m.get('title', '')[:70]}")
        return 0
    if args.trace:
        return _show_trace(args.trace)
    cwd = Path(args.cwd or Path.cwd()).resolve()
    if not cwd.is_dir():
        print(f"caral: --cwd {cwd} is not a directory", file=sys.stderr)
        return 2
    try:
        config = load_config(cwd, _overrides(args))
        if args.print_mode:
            return run_oneshot(args, config, cwd)
        app = build_app(config, cwd, output="tui", resume=args.resume, continue_session=args.cont,
                        trace=not args.no_trace, debug=args.debug)
    except CaralError as e:
        print(f"caral: {e.user_message()}", file=sys.stderr)
        return e.exit_code
    except FileNotFoundError as e:
        print(f"caral: {e}", file=sys.stderr)
        return 2
    try:
        return run_repl(app, initial=" ".join(args.prompt).strip())
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
