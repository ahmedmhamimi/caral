"""
Tests for the colorful theme and the live token/cost display: palette selection, distinct hues per
kind of thing, the per-call meter line, spinner segment, run summary, prompt, and graceful behavior
for unpriced models and with color off.

- Palettes: 256 only when advertised or forced; NO_COLOR/pipes stay plain; every role resolves.
- Renderer: meter line has in/cache/out/cost/run/total; unpriced warns once; /meter off hides the line.
- Spinner: compose() fits the width by dropping low-priority parts and survives a failing extra_fn.
"""

import io
import re

import pytest

from caral.adapters.ui import Terminal, TerminalRenderer
from caral.adapters.ui.renderer import MODE_ROLE
from caral.adapters.ui.spinner import Spinner
from caral.adapters.ui.theme import ROLES, Theme, detect_theme, strip_ansi, visible_len
from caral.domain.events import ModelResponded, RunFinished, RunStarted, StepStarted, ToolStarted
from caral.domain.messages import Usage
from caral.domain.pricing import ModelPrice, PriceBook
from caral.engine.meter import CostMeter

BOOK = PriceBook(bundled={"m": ModelPrice(2.0, 10.0, 0.2, 2.5)})
C256 = Theme(color=True, unicode=True, palette="256")
BASIC = Theme(color=True, unicode=True)
PLAIN = Theme(color=False, unicode=False)


class TTY(io.StringIO):
    def isatty(self):
        return True

    @property
    def encoding(self):
        return "utf-8"


def make(theme=C256, free=False, book=BOOK):
    out = TTY()
    meter = CostMeter(book, free=free)
    r = TerminalRenderer(Terminal(out), theme, status_fn=lambda: (50_000, 200_000), meter=meter)

    def emit(ev):          # same order as the app: meter first, then the renderer
        meter(ev)
        r(ev)
    return r, meter, out, emit


def responded(model="m", usage=None, step=1):
    return ModelResponded(run_id="r", step=step, model=model, stop_reason="end_turn",
                          usage=usage or Usage(1000, 200, 5000, 400), duration_s=0.2, tool_calls=0)


# -- theme -----------------------------------------------------------------------------------
class Stream(io.StringIO):
    def __init__(self, tty=True):
        super().__init__()
        self._t = tty

    def isatty(self):
        return self._t

    encoding = "utf-8"


def test_palette_selection():
    assert detect_theme(Stream(), {"TERM": "xterm-256color"}).palette == "256"
    assert detect_theme(Stream(), {"TERM": "xterm", "COLORTERM": "truecolor"}).palette == "256"
    assert detect_theme(Stream(), {"TERM": "xterm"}).palette == "basic"
    assert detect_theme(Stream(), {"TERM": "xterm-256color", "CARAL_COLORS": "basic"}).palette == "basic"
    assert detect_theme(Stream(), {"TERM": "xterm", "CARAL_COLORS": "256"}).palette == "256"
    plain = detect_theme(Stream(), {"TERM": "xterm-256color", "NO_COLOR": "1"})
    assert not plain.color and plain.paint("accent", "x") == "x" and plain.sgr("tok_in") == ""
    assert not detect_theme(Stream(False), {"TERM": "xterm-256color"}).color


def test_every_role_resolves_in_both_palettes_and_plain():
    for role in ROLES:
        assert BASIC.sgr(role) and C256.sgr(role) and PLAIN.sgr(role) == ""
        assert re.fullmatch(r"(\033\[[0-9;]+m)+", C256.sgr(role))


def test_distinct_things_get_distinct_colors():
    groups = [
        ["cat_read", "cat_edit", "cat_exec", "cat_net", "cat_sub"],
        ["tok_in", "tok_cache", "tok_out"],
        ["money_lo", "money_mid", "money_hi", "money_unknown"],
        ["ok", "warn", "error", "accent", "hunk"],
        ["add", "del", "hunk"],
    ]
    for palette in (C256, BASIC):
        for g in groups:
            codes = [palette.sgr(r) for r in g]
            assert len(set(codes)) == len(codes), (palette.palette, g)
    assert Theme.role_for_category("edit") == "cat_edit" and Theme.role_for_category("weird") == "cat_other"
    assert [Theme.money_role(x, 0.1, 1.0) for x in (0.01, 0.5, 3)] == ["money_lo", "money_mid", "money_hi"]


def test_basic_palette_keeps_the_original_codes():
    assert BASIC.paint("error", "x").startswith("\033[31m") and BASIC.paint("accent", "x").startswith("\033[36m")


# -- meter line ------------------------------------------------------------------------------
def test_meter_line_shows_tokens_cost_run_and_total():
    r, meter, out, emit = make()
    emit(RunStarted(run_id="r", prompt_chars=1, model="m"))
    emit(responded())
    text = strip_ansi(out.getvalue())
    # fresh+written = 1400, cache read 5000, out 200 -> (1000*2 + 400*2.5 + 5000*0.2 + 200*10)/1e6 = 0.006
    assert "in 1.4k" in text and "cache 5.0k" in text and "out 200" in text
    assert "$0.0060" in text and "run $0.0060" in text and "total $0.0060" in text
    assert "\033[38;5;" in out.getvalue()      # truly colored, with distinct hues
    emit(RunStarted(run_id="r2", prompt_chars=1, model="m"))
    emit(responded(usage=Usage(0, 100)))
    last = strip_ansi(out.getvalue()).splitlines()[-1]
    assert "run $0.0010" in last and "total $0.0070" in last      # run resets, total keeps climbing


def test_meter_line_without_color_has_no_escape_codes():
    r, meter, out, emit = make(theme=PLAIN)
    emit(responded())
    assert "\033" not in out.getvalue() and "in 1.4k" in out.getvalue() and "$0.0060" in out.getvalue()


def test_money_color_escalates_with_the_amount():
    r, meter, out, emit = make()
    emit(responded(usage=Usage(10_000_000, 0)))     # $20 in one call
    assert C256.sgr("money_hi") in out.getvalue()
    r2, _m, out2, emit2 = make()
    emit2(responded(usage=Usage(100, 0)))
    assert C256.sgr("money_lo") in out2.getvalue() and C256.sgr("money_hi") not in out2.getvalue()


def test_unpriced_model_shows_tokens_and_warns_once():
    r, meter, out, emit = make()
    emit(responded(model="deepseek-chat"))
    emit(responded(model="deepseek-chat", step=2))
    text = strip_ansi(out.getvalue())
    assert text.count("cost n/a") >= 2 and text.count("no price known for 'deepseek-chat'") == 1
    assert "prices.json" in text and "in 1.4k" in text


def test_local_provider_shows_free():
    r, meter, out, emit = make(free=True)
    emit(responded(model="qwen"))
    assert "local $0" in strip_ansi(out.getvalue())
    assert "local $0" in strip_ansi(r.prompt_text("ask"))


def test_step_meter_can_be_switched_off_and_zero_usage_is_silent():
    r, meter, out, emit = make()
    r.step_meter = False
    emit(responded())
    assert out.getvalue() == ""
    r.step_meter = True
    emit(responded(usage=Usage()))
    assert out.getvalue() == ""


# -- summary, prompt, live segment -----------------------------------------------------------
def test_run_summary_includes_cost_and_cache():
    r, meter, out, emit = make()
    emit(RunStarted(run_id="r", prompt_chars=1, model="m"))
    emit(responded())
    out.truncate(0), out.seek(0)
    emit(RunFinished(run_id="r", status="ok", steps=2, tool_calls=3, duration_s=4.2, usage=Usage(1000, 200, 5000, 400)))
    text = strip_ansi(out.getvalue())
    assert "2 steps" in text and "3 tool calls" in text and "$0.0060" in text and "cached" in text and "context 25%" in text
    assert text.startswith("✓")


def test_prompt_shows_mode_context_and_total_spend():
    r, meter, out, emit = make()
    assert "$" not in strip_ansi(r.prompt_text("ask"))          # nothing spent yet: no noise
    emit(responded())
    p = strip_ansi(r.prompt_text("yolo")).replace("\001", "").replace("\002", "")
    assert "yolo" in p and "25%" in p and "$0.0060" in p and p.rstrip().endswith("❯")
    assert set(MODE_ROLE) >= {"plan", "ask", "auto-edit", "yolo"}
    assert len({C256.sgr(role) for role in ("cat_read", "warn", "accent")}) == 3


def test_live_extra_hidden_until_billed_then_shows_run_spend():
    r, meter, out, emit = make()
    assert r.live_extra() == ""
    emit(responded())
    live = strip_ansi(r.live_extra())
    assert "$0.0060" in live and "1.6k tok" in live


def test_tool_lines_are_colored_by_category():
    r, meter, out, emit = make()
    emit(ToolStarted(call_id="1", name="read_file", summary="a.py", category="read"))
    emit(ToolStarted(call_id="2", name="bash", summary="ls", category="exec"))
    emit(ToolStarted(call_id="3", name="edit_file", summary="b.py", category="edit"))
    r.spinner.stop()
    lines = [l for l in out.getvalue().split("\n") if "\u25b8" in l]
    assert len(lines) == 3 and C256.sgr("cat_read") in lines[0] and C256.sgr("cat_exec") in lines[1] and C256.sgr("cat_edit") in lines[2]


def test_renderer_works_without_a_meter():
    out = TTY()
    r = TerminalRenderer(Terminal(out), C256)
    r(responded())
    r(RunFinished(run_id="r", status="ok", steps=1, tool_calls=0, duration_s=1, usage=Usage(10, 5)))
    assert "$" not in strip_ansi(out.getvalue()) and "1 step" in strip_ansi(out.getvalue())
    assert r.live_extra() == "" and "❯" in strip_ansi(r.prompt_text("ask"))


# -- spinner ---------------------------------------------------------------------------------
def test_spinner_compose_fits_width_by_dropping_least_important_parts():
    sp = Spinner(Terminal(TTY()), C256, extra_fn=lambda: C256.paint("money_lo", "$0.0123") + " · 4.2k tok")
    full = sp.compose("⠋", "Thinking", 7.0, 200)
    assert "Thinking" in full and "7.0s" in full and "$0.0123" in full and "Ctrl-C" in full
    mid = sp.compose("⠋", "Thinking", 7.0, 38)
    assert "Ctrl-C" not in mid and "$0.0123" in mid and visible_len(mid) <= 38
    tiny = sp.compose("⠋", "Thinking", 7.0, 12)
    assert tiny.endswith(C256.RESET) and "Thinking" in tiny and "$" not in tiny
    assert "7.0s" not in sp.compose("⠋", "Thinking", 1.0, 200)          # time appears after 2s


def test_spinner_survives_a_failing_extra_fn():
    def boom():
        raise RuntimeError("display bug")
    sp = Spinner(Terminal(TTY()), PLAIN, extra_fn=boom)
    assert sp.compose("|", "Working", 3.0, 80).startswith("| Working")
