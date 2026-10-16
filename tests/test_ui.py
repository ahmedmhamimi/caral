"""
Tests for the terminal front-end: theme detection and accessibility fallbacks, the streaming
Markdown renderer (including adversarial chunking), the event renderer, approval prompt, and
headless renderers.
"""

import io
import json
import time

import pytest

from caral.adapters.ui import JsonlRenderer, PlainRenderer, Terminal, TerminalRenderer
from caral.adapters.ui.markdown import StreamingMarkdown
from caral.adapters.ui.spinner import Spinner
from caral.adapters.ui.theme import Theme, detect_theme, strip_ansi, visible_len, wrap_nonprinting
from caral.domain.events import (
    Approval, ApprovalRequest, ContextCompacted, Notice, RetryScheduled, RunFinished, StepStarted, TextDelta,
    TodosUpdated, ToolFinished, ToolStarted,
)
from caral.domain.messages import Usage

COLOR = Theme(color=True, unicode=True)
PLAIN = Theme(color=False, unicode=False)


class TTY(io.StringIO):
    def __init__(self, tty=False, encoding="utf-8"):
        super().__init__()
        self._tty, self._enc = tty, encoding

    @property
    def encoding(self):
        return self._enc

    def isatty(self):
        return self._tty


def render(text, theme=COLOR, chunk=None):
    out = io.StringIO()
    md = StreamingMarkdown(out.write, theme)
    if chunk:
        for i in range(0, len(text), chunk):
            md.feed(text[i:i + chunk])
    else:
        md.feed(text)
    md.flush()
    return out.getvalue()


# -- theme -----------------------------------------------------------------------------------
def test_theme_detection_rules():
    assert not detect_theme(TTY(True), {"NO_COLOR": "1"}).color
    assert not detect_theme(TTY(True), {"TERM": "dumb"}).color
    assert not detect_theme(TTY(False), {}).color                         # pipes get no escape codes
    assert detect_theme(TTY(True), {"TERM": "xterm"}).color
    assert detect_theme(TTY(False), {"CLICOLOR_FORCE": "1"}).color
    assert not detect_theme(TTY(True, "ascii"), {}).unicode and not detect_theme(TTY(True), {"CARAL_ASCII": "1"}).unicode
    assert detect_theme(TTY(True, "UTF-8"), {}).unicode


def test_theme_never_encodes_meaning_by_color_alone():
    assert PLAIN.paint("error", "x") == "x" and PLAIN.sgr("error") == ""
    assert COLOR.paint("error", "x").startswith("\033[31m")
    for name in ("ok", "fail", "warn", "tool", "bar", "prompt", "spinner"):
        assert PLAIN.g(name).isascii() and COLOR.g(name)
    assert visible_len(COLOR.paint("bold accent", "hello")) == 5 and strip_ansi("\033[1mx\033[0m") == "x"
    assert wrap_nonprinting("\033[1mx\033[0m") == "\001\033[1m\002x\001\033[0m\002"


# -- markdown --------------------------------------------------------------------------------
def test_markdown_inline_styles_and_reset_at_newline():
    out = render("a **bold** and `code` b\n")
    assert strip_ansi(out) == "a bold and code b\n"
    assert "\033[1m" in out and "\033[36m" in out
    unbalanced = render("**never closed\nnext line\n")
    assert unbalanced.split("\n")[0].endswith("\033[0m") and "\033[1m" not in unbalanced.split("\n")[1]


def test_markdown_blocks():
    out = strip_ansi(render("# Title\n- one\n* two\n> quote\n```python\nx = 1 * 2\n```\nafter\n"))
    lines = out.split("\n")
    assert lines[0] == "Title" and lines[1] == "• one" and lines[2] == "• two" and lines[3] == "│ quote"
    assert "┌ python" in lines[4] and lines[5] == "  │ x = 1 * 2" and lines[6].strip() == "└" and lines[7] == "after"


def test_markdown_code_blocks_are_not_inline_styled():
    out = strip_ansi(render("```\na **b** `c`\n```\n"))
    assert "a **b** `c`" in out


def test_markdown_is_chunking_invariant():
    text = "# H\nSome **bold** and `co*de` text\n- item **x**\n```py\nprint('**')\n```\n1. one\n2*3 and 2**3\n"
    whole = render(text)
    for size in (1, 2, 3, 5, 7):
        assert render(text, chunk=size) == whole, f"chunk size {size} changed output"


def test_markdown_edge_cases():
    assert strip_ansi(render("**")) == "\n"                                    # a lone ** is only a bold toggle
    assert strip_ansi(render("-5 degrees\n")) == "-5 degrees\n"                # '-' not followed by space
    assert strip_ansi(render("2 * 3 = 6\n")) == "2 * 3 = 6\n"
    assert strip_ansi(render("#hashtag\n")) == "#hashtag\n"
    assert strip_ansi(render("`unclosed code\nnext\n")) == "unclosed code\nnext\n"
    assert strip_ansi(render("```")) == "  ┌\n"                                # fence opened, never closed
    assert render("no newline at end").endswith("\n")
    assert strip_ansi(render("a\r\nb\r\n")) == "a\nb\n"
    assert strip_ansi(render("")) == ""


def test_markdown_passthrough_without_color():
    text = "# T\n**b** `c`\n- x\n"
    assert render(text, PLAIN) == text and render("abc", PLAIN) == "abc\n"


# -- spinner / terminal ----------------------------------------------------------------------
def test_spinner_is_silent_when_not_a_tty_and_quick_when_tty():
    quiet = TTY(False)
    sp = Spinner(Terminal(quiet), COLOR)
    sp.start("Thinking")
    time.sleep(0.15)
    sp.stop()
    assert quiet.getvalue() == "" and not sp.running
    tty = TTY(True)
    sp = Spinner(Terminal(tty), PLAIN)
    sp.start("Thinking")
    sp.stop()   # stop before the 300ms grace: nothing should have flickered on screen
    assert "Thinking" not in tty.getvalue() and not sp.running
    sp.start("Working")
    time.sleep(0.55)
    sp.update("Still working")
    time.sleep(0.25)
    sp.stop()
    assert "Working" in tty.getvalue() and tty.getvalue().endswith("\r\033[2K")


def test_terminal_survives_closed_streams():
    s = TTY()
    t = Terminal(s)
    s.close()
    t.write("x")  # must not raise


# -- renderer --------------------------------------------------------------------------------
def make_renderer(theme=PLAIN, status=lambda: (50_000, 200_000)):
    s = TTY(False)
    return TerminalRenderer(Terminal(s), theme, status_fn=status), s


def test_renderer_tool_lifecycle_collapses_output_and_expands_errors():
    r, s = make_renderer()
    r(StepStarted(run_id="r", step=1))
    r(TextDelta(text="Looking.\n"))
    r(ToolStarted(call_id="1", name="read_file", summary="src/app.py", category="read"))
    r(ToolFinished(call_id="1", name="read_file", is_error=False, duration_s=0.04, content="a\nb\nc", chars=5))
    r(ToolStarted(call_id="2", name="bash", summary="pytest", category="exec"))
    r(ToolFinished(call_id="2", name="bash", is_error=True, duration_s=2.0, content="FAILED test_x\nAssertionError", chars=30))
    out = s.getvalue()
    assert "Looking." in out and "> read_file  src/app.py" in out and "+ ok · 40ms · 3 lines" in out
    assert "| a" not in out                                           # read_file output stays collapsed
    assert "x error · 2.0s" in out and "FAILED test_x" in out         # failures expand automatically
    assert r.last_result == ("bash", "FAILED test_x\nAssertionError")


def test_renderer_shell_output_previews_with_overflow_hint():
    r, s = make_renderer()
    r(ToolFinished(call_id="1", name="bash", is_error=False, duration_s=0.1, content="\n".join(f"l{i}" for i in range(20)), chars=60))
    out = s.getvalue()
    assert "l5" in out and "l6" not in out and "14 more lines (/expand to show all)" in out


def test_renderer_diff_display_and_cap():
    r, s = make_renderer()
    diff = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n" + "\n".join(f"+x{i}" for i in range(60))
    r(ToolFinished(call_id="1", name="edit_file", is_error=False, duration_s=0.1, content="Edited", chars=6, display=diff))
    out = s.getvalue()
    assert "-old" in out and "+new" in out and "more diff lines" in out


def test_renderer_run_summary_variants():
    r, s = make_renderer()
    r(RunFinished(run_id="r", status="ok", steps=3, tool_calls=5, duration_s=4.2, usage=Usage(1000, 500, 8000, 0)))
    ok = s.getvalue()
    assert "+ 3 steps · 5 tool calls · 4.2s · 1.5k tokens · 89% cached · context 25%" in ok
    r(RunFinished(run_id="r", status="interrupted", steps=1, tool_calls=0, duration_s=1, usage=Usage()))
    assert "# interrupted" in s.getvalue() and "/undo" in s.getvalue()
    r(RunFinished(run_id="r", status="error", steps=1, tool_calls=0, duration_s=1, usage=Usage(), error="HTTP 401: bad key\n  -> Check your API key"))
    assert "x error" in s.getvalue() and "Check your API key" in s.getvalue()
    r(RunFinished(run_id="r", status="budget", steps=2, tool_calls=2, duration_s=1, usage=Usage(), error="Stopped: token budget reached"))
    assert "budget reached" in s.getvalue()


def test_renderer_misc_events():
    r, s = make_renderer()
    r(TodosUpdated(todos=[{"content": "a", "status": "completed"}, {"content": "b", "status": "in_progress"}, {"content": "c", "status": "pending"}]))
    r(Notice(level="warn", message="careful"))
    r(RetryScheduled(attempt=2, delay_s=3.2, reason="HTTP 429"))
    r(ContextCompacted(before=40, after=7, method="summary"))
    out = s.getvalue()
    assert "[x] a" in out and "[~] b" in out and "[ ] c" in out and "! careful" in out
    assert "HTTP 429; retrying in 3s (attempt 2)" in out and "40 → 7" in out


def test_renderer_unknown_events_are_ignored():
    r, s = make_renderer()
    r(SubStub())
    assert s.getvalue() == ""


class SubStub:
    pass


def test_prompt_text_shows_mode_and_context_and_is_readline_safe():
    r, _ = make_renderer(COLOR)
    p = r.prompt_text("ask")
    assert "\001" in p and strip_ansi(p.replace("\001", "").replace("\002", "")) == "ask · 25% ❯ "
    plain, _ = make_renderer(PLAIN)
    assert plain.prompt_text("yolo") == "yolo · 25% > "


@pytest.mark.parametrize("key,choice,word", [("y", "yes", "allowed"), ("a", "always", "session"), ("n", "no", "denied"), ("esc", "no", "denied")])
def test_approval_prompt(monkeypatch, key, choice, word):
    r, s = make_renderer()
    monkeypatch.setattr("caral.adapters.ui.renderer.read_choice", lambda valid, default=None: key)
    req = ApprovalRequest("c1", "edit_file", "src/app.py", "edit", "modifies files", "--- a/f\n+++ b/f\n-old\n+new")
    a = r.ask(req)
    out = s.getvalue()
    assert a == Approval(choice) and word in out
    assert "? Edit file" in out and "[edit]" in out and "why: modifies files" in out and "+new" in out
    assert "[y] yes" in out and "[a] always" in out and "[f] no, tell Caral why" in out


def test_approval_feedback_path(monkeypatch):
    r, s = make_renderer()
    monkeypatch.setattr("caral.adapters.ui.renderer.read_choice", lambda valid, default=None: "f")
    monkeypatch.setattr("builtins.input", lambda prompt="": "use pathlib")
    a = r.ask(ApprovalRequest("c", "bash", "rm x", "exec", "runs a shell command"))
    assert a == Approval("no", "use pathlib") and "[command]" in s.getvalue()


# -- headless --------------------------------------------------------------------------------
def test_plain_renderer_streams_text_and_sends_diagnostics_to_stderr():
    out, err = io.StringIO(), io.StringIO()
    r = PlainRenderer(out, err)
    r(TextDelta(text="hello"))
    from caral.domain.events import ModelResponded
    r(ModelResponded(run_id="r", step=1, model="m", stop_reason="end_turn", usage=Usage(), duration_s=0, tool_calls=0))
    r(Notice(level="warn", message="w"))
    r(RunFinished(run_id="r", status="error", steps=1, tool_calls=0, duration_s=0, usage=Usage(), error="bad"))
    assert out.getvalue() == "hello\n" and "warn: w" in err.getvalue() and "error: bad" in err.getvalue()


def test_jsonl_renderer_emits_one_valid_json_per_event():
    out = io.StringIO()
    r = JsonlRenderer(out)
    r(TextDelta(text="é"))
    r(RunFinished(run_id="r", status="ok", steps=1, tool_calls=0, duration_s=0.1, usage=Usage(1, 2)))
    rows = [json.loads(l) for l in out.getvalue().splitlines()]
    assert rows[0]["type"] == "TextDelta" and rows[1]["usage"]["output_tokens"] == 2


def test_diff_shown_in_approval_is_not_printed_twice(monkeypatch):
    r, s = make_renderer()
    monkeypatch.setattr("caral.adapters.ui.renderer.read_choice", lambda valid, default=None: "y")
    diff = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new"
    r.ask(ApprovalRequest("c1", "edit_file", "f", "edit", "modifies files", diff))
    r(ToolFinished(call_id="c1", name="edit_file", is_error=False, duration_s=0.1, content="Edited", chars=6, display=diff))
    assert s.getvalue().count("+new") == 1
    r(ToolFinished(call_id="c2", name="edit_file", is_error=False, duration_s=0.1, content="Edited", chars=6, display=diff))
    assert s.getvalue().count("+new") == 2          # an auto-approved edit has no earlier preview, so it is shown once


def test_approval_outcome_is_on_its_own_line(monkeypatch):
    r, s = make_renderer()
    monkeypatch.setattr("caral.adapters.ui.renderer.read_choice", lambda valid, default=None: "y")
    r.ask(ApprovalRequest("c", "bash", "make", "exec", "runs a shell command"))
    lines = s.getvalue().split("\n")
    assert any(l.startswith("  Allow?") and "->" not in l for l in lines) and any(l.strip() == "-> allowed" for l in lines)
