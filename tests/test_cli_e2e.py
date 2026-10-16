"""
End-to-end tests of the real CLI, composition root, OpenAI provider and tools against a local fake
OpenAI-compatible HTTP server: one-shot modes and exit codes, session continuation, diagnostics,
the interactive REPL loop (scripted input), and slash commands.
"""

import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from caral.cli import main


def chunk(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


def tool_call_script(name, args, cid="call_a"):
    return (200, [chunk({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": cid, "function": {"name": name, "arguments": json.dumps(args)}}]}, "finish_reason": "tool_calls"}]}),
                  chunk({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 10}})])


def text_script(text):
    return (200, [chunk({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]}),
                  chunk({"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 100}}})])


class FakeServer:
    def __init__(self, scripts):
        self.scripts, self.requests = list(scripts), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests.append({"GET": self.path, "auth": self.headers.get("authorization")})
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "openai/gpt-oss-120b"}, {"id": "fake"}, {"id": "zeta"}]}).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append(body)
                status, payload = outer.scripts.pop(0)
                self.send_response(status)
                self.send_header("content-type", "text/event-stream" if status == 200 else "application/json")
                self.end_headers()
                if status == 200:
                    for c in payload:
                        self.wfile.write(c)
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.wfile.write(json.dumps(payload).encode())

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    made = []

    def make(scripts):
        s = FakeServer(scripts)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def base(srv, work, *extra):
    return ["--provider", "ollama", "--base-url", srv.url, "--model", "fake", "--cwd", str(work), *extra]


@pytest.fixture(autouse=True)
def no_stdin(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr(sys, "stdin", Tty())


def test_oneshot_text_runs_the_full_tool_loop(tmp_path, capsys, server):
    work = tmp_path / "w"
    work.mkdir()
    srv = server([tool_call_script("write_file", {"path": "hello.txt", "content": "hi from caral\n"}), text_script("Created hello.txt.")])
    rc = main(["-p", "make a file", "--mode", "yolo", *base(srv, work)])
    out = capsys.readouterr()
    assert rc == 0 and (work / "hello.txt").read_text() == "hi from caral\n" and "Created hello.txt." in out.out
    second = srv.requests[1]["messages"]
    assert second[-1]["role"] == "tool" and "Created" in second[-1]["content"]
    assert srv.requests[0]["model"] == "fake" and srv.requests[0]["messages"][0]["role"] == "system"
    assert "Caral" in srv.requests[0]["messages"][0]["content"]


def test_headless_cannot_approve_so_ask_mode_denies_writes(tmp_path, capsys, server):
    work = tmp_path / "w"
    work.mkdir()
    srv = server([tool_call_script("write_file", {"path": "x.txt", "content": "x"}), text_script("could not")])
    assert main(["-p", "write", *base(srv, work)]) == 0
    assert not (work / "x.txt").exists() and "denied" in srv.requests[1]["messages"][-1]["content"].lower()


def test_json_output_format_and_usage(tmp_path, capsys, server):
    work = tmp_path / "w"
    work.mkdir()
    srv = server([text_script("answer")])
    assert main(["-p", "q", "--output-format", "json", *base(srv, work)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "ok" and data["text"] == "answer" and data["usage"]["cache_read"] == 100 and data["usage"]["input_tokens"] == 20


def test_jsonl_output_streams_every_event(tmp_path, capsys, server):
    work = tmp_path / "w"
    work.mkdir()
    srv = server([text_script("answer")])
    assert main(["-p", "q", "--output-format", "jsonl", *base(srv, work)]) == 0
    types = [json.loads(l)["type"] for l in capsys.readouterr().out.splitlines()]
    assert types[0] == "RunStarted" and "TextDelta" in types and types[-1] == "RunFinished"


def test_exit_codes(tmp_path, capsys, server, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    srv = server([(401, {"error": {"message": "invalid api key"}})])
    assert main(["-p", "q", *base(srv, work)]) == 1
    err = capsys.readouterr().err
    assert "invalid api key" in err and "doctor" in err
    srv2 = server([tool_call_script("list_dir", {})])
    assert main(["-p", "q", "--max-steps", "1", *base(srv2, work)]) == 3
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["-p", "x", "--cwd", str(work)]) == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert main(["-p", "x", "--cwd", str(tmp_path / "missing")]) == 2
    assert main(["-p", "--provider", "ollama", "--cwd", str(work)]) == 2          # no prompt at all


def test_piped_stdin_is_appended_to_the_prompt(tmp_path, capsys, server, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    srv = server([text_script("ok")])
    monkeypatch.setattr(sys, "stdin", io.StringIO("ERROR: stack trace here"))
    assert main(["-p", "explain this", *base(srv, work)]) == 0
    sent = srv.requests[0]["messages"][-1]["content"]
    assert "explain this" in sent and "stack trace here" in sent


def test_session_is_saved_and_continue_resumes_it(tmp_path, capsys, server):
    work = tmp_path / "w"
    work.mkdir()
    srv = server([text_script("first answer"), text_script("second answer")])
    assert main(["-p", "remember the number 7", *base(srv, work)]) == 0
    assert main(["-p", "what number?", "-c", *base(srv, work)]) == 0
    history = srv.requests[1]["messages"]
    assert any("remember the number 7" in str(m.get("content")) for m in history) and any("first answer" in str(m.get("content")) for m in history)
    capsys.readouterr()
    assert main(["--list-sessions"]) == 0
    listing = capsys.readouterr().out
    assert "remember the number 7" in listing
    assert main(["--trace"]) == 0
    t = capsys.readouterr().out
    assert "model calls 2" in t and "cache hit" in t


def test_diagnostics(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--doctor", "--cwd", str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert "✗ configuration" in out and "ANTHROPIC_API_KEY" in out
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert main(["--doctor", "--cwd", str(tmp_path)]) == 0 and "All good" in capsys.readouterr().out
    assert main(["--trace", "nope"]) == 2


def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "caral" in capsys.readouterr().out


# -- interactive REPL ------------------------------------------------------------------------
def run_repl_with(monkeypatch, tmp_path, server, scripts, inputs, extra=()):
    work = tmp_path / "w"
    work.mkdir(exist_ok=True)
    srv = server(scripts)
    feed = iter(inputs)

    def fake_input(prompt=""):
        v = next(feed)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr("builtins.input", fake_input)
    return main([*base(srv, work), *extra]), srv


def test_repl_session_with_slash_commands(tmp_path, capsys, server, monkeypatch):
    rc, srv = run_repl_with(monkeypatch, tmp_path, server, [text_script("Hello there")],
                            ["/help", "/mode plan", "/mode", "/nope", "say hi", "/usage", "/model", "/tools", "/exit"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "caral" in out and "Session" in out and "/compact" in out            # welcome + help
    assert "mode set to plan" in out and "mode: plan (read-only)" in out
    assert "Unknown command /nope" in out
    assert "Hello there" in out and "1 step" in out
    assert "session:" in out and "context:" in out and "model: fake" in out and "read_file" in out


def test_repl_typo_suggestions_and_multiline_and_double_ctrl_c(tmp_path, capsys, server, monkeypatch):
    rc, srv = run_repl_with(monkeypatch, tmp_path, server, [text_script("done")],
                            ["/hepl", "line one\\", "line two", KeyboardInterrupt(), KeyboardInterrupt()])
    out = capsys.readouterr().out
    assert rc == 0 and "Did you mean /help" in out and "Press Ctrl-C again" in out
    assert srv.requests[0]["messages"][-1]["content"] == "line one\nline two"


def test_repl_eof_exits_cleanly(tmp_path, capsys, server, monkeypatch):
    rc, _ = run_repl_with(monkeypatch, tmp_path, server, [], [EOFError()])
    assert rc == 0


def test_repl_initial_prompt_runs_first(tmp_path, capsys, server, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    srv = server([text_script("auto")])
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    assert main([*base(srv, work), "fix", "the", "bug"]) == 0
    assert srv.requests[0]["messages"][-1]["content"] == "fix the bug" and "auto" in capsys.readouterr().out


def test_repl_clear_undo_and_error_display(tmp_path, capsys, server, monkeypatch):
    rc, srv = run_repl_with(monkeypatch, tmp_path, server,
                            [(401, {"error": {"message": "bad key"}})], ["hello", "/undo", "/clear", "/sessions", "/exit"])
    out = capsys.readouterr().out
    assert rc == 0 and "x error" in out and "bad key" in out and "Nothing to undo" in out and "fresh conversation" in out


def test_models_command_lists_what_the_key_can_use(tmp_path, capsys, server):
    """Reported bug: a retired preset model gave an opaque 404 with no way to discover valid ids."""
    srv = server([])
    assert main(["--models", *base(srv, tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "* fake" in out and "  openai/gpt-oss-120b" in out and "zeta" in out
    assert srv.requests[0]["GET"].endswith("/models")
    assert main(["--models", "--provider", "ollama", "--base-url", srv.url, "--model", "retired-model", "--cwd", str(tmp_path)]) == 1
    assert "not offered to this key" in capsys.readouterr().err


def test_doctor_online_verifies_model_exists(tmp_path, capsys, server):
    srv = server([])
    assert main(["--doctor", "--online", *base(srv, tmp_path)]) == 0
    assert "model fake available" in capsys.readouterr().out
    assert main(["--doctor", "--online", "--provider", "ollama", "--base-url", srv.url, "--model", "gone", "--cwd", str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert "NOT offered" in out and "openai/gpt-oss-120b" in out


def test_repl_models_command(tmp_path, capsys, server, monkeypatch):
    rc, _ = run_repl_with(monkeypatch, tmp_path, server, [], ["/models", "/model zeta", "/model", "/exit"])
    out = capsys.readouterr().out
    assert rc == 0 and "* fake" in out and "Switch with /model" in out and "model: zeta" in out
