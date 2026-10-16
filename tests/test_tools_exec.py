"""
Tests for bash, glob, grep, web_fetch, todo and task tools, including the audit regressions:
bounded memory for runaway output, background processes not hanging a call, process-group kill.
"""

import os
import time

import httpx

from caral.adapters.tools import default_registry
from caral.adapters.tools.misc import blocked_host, html_to_text
from caral.adapters.tools.search import glob_to_regex
from caral.adapters.tools.shell import BoundedCapture
from caral.config import Config
from caral.domain.tools import ToolContext

REG = default_registry()


def ctx_for(tmp_path, **cfg):
    return ToolContext(cwd=tmp_path, config=Config(bash_timeout=5, **cfg))


def run(tool, ctx, **args):
    return REG.get(tool).run(args, ctx)


def test_bash_basic_exit_codes_and_cwd(tmp_path):
    c = ctx_for(tmp_path)
    assert run("bash", c, command="echo hello").content.strip() == "hello"
    r = run("bash", c, command="echo oops; exit 3")
    assert r.is_error and "exit code 3" in r.content and "oops" in r.content
    (tmp_path / "sub").mkdir()
    run("bash", c, command="cd sub")
    assert c.cwd == tmp_path / "sub" and run("bash", c, command="pwd").content.strip().endswith("sub")
    assert run("bash", c, command="true").content == "(no output)"


def test_bash_timeout_kills_process_group(tmp_path):
    c = ctx_for(tmp_path)
    t = time.time()
    r = run("bash", c, command="sleep 30 & sleep 30", timeout=1)
    assert r.is_error and "timed out" in r.content and time.time() - t < 8


def test_bash_runaway_output_is_memory_bounded(tmp_path):
    """Audit bug: 200MB of output pushed RSS to ~800MB."""
    import resource
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r = run("bash", ctx_for(tmp_path), command="yes xxxxxxxxxxxxxxxxxxxx | head -c 100000000", timeout=30)
    grown_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    assert len(r.content) < 130_000 and "discarded" in r.content and grown_mb < 150


def test_bash_background_process_does_not_hang_or_kill_the_call(tmp_path):
    """Audit bug: `server &` kept the pipe open, so the call timed out and killed everything."""
    t = time.time()
    r = run("bash", ctx_for(tmp_path), command="(sleep 5 >/dev/null 2>&1 &) ; echo started", timeout=20)
    assert not r.is_error and "started" in r.content and time.time() - t < 6


def test_bash_background_with_inherited_pipe_returns_promptly(tmp_path):
    t = time.time()
    r = run("bash", ctx_for(tmp_path), command="sleep 4 & echo started", timeout=20)
    assert "started" in r.content and not r.is_error and time.time() - t < 4


def test_bash_has_no_stdin(tmp_path):
    r = run("bash", ctx_for(tmp_path), command="cat; echo done", timeout=5)
    assert "done" in r.content


def test_bash_handles_invalid_utf8_output(tmp_path):
    r = run("bash", ctx_for(tmp_path), command=r"printf 'a\xffb\n'")
    assert "a" in r.content and "b" in r.content


def test_bounded_capture_unit():
    cap = BoundedCapture(head=10, tail=10)
    for _ in range(100):
        cap.feed("0123456789")
    t = cap.text()
    assert t.startswith("0123456789") and t.endswith("0123456789") and cap.dropped == 980 and "discarded" in t
    small = BoundedCapture()
    small.feed("abc")
    assert small.text() == "abc" and small.dropped == 0


def test_glob_semantics(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "src" / "deep").mkdir(parents=True)
    (tmp_path / "src" / "deep" / "b.py").write_text("")
    (tmp_path / "src" / "c.ts").write_text("")
    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    (tmp_path / "node_modules" / "x" / "ignored.py").write_text("")
    c = ctx_for(tmp_path)
    names = lambda pat: set(run("glob", c, pattern=pat).content.splitlines())
    assert names("*.py") == {"a.py", os.path.join("src", "deep", "b.py")}       # no "/" => any depth
    assert names("**/*.py") == names("*.py")
    assert names("src/**/*.py") == {os.path.join("src", "deep", "b.py")}
    assert names("src/*.{ts,py}") == {os.path.join("src", "c.ts")}
    assert "No files matched" in run("glob", c, pattern="*.rs").content
    assert glob_to_regex("a?c").match("abc") and not glob_to_regex("a?c").match("a/c")


def test_glob_newest_first(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("")
    b.write_text("")
    os.utime(a, (1000, 1000))
    os.utime(b, (2000, 2000))
    assert run("glob", ctx_for(tmp_path), pattern="*.py").content.splitlines() == ["b.py", "a.py"]


def test_grep_modes_and_limits(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    (tmp_path / "b.txt").write_text("foo bar\nFOO\n")
    (tmp_path / "bin.dat").write_bytes(b"foo\x00\x01")
    c = ctx_for(tmp_path)
    content = run("grep", c, pattern="foo").content
    assert "a.py" in content and "b.txt" in content and "bin.dat" not in content
    assert run("grep", c, pattern="foo", output_mode="files", glob="*.py").content.strip() == "a.py"
    assert "FOO" in run("grep", c, pattern="foo", case_insensitive=True).content
    assert "b.txt:2" in run("grep", c, pattern="foo", output_mode="count", case_insensitive=True).content
    assert "No matches" in run("grep", c, pattern="nothere").content
    assert run("grep", c, pattern="(").is_error and run("grep", c, pattern="x", path="nope").is_error
    ctxd = run("grep", c, pattern="pass", context=1, path="a.py").content
    assert "def foo" in ctxd


def test_grep_head_limit_and_huge_lines(tmp_path):
    (tmp_path / "m.txt").write_text("\n".join(f"match {i}" for i in range(300)))
    out = run("grep", ctx_for(tmp_path), pattern="match", head_limit=10).content
    assert out.count("match") >= 10 and "more lines not shown" in out
    (tmp_path / "min.js").write_text("a" * 50_000 + " match")
    assert "No matches" in run("grep", ctx_for(tmp_path), pattern="a+b", path="min.js").content  # skipped, no hang


def test_web_fetch_guards_and_cache(tmp_path, monkeypatch):
    c = ctx_for(tmp_path)
    assert run("web_fetch", c, url="file:///etc/passwd").is_error
    assert "blocked" in run("web_fetch", c, url="http://169.254.169.254/latest/meta-data").content
    assert blocked_host("169.254.169.254") and not blocked_host("127.0.0.1") and not blocked_host("example.com")

    hits = {"n": 0}

    def handler(req):
        hits["n"] += 1
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html><script>x()</script><p>Hello <b>world</b></p></html>")

    real = httpx.Client
    monkeypatch.setattr("caral.adapters.tools.misc.httpx.Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    r1 = run("web_fetch", c, url="https://example.com/a")
    r2 = run("web_fetch", c, url="https://example.com/a")
    assert "Hello world" in r1.content and "x()" not in r1.content and r1.content == r2.content and hits["n"] == 1


def test_web_fetch_rejects_binary_and_errors(tmp_path, monkeypatch):
    real = httpx.Client
    state = {"resp": httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG")}
    monkeypatch.setattr("caral.adapters.tools.misc.httpx.Client", lambda **kw: real(transport=httpx.MockTransport(lambda r: state["resp"]), **kw))
    c = ctx_for(tmp_path)
    assert "Unsupported content type" in run("web_fetch", c, url="https://x.com/i.png").content
    state["resp"] = httpx.Response(404, text="nope")
    assert "HTTP 404" in run("web_fetch", c, url="https://x.com/missing").content


def test_html_to_text():
    assert html_to_text("<div>a</div><style>b{}</style><p>c</p>").split() == ["a", "c"]


def test_todo_tool(tmp_path):
    c = ctx_for(tmp_path)
    r = run("todo_write", c, todos=[{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}])
    assert "more than one" in r.content and len(c.todos) == 2
    assert run("todo_write", c, todos=[{"content": "a", "status": "bogus"}]).is_error
    assert run("todo_write", c, todos=[{"status": "pending"}]).is_error


def test_task_tool_without_spawner(tmp_path):
    assert run("task", ctx_for(tmp_path), description="d", prompt="p").is_error
