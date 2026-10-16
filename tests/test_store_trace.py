"""
Tests for session persistence (JSONL append log, snapshots, crash tolerance, retention) and the
structured trace (redaction, session following, size cap, summaries).
"""

import json
import os
import time

from caral.adapters.store import SessionStore, TraceWriter, format_summary, summarize_trace
from caral.domain.conversation import validate_conversation
from caral.domain.events import ModelResponded, Notice, PermissionResolved, RunFinished, TextDelta, ToolFinished
from caral.domain.messages import Usage, assistant_message, text_block, tool_use_block, user_text
from caral.domain.session import Session


def make_session(tmp_path, n=2, title=None):
    s = Session.new(tmp_path, "m")
    s.messages = [user_text("first task")] + [assistant_message([text_block(f"a{i}")]) for i in range(n)]
    return s


def test_append_only_log_is_incremental(tmp_path):
    st = SessionStore(tmp_path / "s")
    s = make_session(tmp_path, 1)
    st.append(s, s.messages)
    size1 = (st.root / f"{s.id}.jsonl").stat().st_size
    st.append(s, [user_text("more")])
    lines = (st.root / f"{s.id}.jsonl").read_text().splitlines()
    assert len(lines) == 3 and (st.root / f"{s.id}.jsonl").stat().st_size > size1
    st.append(s, [])  # no-op
    assert len((st.root / f"{s.id}.jsonl").read_text().splitlines()) == 3


def test_roundtrip_meta_usage_and_title(tmp_path):
    st = SessionStore(tmp_path / "s")
    s = make_session(tmp_path)
    s.usage = Usage(5, 6, 7, 8)
    s.todos = [{"content": "t", "status": "pending"}]
    s.derive_title()
    st.append(s, s.messages)
    st.save_meta(s)
    loaded = st.load(s.id)
    assert loaded.messages == s.messages and loaded.title == "first task"
    assert loaded.usage == Usage(5, 6, 7, 8) and loaded.todos == s.todos and loaded.persisted == len(s.messages)
    assert st.load(s.id[:-2]).id == s.id


def test_snapshot_replaces_history(tmp_path):
    st = SessionStore(tmp_path / "s")
    s = make_session(tmp_path, 5)
    st.append(s, s.messages)
    s.messages = [user_text("[Conversation summary] x")] + s.messages[-2:]
    st.snapshot(s)
    st.append(s, [user_text("after")])
    assert st.load(s.id).messages[0]["content"][0]["text"].startswith("[Conversation summary") and len(st.load(s.id).messages) == 4


def test_truncated_last_line_after_crash_is_ignored(tmp_path):
    st = SessionStore(tmp_path / "s")
    s = make_session(tmp_path)
    st.append(s, s.messages)
    with open(st.root / f"{s.id}.jsonl", "a") as f:
        f.write('{"t": "m", "msg": {"role": "us')  # process killed mid-write
    assert st.load(s.id).messages == s.messages


def test_load_repairs_dangling_tool_use(tmp_path):
    st = SessionStore(tmp_path / "s")
    s = Session.new(tmp_path, "m")
    s.messages = [user_text("go"), assistant_message([tool_use_block("t1", "bash", {})])]
    st.append(s, s.messages)
    assert validate_conversation(st.load(s.id).messages) == []


def test_list_latest_and_errors(tmp_path):
    st = SessionStore(tmp_path / "s")
    a, b = make_session(tmp_path), Session.new(tmp_path / "other", "m")
    b.messages = [user_text("second")]
    for s, t in ((a, 100), (b, 200)):
        st.append(s, s.messages)
        st.save_meta(s)
        m = json.loads((st.root / f"{s.id}.meta.json").read_text())
        m["updated"] = t
        (st.root / f"{s.id}.meta.json").write_text(json.dumps(m))
    assert [m["id"] for m in st.list()] == [b.id, a.id]
    assert st.latest(tmp_path).id == a.id and st.latest().id == b.id and st.latest(tmp_path / "nowhere") is None
    import pytest
    with pytest.raises(FileNotFoundError):
        st.load("zzz")
    c = Session(id=a.id[:-1] + "Z", cwd="", model="m")   # shares a long prefix with a
    st.append(c, [user_text("x")])
    with pytest.raises(FileNotFoundError, match="ambiguous"):
        st.load(a.id[:-3])


def test_retention_prune(tmp_path):
    st = SessionStore(tmp_path / "s")
    old, new = make_session(tmp_path), make_session(tmp_path)
    old.id = "old-session"
    for s in (old, new):
        st.append(s, s.messages)
        st.save_meta(s)
    ancient = time.time() - 100 * 86400
    os.utime(st.root / "old-session.meta.json", (ancient, ancient))
    assert st.prune(60) == 1 and st.prune(0) == 0
    assert not (st.root / "old-session.jsonl").exists() and (st.root / f"{new.id}.jsonl").exists()


def test_trace_redacts_follows_session_and_skips_deltas(tmp_path):
    current = {"p": tmp_path / "a.jsonl"}
    w = TraceWriter(lambda: current["p"])
    w(TextDelta(text="noise"))
    w(ToolFinished(call_id="c", name="bash", is_error=False, duration_s=0.1, chars=40,
                   content="export ANTHROPIC_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456 " + "x" * 900, display="DIFF"))
    current["p"] = tmp_path / "b.jsonl"      # /clear or /resume changed the session
    w(Notice(level="info", message="hello"))
    w.close()
    a = (tmp_path / "a.jsonl").read_text()
    assert "sk-abcdef" not in a and "[REDACTED]" in a and "noise" not in a and "DIFF" not in a
    assert len(json.loads(a)["content"]) <= 520 and "hello" in (tmp_path / "b.jsonl").read_text()


def test_trace_size_cap(tmp_path):
    w = TraceWriter(tmp_path / "t.jsonl", max_bytes=500)
    for i in range(50):
        w(Notice(level="info", message="x" * 50))
    w.close()
    assert (tmp_path / "t.jsonl").stat().st_size < 800


def test_trace_summary(tmp_path):
    p = tmp_path / "t.jsonl"
    w = TraceWriter(p)
    w(ModelResponded(run_id="r", step=1, model="m", stop_reason="tool_use", usage=Usage(100, 10, 900, 0), duration_s=1.5, tool_calls=2))
    w(ToolFinished(call_id="1", name="bash", is_error=True, duration_s=0.5, content="x", chars=1))
    w(ToolFinished(call_id="2", name="read_file", is_error=False, duration_s=0.1, content="x", chars=1))
    w(PermissionResolved(call_id="1", name="bash", decision="deny", source="policy"))
    w(RunFinished(run_id="r", status="error", steps=1, tool_calls=2, duration_s=2, usage=Usage(), error="boom"))
    w.close()
    s = summarize_trace(p)
    text = format_summary(s)
    assert s["runs"] == 1 and s["denied"] == 1 and s["tool_errors"]["bash"] == 1
    assert "90% cache hit" in text and "bash×1 (1 failed)" in text and "boom" in text
