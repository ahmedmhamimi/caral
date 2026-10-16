"""
Tests for filesystem tools, covering the data-safety fixes from the audit: non-UTF-8 files are
never rewritten, writes are atomic and keep permissions, BOM/CRLF survive, stale detection uses
content hashes, and unchanged re-reads are deduplicated.
"""

import os
import stat
import time

from caral.adapters.tools import default_registry
from caral.adapters.tools._fileio import NotTextError, read_text_file, write_text_file
from caral.config import Config
from caral.domain.tools import ToolContext, coerce_args, validate_args

REG = default_registry()


def ctx_for(tmp_path):
    return ToolContext(cwd=tmp_path, config=Config())


def run(tool, ctx, **args):
    return REG.get(tool).run(args, ctx)


def prep(tmp_path, text, name="f.py"):
    f = tmp_path / name
    f.write_text(text)
    c = ctx_for(tmp_path)
    run("read_file", c, path=name)
    return f, c


def test_read_numbers_and_paging(tmp_path):
    (tmp_path / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    r = run("read_file", ctx_for(tmp_path), path="a.txt", offset=3, limit=2)
    assert "     3\tline3" in r.content and "line5" not in r.content and "offset=5" in r.content


def test_read_errors(tmp_path):
    (tmp_path / "config.py").write_text("x")
    c = ctx_for(tmp_path)
    r = run("read_file", c, path="confg.py")
    assert r.is_error and "config.py" in r.content
    assert run("read_file", c, path=".").is_error
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")
    assert run("read_file", c, path="b.bin").is_error
    (tmp_path / "e.txt").write_text("")
    assert run("read_file", c, path="e.txt").content == "(file is empty)"
    (tmp_path / "s.txt").write_text("a\n")
    assert run("read_file", c, path="s.txt", offset=9).is_error


def test_read_caps_huge_files_and_lines(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join("x" * 200 for _ in range(1000)))
    r = run("read_file", ctx_for(tmp_path), path="big.txt")
    assert len(r.content) < 45_000 and "use offset=" in r.content
    (tmp_path / "wide.txt").write_text("y" * 5000)
    assert "line truncated" in run("read_file", ctx_for(tmp_path), path="wide.txt").content


def test_read_dedupe_saves_tokens_and_invalidates_on_change(tmp_path):
    f, c = prep(tmp_path, "a\nb\n")
    again = run("read_file", c, path="f.py")
    assert "unchanged" in again.content and "     1\ta" not in again.content
    f.write_text("a\nCHANGED\n")
    assert "CHANGED" in run("read_file", c, path="f.py").content
    c.invalidate_views()  # what the engine does after pruning/compaction
    assert "CHANGED" in run("read_file", c, path="f.py").content


def test_write_requires_read_for_existing(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("old")
    c = ctx_for(tmp_path)
    assert run("write_file", c, path="x.txt", content="new").is_error
    run("read_file", c, path="x.txt")
    assert not run("write_file", c, path="x.txt", content="new").is_error and f.read_text() == "new"
    assert not run("write_file", c, path="sub/dir/n.txt", content="hi\n").is_error
    assert run("write_file", c, path=".", content="x").is_error


def test_edit_exact_and_diff(tmp_path):
    f, c = prep(tmp_path, "a = 1\nb = 2\n")
    r = run("edit_file", c, path="f.py", old_string="b = 2", new_string="b = 3")
    assert not r.is_error and f.read_text() == "a = 1\nb = 3\n" and "+b = 3" in r.display


def test_edit_uniqueness_and_replace_all(tmp_path):
    f, c = prep(tmp_path, "x\nx\n")
    r = run("edit_file", c, path="f.py", old_string="x", new_string="y")
    assert r.is_error and "2 times" in r.content and f.read_text() == "x\nx\n"
    assert not run("edit_file", c, path="f.py", old_string="x", new_string="y", replace_all=True).is_error
    assert f.read_text() == "y\ny\n"


def test_edit_flexible_whitespace_reindents(tmp_path):
    f, c = prep(tmp_path, "def f():\n    return 1\n")
    r = run("edit_file", c, path="f.py", old_string="def f():\n  return 1", new_string="def f():\n  return 2")
    assert not r.is_error and f.read_text() == "def f():\n    return 2\n"


def test_edit_error_messages_guide_the_model(tmp_path):
    f, c = prep(tmp_path, "alpha = 1\n")
    r = run("edit_file", c, path="f.py", old_string="alpah = 1", new_string="z")
    assert r.is_error and "alpha = 1" in r.content
    assert "empty" in run("edit_file", c, path="f.py", old_string="", new_string="z").content
    assert "identical" in run("edit_file", c, path="f.py", old_string="alpha", new_string="alpha").content


def test_stale_detection_uses_content_not_mtime(tmp_path):
    f, c = prep(tmp_path, "a\n")
    future = time.time() + 50
    os.utime(f, (future, future))  # touched but unchanged (formatter, checkout): must NOT be stale
    assert not run("edit_file", c, path="f.py", old_string="a", new_string="b").is_error
    f.write_text("external change\n")
    os.utime(f, (future + 9, future + 9))
    r = run("edit_file", c, path="f.py", old_string="external", new_string="x")
    assert r.is_error and "changed on disk" in r.content


def test_multi_edit_is_atomic(tmp_path):
    f, c = prep(tmp_path, "a\nb\n")
    r = run("multi_edit", c, path="f.py", edits=[{"old_string": "a", "new_string": "A"}, {"old_string": "zzz", "new_string": "Z"}])
    assert r.is_error and "Edit #2" in r.content and f.read_text() == "a\nb\n"
    r = run("multi_edit", c, path="f.py", edits=[{"old_string": "a", "new_string": "A"}, {"old_string": "b", "new_string": "B"}])
    assert not r.is_error and f.read_text() == "A\nB\n"
    assert run("multi_edit", c, path="f.py", edits=[]).is_error


def test_crlf_and_bom_preserved(tmp_path):
    f = tmp_path / "w.txt"
    f.write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\n")
    c = ctx_for(tmp_path)
    run("read_file", c, path="w.txt")
    assert not run("edit_file", c, path="w.txt", old_string="two", new_string="2").is_error
    assert f.read_bytes() == b"\xef\xbb\xbfone\r\n2\r\n"


def test_non_utf8_file_is_never_rewritten(tmp_path):
    """Audit bug: editing a Latin-1 file silently replaced é with U+FFFD."""
    f = tmp_path / "latin.txt"
    original = "café au lait\n".encode("latin-1")
    f.write_bytes(original)
    c = ctx_for(tmp_path)
    r = run("read_file", c, path="latin.txt")
    assert "not valid UTF-8" in r.content
    e = run("edit_file", c, path="latin.txt", old_string="au lait", new_string="noir")
    assert e.is_error and "UTF-8" in e.content and f.read_bytes() == original
    w = run("write_file", c, path="latin.txt", content="x")
    assert w.is_error and f.read_bytes() == original


def test_write_preserves_permissions_and_symlinks(tmp_path):
    s = tmp_path / "run.sh"
    s.write_text("#!/bin/sh\n")
    s.chmod(0o755)
    c = ctx_for(tmp_path)
    run("read_file", c, path="run.sh")
    run("write_file", c, path="run.sh", content="#!/bin/sh\necho hi\n")
    assert stat.S_IMODE(s.stat().st_mode) == 0o755
    real = tmp_path / "real.txt"
    real.write_text("one\n")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    run("read_file", c, path="link.txt")
    run("write_file", c, path="link.txt", content="two\n")
    assert link.is_symlink() and real.read_text() == "two\n"


def test_atomic_write_leaves_no_temp_files_and_survives_failure(tmp_path, monkeypatch):
    p = tmp_path / "a.txt"
    write_text_file(p, "hello\n")
    assert [x.name for x in tmp_path.iterdir()] == ["a.txt"]
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    try:
        write_text_file(p, "new\n")
    except OSError:
        pass
    assert p.read_text() == "hello\n" and [x.name for x in tmp_path.iterdir()] == ["a.txt"]


def test_fileio_helpers(tmp_path):
    (tmp_path / "b").write_bytes(b"\x00\x01")
    try:
        read_text_file(tmp_path / "b")
        assert False
    except NotTextError:
        pass


def test_list_dir(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("")
    out = run("list_dir", ctx_for(tmp_path), depth=2).content
    assert "src/" in out and "m.py" in out and "node_modules" not in out
    assert run("list_dir", ctx_for(tmp_path), path="nope").is_error


def test_undo_restores_edits_and_removes_created_files(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("orig")
    c = ctx_for(tmp_path)
    c.begin_turn()
    run("read_file", c, path="u.txt")
    run("edit_file", c, path="u.txt", old_string="orig", new_string="new")
    run("write_file", c, path="created.txt", content="x")
    assert c.undo_last_turn() and f.read_text() == "orig" and not (tmp_path / "created.txt").exists()


def test_arg_hygiene():
    schema = REG.get("read_file").input_schema
    assert validate_args(schema, {}) == "Missing required argument 'path'."
    assert "integer" in validate_args(schema, {"path": "a", "offset": "x"})
    assert coerce_args(schema, {"path": "a", "offset": "5"})["offset"] == 5
    assert "valid JSON" in validate_args(schema, {"_parse_error": "{"})
    me = REG.get("multi_edit").input_schema
    assert coerce_args(me, {"path": "a", "edits": '[{"old_string":"a","new_string":"b"}]'})["edits"][0]["old_string"] == "a"


def test_diff_headers_use_clean_relative_paths(tmp_path):
    f, c = prep(tmp_path, "a\n")
    r = run("edit_file", c, path="f.py", old_string="a", new_string="b")
    assert r.display.startswith("--- a/f.py\n+++ b/f.py")
    outside = tmp_path.parent / "elsewhere.txt"
    w = run("write_file", c, path=str(outside), content="x\n")
    assert "a//" not in w.display and w.display.splitlines()[0].startswith("--- a/")
