"""
Tests for the agent loop with a scripted FakeLLM: tool round-trips, event ordering, approvals,
parallelism, limits, cost controls (small model, compaction, loop guard, budget, read dedupe),
resilience (context-length recovery, mid-stream retry, interrupts, provider errors, crash-repair).
"""

import time

from caral.domain.conversation import validate_conversation
from caral.domain.events import (
    ContextCompacted, ContextPruned, Notice, PermissionResolved, RetryScheduled, RunFinished,
    SubagentFinished, TodosUpdated, ToolFinished, ToolRequested, ToolStarted,
)
from caral.domain.messages import (
    Usage, assistant_message, has_tool_results, tool_result_block, tool_use_block, user_blocks, user_text,
)
from caral.engine.approvers import ScriptedApprover
from caral.errors import ContextLengthError, ProviderError

from conftest import make_agent, text_turn, tool_turn


def test_simple_answer_and_events(tmp_path):
    h = make_agent(tmp_path, [text_turn("hello")])
    r = h.agent.run("hi")
    assert r.ok and r.text == "hello" and r.steps == 1 and r.exit_code == 0
    assert [m["role"] for m in h.agent.messages] == ["user", "assistant"]
    kinds = [type(e).__name__ for e in h.events]
    assert kinds[0] == "RunStarted" and kinds[-1] == "RunFinished" and "TextDelta" in kinds
    assert h.of(RunFinished)[0].usage.output_tokens == 10


def test_tool_roundtrip_and_lifecycle_event_order(tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    (work / "a.txt").write_text("one\n")
    h = make_agent(tmp_path, [
        tool_turn(("read_file", {"path": "a.txt"})),
        tool_turn(("edit_file", {"path": "a.txt", "old_string": "one", "new_string": "two"})),
        text_turn("done")])
    r = h.agent.run("change it")
    assert r.text == "done" and (work / "a.txt").read_text() == "two\n" and r.tool_calls == 2
    edit_events = [type(e).__name__ for e in h.events if getattr(e, "name", "") == "edit_file"]
    assert edit_events == ["ToolRequested", "PermissionResolved", "ToolStarted", "ToolFinished"]
    assert h.of(ToolFinished)[-1].display and "+two" in h.of(ToolFinished)[-1].display


def test_results_follow_calls_and_conversation_stays_valid(tmp_path):
    h = make_agent(tmp_path, [tool_turn(("list_dir", {}), ("glob", {"pattern": "*"})), text_turn("ok")])
    h.agent.run("x")
    ids = [b["id"] for b in h.agent.messages[1]["content"] if b["type"] == "tool_use"]
    assert [b["tool_use_id"] for b in h.agent.messages[2]["content"]] == ids
    assert validate_conversation(h.agent.messages) == []


def test_unknown_tool_and_bad_args_become_errors_not_exceptions(tmp_path):
    h = make_agent(tmp_path, [tool_turn(("nope", {}), ("read_file", {})), text_turn("ok")])
    assert h.agent.run("x").ok
    blocks = h.agent.messages[2]["content"]
    assert blocks[0]["is_error"] and "Unknown tool" in blocks[0]["content"]
    assert blocks[1]["is_error"] and "Missing required" in blocks[1]["content"]


def test_tool_crash_is_contained(tmp_path, monkeypatch):
    h = make_agent(tmp_path, [tool_turn(("list_dir", {})), text_turn("ok")])
    monkeypatch.setattr(h.agent.registry.get("list_dir"), "run", lambda a, c: 1 / 0)
    assert h.agent.run("x").ok
    assert "Tool crashed: ZeroDivisionError" in h.agent.messages[2]["content"][0]["content"]


def test_denied_with_feedback_reaches_model(tmp_path):
    ap = ScriptedApprover([("no", "use pathlib instead")])
    h = make_agent(tmp_path, [tool_turn(("write_file", {"path": "n.py", "content": "x"})), text_turn("ok")], mode="ask", approver=ap)
    h.agent.run("x")
    res = h.agent.messages[2]["content"][0]
    assert res["is_error"] and "use pathlib instead" in res["content"] and not (tmp_path / "proj" / "n.py").exists()
    assert ap.requests[0].diff and "+x" in ap.requests[0].diff          # diff preview shown before approval
    assert any(e.decision == "deny" and e.source == "user" for e in h.of(PermissionResolved))


def test_always_stops_further_prompts(tmp_path):
    ap = ScriptedApprover([("always", "")])
    h = make_agent(tmp_path, [
        tool_turn(("write_file", {"path": "a.py", "content": "1"})),
        tool_turn(("write_file", {"path": "b.py", "content": "2"})), text_turn("ok")], mode="ask", approver=ap)
    h.agent.run("x")
    assert len(ap.requests) == 1 and (tmp_path / "proj" / "b.py").exists()


def test_policy_denial_never_prompts(tmp_path):
    ap = ScriptedApprover([("yes", "")])
    h = make_agent(tmp_path, [tool_turn(("bash", {"command": "rm -rf /"})), text_turn("ok")], mode="ask", approver=ap)
    h.agent.run("x")
    assert ap.requests == [] and "Denied" in h.agent.messages[2]["content"][0]["content"]


def test_parallel_reads_keep_order_and_overlap(tmp_path, monkeypatch):
    work = tmp_path / "proj"
    work.mkdir()
    for n in "abc":
        (work / f"{n}.txt").write_text(n * 3)
    h = make_agent(tmp_path, [tool_turn(*[("read_file", {"path": f"{n}.txt"}) for n in "abc"]), text_turn("ok")])
    tool = h.agent.registry.get("read_file")
    real = tool.run

    def slow(args, ctx):
        time.sleep(0.3)
        return real(args, ctx)

    monkeypatch.setattr(tool, "run", slow)
    t = time.time()
    h.agent.run("x")
    assert time.time() - t < 0.8, "three 0.3s reads should overlap"
    contents = [b["content"] for b in h.agent.messages[2]["content"]]
    assert "aaa" in contents[0] and "bbb" in contents[1] and "ccc" in contents[2]
    assert len(h.of(ToolStarted)) == 3 and len(h.of(ToolRequested)) == 3


def test_mixed_batch_runs_sequentially_in_order(tmp_path):
    ap = ScriptedApprover([("yes", "")])
    h = make_agent(tmp_path, [tool_turn(("write_file", {"path": "a.txt", "content": "A"}), ("list_dir", {})), text_turn("ok")], mode="ask", approver=ap)
    h.agent.run("x")
    assert (tmp_path / "proj" / "a.txt").exists() and len(h.agent.messages[2]["content"]) == 2


def test_step_limit_and_exit_code(tmp_path):
    h = make_agent(tmp_path, [tool_turn(("list_dir", {"depth": i})) for i in range(1, 5)], max_steps=3)
    r = h.agent.run("loop")
    assert r.status == "step_limit" and r.exit_code == 3 and len(h.llm.calls) == 3 and "Send another message" in r.error


def test_max_tokens_truncation_continues(tmp_path):
    cut = text_turn("part one")
    cut.stop_reason = "max_tokens"
    h = make_agent(tmp_path, [cut, text_turn("part two")])
    h.agent.run("x")
    assert len(h.llm.calls) == 2 and "cut off" in h.agent.messages[2]["content"][0]["text"]
    assert any(isinstance(e, Notice) and "max_tokens" in e.message for e in h.events)


def test_empty_model_response_ends_cleanly(tmp_path):
    from caral.domain.messages import Turn
    h = make_agent(tmp_path, [Turn(blocks=[], usage=Usage(5, 0))])
    assert h.agent.run("x").ok and any("empty response" in e.message for e in h.of(Notice))


def test_loop_guard_warns_then_stops(tmp_path):
    h = make_agent(tmp_path, [tool_turn(("list_dir", {})) for _ in range(8)], max_steps=20)
    r = h.agent.run("x")
    assert r.status == "loop" and len(h.llm.calls) == 5
    hint_msgs = [m for m in h.agent.messages if m["role"] == "user" and any("Stop repeating" in b.get("text", "") for b in m["content"] if isinstance(b, dict))]
    assert hint_msgs, "a hint should be injected before stopping"


def test_token_budget_stops_the_run(tmp_path):
    h = make_agent(tmp_path, [tool_turn(("list_dir", {"depth": 1})), tool_turn(("list_dir", {"depth": 2})), text_turn("never")], token_budget=150)
    r = h.agent.run("x")
    assert r.status == "budget" and "budget" in r.error and len(h.llm.calls) == 2


def test_subagent_uses_small_model_readonly_tools_and_reports(tmp_path):
    h = make_agent(tmp_path, [
        tool_turn(("task", {"description": "look", "prompt": "find things"})),
        text_turn("child report"), text_turn("parent done")])
    r = h.agent.run("delegate")
    assert r.text == "parent done"
    assert h.agent.messages[2]["content"][0]["content"] == "child report"
    child = h.llm.calls[1]
    assert child["model"] == "small-model" and h.llm.calls[0]["model"] == "test-model"
    assert {t["name"] for t in child["tools"]} == {"read_file", "list_dir", "glob", "grep", "web_fetch"}
    assert h.of(SubagentFinished)[0].status == "ok"
    assert r.usage.output_tokens == 30  # parent 2 calls + child 1 call, all accounted


def test_subagent_cannot_escalate_to_writes(tmp_path):
    h = make_agent(tmp_path, [
        tool_turn(("task", {"description": "d", "prompt": "p"})),
        tool_turn(("write_file", {"path": "evil.txt", "content": "x"})),   # not in the child's registry
        text_turn("child done"), text_turn("parent done")])
    h.agent.run("x")
    assert not (tmp_path / "proj" / "evil.txt").exists()


def test_read_dedupe_and_prune_invalidation(tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    (work / "a.txt").write_text("hello\n")
    h = make_agent(tmp_path, [tool_turn(("read_file", {"path": "a.txt"})), tool_turn(("read_file", {"path": "a.txt"})), text_turn("ok")])
    h.agent.run("x")
    assert "hello" in h.agent.messages[2]["content"][0]["content"]
    assert "already in your context" in h.agent.messages[4]["content"][0]["content"]


def test_compaction_uses_small_model_and_keeps_pairs_valid(tmp_path):
    h = make_agent(tmp_path, [text_turn("ok")], context_window=3000, prune_threshold=0.3, compact_threshold=0.5)
    msgs = [user_text("start")]
    for i in range(8):
        msgs.append(assistant_message([tool_use_block(f"t{i}", "read_file", {"path": "x"})]))
        msgs.append(user_blocks([tool_result_block(f"t{i}", "y" * 3000)]))
    msgs.append(assistant_message([{"type": "text", "text": "done"}]))
    h.agent.session.messages = msgs
    h.agent.run("go on")
    assert h.of(ContextPruned) or h.of(ContextCompacted)
    assert h.of(ContextCompacted) and h.llm.completions[0]["model"] == "small-model"
    assert validate_conversation(h.agent.messages) == []
    assert "summary" in h.agent.messages[0]["content"][0]["text"].lower()


def test_compaction_failure_falls_back_to_dropping(tmp_path):
    h = make_agent(tmp_path, [text_turn("ok")], context_window=3000, prune_threshold=0.3, compact_threshold=0.5)
    h.llm.summary = ""
    msgs = [user_text("start")]
    for i in range(8):
        msgs.append(assistant_message([tool_use_block(f"t{i}", "read_file", {"path": "x"})]))
        msgs.append(user_blocks([tool_result_block(f"t{i}", "y" * 3000)]))
    msgs.append(assistant_message([{"type": "text", "text": "done"}]))
    h.agent.session.messages = msgs
    assert h.agent.run("go").ok
    assert h.of(ContextCompacted)[0].method == "drop" and validate_conversation(h.agent.messages) == []


def test_context_length_error_triggers_compaction_and_one_retry(tmp_path):
    h = make_agent(tmp_path, [ContextLengthError("too long", 400, kind="context_length"), text_turn("recovered")])
    msgs = [user_text("start")]
    for i in range(6):
        msgs.append(assistant_message([tool_use_block(f"t{i}", "bash", {})]))
        msgs.append(user_blocks([tool_result_block(f"t{i}", "out")]))
    msgs.append(assistant_message([{"type": "text", "text": "x"}]))
    h.agent.session.messages = msgs
    r = h.agent.run("go")
    assert r.ok and r.text == "recovered" and len(h.llm.calls) == 2
    assert any("compacting" in e.message for e in h.of(Notice))
    h2 = make_agent(tmp_path, [ContextLengthError("a", 400), ContextLengthError("b", 400)])
    r2 = h2.agent.run("x")
    assert r2.status == "error" and len(h2.llm.calls) == 2   # only one automatic retry, then a clean error


def test_midstream_retryable_error_is_retried_only_if_nothing_was_shown(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    h = make_agent(tmp_path, [ProviderError("overloaded", 529, True, "overloaded"), text_turn("fine")])
    assert h.agent.run("x").text == "fine" and h.of(RetryScheduled)


def test_provider_error_returns_error_result_and_keeps_conversation_valid(tmp_path):
    h = make_agent(tmp_path, [ProviderError("HTTP 401: bad key", 401, kind="auth", hint="Check your API key")])
    r = h.agent.run("x")
    assert r.status == "error" and r.exit_code == 1 and "bad key" in r.error and "Check your API key" in r.error
    assert validate_conversation(h.agent.messages) == []
    assert h.of(RunFinished)[0].status == "error"


def test_unexpected_exception_does_not_crash_the_agent(tmp_path):
    h = make_agent(tmp_path, [RuntimeError("boom")])
    r = h.agent.run("x")
    assert r.status == "error" and "boom" in r.error


def test_interrupt_during_model_call_leaves_valid_history(tmp_path):
    h = make_agent(tmp_path, [KeyboardInterrupt()])
    r = h.agent.run("go")
    assert r.status == "interrupted" and r.exit_code == 130
    assert [m["role"] for m in h.agent.messages] == ["user", "assistant"] and validate_conversation(h.agent.messages) == []


def test_interrupt_during_tool_fills_every_result(tmp_path, monkeypatch):
    h = make_agent(tmp_path, [tool_turn(("bash", {"command": "echo a"}), ("bash", {"command": "echo b"})), text_turn("x")])
    monkeypatch.setattr(h.agent.registry.get("bash"), "run", lambda a, c: (_ for _ in ()).throw(KeyboardInterrupt()))
    r = h.agent.run("go")
    assert r.status == "interrupted" and validate_conversation(h.agent.messages) == []
    assert all(b["is_error"] for b in h.agent.messages[2]["content"])
    nxt = make_agent(tmp_path, [text_turn("back")])
    nxt.agent.session = h.agent.session
    assert nxt.agent.run("continue").ok   # the session is usable after an interrupt


def test_todo_events_and_session_todos(tmp_path):
    h = make_agent(tmp_path, [tool_turn(("todo_write", {"todos": [{"content": "a", "status": "pending"}]})), text_turn("ok")])
    h.agent.run("x")
    assert h.of(TodosUpdated)[0].todos == [{"content": "a", "status": "pending"}] and h.agent.session.todos[0]["content"] == "a"


def test_undo_after_run(tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    (work / "a.txt").write_text("orig\n")
    h = make_agent(tmp_path, [tool_turn(("read_file", {"path": "a.txt"})), tool_turn(("write_file", {"path": "a.txt", "content": "new\n"})), text_turn("ok")])
    h.agent.run("x")
    assert (work / "a.txt").read_text() == "new\n"
    assert h.agent.undo() and (work / "a.txt").read_text() == "orig\n"


def test_system_prompt_is_cached_for_prompt_cache_stability(tmp_path):
    h = make_agent(tmp_path, [text_turn("a"), text_turn("b")])
    h.agent.run("1")
    h.agent.run("2")
    assert h.llm.calls[0]["system"] is h.llm.calls[1]["system"] or h.llm.calls[0]["system"] == h.llm.calls[1]["system"]
    assert "Caral" in h.llm.calls[0]["system"] and "economical" in h.llm.calls[0]["system"]
    assert h.llm.calls[0]["tools"] is h.llm.calls[1]["tools"]       # cached, byte-stable schema list


def test_context_usage_anchors_on_real_tokens(tmp_path):
    h = make_agent(tmp_path, [text_turn("ok", usage=Usage(1000, 10, 4000, 0))])
    h.agent.run("x")
    used, window = h.agent.context_usage()
    assert 5000 <= used < 5200 and window == 200_000


def test_persistence_and_resume_roundtrip(tmp_path):
    from caral.adapters.store import SessionStore
    store = SessionStore(tmp_path / "s")
    h = make_agent(tmp_path, [text_turn("hello")], store=store)
    h.agent.run("remember me")
    loaded = store.load(h.agent.session.id)
    assert loaded.messages == h.agent.messages and loaded.title == "remember me" and loaded.usage.output_tokens == 10


def test_resume_repairs_crashed_session(tmp_path):
    from caral.adapters.store import SessionStore
    from caral.domain.session import Session
    store = SessionStore(tmp_path / "s")
    s = Session.new(tmp_path, "m")
    s.messages = [user_text("do it"), assistant_message([tool_use_block("t1", "bash", {"command": "x"})])]  # crashed mid-tool
    store.append(s, s.messages)
    store.save_meta(s)
    h = make_agent(tmp_path, [text_turn("ok")], store=store, session=store.load(s.id))
    assert validate_conversation(h.agent.messages) == []
    assert h.agent.run("continue").ok and has_tool_results(h.llm.calls[0]["messages"][-2])


def test_pruning_invalidates_only_the_pruned_files_dedupe(tmp_path):
    """Audit finding: pruning used to clear every dedupe entry, forcing re-reads of files still in context."""
    work = tmp_path / "proj"
    work.mkdir()
    for n in ("old", "new"):
        (work / f"{n}.txt").write_text(n * 400)
    h = make_agent(tmp_path, [text_turn("ok")], context_window=4000, prune_threshold=0.2, compact_threshold=0.9)
    msgs = [user_text("start"),
            assistant_message([tool_use_block("t_old", "read_file", {"path": "old.txt"})]), user_blocks([tool_result_block("t_old", "o" * 2500)])]
    for i in range(3):
        msgs += [assistant_message([{"type": "text", "text": f"m{i}"}]), user_text(f"u{i}")]
    msgs += [assistant_message([tool_use_block("t_new", "read_file", {"path": "new.txt"})]), user_blocks([tool_result_block("t_new", "n" * 100)]),
             assistant_message([{"type": "text", "text": "x"}])]
    h.agent.session.messages = msgs
    h.agent.ctx.remember_view(work / "old.txt", 1, 1000)
    h.agent.ctx.remember_view(work / "new.txt", 1, 1000)
    h.agent.run("go")
    assert h.of(ContextPruned)
    assert not h.agent.ctx.already_viewed(work / "old.txt", 1, 1000)      # its output was pruned: must be re-readable
    assert h.agent.ctx.already_viewed(work / "new.txt", 1, 1000)          # still in context: stays deduped
