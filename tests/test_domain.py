"""
Tests for the pure domain layer.

- conversation validation/repair: orphans, dangling tool_use, empty messages, leading assistant, ordering.
- ensure_unique_ids, TokenEstimator calibration, redaction, textutil, Usage math, Event.to_dict.
"""

import json

from caral.domain.conversation import SYNTHETIC_RESULT, ensure_unique_ids, repair_conversation, validate_conversation
from caral.domain.events import Event, ModelResponded, ToolFinished
from caral.domain.messages import (
    Usage, assistant_message, count_chars, render_transcript, text_block, tool_result_block, tool_use_block,
    user_blocks, user_text,
)
from caral.domain.redact import redact, redact_obj
from caral.domain.textutil import clip, human_duration, human_tokens, make_diff, truncate_middle
from caral.domain.tokens import TokenEstimator


def tu(i, name="bash"):
    return tool_use_block(i, name, {})


def tr(i, text="r", err=False):
    return tool_result_block(i, text, err)


def test_valid_conversation_has_no_problems():
    msgs = [user_text("a"), assistant_message([tu("1")]), user_blocks([tr("1")]), assistant_message([text_block("done")])]
    assert validate_conversation(msgs) == []
    fixed, notes = repair_conversation(msgs)
    assert fixed is msgs and notes == []


def test_dangling_tool_use_at_end_gets_synthetic_result():
    msgs = [user_text("a"), assistant_message([tu("1"), tu("2")])]
    assert validate_conversation(msgs)
    fixed, notes = repair_conversation(msgs)
    assert validate_conversation(fixed) == [] and notes
    results = fixed[-1]["content"]
    assert [b["tool_use_id"] for b in results] == ["1", "2"] and results[0]["content"] == SYNTHETIC_RESULT


def test_partial_results_are_completed_and_ordered_first():
    msgs = [user_text("a"), assistant_message([tu("1"), tu("2")]),
            user_blocks([text_block("also"), tr("2")]), assistant_message([text_block("x")])]
    fixed, _ = repair_conversation(msgs)
    assert validate_conversation(fixed) == []
    kinds = [b["type"] for b in fixed[2]["content"]]
    assert kinds == ["tool_result", "tool_result", "text"]


def test_orphan_results_and_empty_messages_dropped():
    msgs = [user_text("a"), {"role": "assistant", "content": []}, user_blocks([tr("zzz")]), assistant_message([text_block("x")])]
    fixed, notes = repair_conversation(msgs)
    assert validate_conversation(fixed) == [] and len(fixed) == 2 and notes


def test_conversation_must_start_with_user():
    fixed, notes = repair_conversation([assistant_message([text_block("hi")])])
    assert fixed[0]["role"] == "user" and validate_conversation(fixed) == []


def test_unique_ids():
    blocks = [tu("a"), tu("a"), {"type": "tool_use", "id": "", "name": "x", "input": {}}]
    ids = [b["id"] for b in ensure_unique_ids(blocks)]
    assert len(set(ids)) == 3 and all(ids)


def test_estimator_calibrates_and_clamps():
    e = TokenEstimator()
    e.observe(10, 5)  # too small: ignored
    assert e.ratio == 3.6
    for _ in range(20):
        e.observe(40_000, 8_000)  # 5 chars/token
    assert 4.8 < e.ratio <= 5.0
    for _ in range(50):
        e.observe(100_000, 1_000)  # absurd 100 chars/token
    assert e.ratio == 6.0


def test_usage_math():
    u = Usage(100, 50, 400, 25)
    assert u.context_tokens == 525 and u.billable == 175 and round(u.cache_hit_ratio, 2) == 0.76
    u.add(Usage(1, 1, 1, 1))
    assert Usage.from_dict(u.to_dict()) == u and Usage().cache_hit_ratio == 0.0


def test_redaction():
    text = "key sk-abcdefghijklmnop1234 and Bearer abcdefghijklmnopqrstuv and api_key=hunter2hunter2 AKIAABCDEFGHIJKLMNOP"
    out = redact(text)
    assert "sk-abc" not in out and "hunter2" not in out and "AKIA" not in out and out.count("[REDACTED]") >= 4
    assert redact_obj({"a": ["token: supersecretvalue1"]})["a"][0].endswith("[REDACTED]")
    assert redact("nothing secret here") == "nothing secret here"


def test_textutil():
    assert truncate_middle("x" * 100, 40).count("truncated") == 1 and truncate_middle("abc", 40) == "abc"
    assert human_tokens(950) == "950" and human_tokens(12_345) == "12.3k" and human_tokens(250_000) == "250k"
    assert human_duration(0.42) == "420ms" and human_duration(75) == "1m15s"
    assert clip("a  b\nc" * 20, 10).endswith("…") and len(clip("a  b\nc" * 20, 10)) == 10
    assert "+new" in make_diff("old\n", "new\n", "f")


def test_event_to_dict_is_json_safe():
    e = ModelResponded(run_id="r", step=1, model="m", stop_reason="end_turn", usage=Usage(1, 2), duration_s=0.5, tool_calls=0)
    d = e.to_dict()
    assert d["type"] == "ModelResponded" and d["usage"]["output_tokens"] == 2 and "ts" in d
    json.dumps(d)
    assert isinstance(e, Event)
    assert ToolFinished(call_id="c", name="n", is_error=False, duration_s=0, content="x", chars=1).to_dict()["display"] is None


def test_chars_and_transcript():
    msgs = [user_text("hello"), assistant_message([tu("1")]), user_blocks([tr("1", "x" * 5000)])]
    assert count_chars(msgs) > 5000
    t = render_transcript(msgs)
    assert "USER: hello" in t and "called bash" in t and "snip" in t
