"""
Tests for provider adapters over httpx.MockTransport (no network): wire conversion, streaming
assembly, usage/caching normalization, retry policy, error classification, and runtime
capability fallbacks for quirky OpenAI-compatible servers.
"""

import json

import httpx
import pytest

from caral.adapters.providers.anthropic import AnthropicProvider, clean_messages
from caral.adapters.providers.openai_compat import OpenAIProvider, convert_messages, parse_usage
from caral.config import Config
from caral.domain.messages import assistant_message, text_block, tool_result_block, tool_use_block, user_blocks, user_text
from caral.errors import ContextLengthError, ProviderError, classify_http_error


def sse(*events):
    out = ""
    for ev in events:
        if isinstance(ev, tuple):
            out += f"event: {ev[0]}\ndata: {json.dumps(ev[1])}\n\n"
        elif ev == "[DONE]":
            out += "data: [DONE]\n\n"
        else:
            out += f"data: {json.dumps(ev)}\n\n"
    return out.encode()


def provider(cls, handler, **cfg):
    kind = "anthropic" if cls is AnthropicProvider else "openai"
    p = cls(Config(provider="x", kind=kind, model="m", api_key="k", base_url=cfg.pop("base_url", "http://test"), max_retries=2, **cfg))
    p.client = httpx.Client(transport=httpx.MockTransport(handler))
    return p


TOOL = [{"name": "read_file", "description": "d", "input_schema": {"type": "object"}}]


def test_openai_convert_messages():
    msgs = [user_text("hi"),
            assistant_message([text_block("calling"), tool_use_block("c1", "read_file", {"path": "a"})]),
            user_blocks([tool_result_block("c1", "contents"), tool_result_block("c2", "bad", True)])]
    out = convert_messages("SYS", msgs)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"path": "a"}'
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "contents"}
    assert out[4]["content"] == "ERROR: bad"
    assert convert_messages("", [assistant_message([tool_use_block("c", "x", {})])])[0]["content"] is None


def test_openai_usage_normalization():
    u = parse_usage({"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 800}})
    assert (u.input_tokens, u.cache_read, u.output_tokens) == (200, 800, 50) and u.context_tokens == 1000
    d = parse_usage({"prompt_cache_hit_tokens": 700, "prompt_cache_miss_tokens": 300, "completion_tokens": 9})
    assert (d.input_tokens, d.cache_read) == (300, 700)
    assert parse_usage({}).context_tokens == 0


def test_openai_stream_text_tools_usage():
    body = sse({"choices": [{"delta": {"content": "Hel"}}]}, {"choices": [{"delta": {"content": "lo"}}]},
               {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": '{"pa'}}]}}]},
               {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "x"}'}}]}, "finish_reason": "tool_calls"}]},
               {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}}, "[DONE]")
    seen = {}

    def handler(req):
        seen["body"], seen["auth"] = json.loads(req.content), req.headers["authorization"]
        return httpx.Response(200, content=body)

    chunks = []
    turn = provider(OpenAIProvider, handler).stream("sys", [user_text("hi")], TOOL, on_text=chunks.append, model="override", max_tokens=77)
    assert "".join(chunks) == "Hello" == turn.text
    c = turn.tool_calls[0]
    assert (c.id, c.name, c.input) == ("call_1", "read_file", {"path": "x"}) and turn.stop_reason == "tool_use"
    assert turn.usage.input_tokens == 12 and seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "override" and seen["body"]["max_tokens"] == 77 and seen["body"]["tools"][0]["type"] == "function"


def test_openai_uses_max_completion_tokens_on_openai_host_only():
    p = provider(OpenAIProvider, lambda r: None, base_url="https://api.openai.com/v1")
    assert "max_completion_tokens" in p._payload("s", [user_text("x")], [])
    assert "max_tokens" in provider(OpenAIProvider, lambda r: None)._payload("s", [user_text("x")], [])


def test_openai_adapts_when_server_rejects_stream_options():
    calls = []
    ok = sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}, "[DONE]")

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        if "stream_options" in body:
            return httpx.Response(400, json={"error": {"message": "Unknown parameter: stream_options"}})
        return httpx.Response(200, content=ok)

    p = provider(OpenAIProvider, handler)
    assert p.stream("s", [user_text("x")], []).text == "ok" and len(calls) == 2
    p.stream("s", [user_text("y")], [])
    assert len(calls) == 3 and "stream_options" not in calls[2]      # remembered; no wasted request next time


def test_openai_swaps_token_parameter_name_when_rejected():
    calls = []
    ok = sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}, "[DONE]")

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        if "max_completion_tokens" in body:
            return httpx.Response(400, json={"error": {"message": "unsupported parameter: max_completion_tokens"}})
        return httpx.Response(200, content=ok)

    p = provider(OpenAIProvider, handler, base_url="https://api.openai.com/v1")
    assert p.stream("s", [user_text("x")], []).text == "ok" and "max_tokens" in calls[-1]


def test_malformed_and_non_object_tool_arguments(tmp_path):
    for args in ('{"command": "ls', '[1,2]', '"str"'):
        body = sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "bash", "arguments": args}}]}, "finish_reason": "tool_calls"}]}, "[DONE]")
        turn = provider(OpenAIProvider, lambda r, b=body: httpx.Response(200, content=b)).stream("s", [user_text("x")], [])
        assert "_parse_error" in turn.tool_calls[0].input


def test_openai_missing_ids_and_midstream_error():
    body = sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "bash", "arguments": "{}"}}]}}]}, "[DONE]")
    t = provider(OpenAIProvider, lambda r: httpx.Response(200, content=body)).stream("s", [user_text("x")], [])
    assert t.tool_calls[0].id and t.stop_reason == "tool_use"
    err = sse({"error": {"message": "backend exploded"}})
    with pytest.raises(ProviderError, match="backend exploded"):
        provider(OpenAIProvider, lambda r: httpx.Response(200, content=err)).stream("s", [user_text("x")], [])


def test_anthropic_stream_and_cache_usage():
    body = sse(
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 50, "cache_read_input_tokens": 10, "cache_creation_input_tokens": 5}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"command":'}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ' "ls"}'}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
    )
    turn = provider(AnthropicProvider, lambda r: httpx.Response(200, content=body)).stream("s", [user_text("x")], [])
    assert turn.text == "Hi" and turn.tool_calls[0].input == {"command": "ls"} and turn.stop_reason == "tool_use"
    u = turn.usage
    assert (u.input_tokens, u.cache_read, u.cache_write, u.output_tokens) == (50, 10, 5, 9) and u.context_tokens == 65


def test_anthropic_payload_caching_and_overrides():
    p = provider(AnthropicProvider, lambda r: None)
    tools = [{"name": "a", "description": "d", "input_schema": {"type": "object"}}, {"name": "b", "description": "d", "input_schema": {"type": "object"}}]
    payload = p._payload("sys", [user_text("hi")], tools, model="small", max_tokens=123)
    assert payload["model"] == "small" and payload["max_tokens"] == 123
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in payload["tools"][0] and "cache_control" in payload["tools"][1]
    assert "cache_control" in payload["messages"][-1]["content"][-1]
    n = json.dumps(payload).count("cache_control")
    assert n == 3 <= 4          # provider limit is 4 breakpoints
    p.cfg.prompt_cache = False
    assert "cache_control" not in json.dumps(p._payload("sys", [user_text("hi")], tools))


def test_anthropic_cache_marker_does_not_mutate_history():
    p = provider(AnthropicProvider, lambda r: None)
    msgs = [user_text("hi")]
    p._payload("s", msgs, [])
    assert "cache_control" not in json.dumps(msgs)
    assert clean_messages([{"role": "user", "content": [{"type": "text", "text": ""}]}]) == []


def test_anthropic_midstream_overload_is_retryable():
    body = sse(("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
    with pytest.raises(ProviderError) as e:
        provider(AnthropicProvider, lambda r: httpx.Response(200, content=body)).stream("s", [user_text("x")], [])
    assert e.value.retryable and e.value.kind == "overloaded"


def test_retry_with_retry_after_then_success(monkeypatch):
    sleeps = []
    monkeypatch.setattr("caral.adapters.providers.base.time.sleep", sleeps.append)
    n = {"c": 0}
    ok = sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}, "[DONE]")

    def handler(req):
        n["c"] += 1
        return httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow"}}) if n["c"] < 3 else httpx.Response(200, content=ok)

    retries = []
    turn = provider(OpenAIProvider, handler).stream("s", [user_text("x")], [], on_retry=lambda a, d, r: retries.append((a, r)))
    assert turn.text == "ok" and sleeps == [7.0, 7.0] and retries == [(1, "HTTP 429"), (2, "HTTP 429")]


def test_retries_exhausted_and_network_errors(monkeypatch):
    monkeypatch.setattr("caral.adapters.providers.base.time.sleep", lambda s: None)
    with pytest.raises(ProviderError) as e:
        provider(OpenAIProvider, lambda r: httpx.Response(503, text="down")).stream("s", [user_text("x")], [])
    assert e.value.status == 503 and e.value.retryable

    def boom(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError) as e2:
        provider(OpenAIProvider, boom).stream("s", [user_text("x")], [])
    assert e2.value.kind == "network" and "connection" in e2.value.hint.lower()


@pytest.mark.parametrize("status,msg,kind,cls", [
    (401, "invalid x-api-key", "auth", ProviderError), (403, "forbidden", "auth", ProviderError),
    (404, "model not found", "not_found", ProviderError), (429, "rate", "rate_limit", ProviderError),
    (529, "overloaded", "server", ProviderError), (400, "bad field", "bad_request", ProviderError),
    (400, "prompt is too long: 250000 tokens > 200000 maximum", "context_length", ContextLengthError),
    (400, "This model's maximum context length is 8192 tokens", "context_length", ContextLengthError),
    (413, "request too large", "unknown", ProviderError),
])
def test_error_classification(status, msg, kind, cls):
    e = classify_http_error(status, msg)
    assert isinstance(e, cls) and e.kind == kind and e.status == status
    if status in (401, 404, 400, 429):
        assert e.hint


def test_auth_error_is_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(ProviderError) as e:
        provider(AnthropicProvider, handler).stream("s", [user_text("x")], [])
    assert calls["n"] == 1 and "bad key" in str(e.value) and "doctor" in e.value.hint


def test_complete_is_toolless_and_uses_given_model():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "summary"}, "finish_reason": "stop"}]}, "[DONE]"))

    assert provider(OpenAIProvider, handler).complete("sys", "p", max_tokens=99, model="cheap") == "summary"
    assert seen["model"] == "cheap" and "tools" not in seen and seen["max_tokens"] == 99


def test_list_models_openai_and_anthropic_shapes():
    seen = {}

    def handler(req):
        seen["url"], seen["auth"] = str(req.url), req.headers.get("authorization") or req.headers.get("x-api-key")
        return httpx.Response(200, json={"data": [{"id": "b-model"}, {"id": "a-model"}, {"id": "a-model"}, {"junk": 1}]})

    assert provider(OpenAIProvider, handler).list_models() == ["a-model", "b-model"]
    assert seen["url"] == "http://test/models" and seen["auth"] == "Bearer k"
    assert provider(AnthropicProvider, handler).list_models() == ["a-model", "b-model"]
    assert seen["url"].startswith("http://test/v1/models") and seen["auth"] == "k"


def test_list_models_errors_are_typed():
    with pytest.raises(ProviderError) as e:
        provider(OpenAIProvider, lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})).list_models()
    assert e.value.kind == "auth"
    with pytest.raises(ProviderError):
        provider(OpenAIProvider, lambda r: httpx.Response(200, text="<html>")).list_models()

    def boom(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError) as e2:
        provider(OpenAIProvider, boom).list_models()
    assert e2.value.kind == "network"


def test_model_not_found_hint_mentions_the_models_command():
    e = classify_http_error(404, "The model `x` does not exist or you do not have access to it.")
    assert e.kind == "not_found" and "caral --models" in e.hint


# -- prompt-cache markers for Claude behind OpenAI-style gateways (OpenRouter) ---------------
from caral.adapters.providers.openai_compat import add_cache_markers, wants_cache_markers

CLAUDE = "anthropic/claude-sonnet-4.5"
OR_URL = "https://openrouter.ai/api/v1"


def _history():
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result one"},
        {"role": "user", "content": "second prompt"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "result two"},
    ]


def _marked(msgs):
    return [i for i, m in enumerate(msgs) if isinstance(m["content"], list) and any("cache_control" in p for p in m["content"])]


def test_cache_markers_land_on_system_latest_user_and_newest_message():
    h = _history()
    out = add_cache_markers(h)
    assert _marked(out) == [0, 4, 6]
    assert out[0]["content"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert out[6]["content"][0]["text"] == "result two" and out[6]["tool_call_id"] == "c2"
    assert _marked(add_cache_markers(h, "basic")) == [0, 4]          # no newest-message breakpoint
    assert _marked(h) == [] and h[0]["content"] == "SYS"             # input never mutated


def test_cache_markers_skip_empty_content_and_survive_odd_histories():
    assert add_cache_markers([]) == []
    out = add_cache_markers([{"role": "user", "content": ""}, {"role": "assistant", "content": None, "tool_calls": []}])
    assert _marked(out) == []                                        # empty text blocks are invalid for caching
    out = add_cache_markers([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert out[0]["content"][0] == {"type": "text", "text": "a"} and out[0]["content"][1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("base,model,extra,expected", [
    (OR_URL, CLAUDE, {}, True),
    (OR_URL, "openai/gpt-4.1", {}, False),                           # other models cache automatically
    (OR_URL, "deepseek/deepseek-v4.1-flash", {}, False),
    ("https://api.openai.com/v1", "gpt-4.1", {}, False),
    ("https://api.deepseek.com/v1", "deepseek-chat", {}, False),
    (OR_URL, "some/claude-variant", {}, True),
    ("https://gateway.example/v1", "claude-sonnet-5-5", {}, False),  # unknown gateway: not assumed
    ("https://gateway.example/v1", "claude-sonnet-5-5", {"cache_markers": "on"}, True),
    (OR_URL, CLAUDE, {"cache_markers": "off"}, False),
    (OR_URL, CLAUDE, {"prompt_cache": False}, False),
    (OR_URL, CLAUDE, {"cache_markers": "on", "prompt_cache": False}, False),
])
def test_wants_cache_markers(base, model, extra, expected):
    assert wants_cache_markers(Config(provider="x", kind="openai", model=model, base_url=base, **extra), model) is expected


def _ok_stream(usage=None):
    return httpx.Response(200, content=sse(
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": usage or {"prompt_tokens": 10, "completion_tokens": 1}}, "[DONE]"))


def test_openrouter_claude_request_carries_markers_and_other_models_do_not():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return _ok_stream()
    p = provider(OpenAIProvider, handler, base_url=OR_URL)
    msgs = [user_text("hi")]
    p.stream("SYS", msgs, TOOL, model=CLAUDE)
    p.stream("SYS", msgs, TOOL, model="openai/gpt-4.1")
    assert _marked(bodies[0]["messages"]) == [0, 1] and bodies[0]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert _marked(bodies[1]["messages"]) == [] and bodies[1]["messages"][0]["content"] == "SYS"
    assert "cache_control" not in json.dumps(bodies[0]["tools"])      # tools stay plain function schemas


def test_markers_step_down_when_the_server_rejects_them_and_it_is_remembered():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append("cache_control" in req.content.decode())
        if "cache_control" in req.content.decode():
            return httpx.Response(400, json={"error": {"message": "cache_control is not supported on this content"}})
        return _ok_stream()
    p = provider(OpenAIProvider, handler, base_url=OR_URL)
    turn = p.stream("SYS", [user_text("hi"), assistant_message([tool_use_block("c", "x", {})]), user_blocks([tool_result_block("c", "r")])], TOOL, model=CLAUDE)
    assert turn.text == "ok" and seen == [True, True, False] and p._cache_mode == "off"
    p.stream("SYS", [user_text("again")], TOOL, model=CLAUDE)
    assert seen == [True, True, False, False]                          # no more attempts with markers


def test_unrelated_400_is_not_blamed_on_the_markers():
    p = provider(OpenAIProvider, lambda req: httpx.Response(400, json={"error": {"message": "model not found"}}), base_url=OR_URL)
    with pytest.raises(ProviderError):
        p.stream("SYS", [user_text("hi")], TOOL, model=CLAUDE)
    assert p._cache_mode == "full"


def test_openrouter_cache_write_tokens_are_split_out_of_fresh_input():
    u = parse_usage({"prompt_tokens": 3000, "completion_tokens": 20,
                     "prompt_tokens_details": {"cached_tokens": 2000, "cache_write_tokens": 600}})
    assert (u.input_tokens, u.cache_read, u.cache_write, u.output_tokens) == (400, 2000, 600, 20) and u.context_tokens == 3000


def test_cache_markers_setting_is_validated(tmp_path):
    from caral.config import load_config
    from caral.errors import ConfigError
    with pytest.raises(ConfigError, match="cache_markers"):
        load_config(tmp_path, {"provider": "ollama", "cache_markers": "sometimes"})
    assert load_config(tmp_path, {"provider": "ollama", "cache_markers": "off"}).cache_markers == "off"
