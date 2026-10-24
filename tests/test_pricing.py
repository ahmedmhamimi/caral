"""
Tests for the price book, cost arithmetic, price files/fetching and the live cost meter (including the
agent paths that bill outside the main loop: subagents and context summaries).

- Price lookup is exact (no sibling inheritance), cost math covers all four token kinds and the
  long-context tier, OpenRouter parsing converts per-token strings, overrides beat bundled beat fetched.
- CostMeter totals per run / per launch / per model, flags unpriced models, treats local as free.
- Agent integration: usage from subagents and compaction reaches the meter at the right model's rate.
"""

import json

import httpx
import pytest

from caral.adapters.providers.price_fetch import fetch_openrouter_prices
from caral.adapters.store.prices import fetched_age_days, load_price_book, overrides_path, save_fetched
from caral.domain.events import AuxUsage, ContextCompacted, ModelResponded, RunStarted, SubagentFinished
from caral.domain.messages import Usage, assistant_message, tool_result_block, tool_use_block, user_blocks, user_text
from caral.domain.pricing import (
    BUNDLED_PRICES, ModelPrice, PriceBook, cost_usd, format_money, normalize_model_id, parse_openrouter_models,
    parse_overrides,
)
from caral.engine.meter import CostMeter
from conftest import FakeLLM, make_agent, text_turn, tool_turn
from caral.errors import ProviderError


# -- price book ------------------------------------------------------------------------------
def test_normalize_and_exact_lookup():
    assert normalize_model_id("Claude-Sonnet-5.5-20260928") == "claude-sonnet-5-5"
    book = PriceBook()
    assert book.lookup("claude-sonnet-5-5").input == 2.0
    assert book.lookup("anthropic/claude-sonnet-5.5").output == 10.0     # provider prefix and dots
    assert book.lookup("claude-sonnet-5-5-20260928") is not None          # dated snapshot id
    assert book.lookup("gpt-4.1-mini").input == 0.40                       # mini is its own entry
    assert book.lookup("gpt-4.1-nano") is None                             # never inherits a sibling's price
    assert book.lookup("deepseek-chat") is None and book.lookup("") is None


def test_every_bundled_price_is_sane():
    for key, p in BUNDLED_PRICES.items():
        assert key == normalize_model_id(key) and 0 < p.input <= p.output, key
        assert p.cache_read_rate <= p.input * 1.0001 and p.source == "bundled"


def test_cost_covers_all_four_token_kinds():
    sonnet = BUNDLED_PRICES["claude-sonnet-5-5"]
    u = Usage(input_tokens=1000, output_tokens=500, cache_read=2000, cache_write=400)
    assert cost_usd(u, sonnet) == pytest.approx((1000 * 2 + 400 * 2.5 + 2000 * 0.2 + 500 * 10) / 1e6)
    assert cost_usd(Usage(), sonnet) == 0


def test_long_context_tier_applies_only_above_the_threshold():
    haiku = BUNDLED_PRICES["claude-haiku-5-5"]
    small, big = Usage(input_tokens=90_000, output_tokens=1000), Usage(input_tokens=150_000, output_tokens=1000)
    assert cost_usd(small, haiku) == pytest.approx((90_000 * 0.10 + 1000 * 0.50) / 1e6)
    assert cost_usd(big, haiku) == pytest.approx((150_000 * 0.50 + 1000 * 2.50) / 1e6)
    assert cost_usd(big, haiku, tiered=False) == pytest.approx((150_000 * 0.10 + 1000 * 0.50) / 1e6)


def test_unknown_cache_rates_fall_back_to_the_input_rate():
    p = ModelPrice(2.0, 10.0)
    assert cost_usd(Usage(0, 0, cache_read=1_000_000), p) == pytest.approx(2.0)   # conservative, never under-counts


def test_layers_override_wins_then_bundled_then_fetched():
    fetched = {"claude-sonnet-5-5": ModelPrice(99, 99, source="openrouter"), "brand-new-model": ModelPrice(1, 2, source="openrouter")}
    overrides, problems = parse_overrides({"claude-sonnet-5-5": {"input": 1, "output": 5}, "_comment": "x"})
    assert problems == []
    book = PriceBook(fetched=fetched)
    assert book.lookup("claude-sonnet-5-5").input == 2.0 and book.layer_of("claude-sonnet-5-5") == "bundled"
    assert book.lookup("brand-new-model").input == 1 and book.layer_of("brand-new-model") == "fetched"
    book = PriceBook(fetched=fetched, overrides=overrides)
    assert book.lookup("claude-sonnet-5-5").input == 1 and book.layer_of("claude-sonnet-5-5") == "override"
    assert "brand-new-model" in book.known()


def test_parse_overrides_reports_problems_without_raising():
    ok, problems = parse_overrides({"m": {"input": 1, "output": 2, "cache_read": 0.1}, "bad": {"input": "x"}, "neg": {"input": -1, "output": 1}, "str": 5})
    assert list(ok) == ["m"] and ok["m"].cache_read == 0.1 and len(problems) == 3
    assert parse_overrides([1, 2])[1]


def test_parse_openrouter_models():
    payload = {"data": [
        {"id": "anthropic/claude-sonnet-4.5", "pricing": {"prompt": "0.000003", "completion": "0.000015", "input_cache_read": "0.0000003", "input_cache_write": "0.00000375"}},
        {"id": "meta/llama-free:free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "x/dynamic", "pricing": {"prompt": "-1", "completion": "-1"}},
        {"id": "broken"}, "junk",
    ]}
    out = parse_openrouter_models(payload)
    p = out["anthropic/claude-sonnet-4-5"]
    assert (p.input, p.output, p.cache_read, p.cache_write) == pytest.approx((3.0, 15.0, 0.3, 3.75)) and p.source == "openrouter"
    assert out["claude-sonnet-4-5"] is p                       # short id also resolves
    assert not any("free" in k or "dynamic" in k or k == "broken" for k in out)
    assert parse_openrouter_models(None) == {} and parse_openrouter_models([]) == {}


def test_format_money():
    assert (format_money(0.00421), format_money(0.4123), format_money(12.5), format_money(1234.5)) == ("$0.0042", "$0.412", "$12.50", "$1,234.50")


# -- files and fetching ----------------------------------------------------------------------
def test_price_book_loads_overrides_and_survives_corrupt_files(tmp_path):
    book, problems = load_price_book()
    assert problems == [] and book.lookup("claude-sonnet-5-5")
    path = overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"deepseek-chat": {"input": 0.27, "output": 1.10}, "oops": {"input": 1}}))
    book, problems = load_price_book()
    assert book.lookup("deepseek-chat").output == 1.10 and book.layer_of("deepseek-chat") == "override"
    assert len(problems) == 1 and "oops" in problems[0]
    path.write_text("{not json")
    book, problems = load_price_book()
    assert book.lookup("claude-sonnet-5-5") and "cannot read" in problems[0]


def test_fetched_cache_roundtrip_and_fills_gaps():
    assert fetched_age_days() is None
    save_fetched({"new-model": ModelPrice(1.0, 4.0, 0.1, None, "openrouter")})
    assert fetched_age_days() < 0.01
    book, _ = load_price_book()
    assert book.lookup("new-model").output == 4.0 and book.layer_of("new-model") == "fetched"


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_openrouter_prices_success_and_failures():
    ok = {"data": [{"id": "a/b-1", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]}
    prices = fetch_openrouter_prices(_client(lambda r: httpx.Response(200, json=ok)))
    assert prices["a/b-1"].input == pytest.approx(1.0)
    with pytest.raises(ProviderError, match="HTTP 503"):
        fetch_openrouter_prices(_client(lambda r: httpx.Response(503)))
    with pytest.raises(ProviderError, match="did not return JSON"):
        fetch_openrouter_prices(_client(lambda r: httpx.Response(200, text="<html>")))
    with pytest.raises(ProviderError, match="No usable prices"):
        fetch_openrouter_prices(_client(lambda r: httpx.Response(200, json={"data": []})))

    def boom(request):
        raise httpx.ConnectError("down")
    with pytest.raises(ProviderError, match="Could not reach"):
        fetch_openrouter_prices(_client(boom))


# -- meter -----------------------------------------------------------------------------------
BOOK = PriceBook(bundled={"test-model": ModelPrice(1.0, 2.0, 0.1, 1.25), "small-model": ModelPrice(0.1, 0.2, 0.01, 0.125)})


def responded(model, usage):
    return ModelResponded(run_id="r", step=1, model=model, stop_reason="end_turn", usage=usage, duration_s=0.1, tool_calls=0)


def test_meter_run_total_and_per_model_accounting():
    m = CostMeter(BOOK)
    m(RunStarted(run_id="r", prompt_chars=1, model="test-model"))
    m(responded("test-model", Usage(1_000_000, 0)))
    m(responded("test-model", Usage(0, 500_000)))
    assert m.run_cost == pytest.approx(2.0) and m.total_cost == pytest.approx(2.0)
    m(RunStarted(run_id="r2", prompt_chars=1, model="test-model"))
    assert m.run_cost == 0 and m.run_usage.billable == 0 and m.total_cost == pytest.approx(2.0)   # only the run resets
    m(SubagentFinished(description="d", duration_s=1, usage=Usage(1_000_000, 0), status="ok", model="small-model"))
    m(AuxUsage(kind="compaction", model="small-model", usage=Usage(0, 1_000_000)))
    assert m.run_cost == pytest.approx(0.1 + 0.2) and m.total_cost == pytest.approx(2.3)
    by = m.by_model()
    assert by["test-model"].calls == 2 and by["small-model"].calls == 2 and by["small-model"].cost == pytest.approx(0.3)
    assert m.last_call.model == "test-model"     # only main-loop calls count as the "last call"
    snap = m.snapshot()
    assert snap["total"]["cost_usd"] == pytest.approx(2.3) and snap["unpriced_models"] == []
    json.dumps(snap)


def test_meter_flags_unpriced_models_instead_of_guessing():
    m = CostMeter(BOOK)
    m(responded("deepseek-chat", Usage(1000, 100)))
    assert m.total_cost == 0 and m.unpriced == {"deepseek-chat"} and m.last_call.cost is None
    assert m.total_usage.billable == 1100 and m.by_model()["deepseek-chat"].priced is False


def test_meter_local_models_are_free():
    m = CostMeter(BOOK, free=True)
    m(responded("qwen2.5-coder:14b", Usage(5000, 500)))
    assert m.total_cost == 0 and m.last_call.cost == 0 and not m.unpriced and m.total_usage.billable == 5500


def test_meter_ignores_unrelated_events():
    m = CostMeter(BOOK)
    from caral.domain.events import Notice
    m(Notice(level="info", message="x"))
    assert m.total_cost == 0 and m.last_call is None


# -- agent integration -----------------------------------------------------------------------
def wired(tmp_path, turns, **cfg):
    h = make_agent(tmp_path, turns, **cfg)
    meter = CostMeter(BOOK)
    h.bus.subscribe(meter, name="meter")
    return h, meter


def test_agent_run_cost_matches_reported_usage(tmp_path):
    h, meter = wired(tmp_path, [tool_turn(("list_dir", {"depth": 1}), usage=Usage(1000, 100)), text_turn("done", usage=Usage(2000, 50))])
    h.agent.run("go")
    assert meter.total_usage.input_tokens == 3000 and meter.total_usage.output_tokens == 150
    assert meter.total_cost == pytest.approx((3000 * 1.0 + 150 * 2.0) / 1e6)


def test_subagent_usage_is_billed_at_the_small_model_rate(tmp_path):
    h, meter = wired(tmp_path, [
        tool_turn(("task", {"description": "look", "prompt": "find"}), usage=Usage(1_000_000, 0)),
        text_turn("child", usage=Usage(1_000_000, 0)),
        text_turn("parent done", usage=Usage(0, 0))])
    h.agent.run("delegate")
    by = meter.by_model()
    assert by["test-model"].cost == pytest.approx(1.0) and by["small-model"].cost == pytest.approx(0.1)


class UsageLLM(FakeLLM):
    def complete_with_usage(self, system, prompt, max_tokens=2048, model=None):
        self.completions.append({"prompt": prompt, "model": model})
        return self.summary, Usage(input_tokens=2_000_000, output_tokens=0)


def test_compaction_tokens_are_reported_and_billed(tmp_path):
    h, meter = wired(tmp_path, [text_turn("ok", usage=Usage(0, 0))], context_window=3000, prune_threshold=0.3, compact_threshold=0.5)
    h.agent.llm = h.agent.context.llm = UsageLLM([text_turn("ok", usage=Usage(0, 0))])
    msgs = [user_text("start")]
    for i in range(8):
        msgs.append(assistant_message([tool_use_block(f"t{i}", "read_file", {"path": "x"})]))
        msgs.append(user_blocks([tool_result_block(f"t{i}", "y" * 3000)]))
    msgs.append(assistant_message([{"type": "text", "text": "done"}]))
    h.agent.session.messages = msgs
    h.agent.run("go on")
    aux = h.of(AuxUsage)
    assert len(aux) == 1 and aux[0].kind == "compaction" and aux[0].model == "small-model"
    assert meter.by_model()["small-model"].cost == pytest.approx(2_000_000 * 0.1 / 1e6)


def test_compaction_without_usage_support_still_works(tmp_path):
    h, meter = wired(tmp_path, [text_turn("ok", usage=Usage(0, 0))], context_window=3000, prune_threshold=0.3, compact_threshold=0.5)
    msgs = [user_text("start")]
    for i in range(8):
        msgs.append(assistant_message([tool_use_block(f"t{i}", "read_file", {"path": "x"})]))
        msgs.append(user_blocks([tool_result_block(f"t{i}", "y" * 3000)]))
    msgs.append(assistant_message([{"type": "text", "text": "done"}]))
    h.agent.session.messages = msgs
    h.agent.run("go on")                           # FakeLLM has no complete_with_usage: plain complete() path
    assert h.of(ContextCompacted) and h.of(AuxUsage) == []
    assert "small-model" not in meter.by_model()


# -- automatic price fill-in for OpenRouter ---------------------------------------------------
def test_openrouter_unpriced_model_triggers_one_fetch_and_is_cached():
    from caral.app import refresh_prices_if_needed
    from caral.config import Config
    price = ModelPrice(3.0, 15.0, 0.3, 3.75, "openrouter")
    fresh = {"anthropic/claude-sonnet-4-5": price, "claude-sonnet-4-5": price}
    calls = []

    def fetch(**kw):
        calls.append(kw)
        return fresh
    cfg = Config(provider="openrouter", kind="openai", model="anthropic/claude-sonnet-4.5", base_url="https://openrouter.ai/api/v1")
    book = PriceBook()
    assert refresh_prices_if_needed(cfg, book, fetch) is True and calls == [{"timeout": 4.0}]
    assert book.lookup("anthropic/claude-sonnet-4.5").input == 3.0 and book.layer_of("anthropic/claude-sonnet-4.5") == "fetched"
    assert load_price_book()[0].lookup("anthropic/claude-sonnet-4.5").output == 15.0     # persisted for the next launch
    assert refresh_prices_if_needed(cfg, book, fetch) is False and len(calls) == 1       # priced now: no more network


def test_no_fetch_for_other_hosts_priced_models_or_failures():
    from caral.app import refresh_prices_if_needed
    from caral.config import Config

    def boom(**kw):
        raise AssertionError("must not be called")
    direct = Config(provider="anthropic", kind="anthropic", model="mystery", base_url="https://api.anthropic.com")
    assert refresh_prices_if_needed(direct, PriceBook(), boom) is False
    priced = Config(provider="openrouter", kind="openai", model="claude-sonnet-5-5", base_url="https://openrouter.ai/api/v1")
    assert refresh_prices_if_needed(priced, PriceBook(), boom) is False

    def offline(**kw):
        raise ProviderError("offline", kind="network")
    unknown = Config(provider="openrouter", kind="openai", model="x/unknown", base_url="https://openrouter.ai/api/v1")
    assert refresh_prices_if_needed(unknown, PriceBook(), offline) is False
