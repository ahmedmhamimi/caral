"""
Model price book and cost arithmetic. Pure data and math: no I/O, so the engine, UI and tests share
one definition of "what did this cost". Prices are USD per 1M tokens and are list prices, so every
figure derived from them is an estimate, not a bill.

The bundled table only holds models whose prices were checked on BUNDLED_AS_OF; anything else is
unpriced (cost shown as unknown) rather than guessed. Users add or fix entries with overrides, and
`caral --update-prices` can fill gaps from OpenRouter's public model list.

- ModelPrice(input, output, cache_read=None, cache_write=None, source="", note="", over_tokens=0, over=None):
  per-1M-token rates. .rates_for(context_tokens) -> ModelPrice (long-context tier, if any);
  .cache_read_rate / .cache_write_rate -> float (falls back to the input rate when unknown, which
  over-estimates rather than under-estimates).
- BUNDLED_PRICES: dict[str, ModelPrice]; BUNDLED_AS_OF: date string for the table.
- normalize_model_id(model) -> str: lowercase, dots/underscores to "-", strips a trailing -YYYYMMDD.
- PriceBook(bundled=None, fetched=None, overrides=None): layered lookup, overrides win over bundled
  over fetched. .lookup(model) -> ModelPrice | None; .known() -> list[str]; .layer_of(model) -> str.
- cost_usd(usage, price, tiered=True) -> float: fresh input + cache writes + cache reads + output.
- parse_openrouter_models(payload) -> dict[str, ModelPrice]: converts OpenRouter's /models response.
- parse_overrides(data) -> (dict[str, ModelPrice], list[str]): validates a user price file; returns
  the usable entries and a list of human-readable problems.
- format_money(usd) -> str: "$0.0042" / "$0.412" / "$12.50".
- count_models(prices) -> int: distinct models in a price map (full and short id spellings count once).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .messages import Usage

BUNDLED_AS_OF = "2026-10-20"
_DATE_SUFFIX = re.compile(r"-\d{8}$")


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None
    source: str = ""
    note: str = ""
    over_tokens: int = 0            # prompts larger than this use `over` rates (0 = no long-context tier)
    over: "ModelPrice | None" = None

    @property
    def cache_read_rate(self) -> float:
        return self.input if self.cache_read is None else self.cache_read

    @property
    def cache_write_rate(self) -> float:
        return self.input if self.cache_write is None else self.cache_write

    def rates_for(self, context_tokens: int) -> "ModelPrice":
        if self.over is not None and self.over_tokens and context_tokens > self.over_tokens:
            return self.over
        return self


def _anthropic(inp: float, out: float, note: str = "") -> ModelPrice:
    """Anthropic's documented cache multipliers: 5-minute writes cost 1.25x input, reads 0.1x."""
    return ModelPrice(inp, out, round(inp * 0.1, 6), round(inp * 1.25, 6), "bundled",
                      note or "cache rates use Anthropic's standard 0.1x / 1.25x multipliers")


BUNDLED_PRICES: dict[str, ModelPrice] = {
    "claude-sonnet-5-5": _anthropic(2.00, 10.00),
    "claude-fable-5-1": _anthropic(10.00, 50.00),
    "claude-haiku-5-5": ModelPrice(
        0.10, 0.50, 0.01, 0.125, "bundled",
        "prompts over 100k tokens bill at $0.50 in / $2.50 out (cache rates there assumed 5x)",
        over_tokens=100_000, over=ModelPrice(0.50, 2.50, 0.05, 0.625, "bundled"),
    ),
    "deepseek-v4-1-flash": ModelPrice(
        0.15, 0.60, 0.003, None, "bundled",
        "DeepSeek direct API list price; OpenRouter lists roughly double"),
    "kimi-k3": ModelPrice(3.00, 15.00, 0.30, None, "bundled", "Moonshot direct API list price"),
    "gpt-6-astra": ModelPrice(10.00, 50.00, 1.00, 12.50, "bundled"),
    "gpt-5-6-sol": ModelPrice(2.00, 10.00, None, None, "bundled", "cache rates not verified"),
    "gpt-5-6-luna": ModelPrice(0.20, 1.20, None, None, "bundled", "cache rates not verified"),
    "gpt-4-1": ModelPrice(2.00, 8.00, 0.50, None, "bundled"),
    "gpt-4-1-mini": ModelPrice(0.40, 1.60, 0.10, None, "bundled"),
    "muse-spark-1-2": ModelPrice(1.25, 4.25, None, None, "bundled", "OpenRouter list price; cache rates not verified"),
}


def normalize_model_id(model: str) -> str:
    m = (model or "").strip().lower().replace(".", "-").replace("_", "-")
    return _DATE_SUFFIX.sub("", m)


def _candidates(model: str) -> list[str]:
    full = normalize_model_id(model)
    short = normalize_model_id(model.rsplit("/", 1)[-1]) if "/" in model else full
    return [full] if short == full else [full, short]


class PriceBook:
    """Layered price lookup. Exact (normalized) id only: a near match could silently be the wrong
    price tier, so an unknown model stays unknown instead of inheriting a sibling's rate."""

    def __init__(self, bundled: dict[str, ModelPrice] | None = None, fetched: dict[str, ModelPrice] | None = None,
                 overrides: dict[str, ModelPrice] | None = None) -> None:
        self.bundled = dict(BUNDLED_PRICES if bundled is None else bundled)
        self.fetched = dict(fetched or {})
        self.overrides = dict(overrides or {})

    def _layers(self) -> list[tuple[str, dict[str, ModelPrice]]]:
        return [("override", self.overrides), ("bundled", self.bundled), ("fetched", self.fetched)]

    def lookup(self, model: str) -> ModelPrice | None:
        for key in _candidates(model):
            for _name, layer in self._layers():
                if key in layer:
                    return layer[key]
        return None

    def layer_of(self, model: str) -> str:
        for key in _candidates(model):
            for name, layer in self._layers():
                if key in layer:
                    return name
        return ""

    def known(self) -> list[str]:
        return sorted({k for _n, layer in self._layers() for k in layer})


def cost_usd(usage: Usage, price: ModelPrice, tiered: bool = True) -> float:
    r = price.rates_for(usage.context_tokens) if tiered else price
    return (
        usage.input_tokens * r.input + usage.cache_write * r.cache_write_rate
        + usage.cache_read * r.cache_read_rate + usage.output_tokens * r.output
    ) / 1_000_000


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f >= 0 else None


def parse_openrouter_models(payload: Any) -> dict[str, ModelPrice]:
    """OpenRouter reports per-token USD strings; variants like ':free' are skipped (a $0 price for a
    rate-limited free tier would hide real cost on the paid model)."""
    items = payload.get("data", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    out: dict[str, ModelPrice] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mid, pr = str(item.get("id") or ""), item.get("pricing")
        if not mid or ":" in mid or not isinstance(pr, dict):
            continue
        inp, outp = _num(pr.get("prompt")), _num(pr.get("completion"))
        if inp is None or outp is None:
            continue
        cr, cw = _num(pr.get("input_cache_read")), _num(pr.get("input_cache_write"))
        price = ModelPrice(
            inp * 1e6, outp * 1e6,
            None if cr is None else cr * 1e6, None if cw is None else cw * 1e6,
            "openrouter", "fetched from OpenRouter's public model list",
        )
        out[normalize_model_id(mid)] = price
        out.setdefault(normalize_model_id(mid.rsplit("/", 1)[-1]), price)
    return out


def parse_overrides(data: Any) -> tuple[dict[str, ModelPrice], list[str]]:
    """User file format: {"model-id": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5}}
    with prices per 1M tokens; cache fields are optional."""
    prices: dict[str, ModelPrice] = {}
    problems: list[str] = []
    if not isinstance(data, dict):
        return prices, ["price file must be a JSON object keyed by model id"]
    for key, spec in data.items():
        if str(key).startswith("_"):
            continue  # comment keys
        if not isinstance(spec, dict):
            problems.append(f"{key}: expected an object with input/output")
            continue
        inp, outp = _num(spec.get("input")), _num(spec.get("output"))
        if inp is None or outp is None:
            problems.append(f"{key}: needs numeric non-negative 'input' and 'output' (USD per 1M tokens)")
            continue
        cr, cw = _num(spec.get("cache_read")), _num(spec.get("cache_write"))
        prices[normalize_model_id(str(key))] = ModelPrice(inp, outp, cr, cw, "override", "from your price file")
    return prices, problems


def format_money(usd: float) -> str:
    if usd < 0.01:
        return f"${usd:.4f}"
    if usd < 1:
        return f"${usd:.3f}"
    return f"${usd:,.2f}"


def count_models(prices: dict[str, ModelPrice]) -> int:
    return len({id(p) for p in prices.values()})
