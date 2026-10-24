"""
Live token and cost meter. An event subscriber: it listens for every billed model call (the main
loop, subagents, context summaries) and keeps running totals for the current run and for the whole
launch, per model, priced from a PriceBook. Models with no known price are counted in tokens only and
flagged, never guessed. Local providers are free by definition.

Subscribe it to the bus before any renderer so renderers always read up-to-date totals.

- CostMeter(book, free=False): .__call__(event) updates totals from ModelResponded / SubagentFinished /
  AuxUsage and resets the run counters on RunStarted.
  - .run_usage / .total_usage -> Usage: tokens for this run / since launch.
  - .run_cost / .total_cost -> float: USD for priced calls only.
  - .last_call -> CallCost | None: the most recent main-loop call (tokens, cost, model).
  - .unpriced -> set[str]: models seen without a price; .free -> bool; .by_model() -> dict[str, ModelTotals].
  - .price_for(model) -> ModelPrice | None; .snapshot() -> dict (JSON-safe, for /usage and --output-format json).
- CallCost(model, usage, cost): one priced call (cost is None when the model has no price).
- ModelTotals(usage, cost, calls, priced): per-model running totals.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ..domain.events import AuxUsage, ModelResponded, RunStarted, SubagentFinished
from ..domain.messages import Usage
from ..domain.pricing import ModelPrice, PriceBook, cost_usd


@dataclass
class CallCost:
    model: str
    usage: Usage
    cost: float | None


@dataclass
class ModelTotals:
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    calls: int = 0
    priced: bool = True


class CostMeter:
    def __init__(self, book: PriceBook, free: bool = False) -> None:
        self.book = book
        self.free = free
        self.run_usage = Usage()
        self.total_usage = Usage()
        self.run_cost = 0.0
        self.total_cost = 0.0
        self.last_call: CallCost | None = None
        self.unpriced: set[str] = set()
        self._models: dict[str, ModelTotals] = {}
        self._lock = threading.Lock()

    def price_for(self, model: str) -> ModelPrice | None:
        return self.book.lookup(model)

    def __call__(self, event) -> None:
        if isinstance(event, RunStarted):
            with self._lock:
                self.run_usage, self.run_cost = Usage(), 0.0
        elif isinstance(event, ModelResponded):
            self._add(event.model, event.usage, main=True, tiered=True)
        elif isinstance(event, SubagentFinished):
            # A subagent's usage is the sum of several calls, so the long-context tier cannot be
            # applied per call; base rates are used (a small under-estimate only for tiered models).
            self._add(event.model, event.usage, main=False, tiered=False)
        elif isinstance(event, AuxUsage):
            self._add(event.model, event.usage, main=False, tiered=True)

    def _add(self, model: str, usage: Usage, main: bool, tiered: bool) -> None:
        price = None if self.free else self.book.lookup(model)
        cost = 0.0 if self.free else (None if price is None else cost_usd(usage, price, tiered=tiered))
        with self._lock:
            self.run_usage.add(usage)
            self.total_usage.add(usage)
            m = self._models.setdefault(model, ModelTotals())
            m.usage.add(usage)
            m.calls += 1
            if cost is None:
                m.priced = False
                self.unpriced.add(model)
            else:
                m.cost += cost
                self.run_cost += cost
                self.total_cost += cost
            if main:
                self.last_call = CallCost(model, usage, cost)

    def by_model(self) -> dict[str, ModelTotals]:
        with self._lock:
            return dict(self._models)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "run": {"usage": self.run_usage.to_dict(), "cost_usd": round(self.run_cost, 6)},
                "total": {"usage": self.total_usage.to_dict(), "cost_usd": round(self.total_cost, 6)},
                "unpriced_models": sorted(self.unpriced),
                "local_free": self.free,
                "models": {k: {"usage": v.usage.to_dict(), "cost_usd": round(v.cost, 6), "calls": v.calls,
                               "priced": v.priced} for k, v in self._models.items()},
            }
