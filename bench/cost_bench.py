"""
Reproducible cost ablation. Runs the real Caral engine against a scripted model on a file-heavy
session (the model keeps re-reading files, as models do) and measures prompt volume per call, then
prices it with a relative model: fresh tokens 1.0x, cache write 1.25x, cache read 0.1x.

Measured: prompt characters actually sent (from the real message list).
Modelled: price (prefix-cache hits are computed by comparing consecutive prompts byte-for-byte).
The model is scripted, so this isolates harness behavior; real savings depend on how often a model
re-reads and how long sessions run. Run: python bench/cost_bench.py

- Variant(name, dedupe, caching, prune, thresholds): one configuration under test.
- MeteredLLM: FakeLLM that records prompt chars and prefix-cache pricing per call.
- run_variant(v, window_tokens=30_000, steps=30) -> dict: executes one session.
- main() -> None: prints the ablation table.
"""

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from caral.adapters.tools import default_registry
from caral.config import Config
from caral.domain.events import ContextCompacted, ContextPruned
from caral.domain.messages import Usage
from caral.domain.tools import ToolContext
from caral.engine.agent import Agent
from caral.engine.approvers import StaticApprover
from caral.engine.bus import EventBus
from caral.engine.permissions import PermissionManager
from conftest import FakeLLM, text_turn, tool_turn

FILES, FILE_CHARS = 8, 7000


@dataclass
class Variant:
    name: str
    dedupe: bool
    caching: bool
    prune: bool
    thresholds: tuple = (0.70, 0.85)


class MeteredLLM(FakeLLM):
    def __init__(self, turns, caching: bool):
        super().__init__(turns)
        self.caching, self.prev, self.chars, self.cost = caching, [], [], 0.0

    def stream(self, system, messages, tools, on_text=None, on_retry=None, model=None, max_tokens=None):
        fixed = len(system) + len(json.dumps(tools))
        parts = [json.dumps(m, sort_keys=True) for m in messages]
        total = fixed + sum(map(len, parts))
        self.chars.append(total)
        if self.caching:
            common = fixed
            for a, b in zip(self.prev, parts):
                if a != b:
                    break
                common += len(a)
            self.cost += (common * 0.1 + (total - common) * 1.25) / 3.6
        else:
            self.cost += total / 3.6
        self.prev = parts
        turn = super().stream(system, messages, tools, on_text, on_retry, model, max_tokens)
        turn.usage = Usage(0, 0)
        return turn


def run_variant(v: Variant, window_tokens: int = 30_000, steps: int = 30) -> dict:
    work = Path(tempfile.mkdtemp())
    for i in range(FILES):
        (work / f"mod{i}.py").write_text("def f(): pass  # filler line\n" * (FILE_CHARS // 30))
    turns = [tool_turn(("read_file", {"path": f"mod{s % FILES}.py"}), ("read_file", {"path": f"mod{(s * 3 + 1) % FILES}.py"}))
             for s in range(steps)] + [text_turn("done")]
    pt, ct = v.thresholds if v.prune else (0.97, 0.99)
    cfg = Config(provider="anthropic", kind="anthropic", model="m", small_model="s", api_key="k",
                 context_window=window_tokens, prune_threshold=pt, compact_threshold=ct, max_steps=steps + 2)
    llm, bus, ev = MeteredLLM(turns, v.caching), EventBus(), []
    bus.subscribe(ev.append)
    original = ToolContext.already_viewed
    if not v.dedupe:
        ToolContext.already_viewed = lambda self, *a: False  # type: ignore[method-assign]
    try:
        Agent(cfg, llm, default_registry(), PermissionManager("yolo", work, cfg), StaticApprover("yes"), bus, cwd=work).run("review")
    finally:
        ToolContext.already_viewed = original  # type: ignore[method-assign]
    return {"chars": sum(llm.chars), "peak": max(llm.chars), "cost": llm.cost,
            "prunes": sum(isinstance(e, ContextPruned) for e in ev), "compactions": sum(isinstance(e, ContextCompacted) for e in ev)}


VARIANTS = [
    Variant("naive harness", False, False, False),
    Variant("+ prompt caching", False, True, False),
    Variant("+ read dedupe", True, True, False),
    Variant("+ prune/compact @70/85 (Caral default)", True, True, True),
    Variant("  eager prune @50/75 (old default, for contrast)", True, True, True, (0.50, 0.75)),
]


def main() -> None:
    rows = [(v, run_variant(v)) for v in VARIANTS]
    base = rows[0][1]["cost"]
    print(f"{'configuration':52} {'prompt chars':>13} {'rel. cost':>10} {'vs naive':>9} {'prunes':>7} {'compact':>8}")
    for v, r in rows:
        print(f"{v.name:52} {r['chars']:>13,} {r['cost']:>10,.0f} {(r['cost'] - base) / base:>+9.0%} {r['prunes']:>7} {r['compactions']:>8}")


if __name__ == "__main__":
    main()
