# Caral architecture

## Decision: hexagonal core + typed event stream

An agent harness has a tiny, stable core (a loop) surrounded by things that change constantly:
LLM providers, tools, terminals, storage, policies. The architecture is chosen to make every
one of those a plug, and to make everything the core does observable.

**Ports and adapters (hexagonal).** The engine depends on three narrow interfaces (`ports.py`):
`LLM`, `Approver`, `Store`. Concrete classes live in `adapters/`. Swapping Anthropic for a local
model, a terminal for a web UI, or JSONL files for SQLite never touches the engine.

**Event-sourced observability.** The engine never prints and never logs ad hoc. It emits typed
events (`domain/events.py`) on a bus. The terminal UI, the JSONL trace, `--output-format jsonl`
and the tests are just subscribers. One mechanism gives traceability, a replaceable front-end
and testability.

**Policy as an ordered rule chain.** Permissions are a list of small functions
(`engine/permissions.py`); the first that returns a decision wins. Adding behaviour means
inserting a rule, not editing a 100-line function.

**Fail closed, repair on load.** Conversation invariants (every tool_use answered) are
checked and repaired (`domain/conversation.py`) so a crash can never wedge a session.

## Layers and the one-way dependency rule

```
cli.py / app.py / commands.py      composition root + shell (may import anything)
        |
adapters/  providers(+price_fetch)  tools  store(+prices)  ui     concrete I/O; siblings never import each other
        |
engine/    agent  executor  permissions  context  loopguard  bus  prompt  meter
        |
domain/    messages  events  pricing  tools(contract)  conversation  shellsafety  tokens  redact ...
errors.py  config.py                        leaves
```

Rules are enforced by `tests/test_architecture.py` (AST-based), so they cannot rot:
domain imports only domain+errors; engine imports only domain/engine/ports/errors/config;
adapters never import the engine or each other; nothing below the shell imports `app/cli`;
no `print` outside UI/CLI; every module has a docstring listing its public API.

## Extension points (how to add things)

| To add... | Do this | Touches |
|---|---|---|
| A tool | Subclass `Tool`, register in `default_registry()` or expose via the `caral.tools` entry-point group | one file |
| An LLM provider | Subclass `Provider` (stream + complete), add to `create_provider` | one file |
| A permission rule | `policy.add_rule(fn)` | one function |
| A front-end (web, IDE) | Subscribe to the bus, implement `Approver` | one adapter |
| A slash command | `registry.register(SlashCommand(...))` | one function |
| A metric / audit sink | `bus.subscribe(fn)` | one function |
| A storage backend | Implement the `Store` port | one adapter |

## Traceability

Every run has a `run_id`; every tool call a `call_id`. The trace
(`~/.local/share/caral/traces/<session>.jsonl`) records each model call (tokens, cache hits,
latency), tool request, permission decision (and who made it), tool result summary, retry,
prune and compaction, with secrets redacted. `caral --trace` / `/trace` summarise it.
Sessions are append-only JSONL, so the exact conversation is reproducible.

## Cost architecture

1. **Prompt-cache stability** - system prompt built once per session, tool schemas cached and ordered,
   cache breakpoints on system/tools/newest message (3 of the 4 allowed). Claude behind OpenRouter gets
   equivalent breakpoints (system, latest user message, newest message) on the OpenAI-style wire, with
   step-down fallback if the gateway rejects them.
2. **Read dedupe** - re-reading an unchanged range already in context returns a one-line notice.
3. **Right-sized output** - tool output cap 16k chars, read cap 40k chars/1000 lines, grep 100 lines,
   edits return a one-line result (the diff is UI-only and never re-sent to the model).
4. **Cheap model for cheap work** - `small_model` runs summarization and read-only subagents.
5. **Pruning/compaction only when the window actually fills** (70% / 85%). Both rewrite the cached
   prefix, so eager use is a net loss (measured, see AUDIT.md).
6. **Waste guards** - loop detector (warn at 3 identical repeats, stop at 5), per-run token budget,
   one automatic recovery from context-length errors, no retry of auth/bad-request errors.
7. **Self-calibrating token estimates** - chars/token adapts to real provider usage.
8. **Web fetch cache** - 15-minute per-session cache.

## Spend metering

`domain/pricing` is pure data and arithmetic (price table, layered lookup, cost of a `Usage`).
`engine/meter.CostMeter` is just another bus subscriber: it prices every `ModelResponded`,
`SubagentFinished` (carries the model) and `AuxUsage` (context summaries) event, so the agent never
knows about money. The app subscribes it before any renderer, so UIs always read current totals.
Price files (overrides, fetched cache) live in `adapters/store/prices`; the OpenRouter fetch lives in
`adapters/providers/price_fetch`. Lookup is exact-id only: an unknown model is reported as unpriced
instead of inheriting a sibling's rate. Known approximations: list prices only (no discounts, batch or
off-peak rates), subagent usage is priced at base rates (no long-context tier), and a resumed session's
earlier spend is not included (the meter counts since launch).

## What is deliberately not here (yet)

MCP client, hooks, OS-level sandboxing, image input, extended-thinking blocks, Windows support
(the shell tool and key reader have Windows branches that are untested).
