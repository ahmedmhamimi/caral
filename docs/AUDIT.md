# Audit report

Method: reproduce first, fix second, lock with a test. Every item below has a regression test.
The suite is 326 tests (~12 s): unit, property-style (chunking invariance), end-to-end against a
fake HTTP server, real-terminal (pty) checks, and architecture-fitness tests.

## A. Defects found in the previous version (all fixed)

| # | Severity | Defect (reproduced) | Fix |
|---|---|---|---|
| 1 | **Security** | `ls & rm -rf build` classified *safe* (single `&` chains a command) -> auto-ran | `&`, `\|&` split segments |
| 2 | **Security** | `LD_PRELOAD=x ls`, `GIT_EXTERNAL_DIFF=... git diff` classified safe | any env assignment => not safe |
| 3 | **Security** | `tree -o f`, `rg --pre cmd`, `uniq in out`, `date -s`, `sort -o`, `find -fprintf` classified safe | per-command flag checks |
| 4 | **Security** | Blanket allow-rule for interpreters (`python`, `bash -c`, `env`, `xargs`) after one "always" | interpreters match the exact command only |
| 5 | **Security** | Symlink inside project pointing at `~/.ssh` / outside root bypassed path rules | `realpath` in all path checks |
| 6 | **Privacy** | `.env` files read silently (secrets sent to the model provider) | `.env*` is credential-like: asks (not `.env.example`) |
| 7 | **Security** | `sudo` / writes to `~/.ssh` via shell allowed in yolo | always ask (sudo, credential paths) |
| 8 | **Security** | Hostile repo config could run code via `git status` fsmonitor during prompt build | internal git calls disable fsmonitor/hooks |
| 9 | **Security** | web_fetch could hit cloud metadata (169.254.169.254) | link-local blocked |
| 10 | **Data loss** | Editing a Latin-1 file rewrote `é` as U+FFFD | strict UTF-8; refuse to edit non-UTF-8 |
| 11 | Data loss | Non-atomic writes (crash = truncated file); BOM dropped | temp file + fsync + rename; BOM/CRLF preserved; mode kept; symlinks written through |
| 12 | Correctness | Stale-file check used mtime only (false "changed" after formatters; misses same-mtime edits) | size+mtime fast path, SHA-1 confirmation |
| 13 | **Resource** | `yes \| head -c 200MB` -> 789 MB RSS | bounded capture (head+tail, 120 KB) -> <150 MB growth |
| 14 | Correctness | `server &` kept the pipe open: call "timed out" and killed the server | exit-aware reader with grace period |
| 15 | Concurrency | Compaction mutated shared `cfg.max_tokens` (race with parallel subagents) | per-call `max_tokens`/`model` params |
| 16 | Correctness | `-p` mode returned exit 0 and empty output on API error | typed `RunResult`, exit codes 0/1/2/3/130 |
| 17 | Robustness | Crash mid-tool left a dangling tool_use -> provider 400 forever on resume | validate + repair on load and after errors |
| 18 | Robustness | Context-length error ended the session | auto-compact + one retry, emergency drop fallback |
| 19 | Robustness | Mid-stream overload error not retried | retry when nothing was shown yet |
| 20 | Robustness | Servers rejecting `stream_options` / `max_completion_tokens` failed outright | runtime capability fallback, remembered |
| 21 | Performance | Session rewritten in full every step (O(n^2) I/O) | append-only JSONL + snapshot on compaction |
| 22 | UX | ANSI codes in the `input()` prompt corrupted readline cursor math | zero-width markers |
| 23 | UX | `/help` crash (found by the new e2e test); a failing slash command could kill the REPL | fixed; dispatch is exception-safe |
| 24 | Cost | Pruning at 50% *increased* cost 5x in an ablation (it invalidates the cached prefix and dedupe) | later thresholds (70/85) and targeted invalidation |
| 26 | **UX/blocker** | Only `GROQ_API_KEY` set, yet Caral demanded `ANTHROPIC_API_KEY` (hard-coded default provider; reported by a user) | provider auto-detected from the first key found; errors name the key you do have |
| 27 | **UX/blocker** | Groq preset pointed at a model Groq no longer serves; the 404 gave no way to discover valid ids (reported by a user) | preset updated; `caral --models`, `caral --doctor --online`, `/models`; the 404 hint names them |
| 25 | UX | Diff printed twice; `a//abs/path` headers; outcome glued to prompt | fixed, tested in a real pty |

## B. Cost ablation (real engine, scripted model)

`python bench/cost_bench.py` - 30 steps re-reading 8 files, 30k-token window.
Prompt chars are measured; price is modelled (fresh 1.0x, cache write 1.25x, cache read 0.1x).

```
configuration                                         prompt chars  rel. cost  vs naive  prunes  compact
naive harness                                            2,647,013    735,281       +0%      10        0
+ prompt caching                                         2,647,763    371,362      -49%      10        0
+ read dedupe                                            2,497,093     98,185      -87%       0        0
+ prune/compact @70/85 (Caral default)                   1,865,969    187,971      -74%       6        0
  eager prune @50/75 (old default, for contrast)         1,570,516    250,815      -66%      17        0
```

Reading it honestly: caching and read-dedupe do the heavy lifting (-87%). Pruning/compaction is a
safety valve that costs something whenever the window genuinely fills; the 70/85 defaults cost
less than the old eager 50/75 (-74% vs -66%). Without a safety valve a session of this size would
overflow a real 30k window and fail. Real-world savings depend on the model's re-read habits.

## C. Known limitations (stated, not hidden)

- **Bash is not sandboxed.** The classifier is a fail-closed heuristic against accidents, not a
  security boundary. A hostile repo can still run code if you approve a command or run it in yolo.
  The model running `git status` in a repo with a malicious `.git/config` (fsmonitor) is covered
  only for Caral's own calls.
- Never exercised against live provider APIs in this build environment (no keys): wire formats are
  tested against mocked and fake servers following the public API shapes.
- Windows paths are written but untested. Token counts are estimates (self-calibrating).
- Streaming Markdown does not render tables, links or word-wrap; ripgrep is optional.
- Not implemented: MCP, hooks, image input, extended thinking, OS sandbox.
