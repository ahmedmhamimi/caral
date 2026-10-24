# Caral

A provider-agnostic terminal coding agent. Bring any API key.

Caral reads and edits your code, runs commands and searches your project from the terminal, using
the LLM of your choice. It asks before changing anything, shows a diff first, and can undo what it did.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/ahmedmhamimi/caral.git
cd caral
pip install -e .
```

## Quick start

```bash
export GROQ_API_KEY=...      # or ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, ...
caral --doctor --online      # checks your setup, key and model
cd /path/to/your/project
caral
```

The provider is detected from whichever API key is set. Force one with `--provider`, for example
`caral --provider openai --model gpt-4.1`. Run `caral --models` to list the models your key can use.

For a local model with no API key: `caral --provider ollama --model qwen2.5-coder:14b`.
Any OpenAI-compatible server works with `--base-url http://host:port/v1 --model <id>`.

Supported providers: `anthropic`, `openai`, `openrouter`, `groq`, `deepseek`, `together`, `ollama`,
`lmstudio`.

## Usage

```bash
caral                                        # interactive session
caral "fix the failing test"                 # interactive, starting with a prompt
caral -p "summarize this repo" --mode plan   # one-shot: print the answer and exit
git diff | caral -p "review this diff"       # piped input is added to the prompt
caral -c                                     # continue the latest session in this directory
```

When Caral wants to edit a file or run a command it asks, and you answer with one key:
`y` allow, `n` deny, `a` always allow this kind of action for the session, `f` deny and explain why.

### Slash commands

| Command | Does |
|---|---|
| `/help` | list all commands |
| `/mode plan\|ask\|auto-edit\|yolo` | change the permission mode |
| `/undo` | revert file changes from the last turn |
| `/compact`, `/clear` | free up context, or start fresh |
| `/models`, `/model <id>` | list models your key can use, or switch model |
| `/usage` (`/cost`) | tokens, estimated spend, per-model breakdown, cache-hit rate, context use |
| `/prices [list\|<model>\|update]` | show the price Caral uses for a model, list the table, or refresh it from OpenRouter |
| `/meter on\|off` | show or hide the token/cost line printed after every model call |
| `/trace` | summary of this session's trace |
| `/sessions`, `/resume <id>` | list or reload saved sessions |
| `/init` | have Caral write a `CARAL.md` for your project |

Ctrl-C interrupts a running turn. End a line with `\` to continue it on the next line.

## Permission modes

| Mode | Behavior |
|---|---|
| `plan` | read-only |
| `ask` (default) | reads and safe commands run; edits and other commands ask first |
| `auto-edit` | edits inside the project are automatic; commands still ask |
| `yolo` | everything runs except hard-denied operations |

Credential files (`~/.ssh`, `.env`, cloud credentials) always ask, and destructive commands such as
`rm -rf /` are always denied. Caral is not a sandbox: commands run on your machine with your
permissions, so review what you approve.

## Live spend and colors

Every model call prints a counter line (tokens in / cached / out, that call's cost, then this prompt's
and this launch's running total). The spinner shows the running cost while you wait, the prompt shows
the launch total, and each run ends with a summary that includes cost. Amounts turn amber and red as
they grow. Tool lines are colored by what they do (read, edit, run, network, subagent), tokens by kind,
diffs green/red, headings and code in Markdown replies each get their own hue.

```
  ▸ read_file  src/background.js
    ✓ ok · 4ms · 212 lines
  ◆ in 1.4k │ cache 5.0k │ out 200 │ $0.0060 │ run $0.0060 · total $0.0210
✓ 2 steps · 1 tool call · 4.2s · 1.6k tokens · $0.0060 · 78% cached · context 25%
```

- **Costs are estimates**: tokens reported by the provider times list prices. Your provider's usage
  page is the bill. Cache reads and writes are priced separately; prompts above a model's
  long-context threshold use its higher tier where one is known.
- **Prices**: a built-in table (checked 2026-10-20) covers the models listed by `/prices list`. Anything
  else shows `cost n/a` (tokens are still counted); Caral never borrows a similar model's price.
  Add or fix entries in `~/.config/caral/prices.json` (USD per 1M tokens, cache fields optional;
  override wins over built-in):

  ```json
  { "deepseek-chat": { "input": 0.27, "output": 1.10, "cache_read": 0.07 },
    "my-local-gateway-model": { "input": 0, "output": 0 } }
  ```

  `caral --update-prices` (or `/prices update`) fetches OpenRouter's public list to fill gaps. OpenRouter
  can price a model differently from the vendor's own API, so built-in and override prices take priority.
- **OpenRouter**: if your model has no known price, Caral fetches OpenRouter's public price list once at
  startup (4 s timeout, cached for later launches), so costs work out of the box there.
- **Local runners** (`ollama`, `lmstudio`) are shown as free. A custom provider on localhost is not
  assumed free (it may proxy a paid API): give its model a zero price as above.
- `caral -p ... --output-format json` includes `cost_usd` and `unpriced_models`.
- Colors: 256-color palette when the terminal advertises it, otherwise the 16 base colors. Force with
  `CARAL_COLORS=basic|256`; `NO_COLOR=1` or a pipe turns color off. Every status also has a symbol and a word.

## Prompt caching

Anthropic's API caches automatically-marked prefixes, and so do OpenAI and DeepSeek by themselves. Claude
models behind an OpenAI-style gateway (OpenRouter) cache only where the request marks breakpoints, so
Caral marks the system message, the latest user message and the newest message for them. The `cache` figure
in the counter line should climb from the second call on; if it stays at 0, caching is not happening.
Settings in `~/.config/caral/config.json`: `"cache_markers": "auto"` (default: Claude on OpenRouter) | `"on"`
(any gateway that fronts Claude, e.g. a LiteLLM proxy) | `"off"`; `"prompt_cache": false` disables all of it.
If a server rejects the markers, Caral steps down (fewer breakpoints, then none) and remembers for the session.

## Configuration

Optional file `~/.config/caral/config.json`:

```json
{
  "provider": "groq",
  "model": "openai/gpt-oss-120b",
  "permission_mode": "ask",
  "allow_commands": ["pytest"],
  "deny_commands": ["git push"]
}
```

Put guidance for the agent in a `CARAL.md` file in your project. Sessions and traces are stored in
`~/.local/share/caral/`.

More detail is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/AUDIT.md`](docs/AUDIT.md).

## Development

```bash
pip install -e '.[dev]'
pytest
```
