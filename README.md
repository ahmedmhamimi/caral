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
| `/usage` | tokens, cache-hit rate and context use |
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
