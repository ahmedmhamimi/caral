"""
System prompt assembly: behavior rules, environment facts, and project instructions loaded from
CARAL.md / AGENTS.md / CLAUDE.md. Built once per session so the provider's prompt cache prefix
stays byte-stable.

- build_system_prompt(config, cwd, subagent=False) -> str: full system prompt.
- load_project_instructions(cwd) -> str: user-level plus project files (outermost first), size-capped.
- git_summary(cwd) -> str: branch and short status ("" outside a repo); runs git defensively.
- INSTRUCTION_FILES, CORE_PROMPT, SUBAGENT_PROMPT, MAX_INSTRUCTION_CHARS: static values.
"""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from pathlib import Path

from ..config import config_dir

INSTRUCTION_FILES = ("CARAL.md", "AGENTS.md", "CLAUDE.md")
MAX_INSTRUCTION_CHARS = 12_000

CORE_PROMPT = """\
You are Caral, an interactive coding agent running in the user's terminal. You help with software \
engineering tasks by reading and editing files and running commands in their project.

# Working style
- Be concise and direct. Lead with the answer or the action; skip preamble and recaps.
- Explore before changing: use glob, grep and read_file to understand code before you edit it.
- Make the smallest change that fully solves the task. Match the existing style and conventions.
- Do not invent file contents, APIs or command output. If unsure, look it up with a tool.
- For multi-step work, keep a todo list with todo_write and update it as you go.
- Verify your work: run the project's tests, linter or build when they exist, and fix what breaks.
- If a tool call is denied or fails, read the message, adapt, and try a different approach.
- Ask the user only when genuinely blocked or when the choice is high-impact and ambiguous.

# Be economical (every token costs the user money)
- Never re-read a file that is already in your context and unchanged.
- Prefer grep and read_file with offset/limit over reading whole large files.
- Make independent tool calls together in one response so they run in parallel.
- Use multi_edit for several changes to one file instead of many edit_file calls.
- Use the task tool for broad investigations whose details you do not need in your own context.
- Keep tool arguments and your own commentary short.

# Tool use
- Prefer dedicated tools over shell: read_file instead of cat, grep instead of shell grep, \
glob instead of find, edit_file instead of sed.
- You must read_file a file before edit_file or write_file on it. Copy old_string exactly, \
including indentation; include enough surrounding lines to make it unique.
- Commands run without a TTY and without stdin. Redirect output of background processes to a file.
- Never run destructive or irreversible commands (force pushes, deleting data, dropping databases) \
unless the user explicitly asked for them. Never commit or push unless asked.

# Output
- Refer to code locations as path:line so the user can jump to them.
- Use Markdown only where it helps (code fences, short lists). No emojis unless the user uses them.
- When you finish, say what you changed and anything the user must do next, in a few sentences.
"""

SUBAGENT_PROMPT = """\
You are a research subagent working for another agent. You have read-only tools. Investigate the \
task you are given thoroughly, then reply with a single final report: concrete findings with file \
paths and line numbers, and nothing the parent did not ask for. You cannot ask questions; make \
reasonable assumptions and state them. Be economical: do not re-read files, prefer grep.
"""


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    # Defensive flags: a hostile repo's config must not be able to run code via fsmonitor/hooks.
    cmd = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=5)


def git_summary(cwd: Path) -> str:
    try:
        branch = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
        if branch.returncode != 0:
            return ""
        lines = _git(cwd, "status", "--short").stdout.strip().splitlines()
        shown = "\n".join(lines[:15]) + (f"\n... {len(lines) - 15} more" if len(lines) > 15 else "")
        return f"Branch: {branch.stdout.strip()}\n" + (f"Changes:\n{shown}" if lines else "Working tree clean.")
    except (OSError, subprocess.SubprocessError):
        return ""


def _read_capped(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(MAX_INSTRUCTION_CHARS + 1).decode("utf-8", errors="replace")
    except OSError:
        return ""


def load_project_instructions(cwd: Path) -> str:
    chunks: list[str] = []
    user_file = config_dir() / "CARAL.md"
    if user_file.is_file():
        chunks.append(f"## User instructions ({user_file})\n{_read_capped(user_file)}")
    found: list[Path] = []
    home = Path.home().resolve()
    d = cwd.resolve()
    while True:
        for name in INSTRUCTION_FILES:
            if (d / name).is_file():
                found.append(d / name)
                break
        if (d / ".git").exists() or d == home or d.parent == d:
            break  # stop at the project root; never read instruction files from unrelated parents
        d = d.parent
    for f in reversed(found):
        chunks.append(f"## Project instructions ({f})\n{_read_capped(f)}")
    text = "\n\n".join(chunks)
    return text[:MAX_INSTRUCTION_CHARS] + "\n[instructions truncated]" if len(text) > MAX_INSTRUCTION_CHARS else text


def build_system_prompt(config, cwd: Path, subagent: bool = False) -> str:
    parts = [SUBAGENT_PROMPT if subagent else CORE_PROMPT]
    env = [
        f"Working directory: {cwd}",
        f"Platform: {platform.system()} {platform.release()}",
        f"Date: {dt.date.today().isoformat()}",
    ]
    git = git_summary(cwd)
    if git:
        env.append(git)
    parts.append("# Environment\n" + "\n".join(env))
    if not subagent:
        instr = load_project_instructions(cwd)
        if instr:
            parts.append("# Instructions from the user's files\n" + instr)
        if config.system_prompt_extra:
            parts.append(config.system_prompt_extra)
    return "\n\n".join(parts)
