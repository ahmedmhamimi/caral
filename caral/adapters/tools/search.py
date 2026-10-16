"""
Search tools. glob walks the tree with directory pruning (never descends into node_modules, .git,
...) and supports ** and {a,b}; a pattern without "/" matches at any depth. grep uses ripgrep
when installed (fast, honors .gitignore) and a bounded pure-Python fallback otherwise.

- GlobTool: glob(pattern, path=".") -> files, newest first, capped.
- GrepTool: grep(pattern, path, glob, case_insensitive, output_mode, context, head_limit=100).
- glob_to_regex(pattern) -> re.Pattern: translates a glob to an anchored regex over POSIX relative paths.
- walk_files(base) -> Iterator[Path]: pruned directory walk.
- _rg_search(args, base, ctx) -> ToolResult | None; _py_search(args, base, ctx) -> ToolResult.
- _limit(lines, head_limit) -> str: caps output lines and notes the remainder.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from ...domain.tools import IGNORE_DIRS, Tool, ToolContext, ToolResult, resolve_path

MAX_FILE_BYTES = 2_000_000
MAX_SCAN_FILES = 200_000
MAX_LINE_FOR_REGEX = 2000


def _rel(path: Path, ctx: ToolContext) -> str:
    try:
        return str(path.relative_to(ctx.cwd))
    except ValueError:
        return str(path)


def walk_files(base: Path) -> Iterator[Path]:
    for root, dirs, names in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for n in sorted(names):
            yield Path(root) / n


def _glob_body(pat: str) -> str:
    out, i, n = [], 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
                continue
            if pat.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pat.find("]", i + 2)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pat[i + 1 : j]
                body = "^" + body[1:] if body.startswith("!") else body
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        elif c == "{":
            j = pat.find("}", i)
            if j == -1:
                out.append(re.escape(c))
            else:
                out.append("(?:" + "|".join(_glob_body(a) for a in pat[i + 1 : j].split(",")) + ")")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def glob_to_regex(pattern: str) -> re.Pattern:
    pattern = pattern.replace("\\", "/").lstrip("./") if pattern.startswith("./") else pattern.replace("\\", "/")
    prefix = "(?:.*/)?" if "/" not in pattern else ""
    return re.compile("^" + prefix + _glob_body(pattern) + "$")


class GlobTool(Tool):
    name = "glob"
    category = "read"
    description = (
        "Find files by glob pattern, e.g. '**/*.py', 'src/**/*.{ts,tsx}'. A pattern without '/' "
        "(like '*.py') matches at any depth. Results are newest first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Directory to search. Default: cwd."},
        },
        "required": ["pattern"],
    }

    def summary(self, args: dict) -> str:
        return str(args.get("pattern", ""))

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        base = resolve_path(ctx, args.get("path") or ".")
        if not base.is_dir():
            return ToolResult(f"Not a directory: {base}", True)
        try:
            rx = glob_to_regex(args["pattern"])
        except re.error as e:
            return ToolResult(f"Bad glob pattern: {e}", True)
        hits: list[tuple[float, Path]] = []
        scanned = 0
        for p in walk_files(base):
            scanned += 1
            if scanned > MAX_SCAN_FILES:
                break
            if rx.match(p.relative_to(base).as_posix()):
                try:
                    hits.append((p.stat().st_mtime, p))
                except OSError:
                    continue
        if not hits:
            return ToolResult("No files matched.")
        hits.sort(key=lambda t: t[0], reverse=True)
        lines = [_rel(p, ctx) for _, p in hits[:150]]
        if len(hits) > 150:
            lines.append(f"[{len(hits) - 150} more not shown; narrow the pattern]")
        return ToolResult("\n".join(lines))


class GrepTool(Tool):
    name = "grep"
    category = "read"
    description = (
        "Search file contents with a regular expression. output_mode: 'content' (file:line:text), "
        "'files' (names only) or 'count'. Use glob to restrict file types."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex to search for."},
            "path": {"type": "string", "description": "File or directory. Default: cwd."},
            "glob": {"type": "string", "description": "Only files matching this glob, e.g. '*.py'."},
            "case_insensitive": {"type": "boolean"},
            "output_mode": {"type": "string", "enum": ["content", "files", "count"]},
            "context": {"type": "integer", "description": "Lines of context around matches."},
            "head_limit": {"type": "integer", "description": "Max output lines. Default 100."},
        },
        "required": ["pattern"],
    }

    def summary(self, args: dict) -> str:
        return f"{args.get('pattern', '')!r} in {args.get('path', '.')}"

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        base = resolve_path(ctx, args.get("path") or ".")
        if not base.exists():
            return ToolResult(f"Path not found: {base}", True)
        try:
            re.compile(args["pattern"])
        except re.error as e:
            return ToolResult(f"Invalid regex: {e}", True)
        return _rg_search(args, base, ctx) or _py_search(args, base, ctx)


def _limit(lines: list[str], head_limit: int) -> str:
    if not lines:
        return "No matches found."
    if len(lines) > head_limit:
        extra = len(lines) - head_limit
        lines = lines[:head_limit] + [f"[{extra} more lines not shown; narrow your search]"]
    return "\n".join(lines)


def _rg_search(args: dict, base: Path, ctx: ToolContext) -> ToolResult | None:
    rg = shutil.which("rg")
    if not rg:
        return None
    mode = args.get("output_mode") or "content"
    cmd = [rg, "--no-heading", "--color", "never", "--max-columns", "200", "--max-columns-preview"]
    if args.get("case_insensitive"):
        cmd.append("-i")
    if args.get("glob"):
        cmd += ["-g", args["glob"]]
    if mode == "files":
        cmd.append("-l")
    elif mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")
        if args.get("context"):
            cmd += ["-C", str(min(10, int(args["context"])))]
    cmd += ["-e", args["pattern"], "--", str(base)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60, cwd=ctx.cwd)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode == 2 and not proc.stdout:
        return ToolResult(f"grep failed: {proc.stderr.strip()[:300]}", True)
    prefix = str(ctx.cwd) + os.sep
    lines = [l.replace(prefix, "", 1) for l in proc.stdout.splitlines()]
    return ToolResult(_limit(lines, int(args.get("head_limit") or 100)))


def _py_search(args: dict, base: Path, ctx: ToolContext) -> ToolResult:
    rx = re.compile(args["pattern"], re.IGNORECASE if args.get("case_insensitive") else 0)
    mode = args.get("output_mode") or "content"
    ctx_lines = min(10, max(0, int(args.get("context") or 0)))
    inc = args.get("glob")
    out: list[str] = []
    files = [base] if base.is_file() else (
        p for p in walk_files(base) if not inc or fnmatch.fnmatch(p.name, inc)
    )
    scanned = 0
    for f in files:
        scanned += 1
        if scanned > MAX_SCAN_FILES:
            break
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
            raw = f.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        lines = raw.decode("utf-8", errors="replace").split("\n")
        # Bound regex cost on pathological minified lines (ReDoS / huge-line protection).
        hits = [i for i, l in enumerate(lines) if len(l) <= MAX_LINE_FOR_REGEX and rx.search(l)]
        if not hits:
            continue
        rel = _rel(f, ctx)
        if mode == "files":
            out.append(rel)
        elif mode == "count":
            out.append(f"{rel}:{len(hits)}")
        else:
            shown: set[int] = set()
            hit_set = set(hits)
            for h in hits:
                for i in range(max(0, h - ctx_lines), min(len(lines), h + ctx_lines + 1)):
                    if i not in shown:
                        shown.add(i)
                        sep = ":" if i in hit_set else "-"
                        out.append(f"{rel}{sep}{i + 1}{sep}{lines[i][:200]}")
    return ToolResult(_limit(out, int(args.get("head_limit") or 100)))
