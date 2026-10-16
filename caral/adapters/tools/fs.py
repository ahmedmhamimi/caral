"""
Filesystem tools: read, write, exact-string edit, atomic multi-edit, directory listing. Safety
properties: read-before-write, stale-file detection by content hash, refusal to rewrite non-UTF-8
files, atomic writes preserving permissions, and read dedupe (an unchanged file already in
context is not re-sent, which saves tokens).

- ReadFileTool: read_file(path, offset=1, limit=1000) -> numbered lines with paging hints.
- WriteFileTool: write_file(path, content). EditFileTool: edit_file(path, old_string, new_string, replace_all).
- MultiEditTool: multi_edit(path, edits) -> atomic batch of edits. ListDirTool: list_dir(path, depth).
- EditError: edit could not be applied.
- apply_edit(text, old, new, replace_all=False) -> (str, str): applies one edit with whitespace-tolerant fallback.
- MAX_READ_CHARS / MAX_LINE_CHARS: output caps that bound token cost.
- _shown(path, ctx) -> str: display path for diffs (relative to cwd when inside it).
"""

from __future__ import annotations

import difflib
from pathlib import Path

from ...domain.textutil import make_diff
from ...domain.tools import IGNORE_DIRS, Tool, ToolContext, ToolResult, resolve_path
from ._fileio import NotTextError, looks_binary, read_text_file, write_text_file

MAX_READ_BYTES = 4_000_000
MAX_READ_CHARS = 40_000
MAX_LINE_CHARS = 1000


class EditError(Exception):
    pass


def _shown(path: Path, ctx: ToolContext) -> str:
    """Path as displayed in diffs: relative to the working directory when inside it."""
    try:
        return str(path.relative_to(ctx.cwd))
    except ValueError:
        return str(path)


def _similar(path: Path) -> list[str]:
    if not path.parent.is_dir():
        return []
    return difflib.get_close_matches(path.name, [p.name for p in path.parent.iterdir()], n=3, cutoff=0.5)


class ReadFileTool(Tool):
    name = "read_file"
    category = "read"
    description = (
        "Read a text file. Returns lines prefixed with line numbers. Use offset/limit to page through "
        "large files. Re-reading an unchanged range already in context returns a short notice. "
        "You must read a file before editing or overwriting it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (absolute or relative to cwd)."},
            "offset": {"type": "integer", "description": "1-based first line. Default 1."},
            "limit": {"type": "integer", "description": "Max lines. Default 1000."},
        },
        "required": ["path"],
    }

    def summary(self, args: dict) -> str:
        return str(args.get("path", ""))

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args["path"])
        if not path.exists():
            hint = _similar(path)
            return ToolResult(f"File not found: {path}." + (f" Did you mean: {', '.join(hint)}?" if hint else ""), True)
        if path.is_dir():
            return ToolResult(f"{path} is a directory. Use list_dir instead.", True)
        offset = max(1, int(args.get("offset") or 1))
        limit = min(5000, max(1, int(args.get("limit") or 1000)))
        if ctx.already_viewed(path, offset, limit):
            ctx.mark_read(path)
            return ToolResult(
                f"{path} is unchanged since you read lines {offset}-{offset + limit - 1}; its contents are "
                "already in your context above. Use them, or read a different range."
            )
        try:
            with open(path, "rb") as f:
                raw = f.read(MAX_READ_BYTES + 1)
        except OSError as e:
            return ToolResult(f"Cannot read {path}: {e}", True)
        if looks_binary(raw):
            return ToolResult(f"{path} looks like a binary file; refusing to print it.", True)
        too_big = len(raw) > MAX_READ_BYTES
        text = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        lossy = "\ufffd" in text and b"\xef\xbf\xbd" not in raw
        lines = text.replace("\r\n", "\n").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        total = len(lines)
        ctx.mark_read(path)
        if total == 0:
            return ToolResult("(file is empty)")
        if offset > total:
            return ToolResult(f"offset {offset} is past the end of the file ({total} lines).", True)
        out, chars, end = [], 0, offset - 1
        for i, line in enumerate(lines[offset - 1 : offset - 1 + limit], start=offset):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + f"... [line truncated, {len(line)} chars]"
            row = f"{i:>6}\t{line}"
            if chars + len(row) > MAX_READ_CHARS and out:
                break
            out.append(row)
            chars += len(row) + 1
            end = i
        footer = ""
        if end < total:
            footer = f"\n\n[showing lines {offset}-{end} of {total}; use offset={end + 1} to continue]"
        elif too_big:
            footer = "\n\n[file larger than 4MB; output cut off]"
        if lossy:
            footer += "\n[note: file is not valid UTF-8; shown lossily and it cannot be edited with these tools]"
        if end == offset - 1 + min(limit, total - offset + 1):
            ctx.remember_view(path, offset, limit)
        return ToolResult("\n".join(out) + footer)


class WriteFileTool(Tool):
    name = "write_file"
    category = "edit"
    description = (
        "Create a new file or completely overwrite an existing one. Prefer edit_file for changes to "
        "existing files. Existing files must be read with read_file first."
    )
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string", "description": "Full file contents."}},
        "required": ["path", "content"],
    }

    def summary(self, args: dict) -> str:
        return f"{args.get('path', '')} ({len(str(args.get('content', '')).splitlines())} lines)"

    def preview(self, args: dict, ctx: ToolContext) -> str | None:
        path = resolve_path(ctx, args["path"])
        old = read_text_file(path).text if path.is_file() else ""
        return make_diff(old, args["content"], _shown(path, ctx))

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args["path"])
        content = args["content"]
        if path.is_dir():
            return ToolResult(f"{path} is a directory.", True)
        old, crlf, bom, existed = "", False, False, path.exists()
        if existed:
            err = ctx.check_fresh(path)
            if err:
                return ToolResult(err, True)
            try:
                tf = read_text_file(path)
            except NotTextError as e:
                return ToolResult(str(e), True)
            old, crlf, bom = tf.text, tf.crlf, tf.bom
        try:
            ctx.backup(path)
            write_text_file(path, content, crlf, bom)
        except OSError as e:
            return ToolResult(f"Cannot write {path}: {e}", True)
        ctx.mark_read(path)
        ctx.invalidate_views()
        return ToolResult(
            f"{'Overwrote' if existed else 'Created'} {path} ({len(content.splitlines())} lines).",
            display=make_diff(old, content, _shown(path, ctx)),
        )


def _flexible_spans(text: str, old: str) -> list[tuple[int, int]]:
    old_lines = [l.strip() for l in old.strip("\n").split("\n")]
    if not old_lines or not any(old_lines):
        return []
    text_lines = text.split("\n")
    offsets, pos = [], 0
    for l in text_lines:
        offsets.append(pos)
        pos += len(l) + 1
    n = len(old_lines)
    return [
        (offsets[i], offsets[i + n - 1] + len(text_lines[i + n - 1]))
        for i in range(len(text_lines) - n + 1)
        if all(text_lines[i + j].strip() == old_lines[j] for j in range(n))
    ]


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent(new: str, old: str, matched: str) -> str:
    """Map indentation levels used in old/new onto the indentation the file actually has."""
    mapping: dict[str, str] = {}
    for ol, ml in zip(old.strip("\n").split("\n"), matched.split("\n")):
        if ol.strip():
            mapping.setdefault(_indent(ol), _indent(ml))
    out = []
    for line in new.split("\n"):
        ind = _indent(line)
        out.append(mapping[ind] + line[len(ind):] if line.strip() and ind in mapping else line)
    return "\n".join(out)


def apply_edit(text: str, old: str, new: str, replace_all: bool = False) -> tuple[str, str]:
    if old == "":
        raise EditError("old_string must not be empty. Use write_file to create files.")
    if old == new:
        raise EditError("old_string and new_string are identical; nothing to change.")
    count = text.count(old)
    if count == 1 or (count > 1 and replace_all):
        return text.replace(old, new), f"replaced {count} occurrence(s)"
    if count > 1:
        raise EditError(
            f"old_string appears {count} times. Add surrounding context to make it unique, "
            "or set replace_all=true to change every occurrence."
        )
    spans = _flexible_spans(text, old)
    if len(spans) == 1:
        s, e = spans[0]
        adj = new.strip("\n") if old != old.strip("\n") else new
        return text[:s] + _reindent(adj, old, text[s:e]) + text[e:], "matched ignoring whitespace differences"
    if len(spans) > 1:
        raise EditError(f"old_string matches {len(spans)} places when ignoring whitespace; add more context.")
    first = old.strip().split("\n")[0].strip()
    near = difflib.get_close_matches(first, [l.strip() for l in text.split("\n")], n=1, cutoff=0.6)
    raise EditError("old_string was not found in the file. Re-read it and copy the text exactly."
                    + (f" Closest line in file: {near[0]!r}." if near else ""))


def _load_for_edit(args: dict, ctx: ToolContext):
    path = resolve_path(ctx, args["path"])
    if not path.is_file():
        return path, None, ToolResult(f"File not found: {path}", True)
    err = ctx.check_fresh(path)
    if err:
        return path, None, ToolResult(err, True)
    try:
        return path, read_text_file(path), None
    except NotTextError as e:
        return path, None, ToolResult(str(e), True)


def _commit(path: Path, tf, new_text: str, ctx: ToolContext, label: str) -> ToolResult:
    try:
        ctx.backup(path)
        write_text_file(path, new_text, tf.crlf, tf.bom)
    except OSError as e:
        return ToolResult(f"Cannot write {path}: {e}", True)
    ctx.mark_read(path)
    ctx.invalidate_views()
    return ToolResult(f"{label}", display=make_diff(tf.text, new_text, _shown(path, ctx)))


class EditFileTool(Tool):
    name = "edit_file"
    category = "edit"
    description = (
        "Replace an exact string in an existing file. old_string must match the file (including "
        "whitespace) and be unique unless replace_all is true. Read the file first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def summary(self, args: dict) -> str:
        return str(args.get("path", ""))

    def preview(self, args: dict, ctx: ToolContext) -> str | None:
        path = resolve_path(ctx, args["path"])
        old = read_text_file(path).text
        new, _ = apply_edit(old, args["old_string"], args["new_string"], bool(args.get("replace_all")))
        return make_diff(old, new, _shown(path, ctx))

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path, tf, err = _load_for_edit(args, ctx)
        if err:
            return err
        try:
            new_text, note = apply_edit(tf.text, args["old_string"], args["new_string"], bool(args.get("replace_all")))
        except EditError as e:
            return ToolResult(str(e), True)
        return _commit(path, tf, new_text, ctx, f"Edited {path} ({note}).")


class MultiEditTool(Tool):
    name = "multi_edit"
    category = "edit"
    description = (
        "Apply several edits to one file in order, atomically: if any edit fails, none are written. "
        "Each edit has old_string, new_string and optional replace_all."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"}, "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["path", "edits"],
    }

    def summary(self, args: dict) -> str:
        return f"{args.get('path', '')} ({len(args.get('edits', []))} edits)"

    @staticmethod
    def _apply_all(text: str, edits: list) -> str:
        for i, e in enumerate(edits, start=1):
            if not isinstance(e, dict) or "old_string" not in e or "new_string" not in e:
                raise EditError(f"Edit #{i} needs old_string and new_string.")
            try:
                text, _ = apply_edit(text, e["old_string"], e["new_string"], bool(e.get("replace_all")))
            except EditError as ex:
                raise EditError(f"Edit #{i} failed, nothing was written: {ex}") from ex
        return text

    def preview(self, args: dict, ctx: ToolContext) -> str | None:
        path = resolve_path(ctx, args["path"])
        old = read_text_file(path).text
        return make_diff(old, self._apply_all(old, args["edits"]), _shown(path, ctx))

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        if not args["edits"]:
            return ToolResult("No edits provided.", True)
        path, tf, err = _load_for_edit(args, ctx)
        if err:
            return err
        try:
            new_text = self._apply_all(tf.text, args["edits"])
        except EditError as e:
            return ToolResult(str(e), True)
        return _commit(path, tf, new_text, ctx, f"Applied {len(args['edits'])} edits to {path}.")


class ListDirTool(Tool):
    name = "list_dir"
    category = "read"
    description = "List files and directories as a tree. Skips .git, node_modules, caches and similar."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory. Default: cwd."},
            "depth": {"type": "integer", "description": "Levels to descend (1-4). Default 1."},
        },
    }

    def summary(self, args: dict) -> str:
        return str(args.get("path", "."))

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = resolve_path(ctx, args.get("path") or ".")
        if not path.is_dir():
            return ToolResult(f"Not a directory: {path}", True)
        depth = min(4, max(1, int(args.get("depth") or 1)))
        lines, count, cap = [f"{path}/"], 0, 400

        def walk(d: Path, level: int, prefix: str) -> None:
            nonlocal count
            try:
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError as e:
                lines.append(f"{prefix}[unreadable: {e.strerror}]")
                return
            for p in entries:
                if count >= cap:
                    return
                if p.is_dir() and p.name in IGNORE_DIRS:
                    continue
                count += 1
                is_dir = p.is_dir() and not p.is_symlink()
                lines.append(f"{prefix}{p.name}{'/' if is_dir else ''}")
                if is_dir and level < depth:
                    walk(p, level + 1, prefix + "  ")

        walk(path, 1, "  ")
        if count >= cap:
            lines.append(f"[listing capped at {cap} entries; list a subdirectory]")
        return ToolResult("\n".join(lines))
