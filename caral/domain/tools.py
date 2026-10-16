"""
Tool contract: everything an adapter needs to implement a tool, plus the per-session state tools
share. Tools never import the engine; the engine never imports concrete tools.

- ToolResult(content, is_error=False, display=None): outcome of one run (display = UI-only extras like diffs).
- Fingerprint: (mtime_ns, size, sha1) identity of a file's content.
- ToolContext: mutable session state (cwd, read tracking, read dedupe cache, undo backups, todos, web cache).
  - .begin_turn() -> None / .backup(path) -> None / .undo_last_turn() -> list[str]: undo support.
  - .mark_read(path) -> None / .check_fresh(path) -> str | None: read-before-write and staleness checks.
  - .remember_view(path, offset, limit) -> None / .already_viewed(path, offset, limit) -> bool: read dedupe.
  - .invalidate_views() -> None: forgets all dedupe entries (history compacted).
  - .forget_view(path) -> None: forgets dedupe entries for one file (its output was pruned).
- Tool: base class (name, description, input_schema, category read|edit|exec|net).
  - .run(args, ctx) -> ToolResult; .preview(args, ctx) -> str | None; .summary(args) -> str; .schema() -> dict; .parallel_safe -> bool.
- ToolRegistry: .register(tool), .get(name), .names(), .schemas() (cached), .subset(names).
- resolve_path(ctx, raw) -> Path: expands ~ and resolves relative to ctx.cwd.
- file_fingerprint(path) -> Fingerprint: stat + streaming sha1.
- coerce_args(schema, args) -> dict / validate_args(schema, args) -> str | None: argument hygiene.
- IGNORE_DIRS: directory names skipped by listing/search tools.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", "dist", "build", ".next", ".cache", "target", ".idea", ".gradle",
}


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    display: str | None = None


class Fingerprint(NamedTuple):
    mtime_ns: int
    size: int
    sha: str


def file_fingerprint(path: Path) -> Fingerprint:
    st = path.stat()
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return Fingerprint(st.st_mtime_ns, st.st_size, h.hexdigest())


@dataclass
class ToolContext:
    cwd: Path
    config: Any
    root: Path = None  # type: ignore[assignment]
    todos: list = field(default_factory=list)
    read_state: dict = field(default_factory=dict)
    view_cache: dict = field(default_factory=dict)
    web_cache: dict = field(default_factory=dict)
    spawn_subagent: Callable[[str, str], str] | None = None
    backups: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.root is None:
            self.root = self.cwd

    # -- undo ------------------------------------------------------------------------------
    def begin_turn(self) -> None:
        self.backups.append({})
        del self.backups[:-50]

    def backup(self, path: Path) -> None:
        if not self.backups:
            self.begin_turn()
        group = self.backups[-1]
        key = str(path)
        if key in group:
            return
        try:
            group[key] = path.read_bytes() if path.exists() else None
        except OSError:
            group[key] = None

    def undo_last_turn(self) -> list[str]:
        restored: list[str] = []
        while self.backups:
            group = self.backups.pop()
            if not group:
                continue
            for p, data in group.items():
                path = Path(p)
                if data is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                self.read_state.pop(p, None)
                restored.append(p)
            break
        self.invalidate_views()
        return restored

    # -- freshness -------------------------------------------------------------------------
    def mark_read(self, path: Path) -> None:
        try:
            self.read_state[str(path)] = file_fingerprint(path)
        except OSError:
            self.read_state.pop(str(path), None)

    def check_fresh(self, path: Path) -> str | None:
        """Return an error string if the file wasn't read yet or its content changed since."""
        key = str(path)
        known = self.read_state.get(key)
        if known is None:
            return f"{path} has not been read yet in this session. Use read_file on it first."
        try:
            st = path.stat()
            if (st.st_mtime_ns, st.st_size) == (known.mtime_ns, known.size):
                return None
            now = file_fingerprint(path)
        except OSError as e:
            return f"Cannot stat {path}: {e}"
        if now.sha == known.sha:  # touched but unchanged (formatters, git checkout of same content)
            self.read_state[key] = now
            return None
        return f"{path} changed on disk since you last read it. Re-read it with read_file."

    # -- read dedupe -----------------------------------------------------------------------
    def remember_view(self, path: Path, offset: int, limit: int) -> None:
        try:
            self.view_cache[(str(path), offset, limit)] = file_fingerprint(path).sha
        except OSError:
            pass

    def already_viewed(self, path: Path, offset: int, limit: int) -> bool:
        sha = self.view_cache.get((str(path), offset, limit))
        if sha is None:
            return False
        try:
            return file_fingerprint(path).sha == sha
        except OSError:
            return False

    def invalidate_views(self) -> None:
        self.view_cache.clear()

    def forget_view(self, path: Path) -> None:
        """Forget dedupe entries for one file (its earlier output was pruned from context)."""
        key = str(path)
        for k in [k for k in self.view_cache if k[0] == key]:
            del self.view_cache[k]


class Tool:
    name = ""
    description = ""
    input_schema: dict = {"type": "object", "properties": {}}
    category = "read"  # read | edit | exec | net

    @property
    def parallel_safe(self) -> bool:
        return self.category == "read"

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

    def preview(self, args: dict, ctx: "ToolContext") -> str | None:
        """Optional: unified diff of what this call would change, without changing anything."""
        return None

    def summary(self, args: dict) -> str:
        return ", ".join(f"{k}={str(v)[:60]!r}" for k, v in list(args.items())[:2])

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._schemas: list[dict] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._schemas = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        if self._schemas is None:  # stable order => stable prompt-cache prefix
            self._schemas = [t.schema() for t in self._tools.values()]
        return self._schemas

    def subset(self, names) -> "ToolRegistry":
        sub = ToolRegistry()
        for n in names:
            if n in self._tools:
                sub.register(self._tools[n])
        return sub


def resolve_path(ctx: ToolContext, raw: str) -> Path:
    p = Path(os.path.expanduser(str(raw)))
    if not p.is_absolute():
        p = ctx.cwd / p
    return Path(os.path.normpath(p))


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def coerce_args(schema: dict, args: dict) -> dict:
    props = schema.get("properties", {})
    out = dict(args)
    for k, v in args.items():
        want = props.get(k, {}).get("type")
        if want == "integer" and isinstance(v, (str, float)) and not isinstance(v, bool):
            try:
                out[k] = int(float(v))
            except ValueError:
                pass
        elif want == "number" and isinstance(v, str):
            try:
                out[k] = float(v)
            except ValueError:
                pass
        elif want == "boolean" and isinstance(v, str) and v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif want == "array" and isinstance(v, str) and v.strip().startswith("["):
            import json

            try:
                out[k] = json.loads(v)
            except ValueError:
                pass
    return out


def validate_args(schema: dict, args: dict) -> str | None:
    if "_parse_error" in args:
        return "Your tool arguments were not valid JSON. Re-send the call with well-formed JSON."
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            return f"Missing required argument '{req}'."
    for k, v in args.items():
        spec = props.get(k)
        if not spec:
            continue
        want = spec.get("type")
        check = _TYPE_CHECKS.get(want)
        if check and not check(v):
            return f"Argument '{k}' must be of type {want}, got {type(v).__name__}."
        if "enum" in spec and v not in spec["enum"]:
            return f"Argument '{k}' must be one of {spec['enum']}."
    return None
