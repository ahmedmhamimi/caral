"""
Miscellaneous tools: cached web fetching, the todo list, and subagent delegation.

- WebFetchTool: web_fetch(url, max_chars=15000) -> page text; 15-minute per-session cache; blocks
  cloud-metadata/link-local addresses; refuses binary content.
- TodoWriteTool: todo_write(todos) -> replaces the session todo list.
- TaskTool: task(description, prompt) -> read-only subagent report.
- html_to_text(html) -> str: strips scripts/styles/tags, keeps paragraph breaks.
- blocked_host(host) -> bool: True for link-local / metadata addresses.
"""

from __future__ import annotations

import ipaddress
import re
import time
from html.parser import HTMLParser

import httpx

from ...domain.textutil import truncate_middle
from ...domain.tools import Tool, ToolContext, ToolResult

_SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "section", "article"}
CACHE_TTL = 900.0


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML
        return re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.parts))
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def blocked_host(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return host.lower() in ("metadata.google.internal",)
    return ip.is_link_local or ip.is_multicast or ip.is_unspecified


class WebFetchTool(Tool):
    name = "web_fetch"
    category = "net"
    description = "Fetch a URL over HTTP(S) and return its text (HTML converted to plain text). Results are cached for 15 minutes."
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full http(s) URL."},
            "max_chars": {"type": "integer", "description": "Max characters. Default 15000."},
        },
        "required": ["url"],
    }

    def summary(self, args: dict) -> str:
        return str(args.get("url", ""))

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        url = args["url"]
        m = re.match(r"^https?://([^/:?#]+)", url)
        if not m:
            return ToolResult("URL must start with http:// or https://", True)
        if blocked_host(m.group(1)):
            return ToolResult("That address is blocked (link-local / metadata endpoint).", True)
        limit = min(60_000, int(args.get("max_chars") or 15_000))
        cached = ctx.web_cache.get(url)
        if cached and time.monotonic() - cached[0] < CACHE_TTL:
            return ToolResult(truncate_middle(cached[1], limit))
        try:
            with httpx.Client(follow_redirects=True, timeout=30.0, max_redirects=5) as c:
                with c.stream("GET", url, headers={"User-Agent": "caral-agent/0.2"}) as r:
                    if r.status_code >= 400:
                        return ToolResult(f"HTTP {r.status_code} fetching {url}", True)
                    ctype = r.headers.get("content-type", "").lower()
                    if ctype and not any(k in ctype for k in ("text", "json", "xml", "javascript", "html")):
                        return ToolResult(f"Unsupported content type '{ctype}'.", True)
                    chunks, size = [], 0
                    for chunk in r.iter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > 2_000_000:
                            break
                    encoding = r.encoding or "utf-8"
        except httpx.HTTPError as e:
            return ToolResult(f"Fetch failed: {e}", True)
        body = b"".join(chunks).decode(encoding, errors="replace")
        if "html" in ctype or body.lstrip().lower().startswith(("<!doctype html", "<html")):
            body = html_to_text(body)
        if len(ctx.web_cache) > 50:
            ctx.web_cache.clear()
        ctx.web_cache[url] = (time.monotonic(), body)
        return ToolResult(truncate_middle(body, limit) or "(empty response)")


class TodoWriteTool(Tool):
    name = "todo_write"
    category = "read"
    description = (
        "Maintain a task list for multi-step work. Send the complete list each time. Keep exactly one "
        "item in_progress; mark items completed as soon as they are done."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    @property
    def parallel_safe(self) -> bool:
        return False

    def summary(self, args: dict) -> str:
        return f"{len(args.get('todos', []))} items"

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        todos = args["todos"]
        for t in todos:
            if not isinstance(t, dict) or "content" not in t or t.get("status") not in ("pending", "in_progress", "completed"):
                return ToolResult("Each todo needs content and a valid status.", True)
        ctx.todos[:] = [{"content": str(t["content"]), "status": t["status"]} for t in todos]
        note = ""
        if sum(t["status"] == "in_progress" for t in ctx.todos) > 1:
            note = " Note: more than one item is in_progress; keep exactly one."
        return ToolResult("Todo list updated." + note)


class TaskTool(Tool):
    name = "task"
    category = "read"
    description = (
        "Delegate a self-contained research job to a read-only subagent (read_file, list_dir, glob, grep, "
        "web_fetch) running in its own context on a cheaper model; returns a final report. Give it a "
        "complete, specific prompt."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short label (3-6 words)."},
            "prompt": {"type": "string", "description": "Full instructions for the subagent."},
        },
        "required": ["description", "prompt"],
    }

    def summary(self, args: dict) -> str:
        return str(args.get("description", ""))

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.spawn_subagent is None:
            return ToolResult("Subagents are not available in this context.", True)
        try:
            report = ctx.spawn_subagent(args["description"], args["prompt"])
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - child failures become tool errors
            return ToolResult(f"Subagent failed: {e}", True)
        return ToolResult(report or "(subagent returned no text)")
