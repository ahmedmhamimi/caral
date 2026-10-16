"""
Shell execution tool. Output is captured with a hard memory bound (head + tail kept, middle
discarded), the whole process group is killed on timeout or Ctrl-C, `cd` persists between calls,
and background processes that keep the pipe open do not hang the call.

- BashTool: bash(command, timeout=None, description=None) -> combined output plus exit code.
- BoundedCapture(head=60_000, tail=60_000): .feed(text); .text() -> str; .dropped -> int.
- find_shell() -> list[str]: shell argv prefix. kill_group(proc) -> None: SIGTERM then SIGKILL.
- CWD_MARK: sentinel printed after each command to learn the final directory.
"""

from __future__ import annotations

import codecs
import collections
import os
import select
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from ...domain.tools import Tool, ToolContext, ToolResult

CWD_MARK = "__CARAL_CWD__"


def find_shell() -> list[str]:
    if os.name == "nt":
        return ["cmd", "/c"]
    return [shutil.which("bash") or "/bin/sh", "-c"]


def kill_group(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class BoundedCapture:
    def __init__(self, head: int = 60_000, tail: int = 60_000) -> None:
        self.head_limit, self.tail_limit = head, tail
        self._head: list[str] = []
        self._head_len = 0
        self._tail: collections.deque[str] = collections.deque()
        self._tail_len = 0
        self.dropped = 0

    def feed(self, text: str) -> None:
        if self._head_len < self.head_limit:
            take = text[: self.head_limit - self._head_len]
            self._head.append(take)
            self._head_len += len(take)
            text = text[len(take):]
        if not text:
            return
        self._tail.append(text)
        self._tail_len += len(text)
        while self._tail_len > self.tail_limit:
            first = self._tail[0]
            over = self._tail_len - self.tail_limit
            if len(first) <= over:
                self._tail.popleft()
                self._tail_len -= len(first)
                self.dropped += len(first)
            else:
                self._tail[0] = first[over:]
                self._tail_len -= over
                self.dropped += over

    def text(self) -> str:
        mid = f"\n... [{self.dropped} characters of output discarded] ...\n" if self.dropped else ""
        return "".join(self._head) + mid + "".join(self._tail)


def _reader(proc: subprocess.Popen, cap: BoundedCapture, stop: threading.Event) -> None:
    fd = proc.stdout.fileno()  # type: ignore[union-attr]
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while not stop.is_set():
        try:
            if os.name != "nt":
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
            data = os.read(fd, 65536)
        except (OSError, ValueError):
            break
        if not data:
            break
        cap.feed(dec.decode(data))
    cap.feed(dec.decode(b"", final=True))


class BashTool(Tool):
    name = "bash"
    category = "exec"
    description = (
        "Run a shell command; returns combined stdout/stderr and the exit code. The working directory "
        "persists between calls. No TTY and no stdin, so avoid interactive programs. Redirect output of "
        "background processes (cmd > log 2>&1 &). Use timeout (seconds) for slow commands."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command."},
            "timeout": {"type": "integer", "description": "Seconds. Default from config."},
            "description": {"type": "string", "description": "Short note on what this does."},
        },
        "required": ["command"],
    }

    def summary(self, args: dict) -> str:
        cmd = str(args.get("command", "")).strip().replace("\n", " ⏎ ")
        return cmd if len(cmd) <= 160 else cmd[:157] + "..."

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        command = args["command"]
        timeout = max(1, min(int(args.get("timeout") or ctx.config.bash_timeout), 3600))
        env = {**os.environ, "GIT_PAGER": "cat", "PAGER": "cat", "GIT_TERMINAL_PROMPT": "0", "TERM": "dumb", "CARAL": "1"}
        script = command if os.name == "nt" else (
            f"{command}\n__caral_rc=$?\nprintf '\\n{CWD_MARK}%s' \"$PWD\"\nexit $__caral_rc"
        )
        try:
            proc = subprocess.Popen(
                find_shell() + [script], cwd=str(ctx.cwd), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=(os.name != "nt"),
            )
        except OSError as e:
            return ToolResult(f"Failed to start shell: {e}", True)

        cap, stop = BoundedCapture(), threading.Event()
        reader = threading.Thread(target=_reader, args=(proc, cap, stop), daemon=True)
        reader.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_group(proc)
        except KeyboardInterrupt:
            kill_group(proc)
            stop.set()
            raise
        # The shell has exited. Give the reader a moment to drain; a backgrounded child may keep the
        # pipe open forever, so never wait for EOF indefinitely.
        deadline = time.monotonic() + 1.5
        while reader.is_alive() and time.monotonic() < deadline:
            reader.join(0.05)
        stop.set()
        reader.join(0.5)
        try:
            proc.stdout.close()  # type: ignore[union-attr]
        except OSError:
            pass

        out = cap.text()
        idx = out.rfind(CWD_MARK)
        if idx != -1:
            new_cwd = out[idx + len(CWD_MARK):].strip()
            out = out[:idx].rstrip("\n")
            if new_cwd and Path(new_cwd).is_dir():
                ctx.cwd = Path(new_cwd)
        if timed_out:
            return ToolResult(f"{out}\n[command timed out after {timeout}s and was killed]".strip(), True)
        rc = proc.returncode
        if rc != 0:
            return ToolResult(f"{out}\n[exit code {rc}]".strip(), True)
        return ToolResult(out if out.strip() else "(no output)")
