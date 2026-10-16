"""
Crash-safe session persistence. Messages are appended to a JSONL log (O(1) per step instead of
rewriting the whole conversation), with a tiny sidecar meta file for fast listing. A truncated
final line from a crash is ignored on load, and loaded conversations are repaired.

- SessionStore(root=None): implements the Store port.
  - .append(session, messages) / .snapshot(session) / .save_meta(session).
  - .load(id_or_prefix) -> Session; .list(limit=20) -> list[dict]; .latest(cwd=None) -> Session | None.
  - .prune(retention_days) -> int: deletes sessions older than the retention window.
- default_root() -> Path: <data_dir>/sessions.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ...config import data_dir
from ...domain.conversation import repair_conversation
from ...domain.messages import Usage
from ...domain.session import Session


def default_root() -> Path:
    return data_dir() / "sessions"


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else default_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _log(self, sid: str) -> Path:
        return self.root / f"{sid}.jsonl"

    def _meta(self, sid: str) -> Path:
        return self.root / f"{sid}.meta.json"

    # -- writing ---------------------------------------------------------------------------
    def append(self, session: Session, messages: list[dict]) -> None:
        if not messages:
            return
        with open(self._log(session.id), "a", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps({"t": "m", "msg": m}, ensure_ascii=False) + "\n")

    def snapshot(self, session: Session) -> None:
        tmp = self._log(session.id).with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"t": "snap", "messages": session.messages}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._log(session.id))

    def save_meta(self, session: Session) -> None:
        session.updated = time.time()
        tmp = self._meta(session.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(session.meta()), encoding="utf-8")
        os.replace(tmp, self._meta(session.id))

    # -- reading ---------------------------------------------------------------------------
    def _resolve(self, key: str) -> str:
        if self._meta(key).is_file() or self._log(key).is_file():
            return key
        matches = sorted({p.name.split(".")[0] for p in self.root.glob(f"{key}*") if p.suffix in (".jsonl", ".json")})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"Session prefix '{key}' is ambiguous ({len(matches)} matches).")
        raise FileNotFoundError(f"No session '{key}'.")

    def load(self, key: str) -> Session:
        sid = self._resolve(key)
        messages: list[dict] = []
        log_path = self._log(sid)
        if log_path.is_file():
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # truncated final line after a crash
                    if rec.get("t") == "snap":
                        messages = list(rec.get("messages", []))
                    elif rec.get("t") == "m":
                        messages.append(rec["msg"])
        meta: dict = {}
        if self._meta(sid).is_file():
            try:
                meta = json.loads(self._meta(sid).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        messages, _ = repair_conversation(messages)
        s = Session(
            id=sid, cwd=meta.get("cwd", ""), model=meta.get("model", ""), created=meta.get("created", 0),
            updated=meta.get("updated", 0), title=meta.get("title", ""), messages=messages,
            usage=Usage.from_dict(meta.get("usage")), todos=meta.get("todos", []),
        )
        s.persisted = len(messages)
        return s

    def list(self, limit: int = 20) -> list[dict]:
        out = []
        for f in self.root.glob("*.meta.json"):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        out.sort(key=lambda m: m.get("updated", 0), reverse=True)
        return out[:limit]

    def latest(self, cwd: Path | None = None) -> Session | None:
        for meta in self.list(limit=500):
            if cwd is None or meta.get("cwd") == str(cwd):
                try:
                    return self.load(meta["id"])
                except (FileNotFoundError, KeyError):
                    continue
        return None

    def prune(self, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for f in self.root.glob("*.meta.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    sid = f.name.split(".")[0]
                    for p in (self._log(sid), self._meta(sid)):
                        p.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
        return removed
