"""
Permission policy as an ordered chain of small rules; the first rule that returns a Check wins.
Add behavior by inserting a rule, never by growing a function. Modes: plan (read-only), ask
(approve edits and commands), auto-edit (project edits automatic), yolo (all but hard denials).

- Decision: ALLOW / ASK / DENY.   Check(decision, reason="", source="policy").
- PolicyRequest: tool, args, ctx, mode, root, config, state (everything a rule may inspect).
- PolicyState: per-session "always allow" memory (edits_always, commands, hosts).
- PermissionManager(mode, root, config, rules=None): .check(tool, args, ctx) -> Check;
  .remember(tool, args, ctx) -> str; .set_mode(mode) -> None; .add_rule(rule, index=None) -> None.
- DEFAULT_RULES: ordered rule list. Each rule: (PolicyRequest) -> Check | None.
  deny_destructive_bash, deny_configured_commands, deny_credential_writes, deny_git_internals,
  ask_sudo_and_credential_commands, ask_credential_reads, plan_mode, allow_reads, allow_safe_bash,
  edit_outside_root, yolo_allows, session_allowances, auto_edit_allows, default_ask.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from ..config import PERMISSION_MODES
from ..domain.shellsafety import (
    all_segments_match, bash_prefix, classify_bash, is_sensitive_path, is_sensitive_text,
    mentions_sudo, rule_matches, split_segments,
)
from ..domain.tools import resolve_path


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class Check:
    decision: Decision
    reason: str = ""
    source: str = "policy"


@dataclass
class PolicyState:
    edits_always: bool = False
    commands: set = field(default_factory=set)
    hosts: set = field(default_factory=set)


@dataclass
class PolicyRequest:
    tool: object
    args: dict
    ctx: object
    mode: str
    root: Path
    config: object
    state: PolicyState

    @property
    def category(self) -> str:
        return self.tool.category  # type: ignore[attr-defined]

    @property
    def command(self) -> str:
        return str(self.args.get("command", ""))

    @property
    def path(self) -> Path | None:
        raw = self.args.get("path") or self.args.get("file_path")
        if not isinstance(raw, str) or not raw:
            return None
        return resolve_path(self.ctx, raw)  # type: ignore[arg-type]

    def inside_root(self, p: Path) -> bool:
        try:
            Path(os.path.realpath(p)).relative_to(self.root)
            return True
        except ValueError:
            return False

    @property
    def host(self) -> str:
        m = re.match(r"^https?://([^/:?#]+)", str(self.args.get("url", "")))
        return m.group(1).lower() if m else ""


Rule = Callable[[PolicyRequest], "Check | None"]


def deny_destructive_bash(r: PolicyRequest) -> Check | None:
    if r.category == "exec" and classify_bash(r.command) == "dangerous":
        return Check(Decision.DENY, "command matches a destructive pattern")
    return None


def deny_configured_commands(r: PolicyRequest) -> Check | None:
    deny = getattr(r.config, "deny_commands", [])
    if r.category == "exec" and deny and any(rule_matches(s, deny) for s in split_segments(r.command)):
        return Check(Decision.DENY, "command blocked by deny_commands")
    return None


def deny_credential_writes(r: PolicyRequest) -> Check | None:
    if r.category == "edit" and r.path is not None and is_sensitive_path(str(r.path)):
        return Check(Decision.DENY, "writes to credential-like paths are never allowed")
    return None


def deny_git_internals(r: PolicyRequest) -> Check | None:
    if r.category == "edit" and r.path is not None and ".git" in r.path.parts:
        return Check(Decision.DENY, "editing inside .git is not allowed; use git commands")
    return None


def ask_sudo_and_credential_commands(r: PolicyRequest) -> Check | None:
    if r.category != "exec" or classify_bash(r.command) == "safe":
        return None
    if mentions_sudo(r.command):
        return Check(Decision.ASK, "uses elevated privileges")
    if is_sensitive_text(r.command):
        return Check(Decision.ASK, "touches a credential-like path")
    return None


def ask_credential_reads(r: PolicyRequest) -> Check | None:
    if r.mode == "yolo":
        return None
    if r.category == "read" and r.path is not None and is_sensitive_path(str(r.path)):
        return Check(Decision.ASK, "reads a credential-like file (its contents would be sent to the model)")
    if r.category == "exec" and classify_bash(r.command) == "safe" and is_sensitive_text(r.command):
        return Check(Decision.ASK, "reads a credential-like file (its contents would be sent to the model)")
    return None


def plan_mode(r: PolicyRequest) -> Check | None:
    if r.mode != "plan":
        return None
    if r.category == "read" or (r.category == "exec" and classify_bash(r.command) == "safe"):
        return Check(Decision.ALLOW)
    reason = {
        "edit": "plan mode is read-only; describe the change instead of making it",
        "exec": "plan mode only allows read-only commands",
        "net": "plan mode blocks network access",
    }.get(r.category, "plan mode is read-only")
    return Check(Decision.DENY, reason)


def allow_reads(r: PolicyRequest) -> Check | None:
    return Check(Decision.ALLOW) if r.category == "read" else None


def allow_safe_bash(r: PolicyRequest) -> Check | None:
    if r.category == "exec" and classify_bash(r.command) == "safe":
        return Check(Decision.ALLOW)
    return None


def edit_outside_root(r: PolicyRequest) -> Check | None:
    if r.category == "edit" and r.path is not None and not r.inside_root(r.path):
        if r.mode == "yolo":
            return Check(Decision.ALLOW)
        return Check(Decision.ASK, "path is outside the project")
    return None


def yolo_allows(r: PolicyRequest) -> Check | None:
    return Check(Decision.ALLOW, "yolo mode") if r.mode == "yolo" else None


def session_allowances(r: PolicyRequest) -> Check | None:
    s = r.state
    if r.category == "edit" and s.edits_always:
        return Check(Decision.ALLOW, "allowed for this session", "session-rule")
    if r.category == "exec":
        rules = set(getattr(r.config, "allow_commands", [])) | s.commands
        if rules and all_segments_match(r.command, rules):
            return Check(Decision.ALLOW, "matches an allow rule", "session-rule")
    if r.category == "net" and r.host and r.host in s.hosts:
        return Check(Decision.ALLOW, "host allowed for this session", "session-rule")
    return None


def auto_edit_allows(r: PolicyRequest) -> Check | None:
    if r.mode == "auto-edit" and r.category == "edit":
        return Check(Decision.ALLOW, "auto-edit mode")
    return None


def default_ask(r: PolicyRequest) -> Check | None:
    reasons = {
        "edit": "modifies files", "exec": "runs a shell command",
        "net": f"fetches {r.host or 'a URL'}",
    }
    return Check(Decision.ASK, reasons.get(r.category, "unclassified tool"))


DEFAULT_RULES: list[Rule] = [
    deny_destructive_bash, deny_configured_commands, deny_credential_writes, deny_git_internals,
    ask_sudo_and_credential_commands, ask_credential_reads, plan_mode, allow_reads, allow_safe_bash,
    edit_outside_root, yolo_allows, session_allowances, auto_edit_allows, default_ask,
]


class PermissionManager:
    def __init__(self, mode: str, root: Path, config, rules: list[Rule] | None = None) -> None:
        self.root = Path(os.path.realpath(root))
        self.config = config
        self.mode = mode
        self.state = PolicyState()
        self.rules: list[Rule] = list(rules if rules is not None else DEFAULT_RULES)

    def set_mode(self, mode: str) -> None:
        if mode not in PERMISSION_MODES:
            raise ValueError(f"mode must be one of {PERMISSION_MODES}")
        self.mode = mode

    def add_rule(self, rule: Rule, index: int | None = None) -> None:
        self.rules.insert(len(self.rules) - 1 if index is None else index, rule)

    def check(self, tool, args: dict, ctx) -> Check:
        req = PolicyRequest(tool, args, ctx, self.mode, self.root, self.config, self.state)
        for rule in self.rules:
            result = rule(req)
            if result is not None:
                return result
        return Check(Decision.ASK, "no rule matched")

    def remember(self, tool, args: dict, ctx) -> str:
        cat = tool.category
        if cat == "edit":
            self.state.edits_always = True
            return "edits inside the project for this session"
        if cat == "exec":
            prefixes = {bash_prefix(s) for s in split_segments(str(args.get("command", "")))}
            prefixes.discard("")
            self.state.commands |= prefixes
            return "commands: " + ", ".join(sorted(prefixes))
        if cat == "net":
            host = PolicyRequest(tool, args, ctx, self.mode, self.root, self.config, self.state).host
            if host:
                self.state.hosts.add(host)
            return f"fetching from {host}"
        return "this tool"
