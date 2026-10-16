"""
Static analysis of shell commands and paths for the permission policy. Heuristic and
deliberately fail-closed: anything it cannot fully understand is "normal" (needs approval),
never "safe". This is a guardrail against accidents, not a security sandbox.

- classify_bash(command) -> "safe" | "dangerous" | "normal".
- split_segments(command) -> list[str]: splits on && || | |& ; & and newlines.
- bash_prefix(segment) -> str: allow-rule key ("git commit", "ls"); interpreters key on the full segment.
- rule_matches(segment, rules) -> bool / all_segments_match(command, rules) -> bool: allow/deny rule matching.
- is_sensitive_path(path) -> bool: credential-like or secret-bearing path (ssh keys, .env, cloud creds).
- is_sensitive_text(text) -> bool: True if any token of a command/text names a sensitive path.
- mentions_sudo(command) -> bool: True if a segment invokes sudo/doas/su.
- SAFE_COMMANDS / SAFE_GIT / DANGEROUS_PATTERNS: static tables.
"""

from __future__ import annotations

import os
import re
import shlex

SAFE_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "echo", "which", "whoami", "date", "uname", "file",
    "stat", "du", "df", "tree", "grep", "egrep", "fgrep", "rg", "diff", "sort", "uniq", "cut", "tr",
    "basename", "dirname", "realpath", "true", "false", "test", "id", "hostname", "nproc", "type",
    "cd", "printf", "sleep", "seq",
}
SAFE_GIT = {
    "status", "diff", "log", "show", "rev-parse", "ls-files", "blame", "describe", "shortlog",
    "ls-tree", "cat-file", "grep",
}
VERSION_FLAGS = {"--version", "-v", "-V", "version"}
INTERPRETERS = {
    "bash", "sh", "zsh", "python", "python3", "node", "ruby", "perl", "php", "env", "xargs", "eval",
    "exec", "sudo", "doas", "su", "nohup", "time", "command", "watch", "timeout", "npx", "uvx",
}

DANGEROUS_PATTERNS = [
    r"\brm\s+(-\S+\s+)*(/|/\*|~|~/|\$HOME|\*)(\s|$|;|&|\|)",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\b[^|;&]*\bof=/dev/",
    r":\s*\(\)\s*\{",
    r">\s*/dev/(sd|nvme|hd|disk)",
    r"\bchmod\s+(-\S+\s+)*0?777\s+/(\s|$)",
    r"\b(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|da)?sh\b",
    r"\b(shutdown|reboot|halt|poweroff)\b",
]

_SEG_SPLIT = re.compile(r"&&|\|\||\|&|;|\||&|\n")
_REDIR_OK = re.compile(r"\d?>&\d|&?>\s*/dev/null|\d>\s*/dev/null")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_TOKEN_SPLIT = re.compile(r"""[\s'"=:;|&<>()]+""")
_SENSITIVE_PATH = re.compile(
    r"(^|/)(\.ssh|\.aws|\.gnupg|\.kube|\.docker/config\.json|\.netrc|\.pgpass|\.npmrc|\.pypirc|"
    r"id_rsa[^/]*|id_ed25519[^/]*|id_ecdsa[^/]*|\.config/caral|etc/shadow|etc/sudoers)(/|$)"
    r"|(^|/)\.env(\.(?!example$|sample$|template$|dist$)[\w.\-]+)?$"
)


def is_sensitive_path(path: str) -> bool:
    p = path.replace("\\", "/")
    real = os.path.realpath(os.path.expanduser(path)).replace("\\", "/") if path else p
    return bool(_SENSITIVE_PATH.search(p) or _SENSITIVE_PATH.search(real))


def is_sensitive_text(text: str) -> bool:
    return any(t and is_sensitive_path(t) for t in _TOKEN_SPLIT.split(text))


def split_segments(command: str) -> list[str]:
    return [s.strip() for s in _SEG_SPLIT.split(command) if s.strip()]


def _tokens(segment: str, strip_assign: bool = True) -> list[str]:
    try:
        toks = shlex.split(segment)
    except ValueError:
        return []
    while strip_assign and toks and _ASSIGN.match(toks[0]):
        toks.pop(0)
    return toks


def mentions_sudo(command: str) -> bool:
    return any((_tokens(s) or [""])[0] in ("sudo", "doas", "su") for s in split_segments(command))


def bash_prefix(segment: str) -> str:
    toks = _tokens(segment)
    if not toks:
        return ""
    if toks[0] in INTERPRETERS:
        return " ".join(toks)  # exact command only; never a blanket allow for interpreters
    if toks[0] in ("git", "npm", "pnpm", "yarn", "cargo", "docker", "pip", "make", "go", "uv") and len(toks) > 1:
        return f"{toks[0]} {toks[1]}"
    return toks[0]


def _positionals(toks: list[str]) -> list[str]:
    return [t for t in toks[1:] if not t.startswith("-")]


def _segment_safe(segment: str) -> bool:
    raw = _tokens(segment, strip_assign=False)
    if not raw or _ASSIGN.match(raw[0]):  # env assignments can inject LD_PRELOAD, GIT_EXTERNAL_DIFF, ...
        return False
    toks = raw
    cmd = toks[0]
    if len(toks) == 2 and toks[1] in VERSION_FLAGS:
        return cmd not in INTERPRETERS or cmd in ("python", "python3", "node")
    if cmd == "git":
        if len(toks) == 1 or "--output" in segment or "--ext-diff" in segment:
            return False
        sub = toks[1]
        return sub in SAFE_GIT or (sub == "branch" and len(toks) == 2)
    if cmd == "find":
        return not any(t in ("-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls") for t in toks)
    if cmd == "sort":
        return not any(t in ("-o", "--output") or t.startswith("--output=") for t in toks)
    if cmd == "tree":
        return not any(t == "-o" or t.startswith("--output") for t in toks)
    if cmd == "rg":
        return not any(t.startswith(("--pre", "--hostname-bin")) for t in toks)
    if cmd == "uniq":
        return len(_positionals(toks)) <= 1
    if cmd == "date":
        return not any(t in ("-s", "--set") or t.startswith("--set=") for t in toks)
    if cmd == "hostname":
        return len(toks) == 1
    return cmd in SAFE_COMMANDS


def classify_bash(command: str) -> str:
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, command):
            return "dangerous"
    if "$(" in command or "`" in command or "<(" in command or ">(" in command:
        return "normal"
    if ">" in _REDIR_OK.sub("", command):
        return "normal"
    segs = split_segments(command)
    return "safe" if segs and all(_segment_safe(s) for s in segs) else "normal"


def rule_matches(segment: str, rules) -> bool:
    toks = _tokens(segment)
    for rule in rules:
        rule = rule.strip()
        if rule.endswith("*"):
            rule = rule[:-1].strip()
        rt = rule.split()
        if rt and toks[: len(rt)] == rt:
            return True
    return False


def all_segments_match(command: str, rules) -> bool:
    segs = split_segments(command)
    return bool(segs) and all(rule_matches(s, rules) for s in segs)
