"""
Configuration: provider presets, layered config files, env vars and CLI overrides. Project-level
config is restricted to a safe key set so an untrusted repo cannot redirect API traffic or turn on
auto-approval. Depends only on caral.errors.

- Config: dataclass of every runtime setting. .describe() -> str renders a redacted summary.
  An empty provider is auto-detected from the first API key found in DETECT_ORDER (keys_found lists all).
- DETECT_ORDER: provider preference order for auto-detection.
- PRESETS: built-in provider presets (anthropic, openai, openrouter, groq, deepseek, together, ollama, lmstudio).
- PROJECT_SAFE_KEYS / PERMISSION_MODES: restrictions and valid modes.
- config_dir() / data_dir() -> Path: XDG-aware user directories.
- load_config(cwd, overrides=None) -> Config: defaults < global file < project file < env < overrides.
- _read_json(path) -> dict: reads a JSON object, {} if missing, ConfigError if invalid.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from .errors import ConfigError

PRESETS: dict[str, dict] = {
    "anthropic": {
        "kind": "anthropic", "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY", "model": "claude-sonnet-5-5", "small_model": "claude-haiku-5-5",
    },
    "openai": {
        "kind": "openai", "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY", "model": "gpt-4.1", "small_model": "gpt-4.1-mini",
    },
    "openrouter": {
        "kind": "openai", "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY", "model": "anthropic/claude-sonnet-4.5",
    },
    "groq": {
        "kind": "openai", "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-120b", "small_model": "openai/gpt-oss-20b",
    },
    "deepseek": {
        "kind": "openai", "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat",
    },
    "together": {
        "kind": "openai", "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY", "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    },
    "ollama": {
        "kind": "openai", "base_url": "http://localhost:11434/v1", "api_key_env": "",
        "model": "qwen2.5-coder:14b", "requires_key": False,
    },
    "lmstudio": {
        "kind": "openai", "base_url": "http://localhost:1234/v1", "api_key_env": "",
        "model": "local-model", "requires_key": False,
    },
}

DETECT_ORDER = ("anthropic", "openai", "openrouter", "groq", "deepseek", "together")
PROJECT_SAFE_KEYS = {
    "model", "small_model", "max_steps", "bash_timeout", "tool_output_limit", "context_window",
    "system_prompt_extra", "token_budget",
}
PERMISSION_MODES = ("plan", "ask", "auto-edit", "yolo")


@dataclass
class Config:
    provider: str = ""  # empty = auto-detect from whichever API key is set
    provider_auto: bool = False
    keys_found: list[str] = field(default_factory=list)
    kind: str = ""
    model: str = ""
    small_model: str = ""  # cheaper model for compaction and subagents (optional)
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    requires_key: bool = True
    max_tokens: int = 8192
    temperature: float | None = None
    prompt_cache: bool = True
    cache_markers: str = "auto"  # auto|on|off: explicit cache breakpoints on OpenAI-style APIs that front Claude (OpenRouter)
    max_tokens_param: str = ""
    permission_mode: str = "ask"
    context_window: int = 200_000
    prune_threshold: float = 0.70
    compact_threshold: float = 0.85
    max_steps: int = 60
    token_budget: int = 0  # max billable tokens per user message (0 = unlimited)
    bash_timeout: int = 120
    tool_output_limit: int = 16_000
    max_retries: int = 4
    session_retention_days: int = 60
    system_prompt_extra: str = ""
    allow_commands: list[str] = field(default_factory=list)
    deny_commands: list[str] = field(default_factory=list)
    providers: dict = field(default_factory=dict)

    def describe(self) -> str:
        key = "set" if self.api_key else ("not required" if not self.requires_key else "MISSING")
        return (
            f"provider      {self.provider} ({self.kind})"
            + (f"  [auto-detected from {self.api_key_env}]" if self.provider_auto else "") + "\n"
            f"model         {self.model}" + (f"  (small: {self.small_model})" if self.small_model else "") + "\n"
            f"base_url      {self.base_url}\n"
            f"api key       {key}\n"
            f"mode          {self.permission_mode}\n"
            f"context       {self.context_window} tokens (prune {self.prune_threshold:.0%}, compact {self.compact_threshold:.0%})\n"
            f"limits        steps={self.max_steps} budget={self.token_budget or 'off'} bash_timeout={self.bash_timeout}s"
        )


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "caral"


def data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "caral"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"Cannot read config {path}: {e}", hint="Fix or delete the file.") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Config {path} must contain a JSON object", hint="Wrap the settings in { }.")
    return data


def _apply(cfg: Config, data: dict, allowed: set[str] | None = None) -> None:
    names = {f.name for f in fields(Config)}
    for k, v in data.items():
        if k in names and (allowed is None or k in allowed):
            setattr(cfg, k, v)


def load_config(cwd: Path, overrides: dict | None = None) -> Config:
    cfg = Config()
    _apply(cfg, _read_json(config_dir() / "config.json"))
    _apply(cfg, _read_json(cwd / ".caral" / "config.json"), allowed=PROJECT_SAFE_KEYS)

    env_map = {
        "CARAL_PROVIDER": "provider", "CARAL_MODEL": "model", "CARAL_BASE_URL": "base_url",
        "CARAL_API_KEY": "api_key", "CARAL_MODE": "permission_mode",
    }
    for env, attr in env_map.items():
        if os.environ.get(env):
            setattr(cfg, attr, os.environ[env])
    if overrides:
        _apply(cfg, {k: v for k, v in overrides.items() if v is not None})

    cfg.keys_found = [n for n in DETECT_ORDER if os.environ.get(PRESETS[n]["api_key_env"])]
    if not cfg.provider:  # nothing chose a provider: use the first one that has a key
        cfg.provider = cfg.keys_found[0] if cfg.keys_found else "anthropic"
        cfg.provider_auto = bool(cfg.keys_found)
    spec = cfg.providers.get(cfg.provider) or PRESETS.get(cfg.provider)
    if spec is None:
        raise ConfigError(
            f"Unknown provider '{cfg.provider}'.",
            hint=f"Built-ins: {', '.join(PRESETS)}. Define custom ones under 'providers' in your config.",
        )
    # A --model override for a different model name should not inherit the preset's small_model
    # only if the user set neither; presets supply defaults for anything still empty.
    for attr in ("kind", "base_url", "api_key_env", "model", "small_model"):
        if not getattr(cfg, attr):
            setattr(cfg, attr, spec.get(attr, ""))
    if "requires_key" in spec and cfg.requires_key is True:
        cfg.requires_key = spec["requires_key"]
    if not cfg.api_key and cfg.api_key_env:
        cfg.api_key = os.environ.get(cfg.api_key_env, "")

    if cfg.kind not in ("anthropic", "openai"):
        raise ConfigError(f"Provider kind must be 'anthropic' or 'openai', got '{cfg.kind}'.")
    if cfg.cache_markers not in ("auto", "on", "off"):
        raise ConfigError(f"cache_markers must be 'auto', 'on' or 'off', got '{cfg.cache_markers}'.")
    if cfg.permission_mode not in PERMISSION_MODES:
        raise ConfigError(f"permission_mode must be one of {PERMISSION_MODES}, got '{cfg.permission_mode}'.")
    for name, lo, hi in (("prune_threshold", 0.1, 0.95), ("compact_threshold", 0.2, 0.98)):
        if not lo <= getattr(cfg, name) <= hi:
            raise ConfigError(f"{name} must be between {lo} and {hi}.")
    if cfg.prune_threshold >= cfg.compact_threshold:
        raise ConfigError("prune_threshold must be lower than compact_threshold.")
    if cfg.max_steps < 1 or cfg.context_window < 2000 or cfg.max_tokens < 16:
        raise ConfigError("max_steps, context_window and max_tokens must be sensible positive numbers.")
    if cfg.requires_key and not cfg.api_key:
        others = [n for n in cfg.keys_found if n != cfg.provider]
        if others:
            n = others[0]
            hint = (f"You have {PRESETS[n]['api_key_env']} set: run `caral --provider {n}` "
                    f"(or export CARAL_PROVIDER={n}), or export {cfg.api_key_env}.")
        else:
            names = ", ".join(PRESETS[n]["api_key_env"] for n in DETECT_ORDER)
            hint = f"Export one of: {names}. Or use a local model: `caral --provider ollama`."
        raise ConfigError(f"No API key found for provider '{cfg.provider}'.", hint=hint)
    return cfg
