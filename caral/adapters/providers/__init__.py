"""
Provider factory: maps a Config to the right LLM adapter.

- create_provider(cfg) -> Provider: AnthropicProvider for kind "anthropic", OpenAIProvider for "openai".
- fetch_openrouter_prices(client=None) -> dict[str, ModelPrice]: live price list (see price_fetch).
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import Provider
from .openai_compat import OpenAIProvider
from .price_fetch import fetch_openrouter_prices

__all__ = ["create_provider", "Provider", "fetch_openrouter_prices"]


def create_provider(cfg) -> Provider:
    if cfg.kind == "anthropic":
        return AnthropicProvider(cfg)
    if cfg.kind == "openai":
        return OpenAIProvider(cfg)
    raise ValueError(f"Unknown provider kind: {cfg.kind}")
