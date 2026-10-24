"""
Fetches current model prices from OpenRouter's public model list (no API key needed). OpenRouter
resells many vendors' models at its own prices, so this is a gap-filler, not the source of truth for
direct-API users: bundled and override prices always win.

- OPENROUTER_MODELS_URL: the endpoint.
- fetch_openrouter_prices(client=None, url=OPENROUTER_MODELS_URL, timeout=20.0) -> dict[str, ModelPrice]:
  one GET, parsed with domain.pricing.parse_openrouter_models; raises ProviderError on network, HTTP or
  JSON problems.
"""

from __future__ import annotations

import httpx

from ...domain.pricing import ModelPrice, parse_openrouter_models
from ...errors import ProviderError

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_openrouter_prices(client: httpx.Client | None = None, url: str = OPENROUTER_MODELS_URL,
                            timeout: float = 20.0) -> dict[str, ModelPrice]:
    own = client is None
    client = client or httpx.Client(follow_redirects=True)
    try:
        r = client.get(url, timeout=timeout)
    except httpx.HTTPError as e:
        raise ProviderError(f"Could not reach {url}: {e}", retryable=True, kind="network",
                            hint="Check your connection; built-in prices keep working.") from e
    finally:
        if own:
            client.close()
    if r.status_code >= 400:
        raise ProviderError(f"{url} returned HTTP {r.status_code}", status=r.status_code, kind="server")
    try:
        prices = parse_openrouter_models(r.json())
    except ValueError as e:
        raise ProviderError(f"{url} did not return JSON", kind="bad_response") from e
    if not prices:
        raise ProviderError("No usable prices in the response", kind="bad_response")
    return prices
