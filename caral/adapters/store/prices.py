"""
Price files on disk. Two layers feed the PriceBook besides the bundled table: a hand-edited
overrides file (config dir, always wins) and a cache of prices fetched from OpenRouter (data dir,
fills gaps only). A corrupt file never breaks startup: the problem is reported and the layer is skipped.

- overrides_path() / fetched_path() -> Path: where the two files live.
- load_price_book() -> (PriceBook, list[str]): bundled + fetched cache + overrides, and any problems found.
- save_fetched(prices) -> Path: writes the fetched cache atomically.
- load_fetched() -> dict[str, ModelPrice]: reads the cache ({} if missing or unreadable).
- fetched_age_days() -> float | None: age of the cache, None if there is none.
"""

from __future__ import annotations

import json
import os
import time

from pathlib import Path

from ...config import config_dir, data_dir
from ...domain.pricing import ModelPrice, PriceBook, parse_overrides


def overrides_path() -> Path:
    return config_dir() / "prices.json"


def fetched_path() -> Path:
    return data_dir() / "prices-fetched.json"


def load_fetched() -> dict[str, ModelPrice]:
    try:
        data = json.loads(fetched_path().read_text(encoding="utf-8"))
        return {
            k: ModelPrice(float(v["input"]), float(v["output"]), v.get("cache_read"), v.get("cache_write"),
                          "openrouter", "fetched from OpenRouter's public model list")
            for k, v in data.get("prices", {}).items()
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def fetched_age_days() -> float | None:
    try:
        return (time.time() - float(json.loads(fetched_path().read_text(encoding="utf-8")).get("fetched_at", 0))) / 86400
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def save_fetched(prices: dict[str, ModelPrice]) -> Path:
    path = fetched_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": time.time(), "source": "openrouter",
        "prices": {k: {"input": v.input, "output": v.output, "cache_read": v.cache_read, "cache_write": v.cache_write}
                   for k, v in sorted(prices.items())},
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_price_book() -> tuple[PriceBook, list[str]]:
    problems: list[str] = []
    overrides = {}
    path = overrides_path()
    if path.is_file():
        try:
            overrides, found = parse_overrides(json.loads(path.read_text(encoding="utf-8")))
            problems += [f"{path.name}: {p}" for p in found]
        except (OSError, ValueError) as e:
            problems.append(f"{path}: cannot read ({e}); using built-in prices")
    return PriceBook(fetched=load_fetched(), overrides=overrides), problems
