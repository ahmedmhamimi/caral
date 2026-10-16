"""
Synchronous in-process event bus. Subscribers are isolated: an exception in one (a UI bug, a full
disk for the trace) is logged and never breaks the agent run.

- EventBus: .subscribe(fn, name=None) -> Callable[[], None] (unsubscribe); .emit(event) -> None.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from ..domain.events import Event

log = logging.getLogger("caral.bus")


class EventBus:
    def __init__(self) -> None:
        self._subs: list[tuple[str, Callable[[Event], None]]] = []
        self._lock = threading.RLock()

    def subscribe(self, fn: Callable[[Event], None], name: str | None = None) -> Callable[[], None]:
        entry = (name or getattr(fn, "__qualname__", "subscriber"), fn)
        with self._lock:
            self._subs.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                if entry in self._subs:
                    self._subs.remove(entry)

        return unsubscribe

    def emit(self, event: Event) -> None:
        with self._lock:  # serialize: parallel tool threads must not interleave UI output
            subs = list(self._subs)
            for name, fn in subs:
                try:
                    fn(event)
                except Exception:  # noqa: BLE001 - subscriber isolation is the point
                    log.exception("event subscriber %s failed on %s", name, type(event).__name__)
