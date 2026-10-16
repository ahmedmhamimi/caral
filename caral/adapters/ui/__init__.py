"""
Terminal UI adapters: theme, streaming Markdown, spinner, single-key input, interactive renderer
and headless renderers.
"""

from .headless import JsonlRenderer, PlainRenderer
from .renderer import TerminalRenderer
from .term import Terminal
from .theme import Theme, detect_theme

__all__ = ["TerminalRenderer", "PlainRenderer", "JsonlRenderer", "Terminal", "Theme", "detect_theme"]
