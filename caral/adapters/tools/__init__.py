"""
Built-in tool set and registry factory (a plugin point: further tools can be registered on the
returned registry, or discovered from the `caral.tools` entry-point group).

- default_registry(include_task=True) -> ToolRegistry: every built-in tool, in a stable order.
- load_plugin_tools(registry) -> list[str]: registers tools from the "caral.tools" entry-point group;
  returns names added; a broken plugin is skipped, never fatal.
"""

from __future__ import annotations

import logging
from importlib import metadata

from ...domain.tools import Tool, ToolRegistry
from .fs import EditFileTool, ListDirTool, MultiEditTool, ReadFileTool, WriteFileTool
from .misc import TaskTool, TodoWriteTool, WebFetchTool
from .search import GlobTool, GrepTool
from .shell import BashTool

log = logging.getLogger("caral.tools")


def default_registry(include_task: bool = True) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (
        ReadFileTool(), ListDirTool(), GlobTool(), GrepTool(), EditFileTool(), MultiEditTool(),
        WriteFileTool(), BashTool(), WebFetchTool(), TodoWriteTool(),
    ):
        reg.register(tool)
    if include_task:
        reg.register(TaskTool())
    return reg


def load_plugin_tools(registry: ToolRegistry) -> list[str]:
    added: list[str] = []
    try:
        eps = metadata.entry_points(group="caral.tools")
    except Exception:  # noqa: BLE001
        return added
    for ep in eps:
        try:
            obj = ep.load()
            tool = obj() if isinstance(obj, type) else obj
            if isinstance(tool, Tool) and tool.name and registry.get(tool.name) is None:
                registry.register(tool)
                added.append(tool.name)
        except Exception:  # noqa: BLE001
            log.exception("failed to load tool plugin %s", ep.name)
    return added
