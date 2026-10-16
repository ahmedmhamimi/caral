"""
Storage adapters: crash-safe session store and structured trace writer.
"""

from .session_store import SessionStore
from .trace import TraceWriter, format_summary, summarize_trace, trace_path

__all__ = ["SessionStore", "TraceWriter", "trace_path", "summarize_trace", "format_summary"]
