"""
Tool execution pipeline: validate -> authorize -> run -> truncate -> emit, for one batch of tool
calls from a single model response. Read-only batches run in parallel; anything else runs in
order so approvals and side effects stay predictable.

- ToolExecutor(registry, policy, approver, bus, ctx, config): .execute(calls, step) ->
  (list[tool_result blocks], interrupted: bool).
  - ._prepare(call) -> (Tool | None, dict, ToolResult | None): lookup, coercion, validation.
  - ._authorize(call, tool, args) -> ToolResult | None: policy + approval; a result means "blocked".
  - ._run(tool, args) -> ToolResult: runs a tool, converting exceptions to error results, truncating output.
  - ._preview(tool, args) -> str | None: asks the tool for a diff of a pending edit (approval prompt).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ..domain.events import (
    Approval, ApprovalRequest, PermissionResolved, TodosUpdated, ToolFinished, ToolRequested,
    ToolStarted,
)
from ..domain.messages import ToolCall, tool_result_block
from ..domain.textutil import truncate_middle
from ..domain.tools import Tool, ToolResult, coerce_args, validate_args
from .permissions import Decision


class ToolExecutor:
    def __init__(self, registry, policy, approver, bus, ctx, config) -> None:
        self.registry = registry
        self.policy = policy
        self.approver = approver
        self.bus = bus
        self.ctx = ctx
        self.config = config

    # -- preparation -----------------------------------------------------------------------
    def _prepare(self, call: ToolCall) -> tuple[Tool | None, dict, ToolResult | None]:
        tool = self.registry.get(call.name)
        if tool is None:
            avail = ", ".join(self.registry.names())
            return None, {}, ToolResult(f"Unknown tool '{call.name}'. Available tools: {avail}.", True)
        args = coerce_args(tool.input_schema, call.input)
        err = validate_args(tool.input_schema, args)
        if err:
            return tool, args, ToolResult(f"Invalid arguments for {tool.name}: {err}", True)
        return tool, args, None

    # -- authorization ---------------------------------------------------------------------
    def _authorize(self, call: ToolCall, tool: Tool, args: dict) -> ToolResult | None:
        check = self.policy.check(tool, args, self.ctx)
        if check.decision == Decision.ALLOW:
            self.bus.emit(PermissionResolved(call_id=call.id, name=tool.name, decision="allow",
                                             source=check.source, reason=check.reason))
            return None
        if check.decision == Decision.DENY:
            self.bus.emit(PermissionResolved(call_id=call.id, name=tool.name, decision="deny",
                                             source="policy", reason=check.reason))
            return ToolResult(f"Denied: {check.reason}.", True)
        preview = self._preview(tool, args) if tool.category == "edit" else None
        approval: Approval = self.approver.ask(
            ApprovalRequest(call.id, tool.name, tool.summary(args), tool.category, check.reason, preview)
        )
        if approval.choice == "always":
            self.policy.remember(tool, args, self.ctx)
        if approval.choice in ("yes", "always"):
            self.bus.emit(PermissionResolved(call_id=call.id, name=tool.name, decision="allow",
                                             source="user", reason=approval.choice))
            return None
        self.bus.emit(PermissionResolved(call_id=call.id, name=tool.name, decision="deny",
                                         source="user", reason=approval.feedback))
        msg = "The user denied this action."
        if approval.feedback:
            msg += f" Their feedback: {approval.feedback}"
        return ToolResult(msg, True)

    def _preview(self, tool: Tool, args: dict) -> str | None:
        """Diff of a pending edit shown in the approval prompt; best-effort, never raises."""
        try:
            return tool.preview(args, self.ctx)
        except Exception:  # noqa: BLE001
            return None

    # -- running ---------------------------------------------------------------------------
    def _run(self, tool: Tool, args: dict) -> ToolResult:
        try:
            res = tool.run(args, self.ctx)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - a crashing tool must not crash the agent
            return ToolResult(f"Tool crashed: {type(e).__name__}: {e}", True)
        res.content = truncate_middle(str(res.content), self.config.tool_output_limit)
        return res

    def _finish(self, call: ToolCall, tool: Tool | None, res: ToolResult, dur: float) -> dict:
        name = tool.name if tool else call.name
        self.bus.emit(ToolFinished(call_id=call.id, name=name, is_error=res.is_error, duration_s=dur,
                                   content=res.content, chars=len(res.content), display=res.display))
        if tool is not None and tool.name == "todo_write" and not res.is_error:
            self.bus.emit(TodosUpdated(todos=list(self.ctx.todos)))
        return tool_result_block(call.id, res.content, res.is_error)

    # -- entry point -----------------------------------------------------------------------
    def execute(self, calls: list[ToolCall], step: int) -> tuple[list[dict], bool]:
        prepared = [self._prepare(c) for c in calls]
        for c, (tool, args, _err) in zip(calls, prepared):
            summary = tool.summary(args) if tool else ""
            self.bus.emit(ToolRequested(call_id=c.id, name=c.name, summary=summary,
                                        category=tool.category if tool else "unknown", step=step))

        if len(calls) > 1 and all(
            t is not None and err is None and t.parallel_safe
            and self.policy.check(t, a, self.ctx).decision == Decision.ALLOW
            for t, a, err in prepared
        ):
            return self._execute_parallel(calls, prepared), False
        return self._execute_sequential(calls, prepared)

    def _execute_parallel(self, calls, prepared) -> list[dict]:
        for c, (t, a, _e) in zip(calls, prepared):
            self.bus.emit(PermissionResolved(call_id=c.id, name=t.name, decision="allow", source="policy"))
            self.bus.emit(ToolStarted(call_id=c.id, name=t.name, summary=t.summary(a), category=t.category))
        try:
            with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
                futures = [pool.submit(self._timed, t, a) for t, a, _e in prepared]
                results = [f.result() for f in futures]
        except KeyboardInterrupt:
            return [tool_result_block(c.id, "Interrupted by the user.", True) for c in calls]
        return [self._finish(c, t, res, dur) for c, (t, _a, _e), (res, dur) in zip(calls, prepared, results)]

    def _timed(self, tool: Tool, args: dict) -> tuple[ToolResult, float]:
        t0 = time.monotonic()
        res = self._run(tool, args)
        return res, time.monotonic() - t0

    def _execute_sequential(self, calls, prepared) -> tuple[list[dict], bool]:
        blocks: list[dict] = []
        for i, (c, (tool, args, err)) in enumerate(zip(calls, prepared)):
            try:
                if err is not None:
                    blocks.append(self._finish(c, tool, err, 0.0))
                    continue
                blocked = self._authorize(c, tool, args)
                if blocked is not None:
                    blocks.append(self._finish(c, tool, blocked, 0.0))
                    continue
                self.bus.emit(ToolStarted(call_id=c.id, name=tool.name, summary=tool.summary(args),
                                          category=tool.category))
                res, dur = self._timed(tool, args)
                blocks.append(self._finish(c, tool, res, dur))
            except KeyboardInterrupt:
                for rest in calls[i:]:
                    blocks.append(tool_result_block(rest.id, "Interrupted by the user.", True))
                return blocks, True
        return blocks, False
