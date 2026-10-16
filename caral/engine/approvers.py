"""
Non-interactive Approver implementations. Interactive approval lives in the terminal adapter.

- StaticApprover(choice="no"): answers every request the same way (headless runs, subagents, tests).
- ScriptedApprover(answers): replays (choice, feedback) pairs in order; records requests (tests).
"""

from __future__ import annotations

from ..domain.events import Approval, ApprovalRequest


class StaticApprover:
    def __init__(self, choice: str = "no", feedback: str = "") -> None:
        self.choice = choice
        self.feedback = feedback

    def ask(self, request: ApprovalRequest) -> Approval:
        return Approval(self.choice, self.feedback)


class ScriptedApprover:
    def __init__(self, answers) -> None:
        self.answers = list(answers)
        self.requests: list[ApprovalRequest] = []

    def ask(self, request: ApprovalRequest) -> Approval:
        self.requests.append(request)
        choice, feedback = self.answers.pop(0) if self.answers else ("no", "")
        return Approval(choice, feedback)
