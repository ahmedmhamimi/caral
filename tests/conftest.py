"""
Shared fixtures and fakes.

- FakeLLM(turns, summary="SUMMARY"): implements the LLM port with scripted Turns/exceptions; records calls.
- tool_turn(*calls, text="") / text_turn(text): Turn builders.
- Harness: agent + llm + recorded events. make_agent(tmp_path, turns, mode="yolo", approver=None, **cfg) -> Harness.
- isolated_dirs (autouse): XDG_* and HOME point into tmp; CARAL_* env cleared.
"""

import copy
import itertools

import pytest

from caral.config import Config
from caral.domain.messages import Turn, Usage, text_block, tool_use_block
from caral.engine.agent import Agent
from caral.engine.approvers import StaticApprover
from caral.engine.bus import EventBus
from caral.engine.permissions import PermissionManager
from caral.adapters.tools import default_registry

_ids = itertools.count(1)


def tool_turn(*calls, text="", usage=None):
    blocks = [text_block(text)] if text else []
    for name, args in calls:
        blocks.append(tool_use_block(f"toolu_{next(_ids)}", name, args))
    return Turn(blocks=blocks, stop_reason="tool_use", usage=usage or Usage(100, 10))


def text_turn(text, usage=None):
    return Turn(blocks=[text_block(text)], stop_reason="end_turn", usage=usage or Usage(100, 10))


class FakeLLM:
    def __init__(self, turns, summary="SUMMARY"):
        self.turns = list(turns)
        self.calls = []
        self.completions = []
        self.summary = summary

    def stream(self, system, messages, tools, on_text=None, on_retry=None, model=None, max_tokens=None):
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools, "model": model})
        item = self.turns.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item()
        if on_text and item.text:
            on_text(item.text)
        return item

    def complete(self, system, prompt, max_tokens=2048, model=None):
        self.completions.append({"prompt": prompt, "model": model})
        return self.summary

    def close(self):
        pass


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for var in ("CARAL_PROVIDER", "CARAL_MODEL", "CARAL_BASE_URL", "CARAL_API_KEY", "CARAL_MODE", "NO_COLOR",
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "TOGETHER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class Harness:
    def __init__(self, agent, llm, events, bus, policy):
        self.agent, self.llm, self.events, self.bus, self.policy = agent, llm, events, bus, policy

    def of(self, cls):
        return [e for e in self.events if isinstance(e, cls)]


def make_agent(tmp_path, turns, mode="yolo", approver=None, store=None, session=None, **cfg):
    work = tmp_path / "proj"
    work.mkdir(exist_ok=True)
    config = Config(provider="anthropic", kind="anthropic", model="test-model", small_model="small-model", api_key="k", **cfg)
    llm = FakeLLM(turns)
    policy = PermissionManager(mode, work, config)
    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    agent = Agent(config, llm, default_registry(), policy, approver or StaticApprover("yes"), bus,
                  session=session, store=store, cwd=work)
    return Harness(agent, llm, events, bus, policy)
