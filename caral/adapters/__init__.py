"""
Adapters layer: concrete implementations of the ports and tool contract (LLM providers, built-in
tools, session/trace storage, terminal UI). Adapters import domain/engine contracts, never each other
(providers, tools, store and ui are siblings; enforced by tests).
"""
