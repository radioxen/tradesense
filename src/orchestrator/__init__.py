"""Orchestrator package."""

from src.orchestrator.contracts import (
    Action,
    Citation,
    Conflict,
    Direction,
    Event,
    ExecutiveDecision,
    FeatureSummary,
    FundamentalSignal,
    HybridSignal,
    OrderType,
    PipelineState,
    Regime,
    RiskGuardianOutput,
    RLAction,
    Scenario,
    TechnicalSignal,
    VetoReason,
)


# Lazy import to avoid circular dependency with agents.base
def get_orchestrator():
    """Get AgentOrchestrator (lazy import)."""
    from src.orchestrator.crew import AgentOrchestrator
    return AgentOrchestrator


def get_simple_orchestrator():
    """Get SimpleOrchestrator (lazy import)."""
    from src.orchestrator.crew import SimpleOrchestrator
    return SimpleOrchestrator


__all__ = [
    "Action",
    "Citation",
    "Conflict",
    "Direction",
    "Event",
    "ExecutiveDecision",
    "FeatureSummary",
    "FundamentalSignal",
    "HybridSignal",
    "OrderType",
    "PipelineState",
    "Regime",
    "RiskGuardianOutput",
    "RLAction",
    "Scenario",
    "TechnicalSignal",
    "VetoReason",
    "get_orchestrator",
    "get_simple_orchestrator",
]

