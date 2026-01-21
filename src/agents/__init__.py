"""Agents package."""

from src.agents.base import BaseAgent, LLMAgent, ModelAgent, RuleBasedAgent
from src.agents.technical import TechnicalAnalystAgent
from src.agents.fundamental import FundamentalAnalystAgent
from src.agents.hybrid import HybridAnalystAgent
from src.agents.executive import ExecutiveAgent
from src.agents.risk_guardian import KillSwitch, RiskGuardian

# Lazy import to avoid circular dependency
def get_researcher_agent():
    """Get ResearcherAgent (lazy import)."""
    from src.agents.researcher import ResearcherAgent
    return ResearcherAgent

__all__ = [
    "BaseAgent",
    "LLMAgent",
    "ModelAgent",
    "RuleBasedAgent",
    "TechnicalAnalystAgent",
    "FundamentalAnalystAgent",
    "HybridAnalystAgent",
    "ExecutiveAgent",
    "RiskGuardian",
    "KillSwitch",
    "get_researcher_agent",
]


