"""Observability package.

Provides comprehensive observability for the trading system:
- MLflow experiment tracking
- CrewAI AMP tracing
- Agent decision logging
- Performance reporting
"""

from src.observability.artifacts import ArtifactStore, ArtifactPaths
from src.observability.mlflow_tracker import DecisionLogger, MLflowTracker
from src.observability.reporting import BacktestReporter, DailyReporter
from src.observability.tracing import (
    TracingManager,
    get_tracing_manager,
    initialize_tracing,
    trace_agent_call,
    trace_llm_call,
)

__all__ = [
    # MLflow
    "DecisionLogger",
    "MLflowTracker",
    # Artifacts
    "ArtifactStore",
    "ArtifactPaths",
    # Reporting
    "BacktestReporter",
    "DailyReporter",
    # Tracing
    "TracingManager",
    "get_tracing_manager",
    "initialize_tracing",
    "trace_agent_call",
    "trace_llm_call",
]
