"""Comprehensive tracing for AI agents.

Provides unified tracing across:
- CrewAI built-in tracing (via AMP platform)
- MLflow experiment tracking and autolog
- Custom agent instrumentation
- OpenTelemetry compatibility

Usage:
    from src.observability.tracing import TracingManager, trace_agent_call
    
    # Initialize tracing at app startup
    tracing = TracingManager()
    tracing.initialize()
    
    # Use decorator for agent methods
    @trace_agent_call("TechnicalAnalyst")
    async def analyze(self, state):
        ...
"""

import functools
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar
from contextlib import contextmanager
import json
import uuid

from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class TracingManager:
    """Unified tracing manager for the trading system.
    
    Coordinates multiple tracing backends:
    - CrewAI AMP tracing
    - MLflow tracing
    - Local file logging
    """
    
    def __init__(
        self,
        mlflow_uri: str = "http://localhost:5000",
        experiment_name: str = "trading_agents",
        enable_crewai: bool = True,
        enable_mlflow: bool = True,
        enable_file_logging: bool = True,
        log_dir: Path | str = "./logs/traces",
    ):
        """Initialize tracing manager.
        
        Args:
            mlflow_uri: MLflow tracking server URI.
            experiment_name: MLflow experiment name.
            enable_crewai: Enable CrewAI AMP tracing.
            enable_mlflow: Enable MLflow tracing.
            enable_file_logging: Enable local file logging.
            log_dir: Directory for trace logs.
        """
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.enable_crewai = enable_crewai
        self.enable_mlflow = enable_mlflow
        self.enable_file_logging = enable_file_logging
        self.log_dir = Path(log_dir)
        
        self._mlflow = None
        self._initialized = False
        self._current_run_id = None
        self._trace_buffer: list[dict] = []
        
    def initialize(self) -> bool:
        """Initialize all tracing backends.
        
        Returns:
            True if at least one backend initialized successfully.
        """
        if self._initialized:
            return True
            
        success = False
        
        # Initialize CrewAI tracing
        if self.enable_crewai:
            try:
                os.environ["CREWAI_TRACING_ENABLED"] = "true"
                logger.info("CrewAI tracing enabled via environment variable")
                success = True
            except Exception as e:
                logger.warning(f"Failed to enable CrewAI tracing: {e}")
        
        # Initialize MLflow
        if self.enable_mlflow:
            try:
                import mlflow
                
                # Set tracking URI
                mlflow.set_tracking_uri(self.mlflow_uri)
                
                # Set experiment
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    mlflow.create_experiment(self.experiment_name)
                mlflow.set_experiment(self.experiment_name)
                
                # Enable CrewAI autolog if available
                if hasattr(mlflow, 'crewai'):
                    mlflow.crewai.autolog()
                    logger.info("MLflow CrewAI autolog enabled")
                else:
                    # Fallback to generic autolog
                    mlflow.autolog(silent=True)
                    logger.info("MLflow generic autolog enabled")
                
                self._mlflow = mlflow
                success = True
                logger.info(f"MLflow initialized: {self.mlflow_uri}")
                
            except ImportError:
                logger.warning("MLflow not installed, skipping MLflow tracing")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")
        
        # Initialize file logging
        if self.enable_file_logging:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                success = True
                logger.info(f"File tracing enabled: {self.log_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize file logging: {e}")
        
        self._initialized = success
        return success
    
    def start_session(self, session_name: str | None = None) -> str:
        """Start a new tracing session.
        
        Args:
            session_name: Optional session name.
            
        Returns:
            Session ID.
        """
        session_id = str(uuid.uuid4())[:8]
        session_name = session_name or f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start MLflow run
        if self._mlflow:
            try:
                run = self._mlflow.start_run(run_name=session_name)
                self._current_run_id = run.info.run_id
                logger.info(f"Started MLflow run: {self._current_run_id}")
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")
        
        # Initialize session file
        if self.enable_file_logging:
            session_file = self.log_dir / f"session_{session_id}.jsonl"
            self._write_trace({
                "type": "session_start",
                "session_id": session_id,
                "session_name": session_name,
                "timestamp": datetime.now().isoformat(),
            }, session_file)
        
        return session_id
    
    def end_session(self, status: str = "completed"):
        """End the current tracing session.
        
        Args:
            status: Session status.
        """
        if self._mlflow and self._current_run_id:
            try:
                self._mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")
                logger.info(f"Ended MLflow run: {self._current_run_id}")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
            self._current_run_id = None
    
    @contextmanager
    def trace_span(self, span_name: str, span_type: str = "agent", **attributes):
        """Context manager for tracing a span.
        
        Args:
            span_name: Name of the span.
            span_type: Type of span (agent, tool, llm, etc).
            **attributes: Additional span attributes.
            
        Yields:
            Span context with trace_id and methods to add data.
        """
        span_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        start_ts = datetime.now().isoformat()
        
        context = {
            "span_id": span_id,
            "span_name": span_name,
            "span_type": span_type,
            "start_time": start_ts,
            "attributes": attributes,
            "events": [],
            "output": None,
            "error": None,
        }
        
        class SpanContext:
            def __init__(self, ctx, manager):
                self.ctx = ctx
                self.manager = manager
                
            def add_event(self, event_name: str, **data):
                """Add an event to the span."""
                self.ctx["events"].append({
                    "name": event_name,
                    "timestamp": datetime.now().isoformat(),
                    "data": data,
                })
                
            def set_attribute(self, key: str, value: Any):
                """Set a span attribute."""
                self.ctx["attributes"][key] = value
                
            def set_output(self, output: Any):
                """Set the span output."""
                self.ctx["output"] = output
        
        span_ctx = SpanContext(context, self)
        
        try:
            yield span_ctx
            
        except Exception as e:
            context["error"] = str(e)
            raise
            
        finally:
            end_time = time.time()
            context["end_time"] = datetime.now().isoformat()
            context["duration_ms"] = (end_time - start_time) * 1000
            
            # Log to MLflow
            if self._mlflow:
                try:
                    self._mlflow.log_metric(f"{span_name}_duration_ms", context["duration_ms"])
                    if context.get("error"):
                        self._mlflow.log_param(f"{span_name}_error", context["error"][:250])
                except Exception:
                    pass
            
            # Log to file
            if self.enable_file_logging:
                self._write_trace(context)
    
    def log_agent_decision(
        self,
        agent_name: str,
        symbol: str,
        decision: dict[str, Any],
        inputs: dict[str, Any] | None = None,
        duration_ms: float = 0,
    ):
        """Log an agent decision.
        
        Args:
            agent_name: Name of the agent.
            symbol: Stock symbol.
            decision: Decision output.
            inputs: Input data.
            duration_ms: Processing duration.
        """
        trace = {
            "type": "agent_decision",
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "symbol": symbol,
            "decision": decision,
            "inputs": inputs,
            "duration_ms": duration_ms,
        }
        
        # Log to MLflow
        if self._mlflow:
            try:
                self._mlflow.log_metrics({
                    f"{agent_name}_{symbol}_duration_ms": duration_ms,
                })
                # Log decision as artifact
                self._mlflow.log_dict(trace, f"decisions/{agent_name}_{symbol}_{datetime.now().strftime('%H%M%S')}.json")
            except Exception as e:
                logger.debug(f"MLflow logging failed: {e}")
        
        # Log to file
        if self.enable_file_logging:
            self._write_trace(trace)
    
    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: float = 0,
        agent_name: str | None = None,
    ):
        """Log an LLM API call.
        
        Args:
            model: Model name.
            prompt: Input prompt (truncated).
            response: Model response (truncated).
            tokens_in: Input tokens.
            tokens_out: Output tokens.
            duration_ms: API call duration.
            agent_name: Calling agent name.
        """
        trace = {
            "type": "llm_call",
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "response": response[:500] + "..." if len(response) > 500 else response,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "duration_ms": duration_ms,
            "agent_name": agent_name,
        }
        
        # Log to MLflow
        if self._mlflow:
            try:
                prefix = f"{agent_name}_" if agent_name else ""
                self._mlflow.log_metrics({
                    f"{prefix}llm_tokens_in": tokens_in,
                    f"{prefix}llm_tokens_out": tokens_out,
                    f"{prefix}llm_duration_ms": duration_ms,
                })
            except Exception:
                pass
        
        # Log to file
        if self.enable_file_logging:
            self._write_trace(trace)
    
    def log_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        confidence: float = 0,
        rationale: str = "",
    ):
        """Log a trade execution.
        
        Args:
            symbol: Stock symbol.
            action: BUY/SELL.
            quantity: Number of shares.
            price: Execution price.
            confidence: Decision confidence.
            rationale: Trade rationale.
        """
        trace = {
            "type": "trade",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "notional": quantity * price,
            "confidence": confidence,
            "rationale": rationale[:200],
        }
        
        # Log to MLflow
        if self._mlflow:
            try:
                self._mlflow.log_metrics({
                    f"trade_{symbol}_qty": quantity,
                    f"trade_{symbol}_price": price,
                    f"trade_{symbol}_notional": quantity * price,
                })
            except Exception:
                pass
        
        # Log to file
        if self.enable_file_logging:
            self._write_trace(trace)
        
        logger.info(f"TRADE: {action} {quantity} {symbol} @ ${price:.2f}")
    
    def log_performance(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ):
        """Log performance metrics.
        
        Args:
            metrics: Performance metrics.
            step: Step number.
        """
        trace = {
            "type": "performance",
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "metrics": metrics,
        }
        
        # Log to MLflow
        if self._mlflow:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass
        
        # Log to file
        if self.enable_file_logging:
            self._write_trace(trace)
    
    def _write_trace(self, trace: dict, filepath: Path | None = None):
        """Write trace to file.
        
        Args:
            trace: Trace data.
            filepath: Optional specific file path.
        """
        if filepath is None:
            date_str = datetime.now().strftime("%Y%m%d")
            filepath = self.log_dir / f"traces_{date_str}.jsonl"
        
        try:
            with open(filepath, "a") as f:
                f.write(json.dumps(trace, default=str) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write trace: {e}")


# Global tracing manager instance
_tracing_manager: TracingManager | None = None


def get_tracing_manager() -> TracingManager:
    """Get or create the global tracing manager."""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def initialize_tracing(
    mlflow_uri: str = "http://localhost:5000",
    experiment_name: str = "trading_agents",
    enable_crewai: bool = True,
    enable_mlflow: bool = True,
    enable_file_logging: bool = True,
) -> TracingManager:
    """Initialize the global tracing manager.
    
    Args:
        mlflow_uri: MLflow server URI.
        experiment_name: MLflow experiment name.
        enable_crewai: Enable CrewAI tracing.
        enable_mlflow: Enable MLflow tracing.
        enable_file_logging: Enable file logging.
        
    Returns:
        Initialized tracing manager.
    """
    global _tracing_manager
    _tracing_manager = TracingManager(
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        enable_crewai=enable_crewai,
        enable_mlflow=enable_mlflow,
        enable_file_logging=enable_file_logging,
    )
    _tracing_manager.initialize()
    return _tracing_manager


def trace_agent_call(agent_name: str):
    """Decorator to trace agent analyze() calls.
    
    Args:
        agent_name: Name of the agent for logging.
        
    Returns:
        Decorator function.
        
    Usage:
        @trace_agent_call("TechnicalAnalyst")
        async def analyze(self, state):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            manager = get_tracing_manager()
            
            # Extract symbol from state if available
            symbol = "UNKNOWN"
            if args and hasattr(args[0], "symbol"):
                symbol = args[0].symbol
            elif len(args) > 1 and hasattr(args[1], "symbol"):
                symbol = args[1].symbol
            
            start_time = time.time()
            
            with manager.trace_span(agent_name, "agent", symbol=symbol) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Log the decision
                    if result is not None:
                        decision_dict = result.model_dump() if hasattr(result, "model_dump") else str(result)
                        span.set_output(decision_dict)
                        
                        manager.log_agent_decision(
                            agent_name=agent_name,
                            symbol=symbol,
                            decision=decision_dict if isinstance(decision_dict, dict) else {"output": decision_dict},
                            duration_ms=(time.time() - start_time) * 1000,
                        )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"{agent_name} error for {symbol}: {e}")
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            manager = get_tracing_manager()
            symbol = "UNKNOWN"
            if args and hasattr(args[0], "symbol"):
                symbol = args[0].symbol
            
            start_time = time.time()
            
            with manager.trace_span(agent_name, "agent", symbol=symbol) as span:
                try:
                    result = func(*args, **kwargs)
                    
                    if result is not None:
                        decision_dict = result.model_dump() if hasattr(result, "model_dump") else str(result)
                        span.set_output(decision_dict)
                        
                        manager.log_agent_decision(
                            agent_name=agent_name,
                            symbol=symbol,
                            decision=decision_dict if isinstance(decision_dict, dict) else {"output": decision_dict},
                            duration_ms=(time.time() - start_time) * 1000,
                        )
                    
                    return result
                except Exception as e:
                    logger.error(f"{agent_name} error for {symbol}: {e}")
                    raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_llm_call(model: str = "unknown", agent_name: str | None = None):
    """Decorator to trace LLM API calls.
    
    Args:
        model: Model name.
        agent_name: Calling agent name.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            manager = get_tracing_manager()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Try to extract token counts and prompt/response
                duration_ms = (time.time() - start_time) * 1000
                
                # Log the call (with limited info to avoid sensitive data)
                manager.log_llm_call(
                    model=model,
                    prompt="[see detailed logs]",
                    response="[see detailed logs]",
                    duration_ms=duration_ms,
                    agent_name=agent_name,
                )
                
                return result
                
            except Exception as e:
                logger.error(f"LLM call error: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            manager = get_tracing_manager()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                manager.log_llm_call(
                    model=model,
                    prompt="[see detailed logs]",
                    response="[see detailed logs]",
                    duration_ms=duration_ms,
                    agent_name=agent_name,
                )
                
                return result
            except Exception as e:
                logger.error(f"LLM call error: {e}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
