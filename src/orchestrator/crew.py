"""Agent orchestration using pipeline pattern.

Coordinates all agents in sequence to produce trading decisions.
"""

from datetime import datetime
from typing import Any
import asyncio

from src.agents.base import BaseAgent
from src.agents.technical import TechnicalAnalystAgent
from src.agents.fundamental import FundamentalAnalystAgent
from src.agents.hybrid import HybridAnalystAgent
from src.agents.executive import ExecutiveAgent
from src.agents.risk_guardian import RiskGuardian
from src.orchestrator.contracts import (
    Action,
    ExecutiveDecision,
    PipelineState,
    RiskGuardianOutput,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


class AgentOrchestrator:
    """Orchestrates the multi-agent trading pipeline.

    Pipeline:
    1. Technical Analyst → TechnicalSignal
    2. Fundamental Analyst → FundamentalSignal
    3. Hybrid Analyst → HybridSignal
    4. Executive → ExecutiveDecision
    5. Risk Guardian → Approval/Veto
    """

    def __init__(
        self,
        technical_agent: TechnicalAnalystAgent | None = None,
        fundamental_agent: FundamentalAnalystAgent | None = None,
        hybrid_agent: HybridAnalystAgent | None = None,
        executive_agent: ExecutiveAgent | None = None,
        risk_guardian: RiskGuardian | None = None,
        openai_api_key: str | None = None,
        perplexity_api_key: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize orchestrator with agents.

        Args:
            technical_agent: Pre-configured technical agent.
            fundamental_agent: Pre-configured fundamental agent.
            hybrid_agent: Pre-configured hybrid agent.
            executive_agent: Pre-configured executive agent.
            risk_guardian: Pre-configured risk guardian.
            openai_api_key: OpenAI API key for LLM agents.
            perplexity_api_key: Perplexity API key for search.
            config: Additional configuration.
        """
        self.config = config or {}

        # Initialize agents
        self.technical = technical_agent or TechnicalAnalystAgent(
            config=config,
        )

        self.fundamental = fundamental_agent or FundamentalAnalystAgent(
            openai_api_key=openai_api_key,
            perplexity_api_key=perplexity_api_key,
            enable_live_search=self.config.get("enable_live_search", False),
            config=config,
        )

        self.hybrid = hybrid_agent or HybridAnalystAgent(
            openai_api_key=openai_api_key,
            use_llm=self.config.get("use_llm", True),
            config=config,
        )

        self.executive = executive_agent or ExecutiveAgent(
            openai_api_key=openai_api_key,
            use_llm=self.config.get("use_llm", True),
            min_confidence=self.config.get("min_confidence", 0.5),
            max_position_pct=self.config.get("max_position_pct", 0.15),
            config=config,
        )

        self.risk_guardian = risk_guardian or RiskGuardian(
            max_position_pct=self.config.get("risk_max_position_pct", 0.20),
            max_daily_loss_pct=self.config.get("risk_max_daily_loss_pct", 0.05),
            max_trades_per_day=self.config.get("risk_max_trades_per_day", 10),
            min_confidence=self.config.get("risk_min_confidence", 0.5),
        )

    async def run_pipeline(self, state: PipelineState) -> PipelineState:
        """Run the full agent pipeline.

        Args:
            state: Initial pipeline state with market data and features.

        Returns:
            Updated state with all agent outputs and final decision.
        """
        logger.info(f"Starting pipeline for {state.symbol} at {state.timestamp}")

        # Step 1 & 2: Technical and Fundamental in parallel
        tech_task = asyncio.create_task(self.technical(state))
        fund_task = asyncio.create_task(self.fundamental(state))

        try:
            state.technical_signal = await tech_task
            logger.info(f"Technical: {state.technical_signal.direction.value} ({state.technical_signal.confidence:.2f})")
        except Exception as e:
            logger.error(f"Technical agent failed: {e}")
            state.technical_signal = None

        try:
            state.fundamental_signal = await fund_task
            logger.info(f"Fundamental: {state.fundamental_signal.direction.value} ({state.fundamental_signal.confidence:.2f})")
        except Exception as e:
            logger.error(f"Fundamental agent failed: {e}")
            state.fundamental_signal = None

        # Step 3: Hybrid synthesis
        try:
            state.hybrid_signal = await self.hybrid(state)
            logger.info(f"Hybrid: signal={state.hybrid_signal.net_signal:.2f}, agreement={state.hybrid_signal.agreement_level:.2f}")
        except Exception as e:
            logger.error(f"Hybrid agent failed: {e}")
            state.hybrid_signal = None

        # Step 4: Executive decision
        try:
            state.executive_decision = await self.executive(state)
            logger.info(f"Executive: {state.executive_decision.action.value} ({state.executive_decision.size_pct:.1%})")
        except Exception as e:
            logger.error(f"Executive agent failed: {e}")
            # Create HOLD decision on failure
            state.executive_decision = ExecutiveDecision(
                action=Action.HOLD,
                symbol=state.symbol,
                size_pct=0.0,
                entry_type="market",
                time_horizon="0",
                confidence=0.0,
                rationale=f"Executive agent error: {e}",
                what_would_change_mind="System recovery",
                decision_id="error",
                timestamp=state.timestamp,
            )

        # Step 5: Risk Guardian
        try:
            state.risk_guardian_output = await self.risk_guardian(state)
            if state.risk_guardian_output.approved:
                logger.info("Risk Guardian: APPROVED")
                state.final_action = state.executive_decision.action
            else:
                logger.warning(f"Risk Guardian: VETOED - {state.risk_guardian_output.veto_reasons}")
                state.final_action = Action.HOLD
        except Exception as e:
            logger.error(f"Risk Guardian failed: {e}")
            state.final_action = Action.HOLD

        # Build final order if approved
        if state.final_action != Action.HOLD and state.executive_decision:
            state.final_order = {
                "symbol": state.symbol,
                "action": state.final_action.value,
                "size_pct": state.risk_guardian_output.modified_size_pct or state.executive_decision.size_pct,
                "order_type": state.executive_decision.entry_type.value,
                "stop_loss_pct": state.executive_decision.stop_loss_pct,
                "take_profit_pct": state.executive_decision.take_profit_pct,
                "rationale": state.executive_decision.rationale,
            }

        logger.info(f"Pipeline complete: final_action={state.final_action.value if state.final_action else 'NONE'}")
        return state

    async def run_backtest_step(
        self,
        state: PipelineState,
    ) -> ExecutiveDecision | None:
        """Run pipeline for a single backtest step.

        Convenience method that runs the pipeline and returns just the decision.

        Args:
            state: Pipeline state for this timestep.

        Returns:
            Executive decision if approved, None otherwise.
        """
        state = await self.run_pipeline(state)

        if state.final_action == Action.HOLD:
            return None

        return state.executive_decision


class SimpleOrchestrator:
    """Simplified orchestrator for quick backtesting.

    Uses only Technical + Executive + Risk Guardian (skips LLM agents).
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        max_position_pct: float = 0.15,
        technical_agent: TechnicalAnalystAgent | None = None,
        executive_agent: ExecutiveAgent | None = None,
        risk_guardian: RiskGuardian | None = None,
    ):
        """Initialize simple orchestrator.

        Args:
            min_confidence: Minimum confidence to trade.
            max_position_pct: Maximum position size.
            technical_agent: Optional pre-configured technical agent.
            executive_agent: Optional pre-configured executive agent.
            risk_guardian: Optional pre-configured risk guardian.
        """
        self.technical = technical_agent or TechnicalAnalystAgent()
        self.executive = executive_agent or ExecutiveAgent(
            use_llm=False,
            min_confidence=min_confidence,
            max_position_pct=max_position_pct,
        )
        self.risk_guardian = risk_guardian or RiskGuardian(
            max_position_pct=max_position_pct,
        )

    async def run(self, state: PipelineState) -> ExecutiveDecision | None:
        """Run simplified pipeline.

        Args:
            state: Pipeline state.

        Returns:
            Decision if approved, None otherwise.
        """
        # Technical analysis
        try:
            state.technical_signal = await self.technical(state)
        except Exception as e:
            logger.error(f"Technical failed: {e}")
            return None

        # Executive decision
        try:
            state.executive_decision = await self.executive(state)
        except Exception as e:
            logger.error(f"Executive failed: {e}")
            return None

        # Risk check
        try:
            state.risk_guardian_output = await self.risk_guardian(state)
        except Exception as e:
            logger.error(f"Risk Guardian failed: {e}")
            return None

        if not state.risk_guardian_output.approved:
            return None

        if state.executive_decision.action == Action.HOLD:
            return None

        return state.executive_decision
