"""Executive Agent - Final decision maker.

Aggregates all signals and produces the final trading decision
with rationale, sizing, and exit strategy.
"""

from datetime import datetime
from typing import Any
import uuid
import json

from src.agents.base import LLMAgent
from src.orchestrator.contracts import (
    Action,
    Direction,
    ExecutiveDecision,
    HybridSignal,
    OrderType,
    PipelineState,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are the Chief Investment Officer making final trading decisions.

Your responsibilities:
1. Review all analyst inputs (technical, fundamental, hybrid)
2. Make a clear BUY, SELL, or HOLD decision
3. Determine appropriate position sizing
4. Set stop-loss and take-profit levels
5. Provide clear rationale for auditing

Decision framework:
- Only trade when you have EDGE (expected return > costs)
- Size positions based on confidence and volatility
- Always know your exit before entry
- Preserve capital - when in doubt, stay out

Output valid JSON with your decision."""


class ExecutiveAgent(LLMAgent[ExecutiveDecision]):
    """Executive Agent - Final decision maker.

    Reviews all signals and makes the final trading decision.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        use_llm: bool = True,
        min_confidence: float = 0.5,
        max_position_pct: float = 0.15,
        config: dict[str, Any] | None = None,
    ):
        """Initialize Executive Agent.

        Args:
            openai_api_key: OpenAI API key.
            use_llm: Whether to use LLM for decision making.
            min_confidence: Minimum confidence to trade.
            max_position_pct: Maximum position size as fraction of equity.
            config: Additional configuration.
        """
        super().__init__(
            name="executive",
            system_prompt=SYSTEM_PROMPT,
            config=config,
        )
        self.openai_api_key = openai_api_key
        self.use_llm = use_llm
        self.min_confidence = min_confidence
        self.max_position_pct = max_position_pct
        self._openai_client = None

    @property
    def description(self) -> str:
        return "Makes final trading decisions with rationale and exit strategy"

    def _build_user_prompt(self, state: PipelineState) -> str:
        """Build user prompt."""
        return ""

    def _parse_response(self, response: str) -> ExecutiveDecision:
        """Parse LLM response."""
        data = json.loads(response)
        return self._create_decision_from_dict(data, state=None)

    def _create_decision_from_dict(
        self,
        data: dict,
        state: PipelineState | None,
    ) -> ExecutiveDecision:
        """Create decision from parsed data."""
        action = Action(data.get("action", "HOLD").upper())

        # Parse order type
        order_type_str = data.get("entry_type", "market").lower()
        if order_type_str == "limit":
            order_type = OrderType.LIMIT
        elif order_type_str == "twap":
            order_type = OrderType.TWAP
        else:
            order_type = OrderType.MARKET

        return ExecutiveDecision(
            action=action,
            symbol=data.get("symbol", state.symbol if state else ""),
            size_pct=min(data.get("size_pct", 0.1), self.max_position_pct),
            size_shares=data.get("size_shares"),
            size_notional=data.get("size_notional"),
            entry_type=order_type,
            entry_price_limit=data.get("entry_price_limit"),
            stop_loss=data.get("stop_loss"),
            stop_loss_pct=data.get("stop_loss_pct", 0.02),
            take_profit=data.get("take_profit"),
            take_profit_pct=data.get("take_profit_pct", 0.04),
            time_horizon=data.get("time_horizon", "1d"),
            confidence=data.get("confidence", 0.5),
            rationale=data.get("rationale", "No rationale provided"),
            what_would_change_mind=data.get("what_would_change_mind", "Adverse price action"),
            technical_signal=None,
            fundamental_signal=None,
            hybrid_signal=None,
            rl_action=None,
            decision_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    async def analyze(self, state: PipelineState) -> ExecutiveDecision:
        """Make final trading decision.

        Args:
            state: Pipeline state with all agent signals.

        Returns:
            Final executive decision.
        """
        hybrid = state.hybrid_signal

        # Fall back to technical if hybrid not available
        if hybrid is None and state.technical_signal is not None:
            return self._technical_only_decision(state)

        # No signals - HOLD
        if hybrid is None:
            return self._hold_decision(state, "No signals available")

        # Check minimum confidence
        if hybrid.confidence < self.min_confidence:
            return self._hold_decision(
                state,
                f"Confidence {hybrid.confidence:.2f} below threshold {self.min_confidence:.2f}",
            )

        # Check for signal strength
        if abs(hybrid.net_signal) < 0.2:
            return self._hold_decision(state, f"Signal strength too weak: {hybrid.net_signal:.2f}")

        # Check for conflicts
        if hybrid.conflicts and hybrid.agreement_level < 0.4:
            return self._hold_decision(
                state,
                f"High disagreement between signals: {hybrid.agreement_level:.2f}",
            )

        # Use LLM if available
        if self.use_llm and self.openai_api_key:
            try:
                return await self._llm_decision(state, hybrid)
            except Exception as e:
                logger.warning(f"LLM decision failed, using rule-based: {e}")

        return self._rule_based_decision(state, hybrid)

    def _hold_decision(self, state: PipelineState, reason: str) -> ExecutiveDecision:
        """Generate HOLD decision."""
        return ExecutiveDecision(
            action=Action.HOLD,
            symbol=state.symbol,
            size_pct=0.0,
            entry_type=OrderType.MARKET,
            time_horizon="0",
            confidence=0.0,
            rationale=reason,
            what_would_change_mind="Stronger signal alignment with higher confidence",
            decision_id=str(uuid.uuid4()),
            timestamp=state.timestamp,
        )

    def _technical_only_decision(self, state: PipelineState) -> ExecutiveDecision:
        """Decision based on technical signal only."""
        tech = state.technical_signal

        if tech.confidence < self.min_confidence:
            return self._hold_decision(state, "Technical confidence too low")

        if tech.direction == Direction.LONG and tech.edge_after_costs > 0:
            action = Action.BUY
        elif tech.direction == Direction.SHORT and tech.edge_after_costs > 0:
            action = Action.SELL
        else:
            return self._hold_decision(state, "No positive edge identified")

        # Size based on confidence
        size_pct = min(tech.confidence * 0.15, self.max_position_pct)

        # Stop loss based on ATR
        atr_pct = state.features.get("atr_percent", 2.0)
        if atr_pct != atr_pct:  # NaN
            atr_pct = 2.0
        stop_loss_pct = min(atr_pct * 1.5 / 100, 0.05)

        return ExecutiveDecision(
            action=action,
            symbol=state.symbol,
            size_pct=size_pct,
            entry_type=OrderType.MARKET,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=stop_loss_pct * 2,  # 2:1 reward/risk
            time_horizon="1h",
            confidence=tech.confidence,
            rationale=f"Technical {tech.direction.value} signal with edge {tech.edge_after_costs:.2f}%",
            what_would_change_mind="RSI divergence or trend break",
            technical_signal=tech,
            decision_id=str(uuid.uuid4()),
            timestamp=state.timestamp,
        )

    def _rule_based_decision(
        self,
        state: PipelineState,
        hybrid: HybridSignal,
    ) -> ExecutiveDecision:
        """Rule-based decision from hybrid signal."""
        # Determine action from direction
        if hybrid.direction == Direction.LONG:
            action = Action.BUY
        elif hybrid.direction == Direction.SHORT:
            action = Action.SELL
        else:
            return self._hold_decision(state, "Neutral signal direction")

        # Size based on confidence and agreement
        base_size = 0.10
        size_pct = min(
            base_size * hybrid.confidence * hybrid.agreement_level,
            self.max_position_pct,
        )

        # Stop loss based on posterior std
        stop_loss_pct = min(max(hybrid.posterior_std * 2, 0.01), 0.05)

        # Take profit based on posterior mean with minimum
        take_profit_pct = max(abs(hybrid.posterior_mean) * 2, stop_loss_pct * 2)

        # Build rationale
        rationale_parts = [
            f"Net signal: {hybrid.net_signal:.2f}",
            f"Agreement: {hybrid.agreement_level:.2f}",
            f"Regime: {hybrid.regime.value}",
        ]

        if hybrid.scenarios:
            top_scenario = max(hybrid.scenarios, key=lambda s: s.probability)
            rationale_parts.append(f"Top scenario: {top_scenario.name} ({top_scenario.probability:.0%})")

        if hybrid.conflicts:
            rationale_parts.append(f"Conflicts resolved: {len(hybrid.conflicts)}")

        rationale = ". ".join(rationale_parts)

        # What would change mind
        if action == Action.BUY:
            change_mind = "Break below support, negative news, or technical breakdown"
        else:
            change_mind = "Break above resistance, positive catalyst, or short squeeze risk"

        return ExecutiveDecision(
            action=action,
            symbol=state.symbol,
            size_pct=size_pct,
            entry_type=OrderType.MARKET,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            time_horizon=self._estimate_time_horizon(hybrid),
            confidence=hybrid.confidence,
            rationale=rationale,
            what_would_change_mind=change_mind,
            technical_signal=state.technical_signal,
            fundamental_signal=state.fundamental_signal,
            hybrid_signal=hybrid,
            rl_action=state.rl_action,
            decision_id=str(uuid.uuid4()),
            timestamp=state.timestamp,
        )

    def _estimate_time_horizon(self, hybrid: HybridSignal) -> str:
        """Estimate holding time based on signal."""
        # Higher volatility = shorter horizon
        if hybrid.posterior_std > 0.02:
            return "4h"
        elif hybrid.posterior_std > 0.01:
            return "1d"
        else:
            return "2d"

    async def _llm_decision(
        self,
        state: PipelineState,
        hybrid: HybridSignal,
    ) -> ExecutiveDecision:
        """Use LLM for complex decision making."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)

        # Build context
        tech_context = ""
        if state.technical_signal:
            tech = state.technical_signal
            tech_context = f"""
TECHNICAL:
- Direction: {tech.direction.value}
- Forecast: {tech.forecast_q50:.2f}%
- Edge: {tech.edge_after_costs:.2f}%
- Confidence: {tech.confidence:.2f}
"""

        fund_context = ""
        if state.fundamental_signal:
            fund = state.fundamental_signal
            fund_context = f"""
FUNDAMENTAL:
- Sentiment: {fund.sentiment_score:.2f}
- Credibility: {fund.credibility_score:.2f}
- News analyzed: {fund.news_count_analyzed}
"""

        user_prompt = f"""Make a trading decision for {state.symbol}:

Current price: ${state.current_price:.2f}
Current position: {state.current_position_pct:.1%}
Portfolio equity: ${state.equity:,.2f}

{tech_context}
{fund_context}

HYBRID SYNTHESIS:
- Net signal: {hybrid.net_signal:.2f}
- Direction: {hybrid.direction.value}
- Agreement: {hybrid.agreement_level:.2f}
- Confidence: {hybrid.confidence:.2f}
- Expected return: {hybrid.posterior_mean:.3%}
- Std dev: {hybrid.posterior_std:.3%}

Output JSON:
{{
    "action": "BUY|SELL|HOLD",
    "size_pct": <0.0 to {self.max_position_pct}>,
    "entry_type": "market|limit",
    "stop_loss_pct": <decimal>,
    "take_profit_pct": <decimal>,
    "time_horizon": "4h|1d|2d",
    "confidence": <0 to 1>,
    "rationale": "detailed reasoning",
    "what_would_change_mind": "..."
}}"""

        response = await client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        result = response.choices[0].message.content
        data = json.loads(result)
        data["symbol"] = state.symbol

        decision = self._create_decision_from_dict(data, state)

        # Attach signals
        decision = ExecutiveDecision(
            **{
                **decision.model_dump(),
                "technical_signal": state.technical_signal,
                "fundamental_signal": state.fundamental_signal,
                "hybrid_signal": hybrid,
            }
        )

        return decision
