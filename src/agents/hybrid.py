"""Hybrid Signal Analyst Agent.

Mediates between Technical and Fundamental analysts.
Resolves conflicts and produces scenario-weighted forecasts.
"""

from datetime import datetime
from typing import Any
import json

import numpy as np

from src.agents.base import LLMAgent
from src.orchestrator.contracts import (
    Conflict,
    Direction,
    FundamentalSignal,
    HybridSignal,
    PipelineState,
    Regime,
    Scenario,
    TechnicalSignal,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a senior signal analyst at a top quantitative hedge fund.

Your role is to synthesize technical and fundamental signals into a unified view.

KEY RESPONSIBILITIES:
1. Identify conflicts between technical and fundamental signals
2. Weight scenarios based on evidence quality
3. Produce a consensus view with calibrated confidence
4. Flag high-uncertainty situations

When signals conflict:
- Consider which has stronger evidence
- Assess the time horizon of each signal
- Weight by historical reliability
- Consider current market regime

Output valid JSON only."""


class HybridAnalystAgent(LLMAgent[HybridSignal]):
    """Hybrid Signal Analyst Agent.

    Synthesizes technical and fundamental signals into a unified view.
    Resolves conflicts and produces scenario-weighted forecasts.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        use_llm: bool = True,
        config: dict[str, Any] | None = None,
    ):
        """Initialize Hybrid Analyst.

        Args:
            openai_api_key: OpenAI API key.
            use_llm: Whether to use LLM for synthesis.
            config: Additional configuration.
        """
        super().__init__(
            name="hybrid_analyst",
            system_prompt=SYSTEM_PROMPT,
            config=config,
        )
        self.openai_api_key = openai_api_key
        self.use_llm = use_llm
        self._openai_client = None

    @property
    def description(self) -> str:
        return "Synthesizes technical and fundamental signals, resolves conflicts, produces scenarios"

    def _build_user_prompt(self, state: PipelineState) -> str:
        """Build prompt from signals."""
        return ""

    def _parse_response(self, response: str) -> HybridSignal:
        """Parse LLM response."""
        data = json.loads(response)
        return self._create_signal_from_dict(data)

    def _create_signal_from_dict(self, data: dict) -> HybridSignal:
        """Create signal from parsed data."""
        scenarios = []
        for s_data in data.get("scenarios", []):
            scenarios.append(Scenario(
                name=s_data.get("name", "scenario"),
                probability=s_data.get("probability", 0.5),
                expected_return=s_data.get("expected_return", 0.0),
                description=s_data.get("description", ""),
            ))

        conflicts = []
        for c_data in data.get("conflicts", []):
            conflicts.append(Conflict(
                description=c_data.get("description", ""),
                technical_view=c_data.get("technical_view", ""),
                fundamental_view=c_data.get("fundamental_view", ""),
                resolution=c_data.get("resolution", ""),
                confidence_in_resolution=c_data.get("confidence", 0.5),
            ))

        net_signal = data.get("net_signal", 0.0)
        if net_signal > 0.3:
            direction = Direction.LONG
        elif net_signal < -0.3:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        return HybridSignal(
            net_signal=net_signal,
            direction=direction,
            scenarios=scenarios if scenarios else [Scenario(
                name="base_case",
                probability=1.0,
                expected_return=0.0,
                description="Default scenario",
            )],
            conflicts=conflicts,
            agreement_level=data.get("agreement_level", 0.5),
            posterior_mean=data.get("posterior_mean", 0.0),
            posterior_std=data.get("posterior_std", 0.01),
            posterior_skew=data.get("posterior_skew", 0.0),
            regime=Regime(data.get("regime", "neutral")),
            regime_confidence=data.get("regime_confidence", 0.5),
            confidence=data.get("confidence", 0.5),
            timestamp=datetime.now(),
        )

    async def analyze(self, state: PipelineState) -> HybridSignal:
        """Synthesize technical and fundamental signals.

        Args:
            state: Pipeline state with technical and fundamental signals.

        Returns:
            Hybrid signal with scenarios and conflicts.
        """
        tech_signal = state.technical_signal
        fund_signal = state.fundamental_signal

        # If either signal is missing, use available one
        if tech_signal is None and fund_signal is None:
            return self._neutral_signal(state)

        if tech_signal is None:
            return self._fundamental_only(fund_signal, state)

        if fund_signal is None:
            return self._technical_only(tech_signal, state)

        # Both signals available - synthesize
        if self.use_llm and self.openai_api_key:
            try:
                return await self._llm_synthesis(tech_signal, fund_signal, state)
            except Exception as e:
                logger.warning(f"LLM synthesis failed, using rule-based: {e}")

        return self._rule_based_synthesis(tech_signal, fund_signal, state)

    def _neutral_signal(self, state: PipelineState) -> HybridSignal:
        """Return neutral signal when no inputs available."""
        return HybridSignal(
            net_signal=0.0,
            direction=Direction.NEUTRAL,
            scenarios=[Scenario(
                name="no_signal",
                probability=1.0,
                expected_return=0.0,
                description="No technical or fundamental signals available",
            )],
            conflicts=[],
            agreement_level=1.0,
            posterior_mean=0.0,
            posterior_std=0.01,
            posterior_skew=0.0,
            regime=Regime.NEUTRAL,
            regime_confidence=0.3,
            confidence=0.2,
            timestamp=state.timestamp,
        )

    def _technical_only(
        self,
        tech: TechnicalSignal,
        state: PipelineState,
    ) -> HybridSignal:
        """Signal based on technical only."""
        # Convert technical direction to float
        if tech.direction == Direction.LONG:
            net_signal = tech.confidence * 0.8
        elif tech.direction == Direction.SHORT:
            net_signal = -tech.confidence * 0.8
        else:
            net_signal = 0.0

        return HybridSignal(
            net_signal=net_signal,
            direction=tech.direction,
            scenarios=[Scenario(
                name="technical_only",
                probability=1.0,
                expected_return=tech.forecast_q50 / 100,
                description=f"Based on technical analysis only: {tech.regime.value} regime",
            )],
            conflicts=[],
            agreement_level=0.5,  # Lower because only one signal
            posterior_mean=tech.forecast_q50 / 100,
            posterior_std=(tech.forecast_q90 - tech.forecast_q10) / 200,  # Approx std
            posterior_skew=0.0,
            regime=tech.regime,
            regime_confidence=tech.confidence,
            confidence=tech.confidence * 0.7,  # Discount for single source
            timestamp=state.timestamp,
        )

    def _fundamental_only(
        self,
        fund: FundamentalSignal,
        state: PipelineState,
    ) -> HybridSignal:
        """Signal based on fundamental only."""
        net_signal = fund.sentiment_score * fund.confidence

        return HybridSignal(
            net_signal=net_signal,
            direction=fund.direction,
            scenarios=[Scenario(
                name="fundamental_only",
                probability=1.0,
                expected_return=fund.sentiment_score * 0.02,  # Rough translation
                description=f"Based on fundamental analysis only, {fund.news_count_analyzed} sources",
            )],
            conflicts=[],
            agreement_level=0.5,
            posterior_mean=fund.sentiment_score * 0.02,
            posterior_std=0.02,
            posterior_skew=0.0,
            regime=Regime.NEUTRAL,
            regime_confidence=0.3,
            confidence=fund.confidence * 0.6,
            timestamp=state.timestamp,
        )

    def _rule_based_synthesis(
        self,
        tech: TechnicalSignal,
        fund: FundamentalSignal,
        state: PipelineState,
    ) -> HybridSignal:
        """Rule-based signal synthesis."""
        # Convert signals to numeric
        tech_signal = 0.0
        if tech.direction == Direction.LONG:
            tech_signal = tech.confidence
        elif tech.direction == Direction.SHORT:
            tech_signal = -tech.confidence

        fund_signal = fund.sentiment_score * fund.confidence

        # Check for conflicts
        conflicts = []
        agreement_level = 1.0

        tech_bullish = tech_signal > 0.2
        tech_bearish = tech_signal < -0.2
        fund_bullish = fund_signal > 0.2
        fund_bearish = fund_signal < -0.2

        if (tech_bullish and fund_bearish) or (tech_bearish and fund_bullish):
            # Major conflict
            agreement_level = 0.2
            conflicts.append(Conflict(
                description="Technical and fundamental signals disagree",
                technical_view=f"{'Bullish' if tech_bullish else 'Bearish'} with conf {tech.confidence:.2f}",
                fundamental_view=f"{'Bullish' if fund_bullish else 'Bearish'} sentiment {fund.sentiment_score:.2f}",
                resolution="Weighted average with higher weight on technical for short-term",
                confidence_in_resolution=0.5,
            ))
        elif (tech_bullish and fund_bullish) or (tech_bearish and fund_bearish):
            # Agreement
            agreement_level = 0.9

        # Weighted average (60% tech, 40% fundamental for short-term trading)
        tech_weight = 0.6
        fund_weight = 0.4
        net_signal = tech_weight * tech_signal + fund_weight * fund_signal

        # Determine direction
        if net_signal > 0.2:
            direction = Direction.LONG
        elif net_signal < -0.2:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        # Build scenarios
        scenarios = []

        # Bullish scenario
        if net_signal > -0.3:
            bull_prob = min(0.5 + net_signal, 0.8)
            scenarios.append(Scenario(
                name="bullish",
                probability=bull_prob,
                expected_return=max(0.005, tech.forecast_q50 / 100 + 0.003),
                description="Positive momentum continues",
            ))

        # Bearish scenario
        if net_signal < 0.3:
            bear_prob = min(0.5 - net_signal, 0.8)
            scenarios.append(Scenario(
                name="bearish",
                probability=bear_prob,
                expected_return=min(-0.005, tech.forecast_q50 / 100 - 0.003),
                description="Momentum reversal or negative news impact",
            ))

        # Normalize scenario probabilities
        total_prob = sum(s.probability for s in scenarios)
        if total_prob > 0:
            scenarios = [
                Scenario(
                    name=s.name,
                    probability=s.probability / total_prob,
                    expected_return=s.expected_return,
                    description=s.description,
                )
                for s in scenarios
            ]

        # Calculate posterior
        posterior_mean = sum(s.probability * s.expected_return for s in scenarios)
        posterior_std = np.sqrt(sum(
            s.probability * (s.expected_return - posterior_mean) ** 2
            for s in scenarios
        ))

        # Combined confidence
        confidence = (tech.confidence * tech_weight + fund.confidence * fund_weight) * agreement_level

        return HybridSignal(
            net_signal=net_signal,
            direction=direction,
            scenarios=scenarios,
            conflicts=conflicts,
            agreement_level=agreement_level,
            posterior_mean=posterior_mean,
            posterior_std=max(posterior_std, 0.001),
            posterior_skew=0.0,
            regime=tech.regime,
            regime_confidence=tech.confidence,
            confidence=confidence,
            timestamp=state.timestamp,
        )

    async def _llm_synthesis(
        self,
        tech: TechnicalSignal,
        fund: FundamentalSignal,
        state: PipelineState,
    ) -> HybridSignal:
        """Use LLM for signal synthesis."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)

        user_prompt = f"""Synthesize these signals for {state.symbol}:

TECHNICAL SIGNAL:
- Direction: {tech.direction.value}
- Confidence: {tech.confidence:.2f}
- Regime: {tech.regime.value}
- Forecast (median): {tech.forecast_q50:.3f}%
- Upside probability: {tech.p_up:.2f}
- Downside probability: {tech.p_down:.2f}

FUNDAMENTAL SIGNAL:
- Sentiment: {fund.sentiment_score:.2f}
- Credibility: {fund.credibility_score:.2f}
- Confidence: {fund.confidence:.2f}
- Direction: {fund.direction.value}
- News analyzed: {fund.news_count_analyzed}
- Risk flags: {', '.join(fund.risk_flags) if fund.risk_flags else 'none'}

Current price: ${state.current_price:.2f}

Respond with JSON:
{{
    "net_signal": <float -1 to 1>,
    "agreement_level": <float 0 to 1>,
    "posterior_mean": <expected return as decimal>,
    "posterior_std": <std dev>,
    "confidence": <float 0 to 1>,
    "regime": "bullish|bearish|neutral|volatile",
    "regime_confidence": <float 0 to 1>,
    "scenarios": [
        {{"name": "...", "probability": <0-1>, "expected_return": <decimal>, "description": "..."}}
    ],
    "conflicts": [
        {{"description": "...", "technical_view": "...", "fundamental_view": "...", "resolution": "...", "confidence": <0-1>}}
    ]
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
        return self._parse_response(result)
