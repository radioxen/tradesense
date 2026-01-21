"""Agent I/O contracts using Pydantic models.

All agents must use these schemas for their inputs and outputs
to ensure strict validation and type safety.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Enums
# =============================================================================


class Direction(str, Enum):
    """Trading direction."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Regime(str, Enum):
    """Market regime classification."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"


class Action(str, Enum):
    """Trading action."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(str, Enum):
    """Order type for execution."""

    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"


class VetoReason(str, Enum):
    """Risk guardian veto reasons."""

    MAX_POSITION_EXCEEDED = "max_position_exceeded"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_TRADES_EXCEEDED = "max_trades_exceeded"
    LOW_CONFIDENCE = "low_confidence"
    DATA_MISSING = "data_missing"
    MARKET_CLOSED = "market_closed"
    LIQUIDITY_LOW = "liquidity_low"


# =============================================================================
# Supporting Models
# =============================================================================


class Citation(BaseModel):
    """Citation for fundamental analysis claims."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="Source name (e.g., 'Reuters', 'SEC Filing')")
    title: str = Field(..., description="Article or document title")
    url: str | None = Field(None, description="URL if available")
    timestamp: datetime = Field(..., description="Publication timestamp")
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Source credibility 0-1")


class Event(BaseModel):
    """Financial event (earnings, guidance, macro, etc.)."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["earnings", "guidance", "macro", "regulatory", "insider", "social"]
    title: str
    timestamp: datetime
    impact_estimate: Literal["high", "medium", "low"]
    sentiment: float = Field(..., ge=-1.0, le=1.0, description="Sentiment -1 to 1")
    citations: list[Citation] = Field(default_factory=list)


class Scenario(BaseModel):
    """Scenario for hybrid analysis."""

    name: str = Field(..., description="Scenario name (e.g., 'bullish continuation')")
    probability: float = Field(..., ge=0.0, le=1.0)
    expected_return: float = Field(..., description="Expected return in this scenario")
    description: str = Field(..., description="Brief description")


class Conflict(BaseModel):
    """Conflict between technical and fundamental signals."""

    description: str
    technical_view: str
    fundamental_view: str
    resolution: str
    confidence_in_resolution: float = Field(..., ge=0.0, le=1.0)


class FeatureSummary(BaseModel):
    """Summary of top features for explainability."""

    feature_name: str
    importance: float
    current_value: float
    interpretation: str


# =============================================================================
# Agent Output Schemas
# =============================================================================


class TechnicalSignal(BaseModel):
    """Output from Technical Analyst Agent.

    Contains probabilistic forecasts and model ensemble outputs.
    """

    model_config = ConfigDict(frozen=True)

    # Probabilistic forecasts (return quantiles)
    forecast_q10: float = Field(..., description="10th percentile return forecast")
    forecast_q50: float = Field(..., description="Median return forecast")
    forecast_q90: float = Field(..., description="90th percentile return forecast")

    # Probability estimates
    p_up: float = Field(..., ge=0.0, le=1.0, description="P(return > threshold)")
    p_down: float = Field(..., ge=0.0, le=1.0, description="P(return < -threshold)")
    threshold_pct: float = Field(default=0.5, description="Threshold for p_up/p_down (in %)")

    # Edge estimate
    edge_after_costs: float = Field(
        ..., description="Expected edge after transaction costs (in %)"
    )

    # Regime and direction
    regime: Regime = Field(..., description="Current market regime")
    direction: Direction = Field(..., description="Signal direction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")

    # Model metadata
    model_version: str = Field(..., description="Ensemble model version")
    data_version: str = Field(..., description="Feature data version")

    # Metrics
    sharpe_oos: float | None = Field(None, description="Out-of-sample Sharpe ratio")
    hit_rate: float | None = Field(None, ge=0.0, le=1.0, description="Win rate")
    calibration_error: float | None = Field(None, description="Probability calibration error")

    # Explainability
    top_features: list[FeatureSummary] = Field(
        default_factory=list, max_length=10, description="Top contributing features"
    )

    # Timing
    time_horizon: str = Field(..., description="Forecast horizon (e.g., '1h', '1d')")
    timestamp: datetime = Field(..., description="Signal generation timestamp")


class FundamentalSignal(BaseModel):
    """Output from Fundamental Analyst Agent.

    Contains news analysis, sentiment, and event impacts.
    """

    model_config = ConfigDict(frozen=True)

    # Sentiment analysis
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Overall sentiment -1 to 1")
    credibility_score: float = Field(
        ..., ge=0.0, le=1.0, description="Credibility of sources analyzed"
    )

    # Events and catalysts
    events: list[Event] = Field(default_factory=list, description="Relevant events")
    catalysts: list[str] = Field(default_factory=list, description="Potential catalysts")
    risks: list[str] = Field(default_factory=list, description="Identified risks")

    # Historical priors
    impact_prior: dict = Field(
        default_factory=dict,
        description="Historical similar event impacts: {'event_type': {'mean_move': x, 'std': y}}",
    )

    # Risk flags
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Risk flags (e.g., 'rumor', 'unverified', 'high_uncertainty')",
    )

    # Direction and confidence
    direction: Direction = Field(..., description="Suggested direction based on fundamentals")
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Citations (mandatory - no claims without sources)
    citations: list[Citation] = Field(
        ..., min_length=0, description="All sources cited (required for claims)"
    )

    # Metadata
    news_count_analyzed: int = Field(..., description="Number of news items analyzed")
    timestamp: datetime = Field(..., description="Analysis timestamp")


class HybridSignal(BaseModel):
    """Output from Hybrid Signal Analyst Agent.

    Mediates between technical and fundamental signals.
    """

    model_config = ConfigDict(frozen=True)

    # Combined signal
    net_signal: float = Field(..., ge=-1.0, le=1.0, description="Combined signal strength")
    direction: Direction = Field(..., description="Final direction recommendation")

    # Scenario analysis
    scenarios: list[Scenario] = Field(
        ..., min_length=1, max_length=5, description="Weighted scenarios"
    )

    # Conflict resolution
    conflicts: list[Conflict] = Field(
        default_factory=list, description="Identified conflicts and resolutions"
    )
    agreement_level: float = Field(
        ..., ge=0.0, le=1.0, description="Agreement between tech and fundamental"
    )

    # Posterior distribution
    posterior_mean: float = Field(..., description="Posterior expected return")
    posterior_std: float = Field(..., description="Posterior standard deviation")
    posterior_skew: float = Field(default=0.0, description="Posterior skewness")

    # Regime synthesis
    regime: Regime = Field(..., description="Synthesized regime label")
    regime_confidence: float = Field(..., ge=0.0, le=1.0)

    # Overall confidence
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime


class RLAction(BaseModel):
    """Output from RL Policy Agent.

    Converts signals to position sizing and timing.
    """

    model_config = ConfigDict(frozen=True)

    # Position target
    position_target: float = Field(
        ..., ge=-1.0, le=1.0, description="Target position as fraction of max (-1 to 1)"
    )
    position_delta: float = Field(..., description="Required position change")

    # Execution parameters
    urgency: float = Field(..., ge=0.0, le=1.0, description="Execution urgency")
    order_type: OrderType = Field(..., description="Recommended order type")
    limit_offset_bps: int = Field(default=0, description="Limit order offset in basis points")

    # RL outputs
    value_estimate: float = Field(..., description="RL value function estimate")
    action_uncertainty: float = Field(..., ge=0.0, le=1.0, description="Policy uncertainty")

    # Risk-adjusted sizing
    risk_adjusted_size: float = Field(
        ..., description="Position size adjusted for current volatility"
    )
    kelly_fraction: float = Field(..., ge=0.0, le=1.0, description="Kelly criterion fraction")

    # Timing
    suggested_entry_window: str = Field(
        ..., description="Suggested entry window (e.g., 'next_5m', 'at_open')"
    )

    # Metadata
    policy_version: str
    timestamp: datetime


class ExecutiveDecision(BaseModel):
    """Output from Executive Agent.

    Final trading decision with full rationale.
    """

    model_config = ConfigDict(frozen=True)

    # Action
    action: Action = Field(..., description="Final action: BUY, SELL, or HOLD")
    symbol: str = Field(..., description="Trading symbol")

    # Sizing
    size_pct: float = Field(..., ge=0.0, le=1.0, description="Position size as % of equity")
    size_shares: int | None = Field(None, description="Number of shares if calculated")
    size_notional: float | None = Field(None, description="Notional value if calculated")

    # Entry/Exit
    entry_type: OrderType = Field(..., description="Order type for entry")
    entry_price_limit: float | None = Field(None, description="Limit price if applicable")
    stop_loss: float | None = Field(None, description="Stop loss price")
    stop_loss_pct: float | None = Field(None, description="Stop loss as % from entry")
    take_profit: float | None = Field(None, description="Take profit price")
    take_profit_pct: float | None = Field(None, description="Take profit as % from entry")

    # Time management
    time_horizon: str = Field(..., description="Expected holding period")
    expiry: datetime | None = Field(None, description="Decision expiry if not executed")

    # Confidence and rationale
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=10, description="Full rationale for decision")
    what_would_change_mind: str = Field(
        ..., description="Conditions that would invalidate this decision"
    )

    # Audit trail
    technical_signal: TechnicalSignal | None = Field(None, description="Input technical signal")
    fundamental_signal: FundamentalSignal | None = Field(
        None, description="Input fundamental signal"
    )
    hybrid_signal: HybridSignal | None = Field(None, description="Input hybrid signal")
    rl_action: RLAction | None = Field(None, description="Input RL action")

    # Metadata
    decision_id: str = Field(..., description="Unique decision ID for tracking")
    timestamp: datetime


class RiskGuardianOutput(BaseModel):
    """Output from Risk Guardian.

    Deterministic risk checks and potential vetoes.
    """

    model_config = ConfigDict(frozen=True)

    # Approval
    approved: bool = Field(..., description="Whether the decision is approved")
    veto_reasons: list[VetoReason] = Field(
        default_factory=list, description="Reasons for veto if not approved"
    )

    # Modifications
    modified_size_pct: float | None = Field(
        None, description="Modified position size if reduced"
    )
    modifications_made: list[str] = Field(
        default_factory=list, description="List of modifications made"
    )

    # Risk metrics
    current_exposure: float = Field(..., description="Current portfolio exposure")
    post_trade_exposure: float = Field(..., description="Exposure after this trade")
    daily_pnl: float = Field(..., description="Current day P&L")
    trades_today: int = Field(..., description="Number of trades executed today")

    # Limits
    position_limit_used: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of position limit used"
    )
    daily_loss_limit_used: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of daily loss limit used"
    )

    # Original decision reference
    original_decision_id: str
    timestamp: datetime


# =============================================================================
# Pipeline State
# =============================================================================


class PipelineState(BaseModel):
    """Complete state passed through the agent pipeline."""

    # Symbol and timing
    symbol: str
    timestamp: datetime
    interval: str

    # Market data context
    current_price: float
    last_n_bars: list[dict] = Field(default_factory=list, description="Last N OHLCV bars")

    # Feature context
    features: dict = Field(default_factory=dict, description="Current feature values")

    # Portfolio context
    current_position: float = Field(default=0.0, description="Current position in shares")
    current_position_pct: float = Field(default=0.0, description="Position as % of equity")
    cash: float
    equity: float
    daily_pnl: float = Field(default=0.0)
    trades_today: int = Field(default=0)

    # Agent outputs (populated as pipeline progresses)
    technical_signal: TechnicalSignal | None = None
    fundamental_signal: FundamentalSignal | None = None
    hybrid_signal: HybridSignal | None = None
    rl_action: RLAction | None = None
    executive_decision: ExecutiveDecision | None = None
    risk_guardian_output: RiskGuardianOutput | None = None

    # Final output
    final_action: Action | None = None
    final_order: dict | None = None
