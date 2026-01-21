"""Risk Guardian Agent - Deterministic risk rule enforcement.

This agent applies strict, deterministic rules to approve, modify,
or veto trading decisions. It is NOT powered by LLM - purely rule-based.
"""

from datetime import datetime
import uuid

from src.agents.base import RuleBasedAgent
from src.orchestrator.contracts import (
    Action,
    ExecutiveDecision,
    PipelineState,
    RiskGuardianOutput,
    VetoReason,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


class RiskGuardian(RuleBasedAgent[RiskGuardianOutput]):
    """Deterministic risk rule enforcement agent.

    Checks all trading decisions against risk limits and can:
    - Approve the decision as-is
    - Reduce position size
    - Veto the entire decision

    This is the final safety check before execution.
    """

    def __init__(
        self,
        max_position_pct: float = 0.20,
        max_daily_loss_pct: float = 0.05,
        max_trades_per_day: int = 10,
        min_confidence: float = 0.5,
        max_portfolio_exposure: float = 0.80,
    ):
        """Initialize Risk Guardian.

        Args:
            max_position_pct: Maximum position size as % of equity (0.20 = 20%).
            max_daily_loss_pct: Maximum daily loss before stop (0.05 = 5%).
            max_trades_per_day: Maximum number of trades per day.
            min_confidence: Minimum confidence score required.
            max_portfolio_exposure: Maximum total portfolio exposure.
        """
        super().__init__(name="risk_guardian")
        
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_trades_per_day = max_trades_per_day
        self.min_confidence = min_confidence
        self.max_portfolio_exposure = max_portfolio_exposure

    @property
    def description(self) -> str:
        return "Deterministic risk rule enforcement - final safety check before execution"

    def _apply_rules(self, state: PipelineState) -> RiskGuardianOutput:
        """Apply all risk rules to the executive decision.

        Args:
            state: Current pipeline state with executive decision.

        Returns:
            Risk guardian output with approval/veto and modifications.
        """
        decision = state.executive_decision
        
        # If no decision or HOLD, auto-approve
        if decision is None or decision.action == Action.HOLD:
            return RiskGuardianOutput(
                approved=True,
                veto_reasons=[],
                modified_size_pct=None,
                modifications_made=[],
                current_exposure=abs(state.current_position_pct),
                post_trade_exposure=abs(state.current_position_pct),
                daily_pnl=state.daily_pnl,
                trades_today=state.trades_today,
                position_limit_used=abs(state.current_position_pct) / self.max_position_pct,
                daily_loss_limit_used=max(0, -state.daily_pnl / state.equity) / self.max_daily_loss_pct,
                original_decision_id=decision.decision_id if decision else "none",
                timestamp=datetime.now(),
            )

        veto_reasons: list[VetoReason] = []
        modifications: list[str] = []
        modified_size = decision.size_pct

        # Check 1: Maximum daily loss
        daily_loss_pct = -state.daily_pnl / state.equity if state.equity > 0 else 0
        if daily_loss_pct >= self.max_daily_loss_pct:
            veto_reasons.append(VetoReason.MAX_DAILY_LOSS)
            logger.warning(
                f"VETO: Daily loss {daily_loss_pct:.2%} exceeds limit {self.max_daily_loss_pct:.2%}"
            )

        # Check 2: Maximum trades per day
        if state.trades_today >= self.max_trades_per_day:
            veto_reasons.append(VetoReason.MAX_TRADES_EXCEEDED)
            logger.warning(
                f"VETO: Trades today {state.trades_today} exceeds limit {self.max_trades_per_day}"
            )

        # Check 3: Minimum confidence
        if decision.confidence < self.min_confidence:
            veto_reasons.append(VetoReason.LOW_CONFIDENCE)
            logger.warning(
                f"VETO: Confidence {decision.confidence:.2f} below minimum {self.min_confidence:.2f}"
            )

        # Check 4: Position size limit
        if decision.size_pct > self.max_position_pct:
            original_size = decision.size_pct
            modified_size = self.max_position_pct
            modifications.append(
                f"Reduced position size from {original_size:.2%} to {modified_size:.2%}"
            )
            logger.info(
                f"Modified: Position size {original_size:.2%} -> {modified_size:.2%}"
            )

        # Check 5: Total portfolio exposure
        current_exposure = abs(state.current_position_pct)
        proposed_additional = decision.size_pct if decision.action == Action.BUY else 0
        post_trade_exposure = current_exposure + proposed_additional
        
        if post_trade_exposure > self.max_portfolio_exposure:
            if not veto_reasons:  # Only if not already vetoed
                # Reduce size to fit within exposure limit
                max_additional = self.max_portfolio_exposure - current_exposure
                if max_additional > 0:
                    modified_size = min(modified_size, max_additional)
                    modifications.append(
                        f"Reduced size to fit exposure limit: {modified_size:.2%}"
                    )
                else:
                    veto_reasons.append(VetoReason.MAX_POSITION_EXCEEDED)
                    logger.warning("VETO: Portfolio exposure limit reached")

        # Determine final approval
        approved = len(veto_reasons) == 0

        # Calculate metrics
        position_limit_used = current_exposure / self.max_position_pct
        daily_loss_limit_used = max(0, daily_loss_pct) / self.max_daily_loss_pct

        return RiskGuardianOutput(
            approved=approved,
            veto_reasons=veto_reasons,
            modified_size_pct=modified_size if modifications else None,
            modifications_made=modifications,
            current_exposure=current_exposure,
            post_trade_exposure=post_trade_exposure if approved else current_exposure,
            daily_pnl=state.daily_pnl,
            trades_today=state.trades_today,
            position_limit_used=position_limit_used,
            daily_loss_limit_used=daily_loss_limit_used,
            original_decision_id=decision.decision_id,
            timestamp=datetime.now(),
        )


class KillSwitch:
    """Emergency kill switch for critical situations.

    Provides immediate trading halt capabilities.
    """

    def __init__(self):
        self._active = False
        self._reason: str | None = None
        self._activated_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def reason(self) -> str | None:
        return self._reason

    def activate(self, reason: str) -> None:
        """Activate the kill switch.

        Args:
            reason: Reason for activation.
        """
        self._active = True
        self._reason = reason
        self._activated_at = datetime.now()
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate(self) -> None:
        """Deactivate the kill switch."""
        logger.warning(f"Kill switch deactivated (was active for {self._reason})")
        self._active = False
        self._reason = None
        self._activated_at = None

    def check(self) -> bool:
        """Check if trading is allowed.

        Returns:
            True if trading is allowed, False if kill switch is active.

        Raises:
            RuntimeError: If kill switch is active.
        """
        if self._active:
            raise RuntimeError(f"Kill switch active: {self._reason}")
        return True
