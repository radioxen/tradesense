"""Event-driven backtesting engine.

Provides a realistic simulation environment with:
- No lookahead bias (only uses data <= current timestamp)
- Configurable slippage and commission models
- Portfolio accounting
- Full audit trail
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import uuid

import pandas as pd
import numpy as np

from src.orchestrator.contracts import Action, ExecutiveDecision, PipelineState
from src.utils.logging import get_logger


logger = get_logger(__name__)


class FillModel(str, Enum):
    """Order fill model."""

    NEXT_OPEN = "next_open"  # Fill at next bar's open
    CURRENT_CLOSE = "current_close"  # Fill at current bar's close
    MIDPOINT = "midpoint"  # Fill at (high + low) / 2


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    initial_capital: float = 100_000.0

    # Commission model
    commission_type: str = "per_share"  # 'per_share', 'fixed', 'percent'
    commission_value: float = 0.005  # $0.005 per share

    # Slippage model
    slippage_bps: float = 5.0  # 5 basis points

    # Fill model
    fill_model: FillModel = FillModel.NEXT_OPEN

    # Constraints
    allow_fractional: bool = False
    min_trade_value: float = 100.0  # Minimum trade notional

    # Portfolio
    max_position_pct: float = 0.20  # Maximum 20% in one position


@dataclass
class Position:
    """Portfolio position."""

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0:
            return 0.0
        return (self.current_price - self.avg_cost) * self.quantity


@dataclass
class Order:
    """Trade order."""

    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: str  # 'market', 'limit'
    limit_price: float | None = None
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None
    fill_price: float | None = None
    status: str = "pending"  # pending, filled, cancelled


@dataclass
class Trade:
    """Executed trade."""

    id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    slippage_bps: float
    timestamp: datetime
    notional: float = 0.0


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""

    timestamp: datetime
    cash: float
    positions: dict[str, Position]
    equity: float
    daily_pnl: float
    drawdown: float
    trades_today: int


class Portfolio:
    """Portfolio manager with position tracking and P&L calculation."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self._daily_start_equity: float = initial_capital
        self._high_water_mark: float = initial_capital
        self._trades_today: int = 0
        self._last_date: datetime | None = None

    @property
    def equity(self) -> float:
        """Calculate total portfolio equity."""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value

    @property
    def daily_pnl(self) -> float:
        """Calculate today's P&L."""
        return self.equity - self._daily_start_equity

    @property
    def drawdown(self) -> float:
        """Calculate current drawdown from high water mark."""
        if self._high_water_mark == 0:
            return 0.0
        return (self._high_water_mark - self.equity) / self._high_water_mark

    @property
    def trades_today(self) -> int:
        return self._trades_today

    def new_day(self, date: datetime) -> None:
        """Reset daily counters for a new trading day."""
        if self._last_date is None or date.date() != self._last_date.date():
            self._daily_start_equity = self.equity
            self._trades_today = 0
            self._last_date = date
            # Update high water mark
            if self.equity > self._high_water_mark:
                self._high_water_mark = self.equity

    def get_position(self, symbol: str) -> Position:
        """Get or create position for symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def execute_trade(self, trade: Trade) -> None:
        """Execute a trade and update portfolio."""
        position = self.get_position(trade.symbol)
        notional = trade.quantity * trade.price

        if trade.side == "BUY":
            # Update average cost
            total_cost = position.avg_cost * position.quantity + notional
            new_quantity = position.quantity + trade.quantity
            if new_quantity > 0:
                position.avg_cost = total_cost / new_quantity
            position.quantity = new_quantity

            # Deduct cash (including commission)
            self.cash -= notional + trade.commission

        else:  # SELL
            if trade.quantity > position.quantity:
                raise ValueError(f"Cannot sell {trade.quantity}, only have {position.quantity}")

            # Calculate realized P&L
            realized = (trade.price - position.avg_cost) * trade.quantity
            position.realized_pnl += realized

            # Update position
            position.quantity -= trade.quantity

            # Add cash (minus commission)
            self.cash += notional - trade.commission

        position.current_price = trade.price
        self._trades_today += 1

    def get_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=dict(self.positions),
            equity=self.equity,
            daily_pnl=self.daily_pnl,
            drawdown=self.drawdown,
            trades_today=self._trades_today,
        )


class ExecutionSimulator:
    """Simulates order execution with realistic slippage and commission."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade."""
        if self.config.commission_type == "per_share":
            return abs(quantity) * self.config.commission_value
        elif self.config.commission_type == "fixed":
            return self.config.commission_value
        elif self.config.commission_type == "percent":
            return abs(quantity) * price * self.config.commission_value / 100
        return 0.0

    def calculate_slippage(self, price: float, side: str) -> float:
        """Calculate slippage (always adverse)."""
        slippage_pct = self.config.slippage_bps / 10000
        if side == "BUY":
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)

    def fill_order(
        self,
        order: Order,
        current_bar: dict,
        next_bar: dict | None = None,
    ) -> Trade | None:
        """Fill an order based on the fill model."""
        # Determine fill price based on model
        if self.config.fill_model == FillModel.NEXT_OPEN:
            if next_bar is None:
                return None  # Can't fill without next bar
            base_price = next_bar["open"]
        elif self.config.fill_model == FillModel.CURRENT_CLOSE:
            base_price = current_bar["close"]
        else:  # MIDPOINT
            base_price = (current_bar["high"] + current_bar["low"]) / 2

        # Apply slippage
        fill_price = self.calculate_slippage(base_price, order.side)

        # Calculate commission
        commission = self.calculate_commission(order.quantity, fill_price)

        # Create trade
        trade = Trade(
            id=str(uuid.uuid4()),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage_bps=self.config.slippage_bps,
            timestamp=datetime.now(),
            notional=order.quantity * fill_price,
        )

        # Update order status
        order.status = "filled"
        order.filled_at = trade.timestamp
        order.fill_price = fill_price

        return trade


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(
        self,
        config: BacktestConfig,
        decision_callback: Callable[[PipelineState], ExecutiveDecision | None],
    ):
        """Initialize backtest engine.

        Args:
            config: Backtest configuration.
            decision_callback: Async function that takes pipeline state and returns decision.
        """
        self.config = config
        self.decision_callback = decision_callback
        self.portfolio = Portfolio(config.initial_capital)
        self.simulator = ExecutionSimulator(config)

        # Results storage
        self.trades: list[Trade] = []
        self.orders: list[Order] = []
        self.decisions: list[dict] = []
        self.equity_curve: list[dict] = []

    async def run(
        self,
        symbol: str,
        data: pd.DataFrame,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run backtest on historical data.

        Args:
            symbol: Trading symbol.
            data: OHLCV dataframe with timestamp index.
            features: Features dataframe aligned with data.

        Returns:
            Dictionary with backtest results and metrics.
        """
        logger.info(f"Starting backtest for {symbol}: {len(data)} bars")

        # Ensure data is sorted by timestamp
        data = data.sort_values("timestamp").reset_index(drop=True)
        features = features.sort_values("timestamp").reset_index(drop=True)

        n_bars = len(data)
        pending_orders: list[Order] = []

        for i in range(n_bars):
            current_bar = data.iloc[i].to_dict()
            timestamp = pd.Timestamp(current_bar["timestamp"])

            # Get next bar for fill (if available)
            next_bar = data.iloc[i + 1].to_dict() if i + 1 < n_bars else None

            # New day check
            self.portfolio.new_day(timestamp)

            # Update portfolio prices
            self.portfolio.update_prices({symbol: current_bar["close"]})

            # Fill pending orders from previous bar
            for order in pending_orders[:]:
                trade = self.simulator.fill_order(order, current_bar, next_bar)
                if trade:
                    self.portfolio.execute_trade(trade)
                    self.trades.append(trade)
                    pending_orders.remove(order)
                    logger.debug(f"Filled order: {order.side} {order.quantity} @ {trade.price}")

            # Build pipeline state
            position = self.portfolio.get_position(symbol)
            state = PipelineState(
                symbol=symbol,
                timestamp=timestamp,
                interval="1h",
                current_price=current_bar["close"],
                last_n_bars=data.iloc[max(0, i - 10):i + 1].to_dict("records"),
                features=features.iloc[i].to_dict() if i < len(features) else {},
                current_position=position.quantity,
                current_position_pct=position.market_value / self.portfolio.equity if self.portfolio.equity > 0 else 0,
                cash=self.portfolio.cash,
                equity=self.portfolio.equity,
                daily_pnl=self.portfolio.daily_pnl,
                trades_today=self.portfolio.trades_today,
            )

            # Get decision from callback
            try:
                decision = await self.decision_callback(state)

                if decision and decision.action != Action.HOLD:
                    # Create order
                    order = self._create_order(decision, state)
                    if order:
                        self.orders.append(order)
                        pending_orders.append(order)

                        self.decisions.append({
                            "timestamp": timestamp,
                            "action": decision.action.value,
                            "size_pct": decision.size_pct,
                            "confidence": decision.confidence,
                            "rationale": decision.rationale,
                        })

            except Exception as e:
                logger.error(f"Decision error at {timestamp}: {e}")

            # Record equity curve
            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": self.portfolio.equity,
                "cash": self.portfolio.cash,
                "position_value": self.portfolio.equity - self.portfolio.cash,
                "daily_pnl": self.portfolio.daily_pnl,
                "drawdown": self.portfolio.drawdown,
            })

        # Calculate final metrics
        metrics = self._calculate_metrics()
        logger.info(f"Backtest complete: {metrics['total_return']:.2%} return, {len(self.trades)} trades")

        return {
            "metrics": metrics,
            "trades": self.trades,
            "equity_curve": pd.DataFrame(self.equity_curve),
            "decisions": self.decisions,
        }

    def _create_order(
        self,
        decision: ExecutiveDecision,
        state: PipelineState,
    ) -> Order | None:
        """Create an order from executive decision."""
        # Calculate quantity
        if decision.action == Action.BUY:
            # Calculate how much we can buy
            max_notional = self.portfolio.equity * decision.size_pct
            quantity = max_notional / state.current_price

            if not self.config.allow_fractional:
                quantity = int(quantity)

            if quantity * state.current_price < self.config.min_trade_value:
                return None

            return Order(
                id=str(uuid.uuid4()),
                symbol=state.symbol,
                side="BUY",
                quantity=quantity,
                order_type=decision.entry_type.value,
                limit_price=decision.entry_price_limit,
            )

        elif decision.action == Action.SELL:
            position = self.portfolio.get_position(state.symbol)
            if position.quantity <= 0:
                return None

            # Sell specified percentage of position
            quantity = position.quantity * decision.size_pct

            if not self.config.allow_fractional:
                quantity = int(quantity)

            if quantity <= 0:
                return None

            return Order(
                id=str(uuid.uuid4()),
                symbol=state.symbol,
                side="SELL",
                quantity=quantity,
                order_type=decision.entry_type.value,
                limit_price=decision.entry_price_limit,
            )

        return None

    def _calculate_metrics(self) -> dict[str, Any]:
        """Calculate backtest performance metrics."""
        if not self.equity_curve:
            return {}

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
        equity_df = equity_df.set_index("timestamp")

        # Basic returns
        initial_equity = self.config.initial_capital
        final_equity = self.portfolio.equity
        total_return = (final_equity - initial_equity) / initial_equity

        # Daily returns for Sharpe
        equity_df["returns"] = equity_df["equity"].pct_change()
        daily_returns = equity_df["returns"].dropna()

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Maximum drawdown
        rolling_max = equity_df["equity"].expanding().max()
        drawdown = (rolling_max - equity_df["equity"]) / rolling_max
        max_drawdown = drawdown.max()

        # Win rate (based on closed trades)
        trade_pnls = self._calculate_trade_pnls()
        closed_trades = len(trade_pnls)
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        win_rate = winning_trades / closed_trades if closed_trades > 0 else 0.0
        num_trades = len(self.trades)

        # CAGR
        if len(equity_df) > 1:
            days = (equity_df.index[-1] - equity_df.index[0]).days
            if days > 0:
                cagr = (final_equity / initial_equity) ** (365 / days) - 1
            else:
                cagr = 0.0
        else:
            cagr = 0.0

        return {
            "initial_capital": initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_trades": num_trades,
            "closed_trades": closed_trades,
            "win_rate": win_rate,
            "total_commission": sum(t.commission for t in self.trades),
        }

    def _calculate_trade_pnls(self) -> list[float]:
        """Calculate realized P&L per closed trade."""
        trade_pnls: list[float] = []
        positions: dict[str, dict[str, float]] = {}

        for trade in self.trades:
            pos = positions.get(trade.symbol, {"qty": 0.0, "avg_cost": 0.0})

            if trade.side == "BUY":
                total_cost = pos["avg_cost"] * pos["qty"] + trade.quantity * trade.price
                pos["qty"] += trade.quantity
                pos["avg_cost"] = total_cost / pos["qty"] if pos["qty"] > 0 else 0.0
            else:
                sell_qty = min(trade.quantity, pos["qty"])
                if sell_qty > 0:
                    realized = (trade.price - pos["avg_cost"]) * sell_qty - trade.commission
                    trade_pnls.append(realized)
                    pos["qty"] -= sell_qty
                    if pos["qty"] <= 0:
                        pos = {"qty": 0.0, "avg_cost": 0.0}

            positions[trade.symbol] = pos

        return trade_pnls
