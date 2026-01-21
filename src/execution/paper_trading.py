"""Paper trading loop for live trading.

Runs the agent pipeline on a schedule against live market data.
"""

import asyncio
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Any, Callable
import uuid
import pytz

from src.execution.broker import Broker, AlpacaBroker, OrderSide, OrderType as BrokerOrderType
from src.data.market_data import get_provider
from src.data.feature_store import FeatureBuilder
from src.orchestrator.contracts import Action, PipelineState
from src.orchestrator.crew import AgentOrchestrator
from src.agents.risk_guardian import KillSwitch
from src.utils.logging import get_logger


logger = get_logger(__name__)


class TradingSchedule:
    """Market trading schedule."""

    def __init__(
        self,
        timezone: str = "America/New_York",
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
        trading_days: list[int] | None = None,  # 0=Monday, 6=Sunday
    ):
        """Initialize trading schedule.

        Args:
            timezone: Market timezone.
            market_open: Market open time.
            market_close: Market close time.
            trading_days: Days when market is open (0-6).
        """
        self.tz = pytz.timezone(timezone)
        self.market_open = market_open
        self.market_close = market_close
        self.trading_days = trading_days or [0, 1, 2, 3, 4]  # Mon-Fri

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Check if market is currently open."""
        if dt is None:
            dt = datetime.now(self.tz)
        else:
            dt = dt.astimezone(self.tz)

        # Check day
        if dt.weekday() not in self.trading_days:
            return False

        # Check time
        current_time = dt.time()
        return self.market_open <= current_time <= self.market_close

    def next_open(self, dt: datetime | None = None) -> datetime:
        """Get next market open time."""
        if dt is None:
            dt = datetime.now(self.tz)
        else:
            dt = dt.astimezone(self.tz)

        # If market is open now, return now
        if self.is_market_open(dt):
            return dt

        # Find next trading day
        next_dt = dt
        while True:
            # If before market open today and it's a trading day
            if next_dt.weekday() in self.trading_days and next_dt.time() < self.market_open:
                return next_dt.replace(
                    hour=self.market_open.hour,
                    minute=self.market_open.minute,
                    second=0,
                    microsecond=0,
                )

            # Move to next day
            next_dt = (next_dt + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            if next_dt.weekday() in self.trading_days:
                return next_dt.replace(
                    hour=self.market_open.hour,
                    minute=self.market_open.minute,
                )

    def time_until_close(self, dt: datetime | None = None) -> timedelta:
        """Get time remaining until market close."""
        if dt is None:
            dt = datetime.now(self.tz)
        else:
            dt = dt.astimezone(self.tz)

        close_dt = dt.replace(
            hour=self.market_close.hour,
            minute=self.market_close.minute,
            second=0,
            microsecond=0,
        )

        return close_dt - dt


class PaperTradingLoop:
    """Paper trading loop.

    Runs the agent pipeline on a schedule, executing trades
    through the broker.
    """

    def __init__(
        self,
        broker: Broker,
        orchestrator: AgentOrchestrator,
        symbols: list[str],
        interval_minutes: int = 60,
        data_lookback_days: int = 30,
        schedule: TradingSchedule | None = None,
        kill_switch: KillSwitch | None = None,
        on_decision: Callable[[PipelineState], None] | None = None,
        on_order: Callable[[Any], None] | None = None,
    ):
        """Initialize paper trading loop.

        Args:
            broker: Broker instance.
            orchestrator: Agent orchestrator.
            symbols: Symbols to trade.
            interval_minutes: Minutes between runs.
            data_lookback_days: Days of historical data for features.
            schedule: Trading schedule.
            kill_switch: Emergency kill switch.
            on_decision: Callback for decisions.
            on_order: Callback for orders.
        """
        self.broker = broker
        self.orchestrator = orchestrator
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self.data_lookback_days = data_lookback_days
        self.schedule = schedule or TradingSchedule()
        self.kill_switch = kill_switch or KillSwitch()
        self.on_decision = on_decision
        self.on_order = on_order

        self.feature_builder = FeatureBuilder()
        self._running = False
        self._run_id = str(uuid.uuid4())[:8]

    async def start(self):
        """Start the trading loop."""
        logger.info(f"Starting paper trading loop (run_id={self._run_id})")
        self._running = True

        while self._running:
            try:
                # Check kill switch
                if self.kill_switch.is_active:
                    logger.warning("Kill switch active, stopping")
                    break

                # Wait for market open
                if not self.schedule.is_market_open():
                    next_open = self.schedule.next_open()
                    wait_seconds = (next_open - datetime.now(self.schedule.tz)).total_seconds()
                    logger.info(f"Market closed. Next open: {next_open}")
                    await asyncio.sleep(min(wait_seconds, 60))  # Check every minute
                    continue

                # Run trading cycle
                await self._run_cycle()

                # Wait for next interval
                await asyncio.sleep(self.interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    def stop(self):
        """Stop the trading loop."""
        logger.info("Stopping paper trading loop")
        self._running = False

    async def _run_cycle(self):
        """Run one trading cycle for all symbols."""
        logger.info(f"Starting trading cycle at {datetime.now()}")

        # Get account state
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()
        position_map = {p.symbol: p for p in positions}

        logger.info(f"Account equity: ${account.equity:,.2f}")

        for symbol in self.symbols:
            try:
                await self._process_symbol(symbol, account, position_map.get(symbol))
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    async def _process_symbol(
        self,
        symbol: str,
        account: Any,
        position: Any | None,
    ):
        """Process a single symbol."""
        logger.info(f"Processing {symbol}")

        # Get market data
        provider = get_provider("yfinance")
        end = datetime.now()
        start = end - timedelta(days=self.data_lookback_days)

        data = await provider.fetch_ohlcv(symbol, start, end, "1h")
        if data.empty:
            logger.warning(f"No data for {symbol}")
            return

        # Build features
        features_df = self.feature_builder.build_features(data)
        latest_features = features_df.iloc[-1].to_dict()
        current_price = float(data.iloc[-1]["close"])

        # Build state
        current_qty = position.qty if position else 0
        current_value = current_qty * current_price
        current_pct = current_value / account.equity if account.equity > 0 else 0

        state = PipelineState(
            symbol=symbol,
            timestamp=datetime.now(),
            interval="1h",
            current_price=current_price,
            last_n_bars=data.tail(20).to_dict("records"),
            features=latest_features,
            current_position=current_qty,
            current_position_pct=current_pct,
            cash=account.cash,
            equity=account.equity,
            daily_pnl=0.0,  # Would need to track this
            trades_today=0,  # Would need to track this
        )

        # Run pipeline
        state = await self.orchestrator.run_pipeline(state)

        # Callback
        if self.on_decision:
            self.on_decision(state)

        # Execute if approved
        if state.final_action and state.final_action != Action.HOLD:
            await self._execute_decision(state, account)

    async def _execute_decision(
        self,
        state: PipelineState,
        account: Any,
    ):
        """Execute a trading decision."""
        decision = state.executive_decision
        if not decision:
            return

        # Calculate quantity
        if decision.action == Action.BUY:
            notional = account.equity * decision.size_pct
            qty = int(notional / state.current_price)

            if qty <= 0:
                logger.info(f"Order too small for {state.symbol}")
                return

            # Submit order
            order = await self.broker.submit_order(
                symbol=state.symbol,
                qty=qty,
                side=OrderSide.BUY,
                order_type=BrokerOrderType.MARKET,
            )

            logger.info(f"BUY order submitted: {qty} {state.symbol}")

            if self.on_order:
                self.on_order(order)

        elif decision.action == Action.SELL:
            # Get current position
            position = await self.broker.get_position(state.symbol)
            if not position or position.qty <= 0:
                logger.info(f"No position to sell for {state.symbol}")
                return

            qty = int(position.qty * decision.size_pct)
            if qty <= 0:
                qty = int(position.qty)  # Sell all

            order = await self.broker.submit_order(
                symbol=state.symbol,
                qty=qty,
                side=OrderSide.SELL,
                order_type=BrokerOrderType.MARKET,
            )

            logger.info(f"SELL order submitted: {qty} {state.symbol}")

            if self.on_order:
                self.on_order(order)


async def run_paper_trading(
    symbols: list[str],
    api_key: str,
    api_secret: str,
    openai_api_key: str | None = None,
    interval_minutes: int = 60,
    paper: bool = True,
):
    """Run paper trading.

    Args:
        symbols: Symbols to trade.
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        openai_api_key: OpenAI API key for LLM agents.
        interval_minutes: Minutes between cycles.
        paper: Use paper trading.
    """
    # Initialize broker
    broker = AlpacaBroker(
        api_key=api_key,
        api_secret=api_secret,
        paper=paper,
    )

    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        openai_api_key=openai_api_key,
        config={"use_llm": openai_api_key is not None},
    )

    # Initialize loop
    loop = PaperTradingLoop(
        broker=broker,
        orchestrator=orchestrator,
        symbols=symbols,
        interval_minutes=interval_minutes,
    )

    # Start trading
    await loop.start()
