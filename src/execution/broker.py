"""Broker interface and implementations.

Provides abstract broker interface and concrete implementations
for order execution (Alpaca, simulation).
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid

from src.utils.logging import get_logger


logger = get_logger(__name__)


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    PENDING_NEW = "pending_new"
    NEW = "new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    DONE_FOR_DAY = "done_for_day"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    REPLACED = "replaced"
    STOPPED = "stopped"
    SUSPENDED = "suspended"


class TimeInForce(str, Enum):
    """Time in force."""
    DAY = "day"
    GTC = "gtc"  # Good til canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


@dataclass
class Order:
    """Order representation."""
    id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None
    canceled_at: datetime | None = None
    submitted_at: datetime | None = None
    extended_hours: bool = False


@dataclass
class Position:
    """Position representation."""
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float  # Percent
    current_price: float
    side: str  # "long" or "short"


@dataclass
class Account:
    """Account representation."""
    account_id: str
    status: str
    currency: str
    cash: float
    portfolio_value: float
    buying_power: float
    equity: float
    last_equity: float
    long_market_value: float
    short_market_value: float
    initial_margin: float
    maintenance_margin: float
    daytrade_count: int
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    pattern_day_trader: bool


class Broker(ABC):
    """Abstract broker interface."""

    @abstractmethod
    async def get_account(self) -> Account:
        """Get account information."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        pass

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit an order."""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_orders(
        self,
        status: str = "open",
        limit: int = 100,
    ) -> list[Order]:
        """Get orders."""
        pass

    @abstractmethod
    async def close_position(self, symbol: str) -> Order | None:
        """Close a position."""
        pass

    @abstractmethod
    async def close_all_positions(self) -> list[Order]:
        """Close all positions."""
        pass


class AlpacaBroker(Broker):
    """Alpaca broker implementation.

    Supports both paper and live trading through the Alpaca API.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
    ):
        """Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key.
            api_secret: Alpaca API secret.
            paper: Use paper trading (default True).
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self._client = None

    async def _get_client(self):
        """Get or create Alpaca trading client."""
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
                self._client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                    paper=self.paper,
                )
            except ImportError:
                raise ImportError("alpaca-py package required: pip install alpaca-py")
        return self._client

    async def get_account(self) -> Account:
        """Get account information."""
        client = await self._get_client()
        acc = client.get_account()

        return Account(
            account_id=acc.id,
            status=str(acc.status),
            currency=acc.currency,
            cash=float(acc.cash),
            portfolio_value=float(acc.portfolio_value),
            buying_power=float(acc.buying_power),
            equity=float(acc.equity),
            last_equity=float(acc.last_equity),
            long_market_value=float(acc.long_market_value),
            short_market_value=float(acc.short_market_value),
            initial_margin=float(acc.initial_margin),
            maintenance_margin=float(acc.maintenance_margin),
            daytrade_count=acc.daytrade_count,
            trading_blocked=acc.trading_blocked,
            transfers_blocked=acc.transfers_blocked,
            account_blocked=acc.account_blocked,
            pattern_day_trader=acc.pattern_day_trader,
        )

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        client = await self._get_client()
        positions = client.get_all_positions()

        return [
            Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                cost_basis=float(p.cost_basis),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
                current_price=float(p.current_price),
                side=str(p.side),
            )
            for p in positions
        ]

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        client = await self._get_client()
        try:
            p = client.get_open_position(symbol)
            return Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                cost_basis=float(p.cost_basis),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
                current_price=float(p.current_price),
                side=str(p.side),
            )
        except Exception:
            return None

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit an order."""
        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
            StopOrderRequest,
            StopLimitOrderRequest,
        )
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce as AlpacaTIF

        client = await self._get_client()

        # Map enums
        alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL
        alpaca_tif = getattr(AlpacaTIF, time_in_force.value.upper())

        # Build request based on order type
        if order_type == OrderType.MARKET:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                client_order_id=client_order_id,
            )
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("limit_price required for limit order")
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise ValueError("stop_price required for stop order")
            request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                stop_price=stop_price,
                client_order_id=client_order_id,
            )
        elif order_type == OrderType.STOP_LIMIT:
            if limit_price is None or stop_price is None:
                raise ValueError("limit_price and stop_price required for stop-limit order")
            request = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id,
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        # Submit order
        order = client.submit_order(request)

        logger.info(f"Submitted order: {order.id} {side.value} {qty} {symbol}")

        return Order(
            id=str(order.id),
            client_order_id=order.client_order_id or "",
            symbol=order.symbol,
            side=OrderSide(order.side.value),
            order_type=OrderType(order.order_type.value),
            qty=float(order.qty),
            limit_price=float(order.limit_price) if order.limit_price else None,
            stop_price=float(order.stop_price) if order.stop_price else None,
            time_in_force=TimeInForce(order.time_in_force.value),
            status=OrderStatus(order.status.value),
            filled_qty=float(order.filled_qty) if order.filled_qty else 0.0,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            created_at=order.created_at,
            submitted_at=order.submitted_at,
        )

    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        client = await self._get_client()
        try:
            order = client.get_order_by_id(order_id)
            return Order(
                id=str(order.id),
                client_order_id=order.client_order_id or "",
                symbol=order.symbol,
                side=OrderSide(order.side.value),
                order_type=OrderType(order.order_type.value),
                qty=float(order.qty),
                limit_price=float(order.limit_price) if order.limit_price else None,
                stop_price=float(order.stop_price) if order.stop_price else None,
                time_in_force=TimeInForce(order.time_in_force.value),
                status=OrderStatus(order.status.value),
                filled_qty=float(order.filled_qty) if order.filled_qty else 0.0,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                created_at=order.created_at,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                canceled_at=order.canceled_at,
            )
        except Exception:
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        client = await self._get_client()
        try:
            client.cancel_order_by_id(order_id)
            logger.info(f"Canceled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_orders(
        self,
        status: str = "open",
        limit: int = 100,
    ) -> list[Order]:
        """Get orders."""
        from alpaca.trading.requests import GetOrdersRequest

        client = await self._get_client()
        request = GetOrdersRequest(status=status, limit=limit)
        orders = client.get_orders(filter=request)

        return [
            Order(
                id=str(o.id),
                client_order_id=o.client_order_id or "",
                symbol=o.symbol,
                side=OrderSide(o.side.value),
                order_type=OrderType(o.order_type.value),
                qty=float(o.qty),
                limit_price=float(o.limit_price) if o.limit_price else None,
                stop_price=float(o.stop_price) if o.stop_price else None,
                time_in_force=TimeInForce(o.time_in_force.value),
                status=OrderStatus(o.status.value),
                filled_qty=float(o.filled_qty) if o.filled_qty else 0.0,
                filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                created_at=o.created_at,
            )
            for o in orders
        ]

    async def close_position(self, symbol: str) -> Order | None:
        """Close a position."""
        client = await self._get_client()
        try:
            order = client.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            return Order(
                id=str(order.id),
                client_order_id=order.client_order_id or "",
                symbol=order.symbol,
                side=OrderSide(order.side.value),
                order_type=OrderType(order.order_type.value),
                qty=float(order.qty),
                status=OrderStatus(order.status.value),
                created_at=order.created_at,
            )
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return None

    async def close_all_positions(self) -> list[Order]:
        """Close all positions."""
        client = await self._get_client()
        orders = client.close_all_positions(cancel_orders=True)
        logger.info(f"Closed all positions: {len(orders)} orders")
        return [
            Order(
                id=str(o.id),
                client_order_id=o.client_order_id or "",
                symbol=o.symbol,
                side=OrderSide(o.side.value),
                order_type=OrderType(o.order_type.value),
                qty=float(o.qty),
                status=OrderStatus(o.status.value),
                created_at=o.created_at,
            )
            for o in orders
        ]


class SimulatedBroker(Broker):
    """Simulated broker for backtesting and testing.

    Maintains virtual positions and fills orders instantly.
    """

    def __init__(
        self,
        initial_cash: float = 100_000,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
    ):
        """Initialize simulated broker.

        Args:
            initial_cash: Starting cash balance.
            slippage_bps: Slippage in basis points.
            commission_per_share: Commission per share.
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share

        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._prices: dict[str, float] = {}

    def set_price(self, symbol: str, price: float):
        """Set current price for a symbol (for simulation)."""
        self._prices[symbol] = price

    async def get_account(self) -> Account:
        """Get account information."""
        long_value = sum(
            p.market_value for p in self._positions.values() if p.qty > 0
        )
        short_value = sum(
            abs(p.market_value) for p in self._positions.values() if p.qty < 0
        )
        equity = self.cash + long_value - short_value

        return Account(
            account_id="SIM001",
            status="ACTIVE",
            currency="USD",
            cash=self.cash,
            portfolio_value=equity,
            buying_power=self.cash,
            equity=equity,
            last_equity=equity,
            long_market_value=long_value,
            short_market_value=short_value,
            initial_margin=0.0,
            maintenance_margin=0.0,
            daytrade_count=0,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            pattern_day_trader=False,
        )

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit and fill order immediately (market orders)."""
        order_id = str(uuid.uuid4())
        client_id = client_order_id or str(uuid.uuid4())

        # Get price
        price = self._prices.get(symbol, 100.0)

        # Apply slippage
        slippage = price * self.slippage_bps / 10000
        if side == OrderSide.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage

        # Calculate commission
        commission = qty * self.commission_per_share

        # Create order
        order = Order(
            id=order_id,
            client_order_id=client_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            status=OrderStatus.FILLED,
            filled_qty=qty,
            filled_avg_price=fill_price,
            filled_at=datetime.now(),
            submitted_at=datetime.now(),
        )

        self._orders[order_id] = order

        # Update position
        position = self._positions.get(symbol)
        if side == OrderSide.BUY:
            if position:
                # Average up
                total_qty = position.qty + qty
                total_cost = position.cost_basis + qty * fill_price
                position.qty = total_qty
                position.avg_entry_price = total_cost / total_qty if total_qty > 0 else 0
                position.cost_basis = total_cost
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=fill_price,
                    market_value=qty * price,
                    cost_basis=qty * fill_price,
                    unrealized_pl=0.0,
                    unrealized_plpc=0.0,
                    current_price=price,
                    side="long",
                )
            self.cash -= qty * fill_price + commission
        else:  # SELL
            if position:
                position.qty -= qty
                if position.qty <= 0:
                    del self._positions[symbol]
            self.cash += qty * fill_price - commission

        # Update position market values
        for sym, pos in self._positions.items():
            pos.current_price = self._prices.get(sym, pos.current_price)
            pos.market_value = pos.qty * pos.current_price
            pos.unrealized_pl = pos.market_value - pos.cost_basis
            pos.unrealized_plpc = pos.unrealized_pl / pos.cost_basis if pos.cost_basis > 0 else 0

        logger.info(f"Simulated fill: {side.value} {qty} {symbol} @ {fill_price:.2f}")
        return order

    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELED
            order.canceled_at = datetime.now()
            return True
        return False

    async def get_orders(
        self,
        status: str = "open",
        limit: int = 100,
    ) -> list[Order]:
        """Get orders."""
        orders = list(self._orders.values())
        if status == "open":
            orders = [o for o in orders if o.status in [OrderStatus.PENDING, OrderStatus.NEW]]
        return orders[:limit]

    async def close_position(self, symbol: str) -> Order | None:
        """Close a position."""
        position = self._positions.get(symbol)
        if not position:
            return None

        side = OrderSide.SELL if position.qty > 0 else OrderSide.BUY
        return await self.submit_order(symbol, abs(position.qty), side)

    async def close_all_positions(self) -> list[Order]:
        """Close all positions."""
        orders = []
        for symbol in list(self._positions.keys()):
            order = await self.close_position(symbol)
            if order:
                orders.append(order)
        return orders


def get_broker(
    broker_type: str = "alpaca",
    **kwargs,
) -> Broker:
    """Factory function to get broker instance.

    Args:
        broker_type: "alpaca" or "simulated".
        **kwargs: Broker-specific arguments.

    Returns:
        Broker instance.
    """
    if broker_type == "alpaca":
        return AlpacaBroker(**kwargs)
    elif broker_type == "simulated":
        return SimulatedBroker(**kwargs)
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
