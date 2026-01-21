"""Execution package."""

from src.execution.broker import (
    Account,
    AlpacaBroker,
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    SimulatedBroker,
    TimeInForce,
    get_broker,
)
from src.execution.paper_trading import (
    PaperTradingLoop,
    TradingSchedule,
    run_paper_trading,
)
from src.execution.hil import (
    ApprovalRequest,
    ApprovalStatus,
    ConsoleNotifier,
    HILApprovalManager,
    NotificationChannel,
    SlackNotifier,
    TelegramNotifier,
)

__all__ = [
    # Broker
    "Account",
    "AlpacaBroker",
    "Broker",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "SimulatedBroker",
    "TimeInForce",
    "get_broker",
    # Paper Trading
    "PaperTradingLoop",
    "TradingSchedule",
    "run_paper_trading",
    # HIL
    "ApprovalRequest",
    "ApprovalStatus",
    "ConsoleNotifier",
    "HILApprovalManager",
    "NotificationChannel",
    "SlackNotifier",
    "TelegramNotifier",
]
