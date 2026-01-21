"""Background scheduler for automated trading."""

import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import structlog

from api.state import AppState

logger = structlog.get_logger()

scheduler: AsyncIOScheduler | None = None


async def hourly_analysis(state: AppState):
    """Run hourly analysis of positions."""
    logger.info("Starting scheduled position analysis")
    
    try:
        # Get current positions
        positions = await state.broker.get_positions()
        
        if not positions:
            logger.info("No positions to analyze")
            return
        
        from src.data.market_data import get_provider
        from src.data.feature_store import FeatureBuilder
        from datetime import timedelta
        
        provider = get_provider("yfinance")
        feature_builder = FeatureBuilder()
        
        for pos in positions:
            try:
                # Fetch latest data
                end = datetime.now()
                start = end - timedelta(days=30)
                data = await provider.fetch_ohlcv(pos.symbol, start, end, "1d")
                
                if data.empty:
                    continue
                
                features = feature_builder.build_features(data)
                latest = features.iloc[-1] if not features.empty else {}
                
                rsi = latest.get("rsi", 50)
                pnl_pct = pos.unrealized_plpc * 100
                
                # Simple decision logic
                action = "hold"
                reasoning = ""
                
                # Take profit at 5%+
                if pnl_pct >= 5:
                    action = "sell"
                    reasoning = f"Taking profit at {pnl_pct:.1f}%"
                # Cut losses at -3%
                elif pnl_pct <= -3:
                    action = "sell"
                    reasoning = f"Cutting losses at {pnl_pct:.1f}%"
                # RSI overbought
                elif rsi > 75:
                    action = "sell"
                    reasoning = f"RSI overbought ({rsi:.1f})"
                else:
                    action = "hold"
                    reasoning = f"Holding - P&L {pnl_pct:.1f}%, RSI {rsi:.1f}"
                
                state.log_activity({
                    "type": "scheduled_analysis",
                    "agent": "Scheduler",
                    "symbol": pos.symbol,
                    "action": action.upper(),
                    "details": reasoning,
                })
                
                # Execute if sell recommended
                if action == "sell":
                    from src.execution.broker import OrderSide
                    order = await state.broker.submit_order(
                        symbol=pos.symbol,
                        qty=pos.qty,
                        side=OrderSide.SELL,
                    )
                    
                    state.log_activity({
                        "type": "trade",
                        "action": "SELL",
                        "symbol": pos.symbol,
                        "details": f"Auto-sold {pos.qty} shares - {reasoning}",
                    })
                    
                    logger.info(
                        "Executed scheduled sell",
                        symbol=pos.symbol,
                        qty=pos.qty,
                        reason=reasoning,
                    )
                    
            except Exception as e:
                logger.error(
                    "Failed to analyze position",
                    symbol=pos.symbol,
                    error=str(e),
                )
                
    except Exception as e:
        logger.error("Scheduled analysis failed", error=str(e))


async def start_scheduler(state: AppState):
    """Start the background scheduler."""
    global scheduler
    
    scheduler = AsyncIOScheduler()
    
    # Run every hour during market hours (9:30 AM - 4:00 PM ET)
    scheduler.add_job(
        hourly_analysis,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="30",
            timezone="America/New_York",
        ),
        args=[state],
        id="hourly_analysis",
        name="Hourly Position Analysis",
    )
    
    scheduler.start()
    logger.info("Started background scheduler")


async def stop_scheduler():
    """Stop the background scheduler."""
    global scheduler
    
    if scheduler:
        scheduler.shutdown()
        scheduler = None
        logger.info("Stopped background scheduler")
