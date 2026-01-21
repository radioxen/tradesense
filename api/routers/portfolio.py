"""Portfolio API endpoints."""

from fastapi import APIRouter, Request
import structlog

logger = structlog.get_logger()
router = APIRouter()


@router.get("/account")
async def get_account(request: Request):
    """Get account information."""
    state = request.app.state.app_state
    
    try:
        account = await state.broker.get_account()
        return {
            "account_id": account.account_id,
            "status": account.status,
            "cash": account.cash,
            "portfolio_value": account.portfolio_value,
            "buying_power": account.buying_power,
            "equity": account.equity,
            "long_market_value": account.long_market_value,
            "pnl_today": account.equity - account.last_equity,
            "pnl_today_pct": ((account.equity / account.last_equity) - 1) * 100 if account.last_equity > 0 else 0,
        }
    except Exception as e:
        logger.error("Failed to get account", error=str(e))
        return {"error": str(e)}


@router.get("/positions")
async def get_positions(request: Request):
    """Get all open positions."""
    state = request.app.state.app_state
    
    try:
        positions = await state.broker.get_positions()
        return {
            "count": len(positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pl,
                    "unrealized_pnl_pct": p.unrealized_plpc * 100,
                    "side": p.side,
                }
                for p in positions
            ],
        }
    except Exception as e:
        logger.error("Failed to get positions", error=str(e))
        return {"error": str(e), "positions": []}


@router.get("/orders")
async def get_orders(request: Request, status: str = "all", limit: int = 50):
    """Get recent orders."""
    state = request.app.state.app_state
    
    try:
        orders = await state.broker.get_orders(status=status, limit=limit)
        return {
            "count": len(orders),
            "orders": [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "quantity": o.qty,
                    "filled_quantity": o.filled_qty,
                    "filled_price": o.filled_avg_price,
                    "status": o.status.value,
                    "created_at": o.created_at.isoformat() if o.created_at else None,
                }
                for o in orders
            ],
        }
    except Exception as e:
        logger.error("Failed to get orders", error=str(e))
        return {"error": str(e), "orders": []}


@router.post("/close-all")
async def close_all_positions(request: Request):
    """Close all open positions."""
    state = request.app.state.app_state
    
    try:
        orders = await state.broker.close_all_positions()
        
        state.log_activity({
            "type": "trade",
            "action": "CLOSE_ALL",
            "symbol": "*",
            "details": f"Closed {len(orders)} positions",
        })
        
        return {
            "status": "success",
            "closed_count": len(orders),
        }
    except Exception as e:
        logger.error("Failed to close positions", error=str(e))
        return {"error": str(e)}
