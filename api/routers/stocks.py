"""Stock data API for charts and details."""

from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
import structlog

from src.data.market_data import get_provider

logger = structlog.get_logger()
router = APIRouter()


@router.get("/{symbol}/chart")
async def get_stock_chart(symbol: str, days: int = 30):
    """Get stock price history for charting."""
    try:
        provider = get_provider("yfinance")
        end = datetime.now()
        start = end - timedelta(days=days)
        
        data = await provider.fetch_ohlcv(symbol, start, end, "1d")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        chart_data = []
        for idx, row in data.iterrows():
            chart_data.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })
        
        # Calculate some stats
        if len(chart_data) >= 2:
            first_close = chart_data[0]["close"]
            last_close = chart_data[-1]["close"]
            change = last_close - first_close
            change_pct = (change / first_close) * 100
        else:
            change = 0
            change_pct = 0
        
        return {
            "symbol": symbol,
            "data": chart_data,
            "current_price": chart_data[-1]["close"] if chart_data else 0,
            "change": change,
            "change_pct": change_pct,
            "high_30d": max(d["high"] for d in chart_data) if chart_data else 0,
            "low_30d": min(d["low"] for d in chart_data) if chart_data else 0,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chart data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/quote")
async def get_stock_quote(symbol: str):
    """Get current stock quote."""
    try:
        provider = get_provider("yfinance")
        end = datetime.now()
        start = end - timedelta(days=5)
        
        data = await provider.fetch_ohlcv(symbol, start, end, "1d")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) >= 2 else latest
        
        return {
            "symbol": symbol,
            "price": float(latest["close"]),
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "volume": int(latest["volume"]),
            "prev_close": float(prev["close"]),
            "change": float(latest["close"] - prev["close"]),
            "change_pct": float((latest["close"] - prev["close"]) / prev["close"] * 100),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
