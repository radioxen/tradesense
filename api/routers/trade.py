"""Trading API endpoints."""

import os
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel
import structlog

from src.agents.technical import TechnicalAnalystAgent
from src.agents.fundamental import FundamentalAnalystAgent
from src.agents.executive import ExecutiveAgent
from src.data.market_data import get_provider
from src.data.feature_store import FeatureBuilder
from src.execution.broker import OrderSide

logger = structlog.get_logger()
router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Analyze request."""
    symbols: list[str]


class TradeRequest(BaseModel):
    """Trade execution request."""
    symbol: str
    action: Literal["buy", "sell"]
    quantity: int


class AnalyzePositionsRequest(BaseModel):
    """Request to analyze current positions."""
    pass


@router.post("/analyze")
async def analyze_stocks(request: Request, params: AnalyzeRequest):
    """Analyze stocks with agent team."""
    state = request.app.state.app_state
    configs = state.agent_configs
    
    results = []
    provider = get_provider("yfinance")
    feature_builder = FeatureBuilder()
    
    for symbol in params.symbols:
        logger.info("Analyzing symbol", symbol=symbol)
        
        try:
            # Fetch data
            end = datetime.now()
            start = end - timedelta(days=60)
            data = await provider.fetch_ohlcv(symbol, start, end, "1d")
            
            if data.empty:
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "message": "No data available",
                })
                continue
            
            features = feature_builder.build_features(data)
            latest = data.iloc[-1]
            latest_features = features.iloc[-1] if not features.empty else {}
            
            # Technical Analysis
            tech_config = configs.get("technical", {})
            tech_analysis = {
                "signal": "neutral",
                "confidence": 50,
                "reasoning": "",
            }
            
            if tech_config.get("enabled", True):
                rsi = latest_features.get("rsi", 50)
                macd = latest_features.get("macd", 0)
                macd_signal = latest_features.get("macd_signal", 0)
                
                if rsi < 35 and macd > macd_signal:
                    tech_analysis = {
                        "signal": "bullish",
                        "confidence": 75,
                        "reasoning": f"RSI oversold ({rsi:.1f}), MACD bullish crossover",
                    }
                elif rsi > 70:
                    tech_analysis = {
                        "signal": "bearish",
                        "confidence": 65,
                        "reasoning": f"RSI overbought ({rsi:.1f})",
                    }
                elif macd > macd_signal:
                    tech_analysis = {
                        "signal": "bullish",
                        "confidence": 60,
                        "reasoning": f"MACD bullish, RSI neutral ({rsi:.1f})",
                    }
                else:
                    tech_analysis = {
                        "signal": "neutral",
                        "confidence": 50,
                        "reasoning": f"No clear signal, RSI {rsi:.1f}",
                    }
            
            state.log_activity({
                "type": "analysis",
                "agent": "Technical",
                "symbol": symbol,
                "action": tech_analysis["signal"].upper(),
                "details": tech_analysis["reasoning"],
            })
            
            # Fundamental Analysis (use Perplexity)
            fund_config = configs.get("fundamental", {})
            fund_analysis = {
                "signal": "neutral",
                "confidence": 50,
                "catalysts": [],
                "reasoning": "",
            }
            
            if fund_config.get("enabled", True) and os.getenv("PERPLEXITY_API_KEY"):
                # Query for catalysts
                catalysts = await state.researcher._query_catalysts_batch([symbol])
                cat_data = catalysts.get(symbol, {})
                
                score = cat_data.get("score", 50)
                fund_analysis = {
                    "signal": "bullish" if score > 60 else "bearish" if score < 40 else "neutral",
                    "confidence": score,
                    "catalysts": cat_data.get("catalysts", []),
                    "reasoning": cat_data.get("reasoning", "No significant catalysts"),
                }
            
            state.log_activity({
                "type": "analysis",
                "agent": "Fundamental",
                "symbol": symbol,
                "action": fund_analysis["signal"].upper(),
                "details": fund_analysis["reasoning"],
            })
            
            # Executive Decision
            exec_config = configs.get("executive", {})
            executive_decision = {
                "action": "hold",
                "confidence": 50,
                "quantity": 0,
                "reasoning": "",
            }
            
            if exec_config.get("enabled", True):
                # Combine signals
                tech_bullish = tech_analysis["signal"] == "bullish"
                fund_bullish = fund_analysis["signal"] == "bullish"
                combined_confidence = (
                    tech_analysis["confidence"] * 0.4 +
                    fund_analysis["confidence"] * 0.6
                )
                
                if tech_bullish and fund_bullish and combined_confidence > 65:
                    # Calculate position size (max 10% of portfolio)
                    account = await state.broker.get_account()
                    max_position = account.buying_power * 0.10
                    price = float(latest["close"])
                    quantity = int(max_position / price)
                    
                    executive_decision = {
                        "action": "buy",
                        "confidence": round(combined_confidence),
                        "quantity": max(1, quantity),
                        "reasoning": f"Both technical and fundamental bullish with {combined_confidence:.0f}% confidence",
                    }
                elif tech_analysis["signal"] == "bearish" and fund_analysis["signal"] == "bearish":
                    executive_decision = {
                        "action": "sell",
                        "confidence": round(combined_confidence),
                        "quantity": 0,  # Will sell full position if held
                        "reasoning": "Both signals bearish, recommend exit",
                    }
                else:
                    executive_decision = {
                        "action": "hold",
                        "confidence": round(combined_confidence),
                        "quantity": 0,
                        "reasoning": "Mixed signals, holding position",
                    }
            
            state.log_activity({
                "type": "decision",
                "agent": "Executive",
                "symbol": symbol,
                "action": executive_decision["action"].upper(),
                "details": executive_decision["reasoning"],
            })
            
            results.append({
                "symbol": symbol,
                "status": "success",
                "price": float(latest["close"]),
                "technical": tech_analysis,
                "fundamental": fund_analysis,
                "decision": executive_decision,
            })
            
        except Exception as e:
            logger.error("Analysis failed", symbol=symbol, error=str(e))
            results.append({
                "symbol": symbol,
                "status": "error",
                "message": str(e),
            })
    
    return {"results": results}


@router.post("/execute")
async def execute_trade(request: Request, params: TradeRequest):
    """Execute a trade."""
    state = request.app.state.app_state
    
    logger.info("Executing trade", symbol=params.symbol, action=params.action, qty=params.quantity)
    
    try:
        side = OrderSide.BUY if params.action == "buy" else OrderSide.SELL
        order = await state.broker.submit_order(
            symbol=params.symbol,
            qty=params.quantity,
            side=side,
        )
        
        state.log_activity({
            "type": "trade",
            "action": params.action.upper(),
            "symbol": params.symbol,
            "details": f"{params.action.upper()} {params.quantity} shares @ ${order.filled_avg_price or 'market'}",
        })
        
        return {
            "status": "success",
            "order_id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.qty,
            "filled_price": order.filled_avg_price,
            "status": order.status.value,
        }
        
    except Exception as e:
        logger.error("Trade execution failed", error=str(e))
        state.log_activity({
            "type": "trade",
            "action": "FAILED",
            "symbol": params.symbol,
            "details": str(e),
        })
        return {
            "status": "error",
            "message": str(e),
        }


@router.post("/analyze-positions")
async def analyze_positions(request: Request):
    """Analyze all current positions for hold/sell decisions."""
    state = request.app.state.app_state
    
    positions = await state.broker.get_positions()
    if not positions:
        return {"results": [], "message": "No open positions"}
    
    symbols = [p.symbol for p in positions]
    
    # Analyze each position
    analysis_request = AnalyzeRequest(symbols=symbols)
    analysis = await analyze_stocks(request, analysis_request)
    
    results = []
    for pos in positions:
        pos_analysis = next(
            (a for a in analysis["results"] if a["symbol"] == pos.symbol),
            None
        )
        
        results.append({
            "symbol": pos.symbol,
            "quantity": pos.qty,
            "entry_price": pos.avg_entry_price,
            "current_price": pos.current_price,
            "pnl": pos.unrealized_pl,
            "pnl_pct": pos.unrealized_plpc * 100,
            "decision": pos_analysis["decision"] if pos_analysis else None,
        })
    
    return {"results": results}


@router.get("/activity")
async def get_activity(request: Request, limit: int = 50):
    """Get recent activity log."""
    state = request.app.state.app_state
    return {
        "activities": state.activity_log[:limit],
    }
