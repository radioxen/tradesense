"""Scanner API endpoints."""

from fastapi import APIRouter, Request, BackgroundTasks
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()
router = APIRouter()


class ScanRequest(BaseModel):
    """Scan request parameters."""
    target_count: int = 20
    use_llm: bool = True


class ScanResult(BaseModel):
    """Individual scan result."""
    symbol: str
    score: float
    momentum_score: float
    volume_score: float
    technical_score: float
    catalyst_score: float
    price: float
    rsi: float
    trend: str
    catalysts: list[str] = []


@router.post("/run")
async def run_scan(request: Request, params: ScanRequest, background_tasks: BackgroundTasks):
    """Run market scan for momentum stocks."""
    state = request.app.state.app_state
    
    logger.info("Starting market scan", target_count=params.target_count)
    
    # Log activity
    state.log_activity({
        "type": "scan",
        "action": "STARTED",
        "details": f"Scanning for top {params.target_count} momentum stocks",
    })
    
    # Run scan
    try:
        results = await state.researcher.scan_market(
            target_count=params.target_count,
            use_llm=params.use_llm,
        )
        
        # Convert to response format and store
        scan_data = []
        for r in results[:params.target_count]:
            scan_data.append({
                "symbol": r.symbol,
                "score": round(r.score, 1),
                "momentum_score": round(r.momentum_score, 1),
                "volume_score": round(r.volume_score, 1),
                "technical_score": round(r.technical_score, 1),
                "catalyst_score": round(r.catalyst_score, 1),
                "price": round(r.price, 2),
                "rsi": round(r.rsi, 1),
                "trend": r.short_term_trend,
                "catalysts": r.catalysts,
            })
        
        state.scan_results = scan_data
        
        state.log_activity({
            "type": "scan",
            "action": "COMPLETED",
            "details": f"Found {len(scan_data)} momentum candidates",
        })
        
        logger.info("Scan complete", count=len(scan_data))
        
        return {
            "status": "success",
            "count": len(scan_data),
            "results": scan_data,
        }
        
    except Exception as e:
        logger.error("Scan failed", error=str(e))
        state.log_activity({
            "type": "scan",
            "action": "FAILED",
            "details": str(e),
        })
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/results")
async def get_scan_results(request: Request):
    """Get latest scan results."""
    state = request.app.state.app_state
    return {
        "count": len(state.scan_results),
        "results": state.scan_results,
    }
