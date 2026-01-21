"""FastAPI Trading Backend.

Provides REST API and WebSocket for the trading system.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routers import scan, trade, agents, portfolio, workflow, stocks
from api.state import AppState

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle manager."""
    logger.info("Starting trading API server")
    
    # Initialize state
    app.state.app_state = AppState()
    await app.state.app_state.initialize()
    
    # Start scheduler if enabled
    if os.getenv("ENABLE_SCHEDULER", "false").lower() == "true":
        from api.scheduler import start_scheduler
        await start_scheduler(app.state.app_state)
    
    yield
    
    # Cleanup
    logger.info("Shutting down trading API server")
    if hasattr(app.state, "app_state"):
        await app.state.app_state.shutdown()


app = FastAPI(
    title="TradeSense API",
    description="AI-powered day trading system API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://webapp:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scan.router, prefix="/api/scan", tags=["scan"])
app.include_router(trade.router, prefix="/api/trade", tags=["trade"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(workflow.router, prefix="/api/workflow", tags=["workflow"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "running",
        "service": "TradeSense API",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "api": "up",
            "broker": "connected" if os.getenv("ALPACA_API_KEY") else "not configured",
        },
        "timestamp": datetime.now().isoformat(),
    }


# WebSocket for real-time updates
class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected", total=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected", total=len(self.active_connections))
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, receive any messages
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def get_ws_manager() -> ConnectionManager:
    """Get WebSocket manager for broadcasting."""
    return manager


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
