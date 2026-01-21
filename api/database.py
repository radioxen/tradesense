"""Database module for persistence."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
import json

import structlog

logger = structlog.get_logger()

DB_PATH = Path(__file__).parent.parent / "data" / "trading.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Budget/Capital tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS budget (
            id INTEGER PRIMARY KEY,
            amount REAL NOT NULL,
            action TEXT NOT NULL,
            description TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    
    # Trade history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            total_value REAL NOT NULL,
            confidence REAL,
            reasoning TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    
    # Agent decisions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            agent TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            confidence REAL,
            reasoning TEXT,
            data TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    
    # Workflow runs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            stocks_scanned INTEGER DEFAULT 0,
            stocks_analyzed INTEGER DEFAULT 0,
            trades_executed INTEGER DEFAULT 0,
            result TEXT
        )
    """)
    
    # Portfolio snapshots for charts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY,
            portfolio_value REAL NOT NULL,
            cash REAL NOT NULL,
            positions_value REAL NOT NULL,
            pnl_day REAL,
            pnl_total REAL,
            timestamp TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized", path=str(DB_PATH))


def get_budget() -> float:
    """Get current budget/capital."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(amount) as total FROM budget")
    row = cursor.fetchone()
    conn.close()
    return row["total"] if row and row["total"] else 0.0


def add_budget_entry(amount: float, action: str, description: str = "") -> float:
    """Add a budget entry (deposit/withdraw)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO budget (amount, action, description, timestamp) VALUES (?, ?, ?, ?)",
        (amount, action, description, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return get_budget()


def get_budget_history(limit: int = 50) -> list[dict]:
    """Get budget history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM budget ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_trade(
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    confidence: float = 0,
    reasoning: str = ""
) -> int:
    """Save a trade to database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO trades 
           (symbol, action, quantity, price, total_value, confidence, reasoning, timestamp) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, action, quantity, price, quantity * price, confidence, reasoning, datetime.now().isoformat())
    )
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def get_trades(limit: int = 100) -> list[dict]:
    """Get trade history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_decision(
    workflow_id: str,
    agent: str,
    symbol: str,
    signal: str,
    confidence: float,
    reasoning: str,
    data: dict | None = None
) -> int:
    """Save an agent decision."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO decisions 
           (workflow_id, agent, symbol, signal, confidence, reasoning, data, timestamp) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (workflow_id, agent, symbol, signal, confidence, reasoning, 
         json.dumps(data) if data else None, datetime.now().isoformat())
    )
    decision_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return decision_id


def get_decisions(workflow_id: str | None = None, limit: int = 100) -> list[dict]:
    """Get agent decisions."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if workflow_id:
        cursor.execute(
            "SELECT * FROM decisions WHERE workflow_id = ? ORDER BY timestamp DESC LIMIT ?",
            (workflow_id, limit)
        )
    else:
        cursor.execute(
            "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        d = dict(row)
        if d.get("data"):
            d["data"] = json.loads(d["data"])
        results.append(d)
    return results


def create_workflow(workflow_id: str) -> str:
    """Create a new workflow run."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO workflows (id, status, started_at) VALUES (?, ?, ?)",
        (workflow_id, "running", datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return workflow_id


def update_workflow(
    workflow_id: str,
    status: str | None = None,
    stocks_scanned: int | None = None,
    stocks_analyzed: int | None = None,
    trades_executed: int | None = None,
    result: str | None = None
):
    """Update workflow status."""
    conn = get_connection()
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if status:
        updates.append("status = ?")
        params.append(status)
        if status in ["completed", "failed"]:
            updates.append("completed_at = ?")
            params.append(datetime.now().isoformat())
    
    if stocks_scanned is not None:
        updates.append("stocks_scanned = ?")
        params.append(stocks_scanned)
    
    if stocks_analyzed is not None:
        updates.append("stocks_analyzed = ?")
        params.append(stocks_analyzed)
    
    if trades_executed is not None:
        updates.append("trades_executed = ?")
        params.append(trades_executed)
    
    if result:
        updates.append("result = ?")
        params.append(result)
    
    if updates:
        params.append(workflow_id)
        cursor.execute(
            f"UPDATE workflows SET {', '.join(updates)} WHERE id = ?",
            params
        )
        conn.commit()
    
    conn.close()


def get_workflow(workflow_id: str) -> dict | None:
    """Get workflow by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_workflows(limit: int = 20) -> list[dict]:
    """Get recent workflows."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM workflows ORDER BY started_at DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_portfolio_snapshot(
    portfolio_value: float,
    cash: float,
    positions_value: float,
    pnl_day: float = 0,
    pnl_total: float = 0
):
    """Save a portfolio snapshot for charts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO portfolio_snapshots 
           (portfolio_value, cash, positions_value, pnl_day, pnl_total, timestamp) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        (portfolio_value, cash, positions_value, pnl_day, pnl_total, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_portfolio_history(days: int = 30) -> list[dict]:
    """Get portfolio history for charts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT * FROM portfolio_snapshots 
           WHERE timestamp >= datetime('now', ?) 
           ORDER BY timestamp ASC""",
        (f"-{days} days",)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_performance_stats() -> dict:
    """Calculate performance statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total trades
    cursor.execute("SELECT COUNT(*) as count FROM trades")
    total_trades = cursor.fetchone()["count"]
    
    # Winning trades (simplified - would need position tracking for real P&L)
    cursor.execute("SELECT COUNT(*) as count FROM trades WHERE action = 'sell'")
    sell_trades = cursor.fetchone()["count"]
    
    # Total P&L from trades
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN action = 'sell' THEN total_value ELSE 0 END) as sells,
            SUM(CASE WHEN action = 'buy' THEN total_value ELSE 0 END) as buys
        FROM trades
    """)
    row = cursor.fetchone()
    total_sells = row["sells"] or 0
    total_buys = row["buys"] or 0
    
    # Workflows
    cursor.execute("SELECT COUNT(*) as count FROM workflows")
    total_workflows = cursor.fetchone()["count"]
    
    cursor.execute("SELECT COUNT(*) as count FROM workflows WHERE status = 'completed'")
    completed_workflows = cursor.fetchone()["count"]
    
    conn.close()
    
    return {
        "total_trades": total_trades,
        "total_sells": total_sells,
        "total_buys": total_buys,
        "realized_pnl": total_sells - total_buys,
        "total_workflows": total_workflows,
        "completed_workflows": completed_workflows,
    }


# Initialize database on import
init_db()
