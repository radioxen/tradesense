# TradeSense - Project Overview

## Purpose
AI-powered day trading system using multi-agent collaboration (CrewAI). Agents scan, analyze, and execute trades via Alpaca broker (paper trading).

## Tech Stack
- **Backend:** Python 3.11+, FastAPI, uvicorn (port 8000)
- **Frontend:** Next.js, TypeScript, Tailwind CSS (port 3000)
- **AI/ML:** CrewAI, OpenAI, Perplexity API, PyTorch, scikit-learn, XGBoost, LightGBM, stable-baselines3
- **Data:** Pandas, Polars, NumPy, yfinance, Alpaca API
- **Database:** PostgreSQL (SQLAlchemy, asyncpg, pgvector)
- **Infra:** Docker Compose, MLflow, DVC, Prometheus
- **Package manager:** uv (uv.lock present), pip install -e .

## Agent Architecture
Agents in `src/agents/`: technical, fundamental, hybrid, executive, risk_guardian, researcher, lstm_forecaster, evolution_strategy, base.

## Key Directories
- `src/` - Core Python source (agents, backtest, data, execution, orchestrator, rl, utils, observability)
- `api/` - FastAPI backend (main.py, routers/, scheduler, database, state)
- `webapp/` - Next.js frontend
- `configs/` - YAML configs (agents.yaml, tasks.yaml, default.yaml)
- `models/` - ML models
- `infra/` - Infrastructure
- `Docs/` - Documentation

## Entry Points
- `run-backtest` → `src.cli:run_backtest`
- `run-paper` → `src.cli:run_paper`
- `train-rl` → `src.cli:train_rl`
- API: `cd api && uvicorn main:app --reload --port 8000`
- Web: `cd webapp && npm run dev`
- Docker: `docker compose up -d`
