# TradeSense - AI Day Trading System

A multi-agent AI trading system focused on day trading with real-time monitoring and paper trading support.

## Features

- **AI Agent Team**: Technical, Fundamental, Hybrid, Executive, and Risk Guardian
- **Market Scanner**: Perplexity-driven discovery + day-trading momentum scoring
- **Historical Backtesting**: 60/30 train-test split CLI flow
- **Paper Trading**: Safe testing with Alpaca paper trading API
- **Real-time Monitoring**: WebSocket-based live activity stream
- **Editable Agents**: Customize agent roles, goals, and models from the UI
- **Automated Analysis**: Hourly position analysis during market hours

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TradeSense System                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │   Scanner   │────▶│   Agents    │────▶│   Broker    │   │
│  │ Perplexity │      │ Tech/Fund/  │     │  Alpaca     │   │
│  └─────────────┘     │ Hybrid/Exec │     └─────────────┘   │
│                      │ Risk        │                      │
│                      └─────────────┘                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FastAPI Backend (Port 8000)             │   │
│  │  • /api/scan - Run market scanner                    │   │
│  │  • /api/trade - Analyze & execute trades             │   │
│  │  • /api/portfolio - Account & positions              │   │
│  │  • /api/agents - View/edit agent configs             │   │
│  │  • /ws/stream - Real-time WebSocket updates          │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Next.js Frontend (Port 3000)            │   │
│  │  • Dashboard - Portfolio overview                    │   │
│  │  • Scanner - Find & analyze momentum stocks          │   │
│  │  • Trading - Position management                     │   │
│  │  • Agents - Configure AI agents                      │   │
│  │  • Activity - Real-time decision stream              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Configure API Keys

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:
- `OPENAI_API_KEY` - For LLM agents
- `PERPLEXITY_API_KEY` - For market scanning and catalyst detection
- `ALPACA_API_KEY` & `ALPACA_API_SECRET` - For paper trading

### 2. Run with Docker (Recommended)

```bash
docker compose up -d
```

This starts:
- API Backend: http://localhost:8000
- Web UI: http://localhost:3000

### 3. Or Run Locally

**Backend:**
```bash
pip install -e .
cd api && uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd webapp && npm install && npm run dev
```

**Historical Backtest:**
```bash
python -m src.cli run-historical-backtest --symbol TSLA --interval 1h --train-days 60 --test-days 30
```

## Trading Flow

1. **Scan** - Click "Run Scanner" to find momentum stocks
2. **Select** - Choose 10-20 candidates from scan results
3. **Analyze** - AI agents analyze each stock:
   - Technical: RSI, MACD, volume patterns
   - Fundamental: News, catalysts via Perplexity
   - Hybrid + Executive: Final BUY/SELL/HOLD decision
4. **Execute** - Execute approved recommendations
5. **Monitor** - Hourly automated analysis of positions
6. **Exit** - Stop-loss/take-profit set per decision

## Agent Configuration

Agents are configured in `configs/agents.yaml` and can be edited from the UI.

Each agent has:
- **Name**: Display name
- **Role**: What the agent does
- **Goal**: What the agent optimizes for
- **Model**: LLM model (gpt-5-mini-2025-08-07, gpt-5.2-2025-12-11)
- **Temperature**: Creativity level (0-1)
- **Enabled**: Whether to use this agent

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/scan/run` | POST | Run market scanner |
| `/api/scan/results` | GET | Get latest scan results |
| `/api/trade/analyze` | POST | Analyze stocks with agents |
| `/api/trade/execute` | POST | Execute a trade |
| `/api/trade/analyze-positions` | POST | Analyze current positions |
| `/api/portfolio/account` | GET | Get account info |
| `/api/portfolio/positions` | GET | Get open positions |
| `/api/agents/` | GET | List all agents |
| `/api/agents/{id}` | PUT | Update agent config |

## Risk Management

Built-in safeguards:
- **Max Position Size**: 15% per trade (exec) and 20% portfolio cap (risk guardian)
- **Min Confidence**: 50% required for trades (default)
- **Daily Loss Limits**: Enforced by Risk Guardian
- **Stop/Target**: Set per decision and validated by Risk Guardian

## Project Structure

```
Stock Trading/
├── api/                    # FastAPI backend
│   ├── main.py            # App entrypoint
│   ├── routers/           # API routes
│   └── scheduler.py       # Hourly analysis
├── webapp/                 # Next.js frontend
│   └── src/
│       ├── app/           # Pages
│       ├── components/    # UI components
│       └── lib/           # API client
├── src/                    # Core Python modules
│   ├── agents/            # AI agents
│   ├── data/              # Data providers
│   └── execution/         # Broker integration
├── configs/               # Agent configurations
└── docker-compose.yml     # Docker setup
```

## Disclaimer

This is for educational and paper trading purposes only. Do not use with real money without understanding the risks. Day trading involves significant risk of loss.
