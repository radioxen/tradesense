# TradeSense Development Status

**Last Updated:** January 23, 2026  
**Repository:** https://github.com/radioxen/tradesense  
**Branch:** `dev`

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Completed Work](#completed-work)
5. [Agent Status](#agent-status)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [Known Issues](#known-issues)
9. [Pending Tasks](#pending-tasks)
10. [How to Continue](#how-to-continue)

---

## Project Overview

TradeSense is an AI-powered day trading system designed to achieve consistent profits through multi-agent collaboration. The system uses a crew of specialized AI agents (Scanner, Technical, Fundamental, Risk Manager, Executive) that work together to identify, analyze, and execute trades.

### Goals

- **Primary:** 10-20% daily profit through day trading
- **Approach:** Multi-agent AI system with LLM reasoning
- **Broker:** Alpaca (paper trading initially)
- **Budget:** $10,000 paper trading budget

### Key Features

- Real-time market scanning
- Technical analysis with ML ensemble
- Fundamental analysis with news sentiment
- Risk management and position sizing
- Executive decision-making
- Web UI for monitoring and control

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TradeSense                               │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Next.js)          │  Backend (FastAPI)               │
│  ├─ Dashboard                │  ├─ /api/account                 │
│  ├─ Command Center           │  ├─ /api/workflow                │
│  ├─ Settings                 │  ├─ /api/agents                  │
│  └─ Holdings View            │  └─ /api/positions               │
├─────────────────────────────────────────────────────────────────┤
│                      Agent Orchestration                         │
│  ┌──────────┐ ┌───────────┐ ┌─────────────┐ ┌──────┐ ┌────────┐│
│  │ Scanner  │→│ Technical │→│ Fundamental │→│ Risk │→│ Exec   ││
│  │ Agent    │ │ Agent     │ │ Agent       │ │ Mgr  │ │ Agent  ││
│  └──────────┘ └───────────┘ └─────────────┘ └──────┘ └────────┘│
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                  │  External APIs                   │
│  ├─ SQLite (trading.db)      │  ├─ Alpaca (broker)              │
│  ├─ Market Data (yfinance)   │  ├─ OpenAI (LLM)                 │
│  └─ Models (pickle/json)     │  └─ Perplexity (news search)     │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
Stock Trading/
├── api/                    # FastAPI backend
│   ├── main.py            # App entry point
│   ├── routers/           # API endpoints
│   │   ├── workflow.py    # Trading workflow
│   │   ├── agents.py      # Agent management
│   │   └── ...
│   └── database.py        # SQLite setup
├── src/                    # Core logic
│   ├── agents/            # Trading agents
│   │   ├── technical.py   # Technical Analyst (ENHANCED)
│   │   ├── fundamental.py # Fundamental Analyst
│   │   ├── executive.py   # Executive decision maker
│   │   ├── risk_guardian.py # Risk management
│   │   ├── evolution_strategy.py # ML trading (NEW)
│   │   └── lstm_forecaster.py    # LSTM predictions (NEW)
│   ├── orchestrator/      # Agent coordination
│   │   ├── crew.py        # Pipeline orchestrator
│   │   └── crewai_crew.py # CrewAI integration (NEW)
│   ├── backtest/          # Backtesting engine
│   └── utils/             # Utilities
├── webapp/                 # Next.js frontend
│   └── src/app/           # App router pages
├── configs/               # YAML configurations
│   ├── agents.yaml        # Agent definitions
│   └── tasks.yaml         # Task definitions
├── test_technical_agent.py # Agent testing script (NEW)
└── docker-compose.yml     # Container orchestration
```

---

## Technology Stack

### Backend
- **Python 3.11+**
- **FastAPI** - REST API framework
- **CrewAI** - Agent orchestration framework
- **yfinance** - Market data
- **XGBoost/LightGBM** - ML models
- **SQLite** - Local database

### Frontend
- **Next.js 14** (App Router)
- **TypeScript**
- **Tailwind CSS**
- **Recharts** - Data visualization
- **Lucide Icons**

### AI/ML
- **OpenAI GPT-5** - LLM reasoning
  - `gpt-5-mini-2025-08-07` - Regular agents
  - `gpt-5.2-2025-12-11` - Executive agent
- **Perplexity API** - News search
- **Evolution Strategy** - Neural network trading (experimental)

### Infrastructure
- **Docker/Docker Compose**
- **Alpaca** - Paper trading broker

---

## Completed Work

### Phase 1: Foundation
- [x] Project structure setup
- [x] FastAPI backend with SQLite
- [x] Next.js frontend with Tailwind
- [x] Docker containerization
- [x] Alpaca broker integration

### Phase 2: Agent Framework
- [x] Base agent class (`src/agents/base.py`)
- [x] Pipeline orchestrator (`src/orchestrator/crew.py`)
- [x] CrewAI integration (`src/orchestrator/crewai_crew.py`)
- [x] Agent configuration via YAML
- [x] CrewAI decision output structured JSON + parser in workflow

### Phase 3: Technical Analyst Enhancement (CURRENT)
- [x] Multi-factor signal aggregation (8 components)
- [x] Enhanced confidence calculation
- [x] Evolution Strategy integration (from tradioxen)
- [x] LSTM forecaster module
- [x] Extended data fetching (300+ days)
- [x] Backtest validation with fallback
- [x] Test script for agent validation

### Phase 3b: Scanner Enhancement
- [x] Perplexity-driven universe discovery (200-300 tickers)
- [x] Day-trading score (gap, relative volume, ATR, RSI, breakouts)
- [x] Catalyst ranking with JSON parsing

### Phase 4: UI Development
- [x] Dashboard with portfolio overview
- [x] Command Center for workflow control
- [x] Real-time agent thinking display
- [x] Holdings view with charts

### Phase 5: Backtesting Enhancements
- [x] Historical backtesting flow (60/30 train-test split) via CLI
- [x] Win-rate calculation based on closed-trade P&L

---

## Agent Status

### 1. Technical Analyst ✅ ENHANCED

**File:** `src/agents/technical.py`

**Status:** Production-ready with multi-factor analysis

**Features:**
- 8-component weighted signal aggregation:
  - RSI (mean reversion)
  - MACD (momentum with crossover)
  - Stochastic oscillator
  - Bollinger Bands (extremes)
  - ADX/DI (trend strength)
  - ROC momentum
  - Price vs SMA deviation
  - Volume confirmation multiplier

- Enhanced confidence calculation:
  - Signal agreement scoring
  - Trend strength consideration
  - Volume confirmation
  - Volatility regime adjustment

- Optional Evolution Strategy (disabled by default):
  - Neural network trained via evolution
  - Falls back to rule-based when backtest is poor

**Test Command:**
```bash
python test_technical_agent.py TSLA --days 300
```

---

### 2. Scanner Agent ✅ ENHANCED

**File:** `src/agents/researcher.py`

**Status:** Day-trading optimized scanner with Perplexity-driven universe discovery

**Current Capabilities:**
- Builds a 200-300 ticker universe (indices + Perplexity trending list)
- Lightweight day-trading scores (gap, relative volume, volatility, RSI)
- Perplexity catalyst detection with JSON parsing

**Needs:**
- Earnings calendar integration
- Sector rotation analysis

---

### 3. Fundamental Analyst ⚠️ BASIC

**File:** `src/agents/fundamental.py`

**Status:** Functional with LLM reasoning

**Current Capabilities:**
- News sentiment via Perplexity
- Catalyst identification
- LLM-based analysis

**Needs:**
- Earnings data integration
- SEC filing analysis
- Competitor comparison

---

### 4. Risk Manager ⚠️ BASIC

**File:** `src/agents/risk_guardian.py`

**Status:** Functional with basic rules

**Current Capabilities:**
- Position sizing (max 1% risk per trade)
- Portfolio diversification check
- Daily loss limits

**Needs:**
- Correlation analysis
- VaR calculations
- Dynamic position sizing

---

### 5. Executive Agent ⚠️ BASIC

**File:** `src/agents/executive.py`

**Status:** Functional with LLM reasoning

**Current Capabilities:**
- Synthesizes all agent inputs
- Makes final BUY/SELL/HOLD decisions
- Uses GPT-5.2 for reasoning

**Needs:**
- Trade execution integration
- Performance tracking
- Learning from past decisions

---

## Configuration

### Environment Variables (`.env`)

```bash
# Required
OPENAI_API_KEY=your_openai_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_secret

# Optional
PERPLEXITY_API_KEY=your_perplexity_key
CREWAI_TRACING_ENABLED=true
```

### Agent Configuration (`configs/agents.yaml`)

```yaml
technical_analyst:
  role: "Technical Analyst"
  model: "gpt-5-mini-2025-08-07"
  use_evolution_strategy: false  # Disabled in volatile markets
  evolution_epochs: 500
  threshold_pct: 0.5

executive:
  role: "Executive Trader"
  model: "gpt-5.2-2025-12-11"
  min_confidence: 0.5
  max_position_pct: 0.15
```

---

## Testing

### Test Technical Analyst

```bash
cd "Stock Trading"

# Rule-based only (recommended)
python test_technical_agent.py TSLA --days 300

# With Evolution Strategy (experimental)
python test_technical_agent.py TSLA --days 300 --evolution

# Multiple stocks
python test_technical_agent.py AAPL --days 300
python test_technical_agent.py NVDA --days 300
python test_technical_agent.py SPY --days 300
```

### Expected Output

```
============================================================
  Testing Technical Analyst on TSLA
============================================================

--- Key Indicators ---
  RSI (14):        54.3
  MACD:            -0.45
  ADX:             50.3
  BB Position:     0.53

--- Results ---
  Direction:   NEUTRAL
  Confidence:  46.0%
  Regime:      bearish
```

### Run Full System

```bash
# Start backend
cd api && uvicorn main:app --reload --port 8000

# Start frontend (separate terminal)
cd webapp && npm run dev

# Access UI at http://localhost:3000
```

### Historical Backtest (60/30 Split)

```bash
python -m src.cli run-historical-backtest --symbol TSLA --interval 1h --train-days 60 --test-days 30
```

### Scanner (Day-Trading Discovery)

```bash
python -m src.cli scan-market --count 20
```

---

## Known Issues

### 1. Evolution Strategy Poor Performance
**Issue:** Evolution Strategy backtests show -70% to -98% returns  
**Cause:** High market volatility (1.4-1.8x normal), simple NN architecture  
**Workaround:** Disabled by default, use rule-based signals

### 2. Low Confidence Scores
**Issue:** Most stocks show 25-50% confidence  
**Cause:** High volatility and mixed signals (correct behavior)  
**Note:** This is intentional - system is conservative when uncertain

### 3. CrewAI Tracing Not Appearing
**Issue:** Traces not visible at app.crewai.com  
**Status:** Environment variable set but may need API key verification

### 4. Docker Compose Services
**Issue:** Some Docker services defined but not used  
**Status:** Deprioritized, focusing on core functionality

---

## Pending Tasks

### High Priority
1. **Improve Confidence Scores**
   - Current scores are conservative (correct but low)
   - Consider adjusting thresholds for production

2. **Trade Execution**
   - Connect Executive decisions to Alpaca orders
   - Implement order management

### Medium Priority
4. **Scanner Agent Enhancement**
   - Add volume spike detection
   - Implement sector screening
   - Add earnings calendar

5. **Fundamental Agent Enhancement**
   - Integrate earnings data
   - Add SEC filing analysis

6. **Remove Redundant Code**
   - Clean up test files
   - Refactor to minimal working version

### Low Priority
7. **Docker Services Integration**
   - Use all defined Docker services
   - Add monitoring (Prometheus/Grafana)

8. **CrewAI Tracing**
   - Debug tracing connection
   - Verify API key setup

---

## How to Continue

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/radioxen/tradesense.git
cd tradesense

# Checkout dev branch
git checkout dev

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e .

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Install frontend dependencies
cd webapp && npm install
```

### Development Workflow

1. **Test changes locally:**
   ```bash
   python test_technical_agent.py TSLA --days 300
   ```

2. **Run API server:**
   ```bash
   cd api && uvicorn main:app --reload
   ```

3. **Commit changes:**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin dev
   ```

### Agent Development Pattern

When improving an agent:

1. **Read the current implementation:**
   ```bash
   cat src/agents/<agent_name>.py
   ```

2. **Create a test script** (see `test_technical_agent.py` as example)

3. **Run tests with various stocks:**
   ```bash
   python test_<agent>_agent.py TSLA AAPL NVDA
   ```

4. **Check for edge cases:**
   - Oversold/overbought conditions
   - High volatility periods
   - Low volume stocks

5. **Update confidence calculation** based on test results

### Key Files to Understand

| File | Purpose |
|------|---------|
| `src/agents/technical.py` | Technical analysis (ENHANCED) |
| `src/agents/evolution_strategy.py` | ML trading agent |
| `src/orchestrator/crewai_crew.py` | CrewAI integration |
| `api/routers/workflow.py` | Trading workflow API |
| `test_technical_agent.py` | Agent testing |

---

## Integration with tradioxen

The Technical Analyst was enhanced with techniques from the [tradioxen](https://github.com/radioxen/tradioxen) repository:

### Integrated Components

1. **Evolution Strategy Agent** (`src/agents/evolution_strategy.py`)
   - Neural network trained via population-based evolution
   - Price difference state representation
   - Buy/sell/hold signals with confidence

2. **LSTM Forecaster** (`src/agents/lstm_forecaster.py`)
   - Multi-layer LSTM for price prediction
   - Monte Carlo dropout for uncertainty
   - Ensemble with momentum indicators

### Usage Decision

**Evolution Strategy: DISABLED by default**

Reasons:
- Current market volatility is too high
- Simple NN architecture insufficient
- Rule-based 8-component system performs better

To enable (experimental):
```python
agent = TechnicalAnalystAgent(
    use_evolution_strategy=True,
    evolution_epochs=500,
)
```

---

## Contact

For questions about this project, refer to the development history in:
- Git commit messages
- This documentation
- Code comments in key files

---

*Document generated: January 23, 2026*
