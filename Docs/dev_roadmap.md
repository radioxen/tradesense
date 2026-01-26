# Minimal Multi-Agent Trading System (Local Backtest V1) — Project Roadmap (Epics + Tasks)

## Goal

Build the **minimal** multi-agent trading system that:

- Runs **locally** on a Mac (historical-only first)
- Uses **OpenAI LLMs** for all agent reasoning/decisions
- Uses **Perplexity** for web search (but in V1: only for “simulated fundamentals” during backtests, to avoid non-reproducible results)
- Produces **trade decisions** (BUY/SELL/HOLD + size) with **human-in-the-loop approval**
- Backtests via an **event-driven engine** with no look-ahead bias
- Stores all artifacts (data, signals, prompts, outputs, trades, metrics) locally
- Keeps cost minimal by: batching LLM calls, caching, and only using Perplexity on-demand
- Adds a day-trading scanner that produces 10-20 candidates for downstream analysis

## Non-Goals (V1)

- Live trading execution
- Microsecond latency / LOB
- Full risk modeling (we still add a minimal deterministic “Risk Guardian” veto)
- Multi-market support (crypto/forex later)
- Complex DL forecasting (TFT/PatchTST can be V1.1; V1 uses simple baselines + feature signals)

## V1 Definition of Done

- Given a ticker + historical date range, the system can:
  1. ingest historical OHLCV
  2. compute features
  3. run per-step agent loop (Technical + Fundamental + Hybrid + Executive + Risk Guardian)
  4. generate orders in simulation
  5. output backtest metrics and full audit trail (all agent messages and sources)
- Deterministic reruns (same config → same output) unless explicitly enabling web-search mode
- Scanner can produce 10-20 candidates from a 200-300 ticker universe when live search is enabled

---

# Architecture Snapshot (V1)

**Pipeline:** Data → Feature Store → Event Loop → Agent Orchestrator → Decision → Risk Veto → Simulated Execution → Metrics

**Agents (all AI agents except Risk Guardian, which is deterministic):**

- Technical Analyst Agent (OpenAI): reads features/time-series summary, outputs technical signal + confidence
- Fundamental Analyst Agent (OpenAI + Perplexity): in V1 backtest uses _offline “news snapshots” dataset_; optional “live search mode” for manual spot checks
- Hybrid Mediator Agent (OpenAI): reconciles conflicts, outputs combined thesis + regime label
- Executive Agent (OpenAI): final decision (BUY/SELL/HOLD + size + rationale + confidence)
- Risk Guardian (rule engine, veto/resize): hard constraints + sanity checks

**Core principle:** Research-time reasoning (LLMs) inside backtester; execution is simulated and deterministic.

---

# Epic 0 — Repo & Local Dev Skeleton (Minimal)

### Tasks

- [x] Create mono-repo structure:
  - `src/`
  - `configs/`
  - `data/` (ignored / local)
  - `artifacts/` (runs)
  - `tests/`
- [x] Add configuration management (single YAML):
  - symbols, timeframe, fees, slippage, initial capital
  - agent model selection
  - run flags: `offline_news=true`, `enable_perplexity=false` (default)
- [x] Add minimal CLI:
  - `python -m src.run_backtest --config configs/v1.yaml`
- [x] Add deterministic run id + artifact folder:
  - `artifacts/<run_id>/...`

---

# Epic 1 — Historical Market Data Ingestion (Local, Cheap)

## Scope

Only what we need for backtesting:

- OHLCV at 1h or 1d bars
- Corporate actions adjusted prices (optional; default adjusted close)

### Tasks

- [x] Implement `MarketDataProvider` interface:
  - `fetch_ohlcv(symbol, start, end, interval) -> DataFrame`
- [x] Provide 1 minimal provider (choose one):
  - `yfinance` (cheap/free, sufficient for V1 testing)
  - or a CSV import path
- [x] Save raw OHLCV locally per run:
  - `artifacts/<run_id>/data/ohlcv_<symbol>.parquet`
- [x] Add data validation:
  - monotonic timestamps, missing bars detection, duplicates removal

---

# Epic 2 — Feature Engineering (Minimal Technical Signal Inputs)

## Scope

Compute a **small** feature set to support technical agent reasoning:

- returns, log returns
- rolling volatility
- RSI, MACD, SMA/EMA slopes
- volume z-score
- simple regime proxy (volatility bucket)

### Tasks

- [x] Build `FeatureBuilder`:
  - input: OHLCV
  - output: features dataframe aligned by timestamp
- [x] Persist features:
  - `artifacts/<run_id>/data/features_<symbol>.parquet`
- [x] Add “state summarizer” helper:
  - compress last N bars into a compact JSON summary for LLM prompts (avoid token blowups)

---

# Epic 3 — Offline Fundamental Data (Backtest-Reproducible)

## Scope

Backtests must be reproducible; live Perplexity search is not.
V1 uses an **offline news snapshot dataset** for a symbol/date range.

### Tasks

- [ ] Create `NewsStore` interface:
  - `get_news(symbol, ts, lookback_window) -> list[NewsItem]`
- [ ] Implement **minimal offline dataset** option:
  - input folder: `data/news/<symbol>.jsonl`
  - each item: `{timestamp, source, title, snippet, url(optional)}`
- [ ] Implement `FundamentalContextBuilder`:
  - selects top K items by recency + keyword relevance
  - returns compact JSON payload for LLM

### Optional (V1 toggle, not default)

- [ ] Implement `PerplexitySearchTool` wrapper:
  - only used when `enable_perplexity=true`
  - caches results to `artifacts/<run_id>/perplexity_cache.jsonl`
  - adds “non-reproducible run” warning in report

---

# Epic 4 — Agent Orchestration (CrewAI-Style, Minimal)

## Scope

A simple orchestrator that calls agents in order per event step.

### Tasks

- [x] Define agent I/O contracts (strict JSON schemas):
  - `TechnicalSignal`
  - `FundamentalSignal`
  - `HybridSignal`
  - `ExecutiveDecision`
- [x] Implement `AgentRunner` (Orchestrator):
  - OpenAI chat completions with:
    - system prompt
    - tool outputs
    - strict JSON output (reject + retry once)
- [x] Implement `OrchestratorStep(ts, state)`:
  1. Technical Agent
  2. Fundamental Agent
  3. Hybrid Agent
  4. Executive Agent
  5. Risk Guardian veto
  - returns final action: `ORDER | HOLD`
- [x] Add minimal shared memory:
  - last decision, last 5 actions, open positions summary
  - stored as JSON blob per step

---

# Epic 5 — Agent Definitions (Prompts + Tools)

## Scope

All analysis and decisions are performed by AI agents (OpenAI).
Keep prompts minimal and structured.

### Tasks

- [x] Technical Analyst Agent prompt:
  - Inputs: feature summary, last N bars, open position
  - Outputs (JSON): `direction`, `confidence`, `time_horizon`, `key_reasons`, `invalid_if_low_conf`
- [x] Fundamental Analyst Agent prompt:
  - Inputs: offline news bundle (or perplexity result if enabled)
  - Outputs (JSON): `sentiment`, `confidence`, `catalysts`, `risks`, `citations_required=true`
  - Rule: **no claims without a cited item id**
- [x] Hybrid Mediator Agent prompt:
  - Inputs: tech + fundamental outputs + regime proxy
  - Outputs (JSON): `net_signal`, `regime_label`, `conflicts`, `resolution`, `confidence`
- [x] Executive Agent prompt:
  - Inputs: hybrid + position + portfolio cash + constraints
  - Outputs (JSON): `action`, `size_pct`, `order_type(sim)`, `confidence`, `rationale`
  - Rule: must include a “why now” and a “what would change my mind”

---

# Epic 5b — Scanner / Research Agent (Day Trading Discovery)

## Scope

Lightweight discovery that is fast and cheap, producing a small candidate list
for downstream agent analysis.

### Tasks

- [x] Perplexity-driven ticker discovery (200-300 symbols)
- [x] Shallow day-trading score: gap, relative volume, ATR, RSI, breakouts
- [x] Perplexity catalyst ranking with strict JSON parsing
- [ ] Sector rotation filter
- [ ] Earnings calendar integration

---

# Epic 6 — Risk Guardian (Minimal Deterministic Shield)

## Scope

Even in dev, add hard constraints to prevent nonsense actions.

### Tasks

- [x] Implement rules:
  - max position size (e.g., 20% of equity)
  - max trades per day (config)
  - block trading if data missing
  - minimum confidence threshold (config)
- [x] Implement `risk_guardian(decision, portfolio_state) -> approved_decision`
  - can reduce size or veto to HOLD
- [x] Log every veto with reason code

---

# Epic 7 — Event-Driven Backtesting Engine (Minimal, Correct)

## Scope

Event-driven loop over bars:

- at each timestamp:
  - build state
  - get decision from orchestrator
  - simulate fill
  - update portfolio

### Tasks

- [x] Implement `EventLoop`:
  - iterates over bars (time-ordered)
  - prevents look-ahead (only uses data ≤ current ts)
- [x] Implement execution simulator:
  - slippage model (simple bps)
  - commission model (flat or bps)
  - market order fill at next bar open (configurable)
- [x] Implement portfolio accounting:
  - cash, positions, avg cost, equity curve
- [x] Output metrics:
  - total return, CAGR (if applicable), max drawdown, win rate, avg trade, Sharpe (optional)
- [x] Persist full ledger:
  - orders, fills, positions, equity curve
- [x] Add historical backtest CLI (60/30 train-test split)

---

# Epic 8 — Artifact Store (Everything Auditable, Local)

## Scope

Store every run as a reproducible bundle.

### Tasks

- [ ] Define run artifact structure:
  - `artifacts/<run_id>/config.yaml`
  - `.../data/` parquet
  - `.../agents/` jsonl transcripts (per agent, per step)
  - `.../trades/ledger.parquet`
  - `.../reports/summary.md`
- [ ] Save every prompt + response:
  - include model name, temperature, token counts
- [ ] Add caching to reduce costs:
  - technical agent call cached per `(symbol, ts)` if state unchanged
  - fundamental agent call cached per news bundle hash

---

# Epic 9 — Minimal Reporting (So You Can Evaluate Fast)

## Scope

A single Markdown report per run.

### Tasks

- [ ] Generate `reports/summary.md`:
  - config snapshot
  - metrics table
  - trade list (top 20)
  - “agent agreement stats” (how often aligned vs conflicted)
  - veto stats from Risk Guardian
- [ ] Generate “decision replay” option:
  - print decisions and rationales for a chosen date range

---

# Epic 10 — Paper Cut Hardening (Only What’s Needed)

## Scope

Reduce runtime/cost/failures. No gold plating.

### Tasks

- [ ] Add strict JSON validation with Pydantic (fail fast)
- [ ] Add retry policy (max 1 retry) for agent JSON formatting errors
- [ ] Add timeouts for OpenAI + Perplexity calls
- [ ] Add unit tests only for:
  - event-driven no look-ahead invariant
  - portfolio accounting correctness
  - risk guardian veto logic

---

# Milestones

## M1 — Backtest runs end-to-end (1 symbol)

- Epics: 0–2, 4–8, 9
- Offline news optional (can start with empty fundamental bundle)

## M2 — Add offline fundamentals + citations discipline

- Epics: 3, 5
- Fundamental agent cannot speak without source ids

## M3 — Multi-symbol backtest + basic strategy evaluation

- Extend config + loop multiple tickers (still minimal)

## M4 (Optional) — Enable Perplexity “spot mode”

- For interactive analysis, not for performance backtests
- Cache everything to artifacts

---

# Team Roles (Minimal Dev Team)

- 1x Backend/Quant Engineer (data, backtester, portfolio)
- 1x LLM Engineer (agents, prompts, schemas, caching)
- 1x Full-stack/UX (optional; only if you want a UI; otherwise CLI + markdown report)

No additional roles required for V1.

---

# Suggested Tech Choices (Lowest Cost)

- Data: `yfinance` + parquet
- Backtest: custom event loop (minimal)
- Storage: local filesystem + parquet + jsonl
- Validation: `pydantic`
- LLM: OpenAI Chat Completions
- Search: Perplexity (disabled by default in backtests; enabled for manual runs only)

---

# Acceptance Criteria (Per Epic)

- Epic 1: Historical OHLCV saved, validated
- Epic 2: Features computed and aligned
- Epic 3: Offline news retrieval returns deterministic bundles
- Epic 4–5: Agents return strict JSON outputs per step
- Epic 6: Risk guardian veto works and logs
- Epic 7: Backtest produces correct equity curve and ledger
- Epic 8: Full run is reproducible from stored artifacts
- Epic 9: Summary report generated in Markdown
- Epic 10: Failures are bounded (timeouts, retries, schema validation)
