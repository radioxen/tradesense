-- PostgreSQL initialization script
-- Creates tables for trade ledger, documents, and vector embeddings

-- Enable pgvector extension for document embeddings
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Trading decisions table (full audit trail)
CREATE TABLE IF NOT EXISTS decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol TEXT NOT NULL,
    
    -- Agent outputs (stored as JSONB for flexibility)
    technical_signal JSONB,
    fundamental_signal JSONB,
    hybrid_signal JSONB,
    rl_action JSONB,
    executive_decision JSONB,
    
    -- Risk guardian output
    risk_veto BOOLEAN NOT NULL DEFAULT FALSE,
    veto_reason TEXT,
    risk_guardian_output JSONB,
    
    -- Final action
    final_action TEXT NOT NULL,  -- BUY, SELL, HOLD
    approved_by TEXT,  -- 'system', 'human:<user_id>'
    
    -- Execution details
    order_id TEXT,
    fill_price NUMERIC,
    fill_qty NUMERIC,
    slippage_bps NUMERIC,
    commission NUMERIC,
    
    -- Model versions for reproducibility
    model_versions JSONB,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions (run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions (symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions (timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions (final_action);

-- Portfolio state snapshots
CREATE TABLE IF NOT EXISTS portfolio_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cash NUMERIC NOT NULL,
    positions JSONB NOT NULL DEFAULT '{}',
    equity NUMERIC NOT NULL,
    daily_pnl NUMERIC NOT NULL DEFAULT 0,
    drawdown NUMERIC NOT NULL DEFAULT 0,
    trades_today INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_run ON portfolio_state (run_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_state (timestamp);

-- Trade ledger
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    decision_id UUID REFERENCES decisions(id),
    
    -- Trade details
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- BUY, SELL
    quantity NUMERIC NOT NULL,
    price NUMERIC NOT NULL,
    notional NUMERIC NOT NULL,
    
    -- Execution
    order_type TEXT NOT NULL,  -- market, limit
    order_id TEXT,
    filled_at TIMESTAMPTZ,
    
    -- Costs
    commission NUMERIC DEFAULT 0,
    slippage_bps NUMERIC DEFAULT 0,
    
    -- P&L
    realized_pnl NUMERIC,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_run ON trades (run_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_filled ON trades (filled_at);

-- News and documents store
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source TEXT NOT NULL,  -- reuters, sec, twitter, etc.
    doc_type TEXT NOT NULL,  -- news, filing, social, etc.
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    symbols TEXT[] DEFAULT '{}',
    
    -- Metadata
    published_at TIMESTAMPTZ NOT NULL,
    url TEXT,
    author TEXT,
    credibility_score NUMERIC DEFAULT 0.5,
    
    -- Vector embedding for semantic search
    embedding vector(1536),  -- OpenAI embedding dimension
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);
CREATE INDEX IF NOT EXISTS idx_documents_symbols ON documents USING GIN (symbols);
CREATE INDEX IF NOT EXISTS idx_documents_published ON documents (published_at);

-- IVFFlat index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Agent transcripts (for debugging and analysis)
CREATE TABLE IF NOT EXISTS agent_transcripts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL,
    decision_id UUID REFERENCES decisions(id),
    agent_name TEXT NOT NULL,
    
    -- LLM interaction
    system_prompt TEXT,
    user_prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    
    -- Token usage
    model TEXT,
    temperature NUMERIC,
    prompt_tokens INT,
    completion_tokens INT,
    
    -- Timing
    latency_ms INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transcripts_run ON agent_transcripts (run_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_agent ON agent_transcripts (agent_name);

-- Backtest runs
CREATE TABLE IF NOT EXISTS backtest_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT,
    config JSONB NOT NULL,
    
    -- Universe
    symbols TEXT[] NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    interval TEXT NOT NULL,
    
    -- Results
    total_return NUMERIC,
    cagr NUMERIC,
    sharpe_ratio NUMERIC,
    max_drawdown NUMERIC,
    win_rate NUMERIC,
    total_trades INT,
    
    -- Status
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    error_message TEXT,
    
    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_status ON backtest_runs (status);
CREATE INDEX IF NOT EXISTS idx_backtest_created ON backtest_runs (created_at);

-- MLflow schema (if not using mlflow's own tables)
-- This is just for reference - MLflow creates its own tables

-- Human-in-the-loop approvals
CREATE TABLE IF NOT EXISTS hil_approvals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    decision_id UUID REFERENCES decisions(id) NOT NULL,
    
    -- Request
    notification_channel TEXT NOT NULL,  -- slack, telegram, email
    notification_sent_at TIMESTAMPTZ NOT NULL,
    
    -- Response
    approved BOOLEAN,
    approved_by TEXT,
    response_received_at TIMESTAMPTZ,
    notes TEXT,
    
    -- Timeout
    timeout_at TIMESTAMPTZ NOT NULL,
    auto_approved BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hil_decision ON hil_approvals (decision_id);
CREATE INDEX IF NOT EXISTS idx_hil_pending ON hil_approvals (approved) WHERE approved IS NULL;
