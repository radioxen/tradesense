-- TimescaleDB initialization script
-- Creates hypertables for time-series market data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV bars table
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    interval TEXT NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close NUMERIC,
    PRIMARY KEY (symbol, timestamp, interval)
);

-- Convert to hypertable (partitioned by time)
SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv (symbol);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval ON ohlcv (symbol, interval);

-- Features table
CREATE TABLE IF NOT EXISTS features (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_version INT NOT NULL DEFAULT 1,
    features JSONB NOT NULL,
    PRIMARY KEY (symbol, timestamp, feature_version)
);

SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);

-- Create continuous aggregates for common intervals
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_daily
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 day', timestamp) AS timestamp,
    'daily' AS interval,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    last(adjusted_close, timestamp) AS adjusted_close
FROM ohlcv
WHERE interval = '1h'
GROUP BY symbol, time_bucket('1 day', timestamp)
WITH NO DATA;

-- Refresh policies
SELECT add_continuous_aggregate_policy('ohlcv_daily',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Data retention policy (keep 2 years of data)
SELECT add_retention_policy('ohlcv', INTERVAL '2 years', if_not_exists => TRUE);

-- Compression policy (compress data older than 7 days)
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,interval'
);
SELECT add_compression_policy('ohlcv', INTERVAL '7 days', if_not_exists => TRUE);
