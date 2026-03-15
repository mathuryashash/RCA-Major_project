-- TimescaleDB initialization for RCA System
-- Run automatically on first start via Docker entrypoint

-- Create metrics hypertable
CREATE TABLE IF NOT EXISTS metrics (
    time        TIMESTAMPTZ NOT NULL,
    host        TEXT NOT NULL DEFAULT 'localhost',
    service     TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value       DOUBLE PRECISION,
    unit        TEXT DEFAULT ''
);
SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE);

-- Create index for common queries
CREATE INDEX IF NOT EXISTS idx_metrics_service_time ON metrics (service, time DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics (metric_name, time DESC);

-- Events table for CI/CD webhooks
CREATE TABLE IF NOT EXISTS events (
    id          SERIAL PRIMARY KEY,
    time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type  TEXT NOT NULL,  -- 'deploy', 'config', 'rollback'
    service     TEXT NOT NULL,
    payload     JSONB,
    correlation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events (time DESC);
CREATE INDEX IF NOT EXISTS idx_events_service ON events (service, time DESC);

-- Normalization parameters (rolling stats)
CREATE TABLE IF NOT EXISTS normalization_params (
    metric_name TEXT NOT NULL,
    service     TEXT NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rolling_mean DOUBLE PRECISION,
    rolling_std  DOUBLE PRECISION,
    window_days  INTEGER DEFAULT 7,
    PRIMARY KEY (metric_name, service)
);
