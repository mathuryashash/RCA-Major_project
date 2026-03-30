-- TimescaleDB initialization for RCA System
-- Run automatically on first start via Docker entrypoint

-- Create metrics hypertable with labels support
CREATE TABLE IF NOT EXISTS metrics (
    time        TIMESTAMPTZ NOT NULL,
    metric_name TEXT NOT NULL,
    value       DOUBLE PRECISION,
    labels      JSONB DEFAULT '{}'::jsonb
);
SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE, migrate_data => TRUE);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics (metric_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics (time DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_labels ON metrics USING GIN (labels);

-- Events table for CI/CD webhooks
CREATE TABLE IF NOT EXISTS events (
    id              SERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type      TEXT NOT NULL,
    service         TEXT NOT NULL,
    payload         JSONB,
    correlation_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events (time DESC);
CREATE INDEX IF NOT EXISTS idx_events_service ON events (service, time DESC);

-- Normalization parameters (rolling stats)
CREATE TABLE IF NOT EXISTS normalization_params (
    metric_name     TEXT NOT NULL,
    service         TEXT NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rolling_mean    DOUBLE PRECISION,
    rolling_std     DOUBLE PRECISION,
    window_days     INTEGER DEFAULT 7,
    PRIMARY KEY (metric_name, service)
);

-- Remediation audit log (FR-30)
CREATE TABLE IF NOT EXISTS remediation_audit (
    id              SERIAL PRIMARY KEY,
    incident_id     VARCHAR(255) NOT NULL,
    action_type     VARCHAR(255) NOT NULL,
    command_executed TEXT,
    executor        VARCHAR(255) NOT NULL DEFAULT 'system',
    before_state    VARCHAR(500),
    after_state     VARCHAR(500),
    outcome         VARCHAR(50) NOT NULL DEFAULT 'pending',
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_remediation_audit_incident ON remediation_audit (incident_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_remediation_audit_executor ON remediation_audit (executor, timestamp DESC);

-- Compression policy for old data (TimescaleDB feature)
SELECT add_compression_policy('metrics', INTERVAL '7 days') WHERE NOT EXISTS (
    SELECT 1 FROM timescaledb_information.jobs WHERE hypertable_name = 'metrics' AND proc_name = 'compression_policy'
);
