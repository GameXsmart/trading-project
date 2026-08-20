-- TimescaleDB extras, applied after SQLAlchemy has created the tables.
--
-- Executed statement-by-statement by mie.storage.db.Database._apply_timescale.
-- Individual failures are logged and skipped, so this file is safe to re-run and
-- safe to apply against a plain PostgreSQL without the extension installed.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------- hypertables
-- migrate_data => true lets these run against tables that already hold rows.
SELECT create_hypertable('ohlcv', 'open_time', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('funding_rates', 'ts', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('open_interest', 'ts', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('global_metrics', 'ts', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('data_quality_events', 'detected_at', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE, migrate_data => TRUE);

-- ---------------------------------------------------------------- compression
-- Historical candles are read in wide time ranges per instrument and never updated
-- once final, which is precisely the access pattern columnar compression rewards.
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id, timeframe',
    timescaledb.compress_orderby = 'open_time DESC'
);

-- Two weeks is comfortably past the point where a bar can still be revised.
SELECT add_compression_policy('ohlcv', INTERVAL '14 days', if_not_exists => TRUE);
SELECT add_compression_policy('data_quality_events', INTERVAL '30 days', if_not_exists => TRUE);

-- ------------------------------------------------------------------ retention
-- Quality events are diagnostic, not analytical: keep a year, not forever.
SELECT add_retention_policy('data_quality_events', INTERVAL '365 days', if_not_exists => TRUE);

-- --------------------------------------------------- continuous aggregate demo
-- Deriving higher timeframes from a stored base resolution guarantees that
-- timeframes agree by construction instead of by hope. Phase 2 will extend this
-- pattern to the full ladder; the 1m -> 5m rollup is here as the reference shape.
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5m_from_1m
WITH (timescaledb.continuous) AS
SELECT
    instrument_id,
    time_bucket(INTERVAL '5 minutes', open_time) AS bucket,
    first(open, open_time)  AS open,
    max(high)               AS high,
    min(low)                AS low,
    last(close, open_time)  AS close,
    sum(volume)             AS volume,
    sum(quote_volume)       AS quote_volume,
    sum(trades)             AS trades
FROM ohlcv
WHERE timeframe = '1m' AND is_final
GROUP BY instrument_id, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_5m_from_1m',
    start_offset => INTERVAL '3 days',
    end_offset   => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE);
