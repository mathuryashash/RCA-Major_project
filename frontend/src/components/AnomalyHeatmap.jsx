import React, { useState, useMemo, useCallback } from 'react';

/**
 * FR-32: Anomaly Heatmap
 *
 * Multi-signal heatmap showing anomaly scores across monitored log sources.
 * Rows = sources/signals, Columns = 5-minute time buckets, Cells = avg anomaly score.
 * Pure React SVG — no D3 dependency.
 */

const COLORS = {
  background: '#1e1e2e',
  cellGreen: '#22c55e',
  cellAmber: '#f59e0b',
  cellRed: '#ef4444',
  labelText: '#ffffff',
  timeLabelText: '#888888',
  tooltipBg: '#2a2a3e',
  tooltipBorder: '#444',
  filterText: '#cccccc',
  checkboxAccent: '#7c3aed',
  metricRowBg: '#252540',
  divider: '#333355',
};

const CELL_WIDTH = 30;
const CELL_HEIGHT = 24;
const CELL_GAP = 2;
const LABEL_WIDTH = 120;
const HEADER_HEIGHT = 40;
const TIME_LABEL_HEIGHT = 60;
const METRICS_SEPARATOR_HEIGHT = 8;
const BUCKET_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

function getCellColor(score) {
  if (score < 0.3) return COLORS.cellGreen;
  if (score <= 0.7) return COLORS.cellAmber;
  return COLORS.cellRed;
}

function getCellOpacity(score) {
  // Minimum 0.2 so cells are always slightly visible, max 1.0
  return 0.2 + Math.min(1, Math.max(0, score)) * 0.8;
}

function formatTime(date) {
  const h = date.getHours().toString().padStart(2, '0');
  const m = date.getMinutes().toString().padStart(2, '0');
  return `${h}:${m}`;
}

function formatTimeRange(bucketStart) {
  const start = new Date(bucketStart);
  const end = new Date(bucketStart + BUCKET_INTERVAL_MS);
  return `${formatTime(start)} – ${formatTime(end)}`;
}

/**
 * Group logs into a grid: { [source]: { [bucketTimestamp]: { avgScore, count, topMessage } } }
 */
function processLogs(logs) {
  if (!logs || logs.length === 0) return { sources: [], buckets: [], grid: {} };

  // Parse timestamps and find time range
  const parsed = logs.map((log) => ({
    ...log,
    ts: new Date(log.timestamp).getTime(),
    score: typeof log.anomaly_score === 'number' ? log.anomaly_score : 0,
  }));

  const minTs = Math.min(...parsed.map((l) => l.ts));
  const maxTs = Math.max(...parsed.map((l) => l.ts));

  // Generate bucket boundaries aligned to interval
  const bucketStart = Math.floor(minTs / BUCKET_INTERVAL_MS) * BUCKET_INTERVAL_MS;
  const bucketEnd = Math.ceil((maxTs + 1) / BUCKET_INTERVAL_MS) * BUCKET_INTERVAL_MS;

  const buckets = [];
  for (let t = bucketStart; t < bucketEnd; t += BUCKET_INTERVAL_MS) {
    buckets.push(t);
  }

  // Collect unique sources
  const sourceSet = new Set(parsed.map((l) => l.source));
  const sources = Array.from(sourceSet).sort();

  // Build grid
  const grid = {};
  for (const source of sources) {
    grid[source] = {};
    for (const bucket of buckets) {
      grid[source][bucket] = { totalScore: 0, count: 0, topMessage: '', topScore: -1 };
    }
  }

  for (const log of parsed) {
    const bucket = Math.floor(log.ts / BUCKET_INTERVAL_MS) * BUCKET_INTERVAL_MS;
    const cell = grid[log.source]?.[bucket];
    if (cell) {
      cell.totalScore += log.score;
      cell.count += 1;
      if (log.score > cell.topScore) {
        cell.topScore = log.score;
        cell.topMessage = log.message || '';
      }
    }
  }

  // Compute averages
  for (const source of sources) {
    for (const bucket of buckets) {
      const cell = grid[source][bucket];
      cell.avgScore = cell.count > 0 ? cell.totalScore / cell.count : 0;
    }
  }

  return { sources, buckets, grid };
}

function Tooltip({ data, x, y, containerWidth }) {
  if (!data) return null;

  const tooltipWidth = 260;
  const tooltipHeight = 110;
  // Flip tooltip if it would overflow right edge
  const adjustedX = x + tooltipWidth > containerWidth ? x - tooltipWidth - 10 : x + 10;
  const adjustedY = Math.max(5, y - tooltipHeight / 2);

  return (
    <div
      style={{
        position: 'absolute',
        left: adjustedX,
        top: adjustedY,
        width: tooltipWidth,
        background: COLORS.tooltipBg,
        border: `1px solid ${COLORS.tooltipBorder}`,
        borderRadius: 6,
        padding: '8px 12px',
        color: COLORS.labelText,
        fontSize: 12,
        lineHeight: 1.5,
        pointerEvents: 'none',
        zIndex: 100,
        boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 2 }}>{data.source}</div>
      <div style={{ color: COLORS.timeLabelText }}>{data.timeRange}</div>
      <div>
        Score:{' '}
        <span style={{ color: getCellColor(data.score), fontWeight: 600 }}>
          {data.score.toFixed(3)}
        </span>
      </div>
      {data.message && (
        <div
          style={{
            marginTop: 4,
            color: '#aaa',
            fontSize: 11,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {data.message}
        </div>
      )}
    </div>
  );
}

export default function AnomalyHeatmap({ anomalousMetrics, logs }) {
  const [hiddenSources, setHiddenSources] = useState(new Set());
  const [tooltip, setTooltip] = useState(null);

  // Process log data into heatmap grid
  const { sources, buckets, grid } = useMemo(() => processLogs(logs), [logs]);

  // Process anomalous metrics into a summary row
  const metricsRow = useMemo(() => {
    if (!anomalousMetrics || anomalousMetrics.length === 0) return [];
    return anomalousMetrics.map((m) => ({
      name: m.name,
      score: typeof m.score === 'number' ? m.score : 0,
      firstSeen: m.first_seen || '',
    }));
  }, [anomalousMetrics]);

  const visibleSources = useMemo(
    () => sources.filter((s) => !hiddenSources.has(s)),
    [sources, hiddenSources],
  );

  const toggleSource = useCallback((source) => {
    setHiddenSources((prev) => {
      const next = new Set(prev);
      if (next.has(source)) {
        next.delete(source);
      } else {
        next.add(source);
      }
      return next;
    });
  }, []);

  const handleCellEnter = useCallback(
    (e, source, bucket) => {
      const cell = grid[source]?.[bucket];
      if (!cell || cell.count === 0) return;
      const rect = e.currentTarget.closest('[data-heatmap-container]')?.getBoundingClientRect();
      const svgRect = e.currentTarget.getBoundingClientRect();
      if (!rect) return;
      setTooltip({
        data: {
          source,
          timeRange: formatTimeRange(bucket),
          score: cell.avgScore,
          message: cell.topMessage,
        },
        x: svgRect.left - rect.left + CELL_WIDTH,
        y: svgRect.top - rect.top,
      });
    },
    [grid],
  );

  const handleMetricEnter = useCallback((e, metric) => {
    const rect = e.currentTarget.closest('[data-heatmap-container]')?.getBoundingClientRect();
    const svgRect = e.currentTarget.getBoundingClientRect();
    if (!rect) return;
    setTooltip({
      data: {
        source: metric.name,
        timeRange: metric.firstSeen ? `First seen: ${metric.firstSeen}` : 'N/A',
        score: metric.score,
        message: '',
      },
      x: svgRect.left - rect.left + CELL_WIDTH,
      y: svgRect.top - rect.top,
    });
  }, []);

  const handleMouseLeave = useCallback(() => {
    setTooltip(null);
  }, []);

  // ── Compute SVG dimensions ──
  const hasMetrics = metricsRow.length > 0;
  const hasLogs = visibleSources.length > 0 && buckets.length > 0;

  const metricsColumns = hasMetrics ? metricsRow.length : 0;
  const logColumns = hasLogs ? buckets.length : 0;
  const maxColumns = Math.max(metricsColumns, logColumns, 1);

  const metricsRowCount = hasMetrics ? 1 : 0;
  const separatorHeight = hasMetrics && hasLogs ? METRICS_SEPARATOR_HEIGHT : 0;

  const gridWidth = maxColumns * (CELL_WIDTH + CELL_GAP);
  const gridHeight =
    metricsRowCount * (CELL_HEIGHT + CELL_GAP) +
    separatorHeight +
    visibleSources.length * (CELL_HEIGHT + CELL_GAP);

  const svgWidth = LABEL_WIDTH + gridWidth + 20;
  const svgHeight = HEADER_HEIGHT + gridHeight + TIME_LABEL_HEIGHT + 10;

  // ── Empty state ──
  if (!hasMetrics && (!logs || logs.length === 0)) {
    return (
      <div
        style={{
          width: '100%',
          minHeight: 200,
          background: COLORS.background,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 8,
        }}
      >
        <span style={{ color: '#888', fontSize: 14 }}>No anomaly data available</span>
      </div>
    );
  }

  return (
    <div
      data-heatmap-container
      style={{
        position: 'relative',
        width: '100%',
        background: COLORS.background,
        borderRadius: 8,
        padding: 16,
        overflow: 'auto',
      }}
    >
      {/* ── Source filter checkboxes ── */}
      {sources.length > 0 && (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 12,
            marginBottom: 12,
            paddingBottom: 10,
            borderBottom: `1px solid ${COLORS.divider}`,
          }}
        >
          <span style={{ color: COLORS.timeLabelText, fontSize: 12, marginRight: 4 }}>
            Signals:
          </span>
          {sources.map((source) => (
            <label
              key={source}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                color: COLORS.filterText,
                fontSize: 12,
                cursor: 'pointer',
                userSelect: 'none',
              }}
            >
              <input
                type="checkbox"
                checked={!hiddenSources.has(source)}
                onChange={() => toggleSource(source)}
                style={{ accentColor: COLORS.checkboxAccent }}
              />
              {source}
            </label>
          ))}
        </div>
      )}

      {/* ── SVG Heatmap ── */}
      <div style={{ overflowX: 'auto' }}>
        <svg
          width={svgWidth}
          height={svgHeight}
          style={{ display: 'block', minWidth: svgWidth }}
        >
          {/* ── Metrics summary row ── */}
          {hasMetrics &&
            metricsRow.map((metric, colIdx) => {
              const cx = LABEL_WIDTH + colIdx * (CELL_WIDTH + CELL_GAP);
              const cy = HEADER_HEIGHT;
              const color = getCellColor(metric.score);
              const opacity = getCellOpacity(metric.score);

              return (
                <g key={`metric-${metric.name}`}>
                  {/* Label on first metric row */}
                  {colIdx === 0 && (
                    <text
                      x={LABEL_WIDTH - 8}
                      y={cy + CELL_HEIGHT / 2}
                      textAnchor="end"
                      dominantBaseline="central"
                      fill={COLORS.labelText}
                      fontSize={11}
                      fontWeight={600}
                    >
                      Metrics
                    </text>
                  )}
                  <rect
                    x={cx}
                    y={cy}
                    width={CELL_WIDTH}
                    height={CELL_HEIGHT}
                    rx={3}
                    fill={color}
                    opacity={opacity}
                    style={{ cursor: 'pointer', transition: 'opacity 0.2s' }}
                    onMouseEnter={(e) => handleMetricEnter(e, metric)}
                    onMouseLeave={handleMouseLeave}
                  />
                  {/* Metric name below cell (tiny label) */}
                  <text
                    x={cx + CELL_WIDTH / 2}
                    y={cy - 4}
                    textAnchor="middle"
                    fill={COLORS.timeLabelText}
                    fontSize={9}
                  >
                    {metric.name.length > 5 ? metric.name.slice(0, 5) + '..' : metric.name}
                  </text>
                </g>
              );
            })}

          {/* ── Log source rows ── */}
          {hasLogs &&
            visibleSources.map((source, rowIdx) => {
              const rowY =
                HEADER_HEIGHT +
                metricsRowCount * (CELL_HEIGHT + CELL_GAP) +
                separatorHeight +
                rowIdx * (CELL_HEIGHT + CELL_GAP);

              return (
                <g key={`row-${source}`}>
                  {/* Source label */}
                  <text
                    x={LABEL_WIDTH - 8}
                    y={rowY + CELL_HEIGHT / 2}
                    textAnchor="end"
                    dominantBaseline="central"
                    fill={COLORS.labelText}
                    fontSize={11}
                  >
                    {source.length > 14 ? source.slice(0, 14) + '..' : source}
                  </text>

                  {/* Cells for each time bucket */}
                  {buckets.map((bucket, colIdx) => {
                    const cell = grid[source][bucket];
                    const hasData = cell.count > 0;
                    const score = cell.avgScore;
                    const color = hasData ? getCellColor(score) : '#333';
                    const opacity = hasData ? getCellOpacity(score) : 0.1;
                    const cx = LABEL_WIDTH + colIdx * (CELL_WIDTH + CELL_GAP);

                    return (
                      <rect
                        key={`cell-${source}-${bucket}`}
                        x={cx}
                        y={rowY}
                        width={CELL_WIDTH}
                        height={CELL_HEIGHT}
                        rx={3}
                        fill={color}
                        opacity={opacity}
                        style={{ cursor: hasData ? 'pointer' : 'default', transition: 'opacity 0.2s' }}
                        onMouseEnter={hasData ? (e) => handleCellEnter(e, source, bucket) : undefined}
                        onMouseLeave={hasData ? handleMouseLeave : undefined}
                      />
                    );
                  })}
                </g>
              );
            })}

          {/* ── Time labels on X-axis ── */}
          {hasLogs &&
            buckets.map((bucket, colIdx) => {
              const cx = LABEL_WIDTH + colIdx * (CELL_WIDTH + CELL_GAP) + CELL_WIDTH / 2;
              const labelY =
                HEADER_HEIGHT +
                metricsRowCount * (CELL_HEIGHT + CELL_GAP) +
                separatorHeight +
                visibleSources.length * (CELL_HEIGHT + CELL_GAP) +
                8;

              // Show every label if few buckets, otherwise show every Nth
              const step = buckets.length > 20 ? Math.ceil(buckets.length / 12) : 1;
              if (colIdx % step !== 0) return null;

              return (
                <text
                  key={`time-${bucket}`}
                  x={cx}
                  y={labelY}
                  textAnchor="end"
                  fill={COLORS.timeLabelText}
                  fontSize={10}
                  transform={`rotate(-45, ${cx}, ${labelY})`}
                >
                  {formatTime(new Date(bucket))}
                </text>
              );
            })}

          {/* ── Separator line between metrics and log rows ── */}
          {hasMetrics && hasLogs && (
            <line
              x1={LABEL_WIDTH}
              y1={
                HEADER_HEIGHT +
                metricsRowCount * (CELL_HEIGHT + CELL_GAP) +
                separatorHeight / 2
              }
              x2={LABEL_WIDTH + gridWidth}
              y2={
                HEADER_HEIGHT +
                metricsRowCount * (CELL_HEIGHT + CELL_GAP) +
                separatorHeight / 2
              }
              stroke={COLORS.divider}
              strokeWidth={1}
              strokeDasharray="4 3"
            />
          )}
        </svg>
      </div>

      {/* ── Color legend ── */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          marginTop: 10,
          paddingTop: 8,
          borderTop: `1px solid ${COLORS.divider}`,
        }}
      >
        <span style={{ color: COLORS.timeLabelText, fontSize: 11 }}>Score:</span>
        {[
          { label: '< 0.3', color: COLORS.cellGreen },
          { label: '0.3–0.7', color: COLORS.cellAmber },
          { label: '> 0.7', color: COLORS.cellRed },
        ].map(({ label, color }) => (
          <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span
              style={{
                width: 12,
                height: 12,
                borderRadius: 2,
                background: color,
                display: 'inline-block',
              }}
            />
            <span style={{ color: COLORS.timeLabelText, fontSize: 11 }}>{label}</span>
          </span>
        ))}
      </div>

      {/* ── Tooltip overlay ── */}
      {tooltip && (
        <Tooltip
          data={tooltip.data}
          x={tooltip.x}
          y={tooltip.y}
          containerWidth={svgWidth}
        />
      )}
    </div>
  );
}
