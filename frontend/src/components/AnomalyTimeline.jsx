import React, { useState, useMemo, useCallback } from 'react';
import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import { AlertTriangle, Clock, FileText, X } from 'lucide-react';

const ANOMALY_THRESHOLD = 0.7;

const LEVEL_COLORS = {
  INFO: '#3b82f6',
  WARN: '#f59e0b',
  WARNING: '#f59e0b',
  ERROR: '#ef4444',
  CRITICAL: '#dc2626',
};

const SOURCE_COLORS = [
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#10b981', // emerald
  '#f97316', // orange
  '#ec4899', // pink
  '#6366f1', // indigo
  '#14b8a6', // teal
  '#e11d48', // rose
];

function getLevelColor(level) {
  return LEVEL_COLORS[level?.toUpperCase()] || '#6b7280';
}

function formatTimestamp(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatFullTimestamp(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

/** Pulsing dot for anomalous data points (score > threshold) */
function PulsingDot(props) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null) return null;
  const isAnomalous = payload?.anomaly_score > ANOMALY_THRESHOLD;

  return (
    <g>
      {isAnomalous && (
        <circle
          cx={cx}
          cy={cy}
          r={8}
          fill="none"
          stroke="#ef4444"
          strokeWidth={2}
          opacity={0.6}
        >
          <animate
            attributeName="r"
            from="5"
            to="14"
            dur="1.5s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            from="0.7"
            to="0"
            dur="1.5s"
            repeatCount="indefinite"
          />
        </circle>
      )}
      <circle
        cx={cx}
        cy={cy}
        r={4}
        fill={getLevelColor(payload?.level)}
        stroke="#fff"
        strokeWidth={1.5}
      />
    </g>
  );
}

/** Custom scatter shape: color-coded by log level with pulse animation if anomalous */
function LogMarkerShape(props) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null) return null;
  const color = getLevelColor(payload?.level);
  const isAnomalous = payload?.anomaly_score > ANOMALY_THRESHOLD;

  return (
    <g style={{ cursor: 'pointer' }}>
      {isAnomalous && (
        <circle cx={cx} cy={cy} r={10} fill="none" stroke={color} strokeWidth={2}>
          <animate
            attributeName="r"
            from="6"
            to="16"
            dur="1.4s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            from="0.8"
            to="0"
            dur="1.4s"
            repeatCount="indefinite"
          />
        </circle>
      )}
      <circle cx={cx} cy={cy} r={5} fill={color} stroke="#fff" strokeWidth={2} />
    </g>
  );
}

/** Custom tooltip */
function TimelineTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null;

  const dataPoint = payload[0]?.payload;
  if (!dataPoint) return null;

  return (
    <div
      style={{
        background: '#fff',
        border: '1px solid #e5e7eb',
        borderRadius: 8,
        padding: '10px 14px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        maxWidth: 320,
        fontSize: 13,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <Clock size={13} color="#6b7280" />
        <span style={{ color: '#6b7280' }}>{formatFullTimestamp(dataPoint.timestamp)}</span>
      </div>
      {dataPoint.source && (
        <div style={{ fontWeight: 600, marginBottom: 4, color: '#1f2937' }}>
          {dataPoint.source}
        </div>
      )}
      {dataPoint.message && (
        <div
          style={{
            color: '#374151',
            marginBottom: 6,
            wordBreak: 'break-word',
            lineHeight: 1.4,
          }}
        >
          {dataPoint.message}
        </div>
      )}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: '50%',
            background:
              dataPoint.anomaly_score > ANOMALY_THRESHOLD ? '#ef4444' : '#10b981',
          }}
        />
        <span style={{ fontWeight: 600 }}>
          Anomaly Score: {dataPoint.anomaly_score?.toFixed(3)}
        </span>
      </div>
      {dataPoint.level && (
        <div style={{ marginTop: 4 }}>
          <span
            style={{
              display: 'inline-block',
              padding: '1px 8px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 600,
              color: '#fff',
              background: getLevelColor(dataPoint.level),
            }}
          >
            {dataPoint.level.toUpperCase()}
          </span>
        </div>
      )}
    </div>
  );
}

const AnomalyTimeline = ({ logs, metrics }) => {
  const [selectedLog, setSelectedLog] = useState(null);

  // Process and sort log data, grouped by source
  const { chartData, scatterData, sourceList, sourceColorMap } = useMemo(() => {
    if (!logs || logs.length === 0) {
      return { chartData: [], scatterData: [], sourceList: [], sourceColorMap: {} };
    }

    // Parse and sort logs chronologically
    const parsed = logs
      .map((log) => ({
        ...log,
        _ts: new Date(log.timestamp).getTime(),
        anomaly_score:
          typeof log.anomaly_score === 'number'
            ? log.anomaly_score
            : parseFloat(log.anomaly_score) || 0,
      }))
      .filter((log) => !isNaN(log._ts))
      .sort((a, b) => a._ts - b._ts);

    // Discover unique sources
    const sources = [...new Set(parsed.map((l) => l.source).filter(Boolean))];

    // Assign colors to sources
    const colorMap = {};
    sources.forEach((src, i) => {
      colorMap[src] = SOURCE_COLORS[i % SOURCE_COLORS.length];
    });

    // Build unified chart data array: one entry per timestamp
    // Each entry has anomaly_score_<source> for each source's line
    const timeMap = new Map();

    parsed.forEach((log) => {
      const key = log.timestamp;
      if (!timeMap.has(key)) {
        timeMap.set(key, {
          timestamp: log.timestamp,
          _ts: log._ts,
          displayTime: formatTimestamp(log.timestamp),
        });
      }
      const entry = timeMap.get(key);
      const scoreKey = `score_${log.source}`;
      // If multiple logs at the same timestamp+source, keep the max score
      entry[scoreKey] = Math.max(entry[scoreKey] || 0, log.anomaly_score);
    });

    const unified = Array.from(timeMap.values()).sort((a, b) => a._ts - b._ts);

    // Scatter data: one point per log, positioned at its anomaly score
    const scatter = parsed.map((log) => ({
      timestamp: log.timestamp,
      displayTime: formatTimestamp(log.timestamp),
      anomaly_score: log.anomaly_score,
      source: log.source,
      level: log.level,
      message: log.message,
      template: log.template,
      _ts: log._ts,
    }));

    return {
      chartData: unified,
      scatterData: scatter,
      sourceList: sources,
      sourceColorMap: colorMap,
    };
  }, [logs]);

  // Process metric time series if provided
  const metricData = useMemo(() => {
    if (!metrics?.time_series || metrics.time_series.length === 0) return null;

    return metrics.time_series
      .map((point) => ({
        ...point,
        _ts: new Date(point.timestamp).getTime(),
        anomaly_score:
          typeof point.anomaly_score === 'number'
            ? point.anomaly_score
            : parseFloat(point.anomaly_score) || 0,
        displayTime: formatTimestamp(point.timestamp),
      }))
      .filter((p) => !isNaN(p._ts))
      .sort((a, b) => a._ts - b._ts);
  }, [metrics]);

  // Merge metric data into chart data if present
  const mergedChartData = useMemo(() => {
    if (!metricData) return chartData;

    const timeMap = new Map();

    // Add existing chart data
    chartData.forEach((entry) => {
      timeMap.set(entry.timestamp, { ...entry });
    });

    // Overlay metric data
    metricData.forEach((point) => {
      const key = point.timestamp;
      if (!timeMap.has(key)) {
        timeMap.set(key, {
          timestamp: point.timestamp,
          _ts: point._ts,
          displayTime: point.displayTime,
        });
      }
      const entry = timeMap.get(key);
      entry.score_metric = point.anomaly_score;
    });

    return Array.from(timeMap.values()).sort((a, b) => a._ts - b._ts);
  }, [chartData, metricData]);

  const handleScatterClick = useCallback((data) => {
    if (data?.payload) {
      setSelectedLog(data.payload);
    }
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedLog(null);
  }, []);

  // Empty state
  if (!logs || logs.length === 0) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: 300,
          color: '#9ca3af',
          gap: 12,
        }}
      >
        <FileText size={40} strokeWidth={1.5} />
        <span style={{ fontSize: 14 }}>No log events to display</span>
      </div>
    );
  }

  return (
    <div>
      {/* Chart */}
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart
          data={mergedChartData}
          margin={{ top: 10, right: 20, bottom: 5, left: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="displayTime"
            tick={{ fontSize: 11, fill: '#6b7280' }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: '#6b7280' }}
            tickLine={false}
            axisLine={{ stroke: '#e5e7eb' }}
            label={{
              value: 'Anomaly Score',
              angle: -90,
              position: 'insideLeft',
              style: { fontSize: 12, fill: '#6b7280' },
            }}
          />
          <Tooltip content={<TimelineTooltip />} />
          <Legend
            verticalAlign="bottom"
            height={36}
            iconType="circle"
            wrapperStyle={{ fontSize: 12 }}
          />

          {/* Threshold line at 0.7 */}
          <ReferenceLine
            y={ANOMALY_THRESHOLD}
            stroke="#ef4444"
            strokeDasharray="6 4"
            strokeWidth={1.5}
            label={{
              value: `Threshold (${ANOMALY_THRESHOLD})`,
              position: 'right',
              fill: '#ef4444',
              fontSize: 11,
            }}
          />

          {/* One Line per source */}
          {sourceList.map((source) => (
            <Line
              key={source}
              type="monotone"
              dataKey={`score_${source}`}
              name={source}
              stroke={sourceColorMap[source]}
              strokeWidth={2}
              dot={<PulsingDot />}
              activeDot={{ r: 6 }}
              connectNulls
            />
          ))}

          {/* Metric overlay line if metrics provided */}
          {metricData && (
            <Line
              type="monotone"
              dataKey="score_metric"
              name={metrics?.service || 'Metric'}
              stroke="#a855f7"
              strokeWidth={2}
              strokeDasharray="5 3"
              dot={{ r: 3, fill: '#a855f7' }}
              connectNulls
            />
          )}

          {/* Scatter for log event markers */}
          <Scatter
            name="Log Events"
            data={scatterData}
            dataKey="anomaly_score"
            shape={<LogMarkerShape />}
            onClick={handleScatterClick}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Level legend */}
      <div
        style={{
          display: 'flex',
          gap: 16,
          justifyContent: 'center',
          marginTop: 8,
          fontSize: 12,
          color: '#6b7280',
        }}
      >
        {Object.entries(LEVEL_COLORS)
          .filter(([key]) => key !== 'WARNING') // skip duplicate
          .map(([level, color]) => (
            <div key={level} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: color,
                  display: 'inline-block',
                }}
              />
              {level}
            </div>
          ))}
      </div>

      {/* Selected log detail card */}
      {selectedLog && (
        <div
          style={{
            marginTop: 16,
            background: '#fff',
            border: '1px solid #e5e7eb',
            borderRadius: 8,
            padding: 16,
            boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
            position: 'relative',
          }}
        >
          <button
            onClick={clearSelection}
            style={{
              position: 'absolute',
              top: 10,
              right: 10,
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: 4,
              color: '#9ca3af',
              lineHeight: 1,
            }}
            aria-label="Close detail"
          >
            <X size={18} />
          </button>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 12,
            }}
          >
            <AlertTriangle
              size={18}
              color={
                selectedLog.anomaly_score > ANOMALY_THRESHOLD
                  ? '#ef4444'
                  : '#f59e0b'
              }
            />
            <span style={{ fontWeight: 700, fontSize: 15, color: '#1f2937' }}>
              Log Event Detail
            </span>
            <span
              style={{
                marginLeft: 'auto',
                padding: '2px 10px',
                borderRadius: 4,
                fontSize: 11,
                fontWeight: 700,
                color: '#fff',
                background: getLevelColor(selectedLog.level),
              }}
            >
              {selectedLog.level?.toUpperCase()}
            </span>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '120px 1fr',
              gap: '8px 12px',
              fontSize: 13,
              color: '#374151',
            }}
          >
            <span style={{ fontWeight: 600, color: '#6b7280' }}>Timestamp</span>
            <span>{formatFullTimestamp(selectedLog.timestamp)}</span>

            <span style={{ fontWeight: 600, color: '#6b7280' }}>Source</span>
            <span>{selectedLog.source}</span>

            <span style={{ fontWeight: 600, color: '#6b7280' }}>Anomaly Score</span>
            <span>
              <span
                style={{
                  display: 'inline-block',
                  padding: '1px 8px',
                  borderRadius: 4,
                  fontWeight: 700,
                  fontSize: 12,
                  color: '#fff',
                  background:
                    selectedLog.anomaly_score > ANOMALY_THRESHOLD
                      ? '#ef4444'
                      : '#10b981',
                }}
              >
                {selectedLog.anomaly_score?.toFixed(4)}
              </span>
            </span>

            <span style={{ fontWeight: 600, color: '#6b7280' }}>Message</span>
            <span style={{ wordBreak: 'break-word' }}>{selectedLog.message}</span>

            {selectedLog.template && (
              <>
                <span style={{ fontWeight: 600, color: '#6b7280' }}>Template</span>
                <code
                  style={{
                    background: '#f3f4f6',
                    padding: '4px 8px',
                    borderRadius: 4,
                    fontSize: 12,
                    wordBreak: 'break-all',
                  }}
                >
                  {selectedLog.template}
                </code>
              </>
            )}
          </div>

          {/* Raw log line */}
          <div style={{ marginTop: 12 }}>
            <span
              style={{ fontWeight: 600, fontSize: 12, color: '#6b7280', display: 'block', marginBottom: 4 }}
            >
              Raw Log Line
            </span>
            <pre
              style={{
                background: '#1f2937',
                color: '#e5e7eb',
                padding: 12,
                borderRadius: 6,
                fontSize: 12,
                lineHeight: 1.5,
                overflowX: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
                margin: 0,
              }}
            >
              {`[${selectedLog.timestamp}] [${selectedLog.level?.toUpperCase()}] [${selectedLog.source}] ${selectedLog.message}`}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnomalyTimeline;
