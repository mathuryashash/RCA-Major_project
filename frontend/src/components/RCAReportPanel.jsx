import React, { useState, useMemo, useCallback } from 'react';
import { AlertTriangle, ChevronDown, ChevronRight } from 'lucide-react';

// ---------------------------------------------------------------------------
// Simple Markdown-to-HTML converter (no external library)
// Handles: # ## ###, **bold**, *italic*, - list, > blockquote, ---, `code`
// ---------------------------------------------------------------------------
function renderMarkdown(md) {
  if (!md) return '';

  const escaped = md
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // We need to un-escape the `>` we just escaped so blockquotes still work.
  // Process line-by-line to handle block-level elements, then do inline passes.
  const lines = escaped.split('\n');
  const html = [];
  let inList = false;
  let inBlockquote = false;

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // --- Horizontal rule ---
    if (/^-{3,}$/.test(line.trim())) {
      if (inList) { html.push('</ul>'); inList = false; }
      if (inBlockquote) { html.push('</blockquote>'); inBlockquote = false; }
      html.push('<hr/>');
      continue;
    }

    // --- Headings ---
    const headingMatch = line.match(/^(#{1,3})\s+(.*)/);
    if (headingMatch) {
      if (inList) { html.push('</ul>'); inList = false; }
      if (inBlockquote) { html.push('</blockquote>'); inBlockquote = false; }
      const level = headingMatch[1].length;
      html.push(`<h${level}>${inlineMarkdown(headingMatch[2])}</h${level}>`);
      continue;
    }

    // --- Blockquote (original > was escaped to &gt;) ---
    if (line.startsWith('&gt; ') || line === '&gt;') {
      if (inList) { html.push('</ul>'); inList = false; }
      if (!inBlockquote) { html.push('<blockquote>'); inBlockquote = true; }
      const content = line.replace(/^&gt;\s?/, '');
      html.push(`<p>${inlineMarkdown(content)}</p>`);
      continue;
    } else if (inBlockquote) {
      html.push('</blockquote>');
      inBlockquote = false;
    }

    // --- Unordered list ---
    const listMatch = line.match(/^[-*]\s+(.*)/);
    if (listMatch) {
      if (!inList) { html.push('<ul>'); inList = true; }
      html.push(`<li>${inlineMarkdown(listMatch[1])}</li>`);
      continue;
    } else if (inList) {
      html.push('</ul>');
      inList = false;
    }

    // --- Blank line ---
    if (line.trim() === '') {
      continue;
    }

    // --- Paragraph ---
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }

  if (inList) html.push('</ul>');
  if (inBlockquote) html.push('</blockquote>');

  return html.join('\n');
}

/** Inline markdown: bold, italic, inline code */
function inlineMarkdown(text) {
  return text
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>');
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/** "db_connection_pool" -> "Db Connection Pool" */
function humanize(str) {
  if (!str) return '';
  return str
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Format ISO timestamp to a short human-readable form */
function formatTimestamp(iso) {
  if (!iso) return '--';
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return iso;
  }
}

/** Confidence bar color thresholds */
function confidenceColor(conf) {
  if (conf > 0.7) return '#22c55e';   // green
  if (conf >= 0.4) return '#f59e0b';  // amber
  return '#ef4444';                     // red
}

/** Rank badge color */
function rankColor(rank) {
  if (rank === 1) return '#d4a017'; // gold
  if (rank === 2) return '#a8a9ad'; // silver
  if (rank === 3) return '#cd7f32'; // bronze
  return '#6b7280';                  // gray
}

// ---------------------------------------------------------------------------
// Build a causal chain path for a given cause using the edges
// Walk backwards from the cause through causal_chain edges to build the path
// ---------------------------------------------------------------------------
function buildChainPath(cause, edges) {
  if (!edges || edges.length === 0) return [cause];

  // Build adjacency: to -> from (reverse, since we trace backwards from effect)
  const reverseAdj = {};
  for (const edge of edges) {
    if (!reverseAdj[edge.to]) reverseAdj[edge.to] = [];
    reverseAdj[edge.to].push(edge.from);
  }

  // Walk backwards from the cause
  const path = [cause];
  const visited = new Set([cause]);
  let current = cause;
  const MAX_DEPTH = 10;

  for (let depth = 0; depth < MAX_DEPTH; depth++) {
    const parents = reverseAdj[current];
    if (!parents || parents.length === 0) break;
    // Pick the first parent we haven't visited
    const next = parents.find((p) => !visited.has(p));
    if (!next) break;
    visited.add(next);
    path.unshift(next);
    current = next;
  }

  return path;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const styles = {
  panel: {
    display: 'flex',
    gap: 24,
    flexWrap: 'wrap',
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  },
  leftCol: {
    flex: '1 1 380px',
    minWidth: 320,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  rightCol: {
    flex: '1 1 480px',
    minWidth: 340,
    maxHeight: 700,
    overflowY: 'auto',
    background: '#fff',
    borderRadius: 8,
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    padding: 20,
  },
  sectionTitle: {
    fontSize: '1.1rem',
    fontWeight: 700,
    marginBottom: 8,
    color: '#1f2937',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  causeCard: (isSelected) => ({
    background: isSelected ? '#eff6ff' : '#fff',
    border: isSelected ? '2px solid #3b82f6' : '1px solid #e5e7eb',
    borderRadius: 8,
    padding: '12px 14px',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
    boxShadow: isSelected
      ? '0 0 0 2px rgba(59,130,246,0.2)'
      : '0 1px 2px rgba(0,0,0,0.05)',
  }),
  rankBadge: (rank) => ({
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 28,
    height: 28,
    borderRadius: '50%',
    background: rankColor(rank),
    color: '#fff',
    fontWeight: 700,
    fontSize: 13,
    flexShrink: 0,
  }),
  confidenceBarOuter: {
    height: 8,
    borderRadius: 4,
    background: '#e5e7eb',
    flex: 1,
    overflow: 'hidden',
  },
  confidenceBarInner: (conf) => ({
    height: '100%',
    borderRadius: 4,
    width: `${Math.min(conf * 100, 100)}%`,
    background: confidenceColor(conf),
    transition: 'width 0.3s ease',
  }),
  breadcrumb: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 4,
    fontSize: 12,
    color: '#6b7280',
    marginTop: 6,
  },
  breadcrumbNode: {
    background: '#f3f4f6',
    borderRadius: 4,
    padding: '2px 6px',
    fontSize: 11,
    fontFamily: "'Fira Code', 'Consolas', monospace",
    color: '#374151',
  },
  breadcrumbArrow: {
    color: '#9ca3af',
    fontSize: 11,
  },
  typeBadge: (type) => ({
    display: 'inline-block',
    fontSize: 10,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    padding: '2px 6px',
    borderRadius: 4,
    background: type === 'metric' ? '#dbeafe' : '#fef3c7',
    color: type === 'metric' ? '#1d4ed8' : '#92400e',
  }),
  evidenceToggle: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    cursor: 'pointer',
    background: 'none',
    border: '1px solid #e5e7eb',
    borderRadius: 6,
    padding: '8px 12px',
    fontSize: 13,
    fontWeight: 600,
    color: '#374151',
    width: '100%',
    textAlign: 'left',
    marginTop: 4,
  },
  evidenceContent: {
    background: '#f9fafb',
    border: '1px solid #e5e7eb',
    borderTop: 'none',
    borderRadius: '0 0 6px 6px',
    padding: 12,
    fontSize: 13,
  },
  metricRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '6px 0',
    borderBottom: '1px solid #f3f4f6',
  },
  narrative: {
    lineHeight: 1.7,
    fontSize: 14,
    color: '#1f2937',
  },
  empty: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 48,
    color: '#9ca3af',
    gap: 12,
    textAlign: 'center',
  },
};

// Narrative style overrides injected into dangerouslySetInnerHTML wrapper
const narrativeCSS = `
  .rca-narrative h1 { font-size: 1.3rem; margin: 16px 0 8px; color: #111827; }
  .rca-narrative h2 { font-size: 1.15rem; margin: 14px 0 6px; color: #1f2937; }
  .rca-narrative h3 { font-size: 1rem; margin: 12px 0 4px; color: #374151; }
  .rca-narrative p { margin: 4px 0 10px; }
  .rca-narrative ul { padding-left: 20px; margin: 4px 0 10px; }
  .rca-narrative li { margin-bottom: 4px; }
  .rca-narrative blockquote {
    border-left: 3px solid #d1d5db;
    margin: 8px 0;
    padding: 4px 12px;
    color: #6b7280;
    background: #f9fafb;
    border-radius: 0 4px 4px 0;
  }
  .rca-narrative code {
    font-family: 'Fira Code', Consolas, 'Courier New', monospace;
    background: #f3f4f6;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.9em;
    color: #be185d;
  }
  .rca-narrative hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 12px 0;
  }
`;

// ---------------------------------------------------------------------------
// CauseCard
// ---------------------------------------------------------------------------
function CauseCard({ cause, isSelected, chainPath, onClick }) {
  return (
    <div
      style={styles.causeCard(isSelected)}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
    >
      {/* Top row: badge + name + type */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}
      >
        <span style={styles.rankBadge(cause.rank)}>{cause.rank}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontWeight: 600,
              fontSize: 14,
              color: '#111827',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
            title={humanize(cause.cause)}
          >
            {humanize(cause.cause)}
          </div>
        </div>
        <span style={styles.typeBadge(cause.type)}>{cause.type}</span>
      </div>

      {/* Confidence row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginTop: 8,
        }}
      >
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: confidenceColor(cause.confidence),
            minWidth: 42,
          }}
        >
          {(cause.confidence * 100).toFixed(1)}%
        </span>
        <div style={styles.confidenceBarOuter}>
          <div style={styles.confidenceBarInner(cause.confidence)} />
        </div>
      </div>

      {/* Causal chain breadcrumb */}
      {chainPath.length > 1 && (
        <div style={styles.breadcrumb}>
          {chainPath.map((node, idx) => (
            <React.Fragment key={idx}>
              {idx > 0 && <span style={styles.breadcrumbArrow}>&rarr;</span>}
              <span style={styles.breadcrumbNode}>{humanize(node)}</span>
            </React.Fragment>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// EvidenceSection
// ---------------------------------------------------------------------------
function EvidenceSection({ anomalousMetrics }) {
  const [expanded, setExpanded] = useState({});

  if (!anomalousMetrics || anomalousMetrics.length === 0) return null;

  const toggle = (idx) =>
    setExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }));

  return (
    <div style={{ marginTop: 16 }}>
      <div style={styles.sectionTitle}>
        <AlertTriangle size={16} />
        Evidence — Anomalous Metrics
      </div>
      {anomalousMetrics.map((metric, idx) => {
        const isOpen = !!expanded[idx];
        return (
          <div key={idx} style={{ marginBottom: 4 }}>
            <button
              style={styles.evidenceToggle}
              onClick={() => toggle(idx)}
              aria-expanded={isOpen}
            >
              {isOpen ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
              <span style={{ flex: 1 }}>{humanize(metric.name)}</span>
              <span
                style={{
                  fontSize: 12,
                  color: confidenceColor(metric.score),
                  fontWeight: 700,
                }}
              >
                Score: {metric.score.toFixed(3)}
              </span>
            </button>
            {isOpen && (
              <div style={styles.evidenceContent}>
                <div style={styles.metricRow}>
                  <span style={{ fontWeight: 600, color: '#374151' }}>
                    Metric
                  </span>
                  <span
                    style={{
                      fontFamily: "'Fira Code', Consolas, monospace",
                      fontSize: 12,
                    }}
                  >
                    {metric.name}
                  </span>
                </div>
                <div style={styles.metricRow}>
                  <span style={{ fontWeight: 600, color: '#374151' }}>
                    Anomaly Score
                  </span>
                  <span
                    style={{
                      fontWeight: 700,
                      color: confidenceColor(metric.score),
                    }}
                  >
                    {metric.score.toFixed(4)}
                  </span>
                </div>
                <div style={styles.metricRow}>
                  <span style={{ fontWeight: 600, color: '#374151' }}>
                    First Seen (Onset)
                  </span>
                  <span>{formatTimestamp(metric.first_seen)}</span>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// RCAReportPanel (main export)
// ---------------------------------------------------------------------------
export default function RCAReportPanel({ report, onSelectCause }) {
  const [selectedCause, setSelectedCause] = useState(null);

  // Derive top-5 ranked causes (sorted by rank, capped at 5)
  const topCauses = useMemo(() => {
    if (!report?.ranked_causes) return [];
    return [...report.ranked_causes]
      .sort((a, b) => a.rank - b.rank)
      .slice(0, 5);
  }, [report]);

  // Pre-compute chain paths for each cause
  const chainPaths = useMemo(() => {
    const edges = report?.causal_chain ?? [];
    const map = {};
    for (const cause of topCauses) {
      map[cause.cause] = buildChainPath(cause.cause, edges);
    }
    return map;
  }, [topCauses, report]);

  const handleSelectCause = useCallback(
    (cause) => {
      setSelectedCause((prev) =>
        prev === cause.cause ? null : cause.cause,
      );
      if (onSelectCause) {
        onSelectCause(cause.cause);
      }
    },
    [onSelectCause],
  );

  // Render narrative HTML once
  const narrativeHTML = useMemo(
    () => renderMarkdown(report?.narrative),
    [report],
  );

  // ---- Empty state ----
  if (!report) {
    return (
      <div style={styles.empty}>
        <AlertTriangle size={32} color="#d1d5db" />
        <div style={{ fontSize: 15, fontWeight: 600, color: '#6b7280' }}>
          No RCA report available
        </div>
        <div style={{ fontSize: 13, color: '#9ca3af' }}>
          Trigger an analysis to generate a root cause report.
        </div>
      </div>
    );
  }

  return (
    <div style={styles.panel}>
      {/* Inject narrative CSS */}
      <style>{narrativeCSS}</style>

      {/* ---- LEFT COLUMN: Ranked causes + Evidence ---- */}
      <div style={styles.leftCol}>
        <div style={styles.sectionTitle}>
          Ranked Root Causes
          {report.status && (
            <span
              style={{
                fontSize: 11,
                fontWeight: 600,
                padding: '2px 8px',
                borderRadius: 10,
                background: report.status === 'completed' ? '#dcfce7' : '#fef3c7',
                color: report.status === 'completed' ? '#166534' : '#92400e',
                marginLeft: 'auto',
              }}
            >
              {report.status}
            </span>
          )}
        </div>

        {topCauses.length === 0 ? (
          <div style={{ color: '#9ca3af', fontSize: 13, padding: 12 }}>
            No ranked causes found in this report.
          </div>
        ) : (
          topCauses.map((cause) => (
            <CauseCard
              key={cause.rank}
              cause={cause}
              isSelected={selectedCause === cause.cause}
              chainPath={chainPaths[cause.cause] || [cause.cause]}
              onClick={() => handleSelectCause(cause)}
            />
          ))
        )}

        {/* Evidence section */}
        <EvidenceSection anomalousMetrics={report.anomalous_metrics} />
      </div>

      {/* ---- RIGHT COLUMN: Narrative report ---- */}
      <div style={styles.rightCol}>
        <div
          style={{
            fontSize: '1.1rem',
            fontWeight: 700,
            color: '#111827',
            marginBottom: 4,
          }}
        >
          Incident Report
        </div>
        <div
          style={{
            fontSize: 12,
            color: '#6b7280',
            marginBottom: 16,
            display: 'flex',
            flexWrap: 'wrap',
            gap: 16,
          }}
        >
          {report.incident_id && (
            <span>
              <strong>ID:</strong> {report.incident_id}
            </span>
          )}
          {report.detected_at && (
            <span>
              <strong>Detected:</strong> {formatTimestamp(report.detected_at)}
            </span>
          )}
          {report.root_cause && (
            <span>
              <strong>Root Cause:</strong> {humanize(report.root_cause)}
            </span>
          )}
        </div>

        {narrativeHTML ? (
          <div
            className="rca-narrative"
            style={styles.narrative}
            dangerouslySetInnerHTML={{ __html: narrativeHTML }}
          />
        ) : (
          <div style={{ color: '#9ca3af', fontSize: 13 }}>
            No narrative available for this report.
          </div>
        )}

        {/* Remediation plan summary */}
        {report.remediation_plan && (
          <div style={{ marginTop: 20 }}>
            <div
              style={{
                fontSize: '1rem',
                fontWeight: 700,
                color: '#111827',
                marginBottom: 8,
                paddingTop: 12,
                borderTop: '1px solid #e5e7eb',
              }}
            >
              Remediation Plan
            </div>
            {report.remediation_plan.actions?.map((action, idx) => (
              <div
                key={idx}
                style={{
                  padding: '8px 12px',
                  marginBottom: 6,
                  background: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  borderRadius: 6,
                  fontSize: 13,
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 2 }}>
                  {action.tier && (
                    <span
                      style={{
                        textTransform: 'uppercase',
                        fontSize: 10,
                        fontWeight: 700,
                        color: '#15803d',
                        marginRight: 6,
                      }}
                    >
                      [{action.tier}]
                    </span>
                  )}
                  {action.description}
                </div>
                {action.command && (
                  <code
                    style={{
                      display: 'block',
                      marginTop: 4,
                      fontSize: 12,
                      fontFamily: "'Fira Code', Consolas, monospace",
                      background: '#ecfdf5',
                      padding: '4px 8px',
                      borderRadius: 4,
                      color: '#064e3b',
                      overflowX: 'auto',
                    }}
                  >
                    {action.command}
                  </code>
                )}
              </div>
            ))}
            {report.remediation_plan.prevention && (
              <div
                style={{
                  fontSize: 13,
                  color: '#374151',
                  marginTop: 8,
                  fontStyle: 'italic',
                }}
              >
                <strong>Prevention:</strong> {report.remediation_plan.prevention}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
