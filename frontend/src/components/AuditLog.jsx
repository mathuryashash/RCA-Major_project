import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { FileText, AlertCircle, Loader } from 'lucide-react';

const OUTCOME_STYLES = {
  success: { color: '#16a34a' },
  failure: { color: '#dc2626' },
  skipped: { color: '#9ca3af' },
};

function OutcomeCell({ outcome }) {
  const normalized = (outcome || '').toLowerCase();
  const style = OUTCOME_STYLES[normalized] || { color: '#374151' };

  return (
    <span style={{ fontWeight: 600, ...style }}>
      {outcome}
    </span>
  );
}

function formatTimestamp(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

export default function AuditLog({ incidentId }) {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!incidentId) {
      setEntries([]);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    async function fetchAudit() {
      try {
        const res = await axios.get(`/api/audit/${incidentId}`);
        if (!cancelled) {
          setEntries(res.data.entries || []);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.response?.data?.detail || err.message || 'Failed to load audit log');
          setEntries([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchAudit();

    return () => {
      cancelled = true;
    };
  }, [incidentId]);

  const sorted = useMemo(
    () =>
      [...entries].sort(
        (a, b) => new Date(b.timestamp) - new Date(a.timestamp),
      ),
    [entries],
  );

  if (!incidentId) {
    return (
      <div style={{ padding: '24px', color: '#9ca3af', textAlign: 'center' }}>
        Select an incident to view its audit log.
      </div>
    );
  }

  if (loading) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          padding: '32px',
          color: '#6b7280',
        }}
      >
        <Loader size={18} style={{ animation: 'audit-spin 1s linear infinite' }} />
        <span>Loading audit log…</span>
        <style>{`
          @keyframes audit-spin {
            from { transform: rotate(0deg); }
            to   { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '16px',
          color: '#dc2626',
          backgroundColor: '#fef2f2',
          borderRadius: '6px',
          margin: '16px',
          fontSize: '0.9rem',
        }}
      >
        <AlertCircle size={18} />
        <span>{error}</span>
      </div>
    );
  }

  if (sorted.length === 0) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '8px',
          padding: '32px',
          color: '#9ca3af',
        }}
      >
        <FileText size={32} />
        <span>No audit entries yet</span>
      </div>
    );
  }

  return (
    <div style={{ overflowX: 'auto', padding: '8px' }}>
      <table
        style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '0.85rem',
        }}
      >
        <thead>
          <tr
            style={{
              textAlign: 'left',
              borderBottom: '2px solid #e5e7eb',
            }}
          >
            <th style={thStyle}>Timestamp</th>
            <th style={thStyle}>Action</th>
            <th style={thStyle}>Command</th>
            <th style={thStyle}>Executor</th>
            <th style={thStyle}>Outcome</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((entry, idx) => (
            <tr
              key={`${entry.timestamp}-${idx}`}
              style={{
                borderBottom: '1px solid #f3f4f6',
                backgroundColor: idx % 2 === 0 ? '#ffffff' : '#f9fafb',
              }}
            >
              <td style={tdStyle}>
                <span style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                  {formatTimestamp(entry.timestamp)}
                </span>
              </td>
              <td style={tdStyle}>{entry.action_type || '—'}</td>
              <td style={tdStyle}>
                {entry.command_executed ? (
                  <code
                    style={{
                      padding: '2px 6px',
                      backgroundColor: '#f3f4f6',
                      borderRadius: '4px',
                      fontSize: '0.8rem',
                      wordBreak: 'break-all',
                    }}
                  >
                    {entry.command_executed}
                  </code>
                ) : (
                  '—'
                )}
              </td>
              <td style={tdStyle}>{entry.executor || '—'}</td>
              <td style={tdStyle}>
                <OutcomeCell outcome={entry.outcome} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const thStyle = {
  padding: '10px 12px',
  fontWeight: 600,
  color: '#374151',
  whiteSpace: 'nowrap',
};

const tdStyle = {
  padding: '8px 12px',
  color: '#4b5563',
  verticalAlign: 'top',
};
