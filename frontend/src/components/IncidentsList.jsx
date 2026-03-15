import React, { useMemo } from 'react';

const STATUS_STYLES = {
  complete: { backgroundColor: '#dcfce7', color: '#166534' },
  processing: { backgroundColor: '#fef3c7', color: '#92400e' },
  failed: { backgroundColor: '#fee2e2', color: '#991b1b' },
};

function StatusBadge({ status }) {
  const normalized = (status || '').toLowerCase();
  const style = STATUS_STYLES[normalized] || {
    backgroundColor: '#f3f4f6',
    color: '#374151',
  };

  return (
    <span
      style={{
        display: 'inline-block',
        padding: '2px 8px',
        fontSize: '0.75rem',
        fontWeight: 600,
        borderRadius: '9999px',
        ...style,
      }}
    >
      {status}
    </span>
  );
}

function formatDate(dateStr) {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleString();
}

export default function IncidentsList({ incidents, selectedId, onSelect }) {
  const sorted = useMemo(() => {
    if (!incidents || incidents.length === 0) return [];
    return [...incidents].sort(
      (a, b) => new Date(b.detected_at) - new Date(a.detected_at),
    );
  }, [incidents]);

  if (sorted.length === 0) {
    return (
      <div
        style={{
          padding: '32px 16px',
          textAlign: 'center',
          color: '#9ca3af',
          fontSize: '0.9rem',
        }}
      >
        No incidents yet
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        padding: '8px',
        overflowY: 'auto',
      }}
    >
      {sorted.map((incident) => {
        const isSelected = incident.incident_id === selectedId;

        return (
          <div
            key={incident.incident_id}
            onClick={() => onSelect(incident.incident_id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelect(incident.incident_id);
              }
            }}
            style={{
              padding: '12px',
              borderRadius: '8px',
              border: isSelected ? '2px solid #2563eb' : '1px solid #e5e7eb',
              backgroundColor: isSelected ? '#eff6ff' : '#ffffff',
              cursor: 'pointer',
              transition: 'background-color 0.15s, border-color 0.15s',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '6px',
              }}
            >
              <span
                style={{
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  color: '#111827',
                  fontFamily: 'monospace',
                }}
              >
                {incident.incident_id}
              </span>
              <StatusBadge status={incident.status} />
            </div>

            <div
              style={{
                fontSize: '0.8rem',
                color: '#6b7280',
                marginBottom: '4px',
              }}
            >
              {formatDate(incident.detected_at)}
            </div>

            {incident.root_cause && (
              <div
                style={{
                  fontSize: '0.8rem',
                  color: '#374151',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {incident.root_cause}
              </div>
            )}

            {incident.priority != null && (
              <div
                style={{
                  marginTop: '4px',
                  fontSize: '0.75rem',
                  color: '#6b7280',
                }}
              >
                Priority: {incident.priority}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
