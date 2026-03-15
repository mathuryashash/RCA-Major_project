import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Loader, Plus } from 'lucide-react';

export default function Header({ onAnalyze, loading }) {
  const [healthy, setHealthy] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        const res = await axios.get('/api/health');
        if (!cancelled) {
          setHealthy(res.status === 200);
        }
      } catch {
        if (!cancelled) {
          setHealthy(false);
        }
      }
    }

    checkHealth();

    return () => {
      cancelled = true;
    };
  }, []);

  const dotColor =
    healthy === null
      ? '#9ca3af'
      : healthy
        ? '#22c55e'
        : '#ef4444';

  return (
    <header
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 24px',
        borderBottom: '1px solid #e5e7eb',
        backgroundColor: '#ffffff',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <Activity size={24} />
        <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>
          RCA System
        </h1>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div
          style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          title={
            healthy === null
              ? 'Checking health…'
              : healthy
                ? 'API healthy'
                : 'API unhealthy'
          }
        >
          <span
            style={{
              display: 'inline-block',
              width: 10,
              height: 10,
              borderRadius: '50%',
              backgroundColor: dotColor,
            }}
          />
          <span style={{ fontSize: '0.85rem', color: '#6b7280' }}>
            {healthy === null ? 'Checking…' : healthy ? 'Healthy' : 'Unhealthy'}
          </span>
        </div>

        <button
          onClick={onAnalyze}
          disabled={loading}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 16px',
            fontSize: '0.9rem',
            fontWeight: 600,
            color: '#ffffff',
            backgroundColor: loading ? '#9ca3af' : '#2563eb',
            border: 'none',
            borderRadius: '6px',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? (
            <Loader size={16} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
          ) : (
            <Plus size={16} />
          )}
          {loading ? 'Running…' : 'New Analysis'}
        </button>
      </div>

      {/* inline keyframes for the spinner */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>
    </header>
  );
}
