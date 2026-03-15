import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import {
  Play,
  Square,
  Clock,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
  Shield,
} from 'lucide-react';

/**
 * FR-25 through FR-29, FR-32: Remediation Action Panel
 *
 * Three-tier remediation display:
 *   Tier 1 — Auto-actions with 30-second countdown + confidence gate
 *   Tier 2 — Interactive step-by-step walkthrough checklist
 *   Tier 3 — Read-only advisory / prevention recommendations
 * Plus a prevention checklist fetched from the API.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COUNTDOWN_SECONDS = 30;

const TABS = [
  { key: 'auto', label: 'Auto-Actions' },
  { key: 'walkthrough', label: 'Walkthrough' },
  { key: 'advisory', label: 'Advisory' },
];

// ---------------------------------------------------------------------------
// Inline styles (keeps component self-contained, no external CSS needed)
// ---------------------------------------------------------------------------

const S = {
  panel: {
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    background: '#111827',
    borderRadius: 12,
    border: '1px solid #1f2937',
    color: '#e5e7eb',
    overflow: 'hidden',
  },
  header: {
    padding: '16px 20px',
    borderBottom: '1px solid #1f2937',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    fontSize: 16,
    fontWeight: 600,
  },
  tabs: {
    display: 'flex',
    borderBottom: '1px solid #1f2937',
  },
  tab: (active) => ({
    flex: 1,
    padding: '10px 16px',
    textAlign: 'center',
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: 500,
    background: active ? '#1f2937' : 'transparent',
    color: active ? '#f9fafb' : '#9ca3af',
    borderBottom: active ? '2px solid #3b82f6' : '2px solid transparent',
    transition: 'all 0.15s',
    userSelect: 'none',
  }),
  body: {
    padding: 20,
    maxHeight: 600,
    overflowY: 'auto',
  },
  empty: {
    textAlign: 'center',
    padding: '48px 20px',
    color: '#6b7280',
    fontSize: 14,
  },
  // Confidence gate warning banner
  gateBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '10px 16px',
    background: '#78350f',
    borderBottom: '1px solid #92400e',
    color: '#fbbf24',
    fontSize: 13,
    fontWeight: 500,
  },
  // Tier 1
  actionCard: (status) => ({
    background: status === 'executed' ? '#064e3b' : status === 'cancelled' ? '#1f2937' : '#1c1917',
    border: `1px solid ${status === 'executed' ? '#059669' : status === 'cancelled' ? '#374151' : '#7f1d1d'}`,
    borderRadius: 8,
    padding: 14,
    marginBottom: 10,
  }),
  actionHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  actionTitle: {
    fontWeight: 600,
    fontSize: 14,
  },
  actionDesc: {
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 8,
  },
  codeBlock: {
    background: '#0d1117',
    border: '1px solid #21262d',
    borderRadius: 6,
    padding: '8px 12px',
    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    fontSize: 12,
    color: '#c9d1d9',
    overflowX: 'auto',
    marginBottom: 8,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
  },
  countdown: (seconds) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 13,
    fontWeight: 700,
    fontVariantNumeric: 'tabular-nums',
    color: seconds <= 10 ? '#ef4444' : seconds <= 20 ? '#f59e0b' : '#10b981',
  }),
  btnCancel: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    padding: '5px 12px',
    borderRadius: 6,
    border: '1px solid #7f1d1d',
    background: '#450a0a',
    color: '#fca5a5',
    fontSize: 12,
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'background 0.15s',
  },
  badge: (variant) => {
    const map = {
      executed: { bg: '#064e3b', color: '#34d399', border: '#059669' },
      cancelled: { bg: '#1f2937', color: '#9ca3af', border: '#374151' },
      disabled: { bg: '#78350f', color: '#fbbf24', border: '#92400e' },
    };
    const s = map[variant] || map.disabled;
    return {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 4,
      padding: '3px 10px',
      borderRadius: 9999,
      fontSize: 11,
      fontWeight: 600,
      background: s.bg,
      color: s.color,
      border: `1px solid ${s.border}`,
    };
  },
  // Tier 2
  stepCard: (active, completed) => ({
    background: completed ? '#064e3b' : active ? '#1e293b' : '#111827',
    border: `1px solid ${completed ? '#059669' : active ? '#3b82f6' : '#1f2937'}`,
    borderRadius: 8,
    padding: 14,
    marginBottom: 10,
    opacity: !active && !completed ? 0.5 : 1,
    transition: 'all 0.2s',
  }),
  stepHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  stepNumber: (active, completed) => ({
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 28,
    height: 28,
    borderRadius: '50%',
    fontSize: 12,
    fontWeight: 700,
    background: completed ? '#059669' : active ? '#3b82f6' : '#374151',
    color: '#fff',
    marginRight: 10,
    flexShrink: 0,
  }),
  stepTitle: {
    fontWeight: 600,
    fontSize: 14,
    flex: 1,
  },
  stepMeta: {
    display: 'flex',
    gap: 16,
    flexWrap: 'wrap',
    fontSize: 12,
    color: '#9ca3af',
    marginBottom: 8,
  },
  safetyNote: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 6,
    padding: '8px 12px',
    background: '#78350f',
    border: '1px solid #92400e',
    borderRadius: 6,
    fontSize: 12,
    color: '#fbbf24',
    marginBottom: 8,
  },
  stepLabel: {
    fontSize: 11,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    color: '#6b7280',
    marginBottom: 4,
  },
  btnPrimary: (disabled) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    padding: '7px 16px',
    borderRadius: 6,
    border: 'none',
    background: disabled ? '#374151' : '#2563eb',
    color: disabled ? '#6b7280' : '#fff',
    fontSize: 13,
    fontWeight: 500,
    cursor: disabled ? 'not-allowed' : 'pointer',
    transition: 'background 0.15s',
  }),
  progressBar: {
    height: 4,
    borderRadius: 2,
    background: '#1f2937',
    marginBottom: 16,
    overflow: 'hidden',
  },
  progressFill: (pct) => ({
    height: '100%',
    width: `${pct}%`,
    background: 'linear-gradient(90deg, #3b82f6, #10b981)',
    borderRadius: 2,
    transition: 'width 0.3s',
  }),
  // Tier 3
  advisoryCard: {
    background: '#1e293b',
    border: '1px solid #334155',
    borderRadius: 8,
    padding: 14,
    marginBottom: 10,
  },
  horizonBadge: (horizon) => {
    const map = {
      immediate: { bg: '#7f1d1d', color: '#fca5a5', border: '#991b1b' },
      short_term: { bg: '#78350f', color: '#fbbf24', border: '#92400e' },
      long_term: { bg: '#1e3a5f', color: '#93c5fd', border: '#1d4ed8' },
    };
    const s = map[horizon] || map.long_term;
    return {
      display: 'inline-block',
      padding: '2px 8px',
      borderRadius: 9999,
      fontSize: 11,
      fontWeight: 600,
      background: s.bg,
      color: s.color,
      border: `1px solid ${s.border}`,
    };
  },
  priorityBadge: (priority) => {
    const p = (priority || '').toLowerCase();
    const map = {
      critical: '#ef4444',
      high: '#f59e0b',
      medium: '#3b82f6',
      low: '#6b7280',
    };
    return {
      display: 'inline-block',
      padding: '2px 8px',
      borderRadius: 9999,
      fontSize: 11,
      fontWeight: 600,
      background: 'transparent',
      color: map[p] || '#9ca3af',
      border: `1px solid ${map[p] || '#4b5563'}`,
    };
  },
  advisoryMeta: {
    display: 'flex',
    gap: 10,
    alignItems: 'center',
    flexWrap: 'wrap',
    marginTop: 10,
    fontSize: 12,
  },
  // Prevention checklist
  preventionSection: {
    marginTop: 24,
    padding: '16px 20px',
    borderTop: '1px solid #1f2937',
  },
  preventionHeading: {
    fontSize: 15,
    fontWeight: 600,
    marginBottom: 14,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  preventionGroup: {
    marginBottom: 14,
  },
  preventionGroupTitle: {
    fontSize: 13,
    fontWeight: 600,
    color: '#9ca3af',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 8,
  },
  checkItem: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 8,
    padding: '6px 0',
    fontSize: 13,
    color: '#d1d5db',
  },
  checkbox: {
    marginTop: 2,
    accentColor: '#3b82f6',
    cursor: 'pointer',
  },
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Single Tier-1 auto-action card with countdown */
function AutoActionCard({ action, index, gateBlocked, onExecute }) {
  const [seconds, setSeconds] = useState(COUNTDOWN_SECONDS);
  const [status, setStatus] = useState(gateBlocked ? 'disabled' : 'pending'); // pending | cancelled | executed | disabled
  const intervalRef = useRef(null);

  // Start countdown only when not blocked
  useEffect(() => {
    if (status !== 'pending') return;

    intervalRef.current = setInterval(() => {
      setSeconds((prev) => {
        if (prev <= 1) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
          setStatus('executed');
          onExecute(index);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [status, index, onExecute]);

  // If gate becomes blocked after mount, disable
  useEffect(() => {
    if (gateBlocked && status === 'pending') {
      setStatus('disabled');
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
  }, [gateBlocked, status]);

  const handleCancel = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setStatus('cancelled');
  };

  return (
    <div style={S.actionCard(status)}>
      <div style={S.actionHeader}>
        <span style={S.actionTitle}>{action.action}</span>
        {status === 'pending' && (
          <span style={S.countdown(seconds)}>
            <Clock size={14} />
            {seconds}s
          </span>
        )}
        {status === 'executed' && (
          <span style={S.badge('executed')}>
            <CheckCircle size={12} /> Executed
          </span>
        )}
        {status === 'cancelled' && (
          <span style={S.badge('cancelled')}>
            <Square size={12} /> Cancelled
          </span>
        )}
        {status === 'disabled' && (
          <span style={S.badge('disabled')}>
            <AlertTriangle size={12} /> Blocked
          </span>
        )}
      </div>

      <div style={S.actionDesc}>{action.description}</div>

      <div style={S.codeBlock}>{action.command}</div>

      {status === 'pending' && (
        <button style={S.btnCancel} onClick={handleCancel}>
          <Square size={12} /> Cancel
        </button>
      )}
    </div>
  );
}

/** Tier-2 walkthrough step card */
function StepCard({ step, isActive, isCompleted, onMarkComplete }) {
  const [expanded, setExpanded] = useState(false);

  // Auto-expand active step
  useEffect(() => {
    if (isActive) setExpanded(true);
  }, [isActive]);

  return (
    <div style={S.stepCard(isActive, isCompleted)}>
      <div
        style={{ ...S.stepHeader, cursor: 'pointer' }}
        onClick={() => setExpanded((v) => !v)}
      >
        <div style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
          <span style={S.stepNumber(isActive, isCompleted)}>
            {isCompleted ? <CheckCircle size={14} /> : step.step}
          </span>
          <span style={S.stepTitle}>{step.title}</span>
        </div>
        <ChevronDown
          size={16}
          style={{
            color: '#6b7280',
            transition: 'transform 0.2s',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
        />
      </div>

      {expanded && (
        <div style={{ marginTop: 8 }}>
          <div style={S.stepMeta}>
            <span>
              <Clock size={12} style={{ verticalAlign: -2, marginRight: 4 }} />
              ~{step.est_minutes} min
            </span>
          </div>

          {step.safety_note && (
            <div style={S.safetyNote}>
              <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
              <span>{step.safety_note}</span>
            </div>
          )}

          <div style={S.stepLabel}>Command</div>
          <div style={S.codeBlock}>{step.command}</div>

          <div style={S.stepLabel}>Expected Output</div>
          <div style={S.codeBlock}>{step.expected_output}</div>

          <div style={S.stepLabel}>Verification</div>
          <div style={S.codeBlock}>{step.verification}</div>

          <div style={S.stepLabel}>Rollback</div>
          <div style={S.codeBlock}>{step.rollback}</div>

          {isActive && !isCompleted && (
            <button
              style={S.btnPrimary(false)}
              onClick={() => onMarkComplete(step.step)}
            >
              <CheckCircle size={14} /> Mark Complete
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** Tier-3 advisory card */
function AdvisoryCard({ item }) {
  const horizonLabel = (item.horizon || '').replace('_', ' ');

  return (
    <div style={S.advisoryCard}>
      <div style={{ fontSize: 14, lineHeight: 1.5 }}>{item.recommendation}</div>
      <div style={S.advisoryMeta}>
        <span style={S.horizonBadge(item.horizon)}>{horizonLabel}</span>
        <span style={S.priorityBadge(item.priority)}>{item.priority}</span>
        <span style={{ color: '#6b7280' }}>Owner: {item.owner}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Prevention Checklist
// ---------------------------------------------------------------------------

function PreventionChecklist({ incidentId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [checked, setChecked] = useState({});

  useEffect(() => {
    let cancelled = false;

    async function fetch() {
      try {
        setLoading(true);
        setError(null);
        const res = await axios.get(`/api/remediate/${incidentId}/prevention`);
        if (!cancelled) setData(res.data);
      } catch (err) {
        if (!cancelled) setError(err.message || 'Failed to load prevention checklist');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetch();
    return () => {
      cancelled = true;
    };
  }, [incidentId]);

  const toggle = (sectionKey, idx) => {
    const key = `${sectionKey}-${idx}`;
    setChecked((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  if (loading) {
    return (
      <div style={S.preventionSection}>
        <div style={S.preventionHeading}>
          <Shield size={16} /> Prevention Checklist
        </div>
        <div style={{ color: '#6b7280', fontSize: 13 }}>Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={S.preventionSection}>
        <div style={S.preventionHeading}>
          <Shield size={16} /> Prevention Checklist
        </div>
        <div style={{ color: '#ef4444', fontSize: 13 }}>{error}</div>
      </div>
    );
  }

  if (!data) return null;

  const sections = [
    { key: 'immediate', title: 'Immediate', items: data.immediate || [] },
    { key: 'short_term', title: 'Short-Term', items: data.short_term || [] },
    { key: 'long_term', title: 'Long-Term', items: data.long_term || [] },
  ];

  return (
    <div style={S.preventionSection}>
      <div style={S.preventionHeading}>
        <Shield size={16} /> Prevention Checklist
      </div>
      {sections.map((section) => (
        <div key={section.key} style={S.preventionGroup}>
          <div style={S.preventionGroupTitle}>{section.title}</div>
          {section.items.length === 0 && (
            <div style={{ color: '#4b5563', fontSize: 13 }}>No items</div>
          )}
          {section.items.map((item, idx) => {
            const ck = `${section.key}-${idx}`;
            return (
              <label key={ck} style={S.checkItem}>
                <input
                  type="checkbox"
                  style={S.checkbox}
                  checked={!!checked[ck]}
                  onChange={() => toggle(section.key, idx)}
                />
                <span style={checked[ck] ? { textDecoration: 'line-through', color: '#6b7280' } : undefined}>
                  {typeof item === 'string' ? item : item.description || item.recommendation || JSON.stringify(item)}
                </span>
              </label>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function RemediationPanel({ incidentId, remediationPlan, confidenceGate }) {
  const [activeTab, setActiveTab] = useState('auto');
  const [completedSteps, setCompletedSteps] = useState(new Set());
  const [executedActions, setExecutedActions] = useState(new Set());
  const [executeError, setExecuteError] = useState(null);

  const gateBlocked = confidenceGate != null && confidenceGate.passed === false;

  // -----------------------------------------------------------------------
  // Tier 1: fire execute endpoint when an action's countdown reaches 0
  // -----------------------------------------------------------------------
  const handleAutoExecute = useCallback(
    async (actionIndex) => {
      if (!remediationPlan) return;
      const action = remediationPlan.tier1_auto_actions[actionIndex];
      if (!action) return;

      try {
        setExecuteError(null);
        await axios.post(`/api/remediate/${incidentId}/execute`, {
          actions: [action],
          executor: 'dashboard_operator',
        });
        setExecutedActions((prev) => new Set(prev).add(actionIndex));
      } catch (err) {
        setExecuteError(err?.response?.data?.detail || err.message || 'Execution failed');
      }
    },
    [incidentId, remediationPlan],
  );

  // -----------------------------------------------------------------------
  // Tier 2: mark step complete and unlock next
  // -----------------------------------------------------------------------
  const handleMarkComplete = useCallback((stepNumber) => {
    setCompletedSteps((prev) => new Set(prev).add(stepNumber));
  }, []);

  // -----------------------------------------------------------------------
  // Null plan guard
  // -----------------------------------------------------------------------
  if (!remediationPlan) {
    return (
      <div style={S.panel}>
        <div style={S.header}>
          <Shield size={18} /> Remediation
        </div>
        <div style={S.empty}>
          <AlertTriangle size={24} style={{ marginBottom: 8, color: '#6b7280' }} />
          <div>No remediation plan available.</div>
          <div style={{ marginTop: 4, fontSize: 12 }}>Trigger remediation first.</div>
        </div>
      </div>
    );
  }

  // Derived data
  const autoActions = remediationPlan.tier1_auto_actions || [];
  const walkthrough = remediationPlan.tier2_walkthrough || { total_steps: 0, steps: [] };
  const advisory = remediationPlan.tier3_advisory || [];
  const walkthroughSteps = walkthrough.steps || [];
  const walkthroughProgress =
    walkthroughSteps.length > 0
      ? (completedSteps.size / walkthroughSteps.length) * 100
      : 0;

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------
  return (
    <div style={S.panel}>
      {/* Header */}
      <div style={S.header}>
        <Shield size={18} /> Remediation
      </div>

      {/* Confidence gate warning */}
      {gateBlocked && (
        <div style={S.gateBanner}>
          <AlertTriangle size={16} />
          Confidence gate failed ({((confidenceGate.confidence ?? 0) * 100).toFixed(1)}%
          &lt; {((confidenceGate.threshold ?? 0) * 100).toFixed(1)}% threshold) — auto-actions
          are disabled.
        </div>
      )}

      {/* Execution error */}
      {executeError && (
        <div
          style={{
            ...S.gateBanner,
            background: '#7f1d1d',
            borderColor: '#991b1b',
            color: '#fca5a5',
          }}
        >
          <AlertTriangle size={16} />
          {executeError}
        </div>
      )}

      {/* Tabs */}
      <div style={S.tabs}>
        {TABS.map((t) => (
          <div key={t.key} style={S.tab(activeTab === t.key)} onClick={() => setActiveTab(t.key)}>
            {t.label}
            {t.key === 'auto' && autoActions.length > 0 && (
              <span style={{ marginLeft: 6, opacity: 0.6 }}>({autoActions.length})</span>
            )}
            {t.key === 'walkthrough' && walkthroughSteps.length > 0 && (
              <span style={{ marginLeft: 6, opacity: 0.6 }}>
                {completedSteps.size}/{walkthroughSteps.length}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Tab content */}
      <div style={S.body}>
        {/* ---- Tier 1: Auto-Actions ---- */}
        {activeTab === 'auto' && (
          <>
            {autoActions.length === 0 ? (
              <div style={S.empty}>No auto-actions in this plan.</div>
            ) : (
              autoActions.map((action, idx) => (
                <AutoActionCard
                  key={idx}
                  action={action}
                  index={idx}
                  gateBlocked={gateBlocked}
                  onExecute={handleAutoExecute}
                />
              ))
            )}
          </>
        )}

        {/* ---- Tier 2: Walkthrough ---- */}
        {activeTab === 'walkthrough' && (
          <>
            {walkthroughSteps.length === 0 ? (
              <div style={S.empty}>No walkthrough steps available.</div>
            ) : (
              <>
                {/* Progress bar */}
                <div style={S.progressBar}>
                  <div style={S.progressFill(walkthroughProgress)} />
                </div>

                {walkthroughSteps.map((step) => {
                  const isCompleted = completedSteps.has(step.step);

                  // Active if it's the first incomplete step
                  const firstIncomplete = walkthroughSteps.find(
                    (s) => !completedSteps.has(s.step),
                  );
                  const isActive = firstIncomplete?.step === step.step;

                  return (
                    <StepCard
                      key={step.step}
                      step={step}
                      isActive={isActive}
                      isCompleted={isCompleted}
                      onMarkComplete={handleMarkComplete}
                    />
                  );
                })}
              </>
            )}
          </>
        )}

        {/* ---- Tier 3: Advisory ---- */}
        {activeTab === 'advisory' && (
          <>
            {advisory.length === 0 ? (
              <div style={S.empty}>No advisory recommendations.</div>
            ) : (
              advisory.map((item, idx) => <AdvisoryCard key={idx} item={item} />)
            )}
          </>
        )}
      </div>

      {/* Prevention Checklist (always visible below tabs) */}
      <PreventionChecklist incidentId={incidentId} />
    </div>
  );
}
