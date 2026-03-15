import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';

import Header from './components/Header';
import IncidentsList from './components/IncidentsList';
import CausalGraph from './components/CausalGraph';
import AnomalyTimeline from './components/AnomalyTimeline';
import AnomalyHeatmap from './components/AnomalyHeatmap';
import RCAReportPanel from './components/RCAReportPanel';
import RemediationPanel from './components/RemediationPanel';
import AuditLog from './components/AuditLog';

const TABS = [
  { key: 'timeline', label: 'Timeline' },
  { key: 'heatmap', label: 'Heatmap' },
  { key: 'graph', label: 'Causal Graph' },
  { key: 'report', label: 'RCA Report' },
  { key: 'remediation', label: 'Remediation' },
  { key: 'audit', label: 'Audit Log' },
];

const POLL_INTERVAL = 2000; // ms
const MAX_POLLS = 60; // 2 minutes max

const Dashboard = () => {
  // --- State ---
  const [incidents, setIncidents] = useState([]);
  const [selectedIncidentId, setSelectedIncidentId] = useState(null);
  const [report, setReport] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [remediationPlan, setRemediationPlan] = useState(null);
  const [confidenceGate, setConfidenceGate] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('timeline');
  const [error, setError] = useState(null);

  const pollRef = useRef(null);

  // --- Load incidents on mount ---
  useEffect(() => {
    fetchIncidents();
  }, []);

  const fetchIncidents = async () => {
    try {
      const res = await axios.get('/api/incidents');
      setIncidents(res.data);
    } catch (err) {
      console.warn('Could not fetch incidents:', err.message);
    }
  };

  // --- Trigger new analysis ---
  const handleAnalyze = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post('/api/analyze', {});
      const incidentId = res.data.incident_id;

      // Poll for completion
      let polls = 0;
      pollRef.current = setInterval(async () => {
        polls++;
        try {
          const reportRes = await axios.get(`/api/report/${incidentId}`);
          if (reportRes.data.status === 'complete' || polls >= MAX_POLLS) {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setSelectedIncidentId(incidentId);
            await loadIncidentData(incidentId, reportRes.data);
            setLoading(false);
            fetchIncidents(); // refresh list
          }
        } catch (pollErr) {
          if (polls >= MAX_POLLS) {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setLoading(false);
            setError('Analysis timed out. Try again.');
          }
        }
      }, POLL_INTERVAL);
    } catch (err) {
      setLoading(false);
      setError(`Analysis failed: ${err.response?.data?.detail || err.message}`);
    }
  }, []);

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // --- Select an existing incident ---
  const handleSelectIncident = useCallback(async (incidentId) => {
    setSelectedIncidentId(incidentId);
    setLoading(true);
    setError(null);
    try {
      const reportRes = await axios.get(`/api/report/${incidentId}`);
      await loadIncidentData(incidentId, reportRes.data);
    } catch (err) {
      setError(`Failed to load incident: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  // --- Load all data for an incident ---
  const loadIncidentData = async (incidentId, reportData) => {
    setReport(reportData);

    // Parallel fetch: graph, logs, remediation
    const promises = [
      axios.get(`/api/graph/${incidentId}`).catch(() => null),
      axios.get(`/api/logs/${incidentId}`).catch(() => null),
      axios.post(`/api/remediate/${incidentId}`).catch(() => null),
    ];

    // Fetch metrics for root cause service
    if (reportData.root_cause) {
      promises.push(
        axios.get(`/api/metrics/${reportData.root_cause}`, {
          params: { incident_id: incidentId }
        }).catch(() => null)
      );
    }

    const [graphRes, logsRes, remRes, metricsRes] = await Promise.all(promises);

    if (graphRes?.data) setGraphData(graphRes.data);
    else setGraphData(null);

    if (logsRes?.data?.logs) setLogs(logsRes.data.logs);
    else setLogs([]);

    if (remRes?.data) {
      setRemediationPlan(remRes.data.remediation_plan || null);
      setConfidenceGate(remRes.data.confidence_gate || null);
    } else {
      setRemediationPlan(null);
      setConfidenceGate(null);
    }

    if (metricsRes?.data) setMetrics(metricsRes.data);
    else setMetrics(null);
  };

  // --- Highlight cause in graph (from report panel) ---
  const handleSelectCause = useCallback((cause) => {
    setActiveTab('graph');
    // Graph component could highlight this node
  }, []);

  // --- Render active tab content ---
  const renderTabContent = () => {
    if (!report && !loading) {
      return (
        <div className="empty-state">
          <p>Select an incident or trigger a new analysis to begin.</p>
        </div>
      );
    }

    switch (activeTab) {
      case 'timeline':
        return (
          <AnomalyTimeline
            logs={logs}
            metrics={metrics}
          />
        );

      case 'heatmap':
        return (
          <AnomalyHeatmap
            anomalousMetrics={report?.anomalous_metrics || []}
            logs={logs}
          />
        );

      case 'graph':
        return (
          <CausalGraph
            graphData={graphData}
            onNodeClick={(nodeId) => console.log('Node clicked:', nodeId)}
          />
        );

      case 'report':
        return (
          <RCAReportPanel
            report={report}
            onSelectCause={handleSelectCause}
          />
        );

      case 'remediation':
        return (
          <RemediationPanel
            incidentId={selectedIncidentId}
            remediationPlan={remediationPlan}
            confidenceGate={confidenceGate}
          />
        );

      case 'audit':
        return <AuditLog incidentId={selectedIncidentId} />;

      default:
        return null;
    }
  };

  return (
    <div className="app-layout">
      {/* Header */}
      <Header onAnalyze={handleAnalyze} loading={loading} />

      {/* Error banner */}
      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError(null)} className="dismiss-btn">&times;</button>
        </div>
      )}

      <div className="main-content">
        {/* Sidebar: Incidents list */}
        <aside className="sidebar">
          <IncidentsList
            incidents={incidents}
            selectedId={selectedIncidentId}
            onSelect={handleSelectIncident}
          />
        </aside>

        {/* Main panel */}
        <main className="main-panel">
          {/* Tab bar */}
          <nav className="tab-bar">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                className={`tab-btn ${activeTab === tab.key ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.key)}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Incident header */}
          {report && (
            <div className="incident-header">
              <h2>{report.incident_id}</h2>
              <span className="detected-at">Detected: {report.detected_at}</span>
              {report.root_cause && (
                <span className="root-cause-badge">
                  Root Cause: {report.root_cause.replace(/_/g, ' ')}
                </span>
              )}
            </div>
          )}

          {/* Tab content */}
          <div className="tab-content">
            {loading && !report ? (
              <div className="loading-state">
                <div className="spinner" />
                <p>Running pipeline analysis...</p>
              </div>
            ) : (
              renderTabContent()
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
