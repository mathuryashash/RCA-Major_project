# Design Document – 50‑Anomaly Extension for RCA System

**Date:** 2026‑03‑10

## Overview
This design adds a full catalog of **50 anomaly types** to the Streamlit sidebar and introduces two autonomous sub‑agents:

1. **Tester Agent** – validates each anomaly using synthetic data and asserts detection scores.
2. **Reviewer Agent** – runs security‑focused static analysis (Bandit, Safety, regex scans) on the whole code base.

All changes keep the existing visual theme (dark‑blue header, teal accents, Plotly styling) and respect the current project layout.

---

## 1. Data‑Driven Anomaly Catalog
- **File:** `src/reporting/anomalies.json`
- **Format:** JSON array, each entry with `name`, `metric`, `shape`, `severity`.
```json
{ "name": "01. CPU Sudden Spike", "metric": "cpu_utilization", "shape": "spike", "severity": 0.8 }
```
- UI reads this file at start‑up to populate the sidebar dropdown, eliminating hard‑coded Python lists.

---

## 2. UI Adjustments (Dashboard)
Replace the static selectbox in `src/reporting/dashboard.py` with:
```python
with open(os.path.join(_src, "reporting", "anomalies.json")) as f:
    ANOMALIES = json.load(f)
selected_name = st.sidebar.selectbox("Select Anomaly", [a["name"] for a in ANOMALIES])
anomaly_spec = next(a for a in ANOMALIES if a["name"] == selected_name)
```
All existing Plotly charts, metric‑chip CSS, and colour scheme remain unchanged.

---

## 3. Tester Agent
- **Location:** `tests/test_anomalies.py`
- **Framework:** `pytest` (optionally with `pytest‑xdist` for parallel runs).
- **Workflow per anomaly:**
  1. Generate normal data via `SyntheticMetricsGenerator`.
  2. Inject the failure using `inject_failure_scenario` with parameters from the JSON entry.
  3. Run `AnomalyDetector.detect`.
  4. Assert the target metric’s anomaly score ≥ 0.85 (configurable).
- **Report:** Pytest’s summary; a helper can convert XML to `tests/report_anomalies.md`.

---

## 4. Reviewer Agent
- **Location:** `tools/reviewer.py`
- **Steps:**
  1. Run **Bandit** on `src/` → `reviews/bandit_report.json`.
  2. Run **Safety** on `requirements.txt` → `reviews/safety_report.json`.
  3. Simple regex scan for `eval(`, `exec(`, unsafe `subprocess.Popen`.
  4. Combine findings into `reviews/security_review.md` with sections for critical, medium, and info.
- **Execution:** `python -m tools.reviewer` manually or as a CI job.

---

## 5. CI / Deployment Flow
1. Commit `anomalies.json` and UI changes.
2. CI runs:
   - `pytest -q tests/test_anomalies.py` → guarantees all 50 anomalies are detectable.
   - `python -m tools.reviewer` → produces the security review.
3. Docker image unchanged – the container reads the new JSON at start‑up.

---

## 6. Risks \u0026 Mitigations
| Risk | Mitigation |
|------|------------|
| Large JSON slows UI | File is only a few KB; loaded once per session. |
| Tester false‑positives | Threshold configurable per anomaly; adjust as needed. |
| Reviewer missing custom patterns | Extend regex list in `reviewer.py`; report un‑scanned patterns. |
| New dev dependencies bloat | `bandit` and `safety` listed under `[dev]` in `requirements.txt`. |

---

## 7. Next Steps
1. Populate `src/reporting/anomalies.json` with the 50 entries.
2. Update `dashboard.py` to load the JSON.
3. Implement `tests/test_anomalies.py` and `tools/reviewer.py`.
4. Add CI steps for testing and security review.
5. Run the **writing‑plans** skill to generate a detailed implementation plan.

---

*The design preserves the existing UI theme and adds automated validation and security review pipelines, ready for integration into the final project.*
