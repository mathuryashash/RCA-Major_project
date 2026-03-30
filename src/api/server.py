"""
FastAPI server for the Automated RCA System.

Implements all PRD-specified endpoints (FR-30, Section 11):
  POST /analyze           — Trigger RCA analysis (202 Accepted)
  GET  /report/{id}       — Full RCA report
  GET  /logs/{id}         — Annotated log lines sorted by anomaly contribution
  GET  /metrics/{service} — 60-min anomaly score time series for a service
  GET  /graph/{id}        — Causal DAG as D3.js-compatible node-link JSON
  POST /remediate/{id}    — Trigger remediation engine
  POST /remediate/{id}/execute    — Execute pending Tier 1 actions
  GET  /remediate/{id}/walkthrough — Tier 2 step-by-step walkthrough
  GET  /remediate/{id}/prevention  — Prevention checklist (3 horizons)
  GET  /audit/{id}        — Immutable remediation audit log
  GET  /incidents         — List all incidents
  GET  /health            — System health + diagnostics
  POST /events/deploy     — Ingest deployment events (FR-07)
  POST /events/config     — Ingest config change events (FR-07)
  GET  /events            — List all stored events (debug)
  GET  /events/{id}       — Events correlated to an incident (+/-30 min)
"""

import os
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse

from src.pipeline import PipelineOrchestrator
from src.api.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_active_user,
    get_user,
    verify_token,
    Token,
    User,
)
from src.api.exceptions import (
    RCAException,
    IncidentNotFoundError,
    PipelineError,
    ModelLoadError,
)
from src.api.errors import ErrorResponse
from src.api.webhooks import get_webhook_validator

PRODUCTION = os.environ.get("PRODUCTION", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Automated RCA System API",
    description="AI-powered Root Cause Analysis pipeline",
    version="0.3.0",
)

# CORS — allow the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RCAException)
async def rca_exception_handler(request: Request, exc: RCAException):
    logger.warning(f"RCA error: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=exc.message if not PRODUCTION else "An error occurred",
            code=exc.code,
            details={"incident_id": exc.incident_id}
            if hasattr(exc, "incident_id")
            else None,
            timestamp=datetime.now().isoformat(),
        ).model_dump(),
    )


@app.exception_handler(IncidentNotFoundError)
async def incident_not_found_handler(request: Request, exc: IncidentNotFoundError):
    logger.warning(f"Incident not found: {exc.incident_id}")
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error=exc.message,
            code=exc.code,
            details={"incident_id": exc.incident_id},
            timestamp=datetime.now().isoformat(),
        ).model_dump(),
    )


@app.exception_handler(PipelineError)
async def pipeline_error_handler(request: Request, exc: PipelineError):
    logger.error(f"Pipeline error in stage '{exc.stage}': {exc.message}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=exc.message if not PRODUCTION else "Pipeline execution failed",
            code=exc.code,
            details={"stage": exc.stage} if not PRODUCTION else None,
            timestamp=datetime.now().isoformat(),
        ).model_dump(),
    )


@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request: Request, exc: ModelLoadError):
    logger.error(f"Model load error: {exc.model_name}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=exc.message if not PRODUCTION else "Failed to load analysis model",
            code=exc.code,
            details={"model_name": exc.model_name} if not PRODUCTION else None,
            timestamp=datetime.now().isoformat(),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error" if PRODUCTION else str(exc),
            code="INTERNAL_ERROR",
            timestamp=datetime.now().isoformat(),
        ).model_dump(),
    )


# Pipeline singleton — expensive to init (loads Drain3, TF-IDF, etc.)
pipeline = PipelineOrchestrator()

# In-memory incident store  {incident_id: report_dict}
incidents: Dict[str, Dict[str, Any]] = {}

# In-memory event stores for deploy and config change events (FR-07)
deploy_events: List[Dict[str, Any]] = []
config_events: List[Dict[str, Any]] = []

# In-memory remediation plans  {incident_id: full_plan_dict}
remediation_plans: Dict[str, Dict[str, Any]] = {}

# Thread pool for CPU-bound pipeline work (Granger, TF-IDF, PageRank)
_executor = ThreadPoolExecutor(max_workers=2)

# Track parse failures (FR-03)
_parse_failures = 0

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IncidentStatus(str, Enum):
    processing = "processing"
    complete = "complete"
    failed = "failed"


class RCAInquiry(BaseModel):
    incident_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    services: Optional[List[str]] = []
    priority: Optional[str] = Field(
        default="medium",
        description="Incident priority: low, medium, high, critical",
    )


class AnalyzeResponse(BaseModel):
    incident_id: str
    status: str
    estimated_completion_seconds: int = 180
    poll_url: str


class ExecuteRequest(BaseModel):
    """Request body for Tier 1 action execution."""

    actions: Optional[List[str]] = Field(
        default=None,
        description="List of action names to execute. None = all pending Tier 1 actions.",
    )
    executor: str = Field(
        default="operator",
        description="Identity of the person/system executing the actions.",
    )


class DeployEvent(BaseModel):
    """Deployment event from CI/CD systems (FR-07)."""

    service: str
    version: str
    deployer: str
    timestamp: str
    commit_sha: str
    environment: str = "production"
    rollback_available: bool = False


class ConfigChangeEvent(BaseModel):
    """Configuration change event (FR-07)."""

    service: str
    config_key: str
    old_value: str
    new_value: str
    changer: str
    timestamp: str
    environment: str = "production"


# ---------------------------------------------------------------------------
# Background pipeline execution
# ---------------------------------------------------------------------------


def _run_pipeline(incident_id: str, inquiry: RCAInquiry) -> None:
    """Run the full RCA pipeline synchronously (called from a thread)."""
    try:
        logger.info("Pipeline started for %s", incident_id)
        report = pipeline.run(
            incident_id=incident_id,
            start_time=inquiry.start_time,
            end_time=inquiry.end_time,
            services=inquiry.services if inquiry.services else None,
        )
        incidents[incident_id] = report
        logger.info("Pipeline completed for %s", incident_id)
    except PipelineError:
        raise
    except Exception as exc:
        logger.exception("Pipeline failed for %s", incident_id)
        incidents[incident_id] = {
            "incident_id": incident_id,
            "status": "failed",
            "error": str(exc),
            "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        raise PipelineError(str(exc), stage="full_analysis") from exc


def _require_incident(incident_id: str) -> Dict[str, Any]:
    """Helper: return incident data or raise 404."""
    if incident_id not in incidents:
        raise IncidentNotFoundError(incident_id)
    return incidents[incident_id]


def _require_complete_incident(incident_id: str) -> Dict[str, Any]:
    """Helper: return completed incident or raise 404/409."""
    data = _require_incident(incident_id)
    if data.get("status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="Analysis still in progress. Poll /report/{incident_id} for status.",
        )
    if data.get("status") == "failed":
        error_msg = data.get("error", "unknown error")
        raise PipelineError(error_msg, stage="full_analysis")
    return data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {"message": "Automated RCA System API is live", "version": "0.3.0"}


# ---- Health ----


@app.get("/health")
def health():
    """
    System health: models loaded, log files accessible, data freshness.
    Includes parse_failures counter (FR-03).
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.3.0",
    }


@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login. Returns access and refresh tokens.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes}
    )
    refresh_token = create_refresh_token(data={"sub": user.username})
    return Token(access_token=access_token, refresh_token=refresh_token)


@app.post("/auth/refresh", response_model=Token)
async def refresh(refresh_token: str):
    """
    Refresh access token using a valid refresh token.
    """
    payload = verify_token(refresh_token, token_type="refresh")
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    user = get_user(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes}
    )
    new_refresh_token = create_refresh_token(data={"sub": user.username})
    return Token(access_token=access_token, refresh_token=new_refresh_token)


@app.get("/health/detailed")
def health_detailed(current_user: User = Depends(get_current_active_user)):
    """
    Detailed system health: models loaded, log files accessible, data freshness.
    Includes parse_failures counter (FR-03). Requires authentication.
    """
    # Check log file accessibility
    log_sources = pipeline.config.get("log_sources", [])
    log_status = {}
    for src in log_sources:
        path = os.path.abspath(src["path"])
        log_status[src["label"]] = {
            "path": path,
            "accessible": os.path.exists(path),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        }

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.3.0",
        "models_loaded": {
            "drain3_parser": pipeline.log_parser is not None,
            "tfidf_anomaly_detector": pipeline.log_anomaly_detector is not None,
            "granger_causal": pipeline.causal_engine is not None,
            "nlg_generator": pipeline.nlg_generator is not None,
            "remediation_engine": pipeline.remediation_engine is not None,
            "unified_scorer": pipeline.unified_scorer is not None,
            "logbert": getattr(pipeline, "logbert", None) is not None,
            "lstm_ae": getattr(pipeline, "lstm_ae", None) is not None,
            "temporal_transformer": getattr(pipeline, "temporal_transformer", None)
            is not None,
            "khbn": getattr(pipeline, "khbn", None) is not None,
        },
        "log_sources": log_status,
        "active_incidents": len(
            [v for v in incidents.values() if v.get("status") == "processing"]
        ),
        "total_incidents": len(incidents),
        "parse_failures": _parse_failures,
    }


# ---- Events (FR-07) ----


@app.post("/events/deploy", status_code=202)
async def ingest_deploy_event(
    event: DeployEvent,
    request: Request,
    current_user: User = Depends(get_current_active_user),
):
    """Record a deployment event from CI/CD. Correlated with anomalies within +/-30 min."""
    webhook_validator = get_webhook_validator()
    body = await request.body()
    webhook_validator.require_valid_webhook(request, body)

    event_id = f"EVT-{uuid4().hex[:12]}"
    record = {
        "event_id": event_id,
        "type": "deploy",
        **event.model_dump(),
    }
    deploy_events.append(record)
    logger.info(
        "Stored deploy event %s for %s %s", event_id, event.service, event.version
    )
    return {
        "event_id": event_id,
        "type": "deploy",
        "stored": True,
        "message": "Deployment event recorded. Will be correlated with anomalies within +/-30 minutes.",
    }


@app.post("/events/config", status_code=202)
async def ingest_config_event(
    event: ConfigChangeEvent,
    request: Request,
    current_user: User = Depends(get_current_active_user),
):
    """Record a configuration change event. Correlated with anomalies within +/-30 min."""
    webhook_validator = get_webhook_validator()
    body = await request.body()
    webhook_validator.require_valid_webhook(request, body)

    event_id = f"EVT-{uuid4().hex[:12]}"
    record = {
        "event_id": event_id,
        "type": "config_change",
        **event.model_dump(),
    }
    config_events.append(record)
    logger.info(
        "Stored config event %s for %s.%s", event_id, event.service, event.config_key
    )
    return {
        "event_id": event_id,
        "type": "config_change",
        "stored": True,
        "message": "Configuration change event recorded. Will be correlated with anomalies within +/-30 minutes.",
    }


@app.get("/events")
def list_events(current_user: User = Depends(get_current_active_user)):
    """Return all stored deploy and config events (debug/inspection)."""
    all_events = sorted(
        [*deploy_events, *config_events],
        key=lambda e: e.get("timestamp", ""),
        reverse=True,
    )
    return {
        "total": len(all_events),
        "deploy_count": len(deploy_events),
        "config_count": len(config_events),
        "events": all_events,
    }


@app.get("/events/{incident_id}")
def get_events_for_incident(
    incident_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Return deploy/config events within +/-30 minutes of the incident's detected_at.

    Used by the pipeline to compute the deployment_event_bonus (FR-26).
    """
    data = _require_incident(incident_id)
    detected_at_str = data.get("detected_at")
    if not detected_at_str:
        raise HTTPException(
            status_code=400,
            detail="Incident has no detected_at timestamp yet.",
        )

    # Parse the incident timestamp (supports ISO and simple datetime formats)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            detected_at = datetime.strptime(detected_at_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot parse incident detected_at: {detected_at_str}",
        )

    window = timedelta(minutes=30)
    start = detected_at - window
    end = detected_at + window

    def _in_window(evt: Dict[str, Any]) -> bool:
        ts_str = evt.get("timestamp", "")
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
            try:
                ts = datetime.strptime(ts_str, fmt)
                return start <= ts <= end
            except ValueError:
                continue
        return False

    matched = [e for e in [*deploy_events, *config_events] if _in_window(e)]

    return {
        "incident_id": incident_id,
        "detected_at": detected_at_str,
        "window_minutes": 30,
        "total": len(matched),
        "events": matched,
    }


# ---- Analyze ----


@app.post("/analyze", status_code=202, response_model=AnalyzeResponse)
async def analyze(
    inquiry: RCAInquiry,
    current_user: User = Depends(get_current_active_user),
):
    """
    Kick off an RCA analysis.

    Returns 202 immediately with an incident ID and poll URL.
    The real pipeline runs in a background thread.
    """
    incident_id = inquiry.incident_id or f"INC-{int(time.time())}"

    if incident_id in incidents:
        existing = incidents[incident_id]
        return AnalyzeResponse(
            incident_id=incident_id,
            status=existing.get("status", "processing"),
            estimated_completion_seconds=0
            if existing.get("status") == "complete"
            else 60,
            poll_url=f"/report/{incident_id}",
        )

    # Mark as processing immediately so duplicate requests are idempotent
    incidents[incident_id] = {
        "incident_id": incident_id,
        "status": "processing",
        "priority": inquiry.priority,
    }

    # Dispatch to thread pool (pipeline is CPU-bound)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, incident_id, inquiry)

    return AnalyzeResponse(
        incident_id=incident_id,
        status="processing",
        estimated_completion_seconds=180,
        poll_url=f"/report/{incident_id}",
    )


# ---- Report ----


@app.get("/report/{incident_id}")
def get_report(
    incident_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve the full analysis report for a given incident."""
    return _require_incident(incident_id)


# ---- Incidents listing ----


@app.get("/incidents")
def list_incidents(current_user: User = Depends(get_current_active_user)):
    """List all incidents with their status."""
    return [
        {
            "incident_id": iid,
            "status": data.get("status", "unknown"),
            "detected_at": data.get("detected_at"),
            "root_cause": data.get("root_cause"),
            "priority": data.get("priority", "medium"),
        }
        for iid, data in sorted(incidents.items(), reverse=True)
    ]


# ---- Logs (FR-20, FR-31) ----


@app.get("/logs/{incident_id}")
def get_logs(
    incident_id: str,
    source: Optional[str] = None,
    min_score: float = 0.0,
    current_user: User = Depends(get_current_active_user),
):
    """
    Return annotated log lines around the incident window,
    sorted by anomaly contribution (descending).

    Query params:
      source    — filter to a specific log source (e.g. 'application')
      min_score — only return logs with anomaly_score >= this value
    """
    data = _require_complete_incident(incident_id)
    logs = data.get("annotated_logs", [])

    # Apply filters
    if source:
        logs = [l for l in logs if l.get("source") == source]
    if min_score > 0:
        logs = [l for l in logs if l.get("anomaly_score", 0) >= min_score]

    return {
        "incident_id": incident_id,
        "total": len(logs),
        "logs": logs,
    }


# ---- Metrics (FR-32) ----


@app.get("/metrics/{service}")
def get_metrics(
    service: str,
    incident_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    """
    Last 60-minute anomaly score time series for a service.

    If incident_id is provided, returns the per-record anomaly scores
    from that incident's analysis. Otherwise returns a summary across
    all incidents where the service appears.
    """
    if incident_id:
        data = _require_complete_incident(incident_id)
        logs = data.get("annotated_logs", [])
        series = [
            {
                "timestamp": l.get("timestamp"),
                "anomaly_score": l.get("anomaly_score", 0.0),
                "message": l.get("message", "")[:100],
            }
            for l in logs
            if l.get("source") == service
        ]
        # Sort by timestamp
        series.sort(key=lambda x: x["timestamp"] or "")

        # Get aggregate score from anomalous_metrics
        agg_score = 0.0
        for m in data.get("anomalous_metrics", []):
            if m.get("name") == service:
                agg_score = m.get("score", 0.0)
                break

        # Include metric anomaly scores if available in report
        metric_anomaly_score = None
        metric_anomaly_data = data.get("metric_anomaly_scores", {})
        if service in metric_anomaly_data:
            metric_anomaly_score = metric_anomaly_data[service]

        result = {
            "service": service,
            "incident_id": incident_id,
            "aggregate_anomaly_score": agg_score,
            "time_series": series,
        }
        if metric_anomaly_score is not None:
            result["metric_anomaly_score"] = metric_anomaly_score

        return result

    # No incident_id: aggregate across all completed incidents
    summary = []
    for iid, idata in incidents.items():
        if idata.get("status") != "complete":
            continue
        for m in idata.get("anomalous_metrics", []):
            if m.get("name") == service:
                summary.append(
                    {
                        "incident_id": iid,
                        "detected_at": idata.get("detected_at"),
                        "anomaly_score": m.get("score", 0.0),
                    }
                )
    return {
        "service": service,
        "incidents": summary,
    }


# ---- Causal Graph (FR-33) ----


@app.get("/graph/{incident_id}")
def get_graph(
    incident_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Return the causal DAG as D3.js-compatible node-link JSON.

    Nodes have: id, anomaly_score, is_root_cause
    Edges have: source, target, weight (confidence), p_value (lag)
    """
    data = _require_complete_incident(incident_id)
    graph = data.get("causal_graph", {"nodes": [], "edges": []})
    root_cause = data.get("root_cause")
    ranked = data.get("ranked_causes", [])

    # Build a confidence lookup from ranked causes
    confidence_map = {rc["cause"]: rc["confidence"] for rc in ranked}

    # Enhance nodes for D3.js
    nodes = []
    for n in graph.get("nodes", []):
        node_id = n["id"]
        nodes.append(
            {
                "id": node_id,
                "anomaly_score": n.get("anomaly_score", 0.0),
                "is_root_cause": node_id == root_cause,
                "confidence": confidence_map.get(node_id, 0.0),
                # D3 shape hint: root cause = square, others = circle
                "shape": "square" if node_id == root_cause else "circle",
            }
        )

    # Rename edge keys for D3.js node-link format
    edges = []
    for e in graph.get("edges", []):
        edges.append(
            {
                "source": e.get("from"),
                "target": e.get("to"),
                "weight": e.get("confidence", 0.0),
                "p_value": e.get("lag", 0.0),
                "label": f"p={e.get('lag', 0.0):.3f}",
            }
        )

    return {
        "incident_id": incident_id,
        "root_cause": root_cause,
        "nodes": nodes,
        "links": edges,
    }


# ---- Remediate (FR-25, FR-26) ----


@app.post("/remediate/{incident_id}", status_code=202)
def trigger_remediation(
    incident_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Trigger the Remediation Engine for an incident.
    Returns the safety-classified action plan.
    """
    data = _require_complete_incident(incident_id)
    root_cause = data.get("root_cause")

    if not root_cause:
        raise HTTPException(status_code=400, detail="No root cause identified")

    # Check if we already have a plan from the pipeline
    existing_plan = data.get("remediation_plan")
    if existing_plan and existing_plan.get("root_cause"):
        # Store for execution tracking
        remediation_plans[incident_id] = {
            "plan": existing_plan,
            "status": "pending",
            "executed_actions": [],
            "created_at": datetime.now().isoformat(),
        }
        return {
            "incident_id": incident_id,
            "root_cause": existing_plan.get("root_cause"),
            "confidence": existing_plan.get("confidence_gate", {}).get(
                "confidence", 0.0
            ),
            "confidence_gate": existing_plan.get("confidence_gate"),
            "remediation_plan": {
                "tier1_auto_actions": existing_plan.get("tier1_auto_actions", []),
                "tier2_walkthrough": existing_plan.get("tier2_walkthrough", {}),
                "tier3_advisory": existing_plan.get("tier3_advisory", []),
            },
        }

    # Generate fresh plan if pipeline didn't produce one
    ranked = data.get("ranked_causes", [])
    confidence = ranked[0]["confidence"] if ranked else 0.0
    plan = pipeline.remediation_engine.get_remediation_plan(
        root_cause,
        confidence=confidence,
        context={"service": root_cause, "incident_id": incident_id},
    )
    remediation_plans[incident_id] = {
        "plan": plan,
        "status": "pending",
        "executed_actions": [],
        "created_at": datetime.now().isoformat(),
    }
    return {
        "incident_id": incident_id,
        "root_cause": plan.get("root_cause"),
        "confidence": plan.get("confidence_gate", {}).get("confidence", 0.0),
        "confidence_gate": plan.get("confidence_gate"),
        "remediation_plan": {
            "tier1_auto_actions": plan.get("tier1_auto_actions", []),
            "tier2_walkthrough": plan.get("tier2_walkthrough", {}),
            "tier3_advisory": plan.get("tier3_advisory", []),
        },
    }


# ---- Execute Tier 1 (FR-25, FR-31) ----


@app.post("/remediate/{incident_id}/execute")
def execute_tier1(
    incident_id: str,
    request: ExecuteRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    Confirm and execute pending Tier 1 auto-fix actions.

    In production, this would actually run the commands.
    For now, it marks them as executed and logs to the audit trail.
    """
    data = _require_complete_incident(incident_id)
    plan_data = data.get("remediation_plan", {})

    # Check confidence gate
    gate = plan_data.get("confidence_gate", {})
    if gate and not gate.get("passed", False):
        raise HTTPException(
            status_code=403,
            detail=f"Confidence gate not passed ({gate.get('confidence', 0):.2f} < "
            f"{gate.get('threshold', 0.70):.2f}). Manual override required.",
        )

    tier1_actions = plan_data.get("tier1_auto_actions", [])
    if not tier1_actions:
        raise HTTPException(status_code=400, detail="No Tier 1 actions available")

    # Filter to requested actions if specified
    if request.actions:
        tier1_actions = [a for a in tier1_actions if a.get("action") in request.actions]

    executed = []
    for action in tier1_actions:
        # Log to audit trail (FR-30)
        pipeline.remediation_engine.log_action(
            incident_id=incident_id,
            action_type="tier1_executed",
            command=action.get("command", ""),
            executor=request.executor,
            before_state="degraded",
            after_state="remediation_applied",
            outcome="simulated",  # In production: "success" or "failed"
        )
        executed.append(
            {
                "action": action.get("action"),
                "command": action.get("command"),
                "status": "simulated",
                "message": f"Action '{action.get('action')}' would be executed in production.",
            }
        )

    # Update tracking
    if incident_id in remediation_plans:
        remediation_plans[incident_id]["executed_actions"].extend(executed)
        remediation_plans[incident_id]["status"] = "executed"

    return {
        "incident_id": incident_id,
        "executor": request.executor,
        "executed_actions": executed,
        "total_executed": len(executed),
        "audit_url": f"/audit/{incident_id}",
    }


# ---- Tier 2 Walkthrough (FR-28) ----


@app.get("/remediate/{incident_id}/walkthrough")
def get_walkthrough(
    incident_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Full step-by-step Tier 2 walkthrough with commands.
    Each step includes: step number, title, command, expected output,
    verification, rollback, estimated time, safety note.
    """
    data = _require_complete_incident(incident_id)
    plan = data.get("remediation_plan", {})
    walkthrough = plan.get("tier2_walkthrough", {})

    return {
        "incident_id": incident_id,
        "root_cause": plan.get("root_cause", data.get("root_cause")),
        "confidence_gate": plan.get("confidence_gate", {}),
        "total_steps": walkthrough.get("total_steps", 0),
        "estimated_total_minutes": walkthrough.get("estimated_total_minutes", 0),
        "steps": walkthrough.get("steps", []),
    }


# ---- Prevention Checklist (FR-29) ----


@app.get("/remediate/{incident_id}/prevention")
def get_prevention(
    incident_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Long-term prevention checklist by horizon:
    immediate, short_term, long_term.
    """
    data = _require_complete_incident(incident_id)
    plan = data.get("remediation_plan", {})
    checklist = plan.get("prevention_checklist", {})

    total_items = sum(len(v) for v in checklist.values() if isinstance(v, list))

    return {
        "incident_id": incident_id,
        "root_cause": plan.get("root_cause", data.get("root_cause")),
        "total_items": total_items,
        "prevention_checklist": checklist,
    }


# ---- Audit Log (FR-30) ----


@app.get("/audit/{incident_id}")
def get_audit(
    incident_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Immutable audit log of all remediation actions for an incident.
    Entries include: incident_id, action_type, command_executed,
    executor, before_state, after_state, timestamp, outcome.
    """
    # Don't require the incident to be complete — audit entries can exist
    # even during processing (e.g. plan generation audit)
    if incident_id not in incidents:
        raise HTTPException(status_code=404, detail="Incident not found")

    entries = pipeline.remediation_engine.get_audit_log(incident_id)

    return {
        "incident_id": incident_id,
        "total_entries": len(entries),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
