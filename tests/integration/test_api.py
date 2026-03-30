"""
Integration tests for the FastAPI application (src.api.server).

Tests all major endpoints: health, incidents, events webhooks,
report lookups, analyze, remediate, and audit.
"""

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from datetime import datetime

from src.api.server import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Async HTTP client wired to the FastAPI ASGI app."""
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest_asyncio.fixture
async def auth_client():
    """Async HTTP client with JWT authentication."""
    import os

    os.environ["RCA_WEBHOOK_VALIDATE"] = "false"
    import src.api.webhooks as webhooks

    webhooks._webhook_validator = None
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        login_resp = await c.post(
            "/auth/login", data={"username": "admin", "password": "admin123"}
        )
        token = login_resp.json()["access_token"]
        c.headers["Authorization"] = f"Bearer {token}"
        yield c


def _deploy_event_payload() -> dict:
    """Valid deploy event payload matching the DeployEvent Pydantic model."""
    return {
        "service": "test-api",
        "version": "v1.0.0",
        "deployer": "test",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "commit_sha": "abc123def456",
        "environment": "production",
        "rollback_available": False,
    }


def _config_event_payload() -> dict:
    """Valid config change event payload matching the ConfigChangeEvent Pydantic model."""
    return {
        "service": "redis",
        "config_key": "maxmemory",
        "old_value": "1gb",
        "new_value": "2gb",
        "changer": "test",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": "production",
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_endpoint(client):
    """GET /health returns 200."""
    async with client as c:
        resp = await c.get("/health")
        assert resp.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_response_structure(client):
    """GET /health response contains a 'status' key."""
    async with client as c:
        resp = await c.get("/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_incidents_endpoint(auth_client):
    """GET /incidents returns 200 with a list."""
    resp = await auth_client.get("/incidents")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


# ---------------------------------------------------------------------------
# Docs (Swagger UI)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_docs_endpoint(client):
    """GET /docs returns 200 (Swagger UI)."""
    async with client as c:
        resp = await c.get("/docs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Deploy webhook
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deploy_webhook(auth_client):
    """POST /events/deploy with a valid payload returns 202."""
    resp = await auth_client.post("/events/deploy", json=_deploy_event_payload())
    assert resp.status_code == 202


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deploy_webhook_response_has_id(auth_client):
    """POST /events/deploy response includes an event_id field."""
    resp = await auth_client.post("/events/deploy", json=_deploy_event_payload())
    data = resp.json()
    assert "event_id" in data or "id" in data
    if "event_id" in data:
        assert data["event_id"].startswith("EVT-")


# ---------------------------------------------------------------------------
# Config webhook
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_config_webhook(auth_client):
    """POST /events/config with a valid payload returns 202."""
    resp = await auth_client.post("/events/config", json=_config_event_payload())
    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Events list (post then list)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_events_list(auth_client):
    """POST a deploy event, then GET /events — the event should appear."""
    post_resp = await auth_client.post("/events/deploy", json=_deploy_event_payload())
    assert post_resp.status_code == 202
    event_id = post_resp.json().get("event_id")

    list_resp = await auth_client.get("/events")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert "events" in data
    assert data["total"] >= 1

    event_ids = [e.get("event_id") for e in data["events"]]
    assert event_id in event_ids


# ---------------------------------------------------------------------------
# Report for nonexistent incident
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_report_nonexistent_incident(auth_client):
    """GET /report/NONEXISTENT should return 404."""
    resp = await auth_client.get("/report/NONEXISTENT")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Audit for nonexistent incident
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_audit_nonexistent_incident(auth_client):
    """GET /audit/NONEXISTENT should return 404 (incident not found)."""
    resp = await auth_client.get("/audit/NONEXISTENT")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Analyze endpoint exists
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_endpoint_exists(auth_client):
    """POST /analyze with {} should return 202 or 422, but NOT 404."""
    resp = await auth_client.post("/analyze", json={})
    assert resp.status_code in (200, 202, 422)
    assert resp.status_code != 404


# ---------------------------------------------------------------------------
# Remediate endpoint structure
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remediate_endpoint_structure(auth_client):
    """POST /remediate/INC-TEST should return a meaningful response.

    Since INC-TEST doesn't exist as a completed incident, we expect either:
    - 404 (incident not found)
    - 409 (still processing)
    - 202 with a plan structure (if incident exists and is complete)

    This test verifies the endpoint is routed and responds appropriately.
    """
    resp = await auth_client.post("/remediate/INC-TEST")
    assert resp.status_code in (202, 400, 404, 409, 500)

    if resp.status_code == 202:
        data = resp.json()
        assert "incident_id" in data
        assert "root_cause" in data or "remediation_plan" in data
