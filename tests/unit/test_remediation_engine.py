"""
Unit tests for src.remediation.remediation_engine.RemediationEngine

Tests safety classification (FR-25), confidence gate (FR-31),
remediation plan generation (FR-26/FR-28), audit logging (FR-30),
and the simplified plan interface.
"""

import pytest

from src.remediation.remediation_engine import RemediationEngine

# ---------------------------------------------------------------------------
# Expected rule keys (all 10 scenarios from remediation_rules.yaml)
# ---------------------------------------------------------------------------

ALL_RULE_KEYS = [
    "db_migration_applied",
    "memory_leak_detected",
    "network_partition_detected",
    "thread_pool_exhaustion",
    "dns_ttl_misconfiguration",
    "cpu_runaway_process",
    "connection_leak_bug",
    "cache_ttl_expiry",
    "log_rotation_disabled",
    "consumer_service_crash",
]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Instantiate a fresh RemediationEngine with default config paths."""
    return RemediationEngine()


# ===================================================================
# Rule / Config Loading
# ===================================================================


@pytest.mark.unit
def test_engine_loads_10_rules(engine):
    """Verify the engine loads exactly 10 remediation rule sets."""
    assert len(engine.rules) == 10


@pytest.mark.unit
def test_engine_loads_safety_config(engine):
    """Verify safety_config is loaded and non-empty."""
    assert engine.safety_config is not None
    assert len(engine.safety_config) > 0


# ===================================================================
# Safety Classification (FR-25)
# ===================================================================


@pytest.mark.unit
def test_classify_tier1_actions(engine):
    """Tier 1 patterns: flush_*_cache, restart_*, force_gc."""
    assert engine.classify_safety_tier("flush_query_cache") == "tier1"
    assert engine.classify_safety_tier("restart_api") == "tier1"
    assert engine.classify_safety_tier("force_gc") == "tier1"


@pytest.mark.unit
def test_classify_tier2_actions(engine):
    """Tier 2 patterns: rollback_deployment, rollback_migration, halt_trading."""
    assert engine.classify_safety_tier("rollback_deployment") == "tier2"
    assert engine.classify_safety_tier("rollback_migration") == "tier2"
    assert engine.classify_safety_tier("halt_trading") == "tier2"


@pytest.mark.unit
def test_classify_tier3_actions(engine):
    """Tier 3 patterns: code_fix_*, redesign_*."""
    assert engine.classify_safety_tier("code_fix_websocket") == "tier3"
    assert engine.classify_safety_tier("redesign_architecture") == "tier3"


@pytest.mark.unit
def test_classify_unknown_action(engine):
    """Unknown action names should default to tier3 (safest default)."""
    assert engine.classify_safety_tier("totally_unknown_action") == "tier3"


# ===================================================================
# Confidence Gate (FR-31)
# ===================================================================


@pytest.mark.unit
def test_confidence_gate_passes(engine):
    """Confidence 0.85 exceeds threshold; gate should pass."""
    result = engine.check_confidence_gate(0.85)
    assert result["passed"] is True
    assert result["mode"] == "normal"


@pytest.mark.unit
def test_confidence_gate_fails(engine):
    """Confidence 0.50 is below threshold; gate should fail."""
    result = engine.check_confidence_gate(0.50)
    assert result["passed"] is False
    assert result["mode"] == "advisory_only"


@pytest.mark.unit
def test_confidence_gate_boundary(engine):
    """Confidence exactly at threshold (0.70) should pass."""
    result = engine.check_confidence_gate(0.70)
    assert result["passed"] is True
    assert result["threshold"] == 0.70


# ===================================================================
# Remediation Plan Generation (FR-26 / FR-28)
# ===================================================================


@pytest.mark.unit
def test_get_plan_db_migration(engine):
    """Plan for db_migration_applied should contain all expected top-level keys."""
    plan = engine.get_remediation_plan("db_migration_applied", 0.85)
    expected_keys = {
        "root_cause",
        "confidence_gate",
        "tier1_auto_actions",
        "tier2_walkthrough",
        "tier3_advisory",
        "prevention_checklist",
    }
    assert expected_keys.issubset(set(plan.keys()))


@pytest.mark.unit
def test_get_plan_has_tier1_actions(engine):
    """Plan for db_migration_applied should have at least 1 tier1 action."""
    plan = engine.get_remediation_plan("db_migration_applied", 0.85)
    assert len(plan["tier1_auto_actions"]) >= 1


@pytest.mark.unit
def test_get_plan_has_walkthrough_steps(engine):
    """Plan should have a tier2_walkthrough dict with a 'steps' list."""
    plan = engine.get_remediation_plan("db_migration_applied", 0.85)
    walkthrough = plan["tier2_walkthrough"]
    assert "steps" in walkthrough
    assert isinstance(walkthrough["steps"], list)
    assert len(walkthrough["steps"]) >= 1


@pytest.mark.unit
def test_get_plan_unknown_cause(engine):
    """Unknown root cause should still return a valid plan structure (degraded)."""
    plan = engine.get_remediation_plan("completely_unknown_failure", 0.90)
    assert plan["root_cause"] == "completely_unknown_failure"
    assert "confidence_gate" in plan
    assert "tier1_auto_actions" in plan
    assert "tier2_walkthrough" in plan
    assert "tier3_advisory" in plan
    assert "prevention_checklist" in plan
    # Degraded: no actions
    assert plan["tier1_auto_actions"] == []


@pytest.mark.unit
def test_get_plan_low_confidence_advisory_only(engine):
    """Low confidence (0.50) should produce advisory_only mode."""
    plan = engine.get_remediation_plan("db_migration_applied", confidence=0.50)
    assert plan["confidence_gate"]["mode"] == "advisory_only"
    assert plan["confidence_gate"]["passed"] is False


@pytest.mark.unit
def test_all_10_rules_produce_plans(engine):
    """Every known rule key should produce a plan without raising."""
    for key in ALL_RULE_KEYS:
        plan = engine.get_remediation_plan(key, 0.85)
        assert plan["root_cause"] == key
        assert "confidence_gate" in plan


# ===================================================================
# Audit Log (FR-30)
# ===================================================================


@pytest.mark.unit
def test_log_action_returns_entry(engine):
    """log_action should return a dict with standard audit fields."""
    entry = engine.log_action(
        incident_id="INC-001",
        action_type="tier1_auto",
        command="flush cache",
        executor="system",
    )
    assert isinstance(entry, dict)
    expected_keys = {
        "incident_id",
        "action_type",
        "command_executed",
        "executor",
        "timestamp",
        "outcome",
    }
    assert expected_keys.issubset(set(entry.keys()))
    assert entry["incident_id"] == "INC-001"


@pytest.mark.unit
def test_audit_log_filters_by_incident(engine):
    """get_audit_log should filter entries by incident_id."""
    engine.log_action("INC-001", "tier1_auto", "cmd1", "system")
    engine.log_action("INC-001", "tier2_guided", "cmd2", "engineer")
    engine.log_action("INC-002", "tier1_auto", "cmd3", "system")

    log_001 = engine.get_audit_log("INC-001")
    assert len(log_001) == 2
    assert all(e["incident_id"] == "INC-001" for e in log_001)


@pytest.mark.unit
def test_audit_log_has_timestamp(engine):
    """Each audit log entry should contain an ISO-format timestamp."""
    engine.log_action("INC-001", "tier1_auto", "cmd1", "system")
    entries = engine.get_audit_log("INC-001")
    assert len(entries) == 1
    ts = entries[0]["timestamp"]
    assert isinstance(ts, str)
    # ISO 8601: contains 'T' or '-' separators at minimum
    assert "T" in ts or "-" in ts


# ===================================================================
# Simple Plan (Legacy Interface)
# ===================================================================


@pytest.mark.unit
def test_get_simple_plan(engine):
    """get_simple_plan should return a dict with root_cause, actions, prevention."""
    plan = engine.get_simple_plan("db_migration_applied")
    assert isinstance(plan, dict)
    assert "root_cause" in plan
    assert "actions" in plan
    assert "prevention" in plan
    assert plan["root_cause"] == "db_migration_applied"


# ===================================================================
# Plan Structure Detail Checks
# ===================================================================


@pytest.mark.unit
def test_plan_tier1_has_command(engine):
    """Each tier1 auto-action should have a 'command' key."""
    plan = engine.get_remediation_plan("db_migration_applied", 0.85)
    for action in plan["tier1_auto_actions"]:
        assert "command" in action
        assert isinstance(action["command"], str)
        assert len(action["command"]) > 0


@pytest.mark.unit
def test_plan_prevention_has_horizons(engine):
    """prevention_checklist should have immediate, short_term, long_term keys."""
    plan = engine.get_remediation_plan("db_migration_applied", 0.85)
    prevention = plan["prevention_checklist"]
    assert "immediate" in prevention
    assert "short_term" in prevention
    assert "long_term" in prevention


# ===================================================================
# NFR-22: Determinism
# ===================================================================


@pytest.mark.unit
def test_safety_classification_is_deterministic(engine):
    """NFR-22: Same input must always produce the same safety tier (10 calls)."""
    action = "rollback_deployment"
    results = [engine.classify_safety_tier(action) for _ in range(10)]
    assert all(r == results[0] for r in results), (
        f"Non-deterministic classification: {results}"
    )
