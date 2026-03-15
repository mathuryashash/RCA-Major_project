"""
Unit tests for src.reporting.nlg_generator.NLGGenerator

Tests narrative generation (Markdown and plaintext), and
module-level helper functions (_display_name, _confidence_label,
_severity_label, _lag_display).
"""

import pytest

from src.reporting.nlg_generator import (
    NLGGenerator,
    _confidence_label,
    _display_name,
    _lag_display,
    _severity_label,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def generator():
    """Instantiate NLGGenerator with default template directory."""
    return NLGGenerator()


# ===================================================================
# Narrative Generation (Markdown)
# ===================================================================


@pytest.mark.unit
def test_generate_narrative_returns_string(generator, sample_rca_data):
    """generate_narrative should return a non-empty string."""
    result = generator.generate_narrative(sample_rca_data)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_narrative_contains_incident_id(generator, sample_rca_data):
    """Output should contain the incident ID 'INC-TEST-001'."""
    result = generator.generate_narrative(sample_rca_data)
    assert "INC-TEST-001" in result


@pytest.mark.unit
def test_narrative_contains_root_cause(generator, sample_rca_data):
    """Output should contain the root cause display name."""
    result = generator.generate_narrative(sample_rca_data)
    display = _display_name(sample_rca_data["root_cause"])
    assert display in result


@pytest.mark.unit
def test_narrative_contains_evidence(generator, sample_rca_data):
    """Output should contain evidence detail text."""
    result = generator.generate_narrative(sample_rca_data)
    # At least one evidence detail should appear in the narrative
    found = any(ev["detail"] in result for ev in sample_rca_data["evidence"])
    assert found, "No evidence detail text found in narrative output"


@pytest.mark.unit
def test_narrative_contains_recommendations(generator, sample_rca_data):
    """Output should mention tier-based recommendations."""
    result = generator.generate_narrative(sample_rca_data)
    assert "Tier 1" in result or "Tier 2" in result or "Tier 3" in result


@pytest.mark.unit
def test_narrative_contains_prevention(generator, sample_rca_data):
    """Output should contain prevention checklist items."""
    result = generator.generate_narrative(sample_rca_data)
    # Check for at least one prevention item from any horizon
    prevention = sample_rca_data["prevention"]
    all_items = (
        prevention.get("immediate", [])
        + prevention.get("short_term", [])
        + prevention.get("long_term", [])
    )
    found = any(item in result for item in all_items)
    assert found, "No prevention items found in narrative output"


@pytest.mark.unit
def test_narrative_is_markdown(generator, sample_rca_data):
    """Output should contain Markdown markers (headings, bold, or lists)."""
    result = generator.generate_narrative(sample_rca_data)
    has_heading = "#" in result
    has_bold = "**" in result
    has_list = "- " in result
    assert has_heading or has_bold or has_list, (
        "Narrative output does not appear to be Markdown"
    )


# ===================================================================
# Plaintext Generation
# ===================================================================


@pytest.mark.unit
def test_generate_plaintext_no_markdown(generator, sample_rca_data):
    """Plaintext output should NOT contain '**' or '##'."""
    result = generator.generate_plaintext(sample_rca_data)
    assert "**" not in result
    assert "##" not in result


@pytest.mark.unit
def test_plaintext_contains_content(generator, sample_rca_data):
    """Plaintext should still contain the incident_id and root cause info."""
    result = generator.generate_plaintext(sample_rca_data)
    assert "INC-TEST-001" in result
    # Root cause display name or raw key should be present
    display = _display_name(sample_rca_data["root_cause"])
    assert display in result or sample_rca_data["root_cause"] in result


# ===================================================================
# _display_name helper
# ===================================================================


@pytest.mark.unit
def test_display_name_known_key():
    """_display_name for a known key should return a human-readable string."""
    result = _display_name("db_migration_applied")
    assert isinstance(result, str)
    assert "_" not in result, "Display name should not contain underscores"
    assert len(result) > 0


@pytest.mark.unit
def test_display_name_unknown_key():
    """_display_name for an unknown key should return a title-cased string."""
    result = _display_name("something_random")
    assert isinstance(result, str)
    assert "_" not in result, "Display name should not contain underscores"
    # Should be title-cased: "Something Random"
    assert result[0].isupper()


# ===================================================================
# _confidence_label helper
# ===================================================================


@pytest.mark.unit
def test_confidence_label_high():
    """Confidence 0.9 should map to 'High'."""
    assert _confidence_label(0.9) == "High"


@pytest.mark.unit
def test_confidence_label_moderate():
    """Confidence 0.6 should map to 'Moderate'."""
    assert _confidence_label(0.6) == "Moderate"


@pytest.mark.unit
def test_confidence_label_low():
    """Confidence 0.3 should map to 'Low'."""
    assert _confidence_label(0.3) == "Low"


# ===================================================================
# _severity_label helper
# ===================================================================


@pytest.mark.unit
def test_severity_critical():
    """Max anomaly score 0.95 should map to 'Critical'."""
    assert _severity_label(0.95) == "Critical"


@pytest.mark.unit
def test_severity_low():
    """Max anomaly score 0.2 should map to 'Low'."""
    assert _severity_label(0.2) == "Low"


# ===================================================================
# _lag_display helper
# ===================================================================


@pytest.mark.unit
def test_lag_display_seconds():
    """_lag_display(30.0) should contain 'minute' (30 minutes)."""
    # Note: _lag_display treats the value as minutes
    # 0.5 minutes = 30 seconds
    result = _lag_display(0.5)
    assert "second" in result.lower()


@pytest.mark.unit
def test_lag_display_minutes():
    """_lag_display(120.0) should contain 'minute' or 'hour'."""
    result = _lag_display(120.0)
    assert "minute" in result.lower() or "hour" in result.lower()


# ===================================================================
# Edge Cases
# ===================================================================


@pytest.mark.unit
def test_narrative_with_minimal_data(generator):
    """Minimal rca_data (just incident_id and root_cause) should not crash."""
    minimal = {
        "incident_id": "INC-MINIMAL-001",
        "root_cause": "db_migration_applied",
    }
    result = generator.generate_narrative(minimal)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "INC-MINIMAL-001" in result


@pytest.mark.unit
def test_narrative_with_empty_causal_chain(generator, sample_rca_data):
    """Empty causal_chain should produce output without error."""
    sample_rca_data["causal_chain"] = []
    result = generator.generate_narrative(sample_rca_data)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "INC-TEST-001" in result
