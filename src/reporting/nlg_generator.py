"""
Module 5: NLG Report Generator (FR-29)

Produces human-readable narratives from RCA findings using Jinja2 + spaCy.

Output formats:
  - Markdown (primary, for dashboard rendering)
  - Plain text (stripped Markdown, for CLI / alerting)

Capabilities:
  - Prose-quality narrative with causal verbs and temporal connectives
  - Cited log lines with timestamps as evidence
  - Multi-horizon prevention checklist (immediate / short_term / long_term)
  - Confidence-adaptive language (high/moderate/low)
  - Severity classification based on anomaly scores
  - spaCy-powered sentence quality: proper noun casing, sentence segmentation
"""

import os
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy setup (graceful fallback if not installed)
# ---------------------------------------------------------------------------

_nlp = None

try:
    import spacy

    _nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy loaded: en_core_web_sm")
except Exception as exc:
    logger.warning(
        "spaCy not available (%s). NLG will run without linguistic polish.", exc
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Causal verbs for varied language (rotated per edge)
_CAUSAL_VERBS = [
    "caused",
    "led to",
    "triggered",
    "resulted in",
    "cascaded into",
    "propagated to",
    "introduced",
]

# Human-readable labels for common signal/source names
_DISPLAY_NAMES: Dict[str, str] = {
    "application": "Application Service",
    "database": "Database",
    "system": "System (OS-level)",
    "db_migration": "Database Migration",
    "db_migration_applied": "Database Schema Migration",
    "query_latency": "Query Latency",
    "connection_pool": "Connection Pool",
    "memory_leak_detected": "Memory Leak",
    "network_partition_detected": "Network Partition",
    "thread_pool_exhaustion": "Thread Pool Exhaustion",
    "dns_ttl_misconfiguration": "DNS TTL Misconfiguration",
    "cpu": "CPU Utilization",
    "memory": "Memory Usage",
    "disk": "Disk I/O",
    "network": "Network Traffic",
}


def _display_name(key: str) -> str:
    """Convert a snake_case key to a human-readable display name."""
    if key in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[key]
    # Auto-convert: replace underscores, title-case
    return key.replace("_", " ").title()


def _confidence_label(confidence: float) -> str:
    """Map confidence score to a human-readable label."""
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.70:
        return "Moderate-High"
    if confidence >= 0.50:
        return "Moderate"
    if confidence >= 0.30:
        return "Low"
    return "Very Low"


def _severity_label(max_anomaly_score: float) -> str:
    """Map the maximum anomaly score to a severity label."""
    if max_anomaly_score >= 0.90:
        return "Critical"
    if max_anomaly_score >= 0.70:
        return "High"
    if max_anomaly_score >= 0.50:
        return "Medium"
    return "Low"


def _lag_display(lag_value: float) -> str:
    """Format a lag value (in minutes or as p-value) into readable text."""
    if lag_value < 0.001:
        return "near-simultaneous"
    if lag_value < 1.0:
        return f"{lag_value * 60:.0f} seconds"
    if lag_value < 60:
        return f"{lag_value:.0f} minutes"
    hours = lag_value / 60
    return f"{hours:.1f} hours"


# ---------------------------------------------------------------------------
# spaCy helpers
# ---------------------------------------------------------------------------


def _polish_sentence(text: str) -> str:
    """
    Use spaCy to improve sentence quality:
    - Ensure proper capitalization of the first word
    - Ensure sentences end with a period
    - Normalize whitespace
    """
    if not _nlp or not text:
        return text

    doc = _nlp(text)
    sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        if not s:
            continue
        # Capitalize first character
        s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
        # Ensure ends with punctuation
        if s and s[-1] not in ".!?:":
            s += "."
        sentences.append(s)

    return " ".join(sentences) if sentences else text


def _generate_prose_summary(
    root_cause: str,
    confidence: float,
    causal_chain: List[Dict],
    evidence: List[Dict],
    anomaly_scores: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate a fluent one-sentence executive summary.
    Matches PRD exemplar style: "A [event] at [time] caused [effect],
    leading to [impact]."
    """
    root_display = _display_name(root_cause)

    # Find the top evidence detail
    top_evidence = ""
    if evidence:
        top_ev = max(evidence, key=lambda e: e.get("score", 0))
        top_evidence = top_ev.get("detail", "")[:120]

    # Build causal narrative fragment
    if causal_chain and len(causal_chain) >= 2:
        chain_text = (
            f"{_display_name(causal_chain[0].get('from', ''))} "
            f"propagated through {_display_name(causal_chain[0].get('to', ''))}, "
            f"ultimately affecting {_display_name(causal_chain[-1].get('to', ''))}"
        )
    elif causal_chain:
        chain_text = (
            f"{_display_name(causal_chain[0].get('from', ''))} "
            f"caused degradation in {_display_name(causal_chain[0].get('to', ''))}"
        )
    else:
        chain_text = "anomalous behavior was detected across monitored services"

    # Determine severity
    max_score = 0.0
    if anomaly_scores:
        max_score = max(anomaly_scores.values()) if anomaly_scores else 0.0
    severity = _severity_label(max_score)

    summary = (
        f"A {severity.lower()}-severity incident was detected involving "
        f"{root_display.lower()}. Analysis shows {chain_text}. "
    )

    if top_evidence:
        summary += f'Key indicator: "{top_evidence}". '

    summary += (
        f"Root cause confidence: {confidence * 100:.0f}% "
        f"({_confidence_label(confidence).lower()})."
    )

    return _polish_sentence(summary)


# ---------------------------------------------------------------------------
# NLG Generator class
# ---------------------------------------------------------------------------


class NLGGenerator:
    """
    Module 5: NLG Report Generator.

    Uses Jinja2 external templates + spaCy for prose-quality narrative
    generation. Produces both Markdown and plain-text output per FR-29.
    """

    def __init__(self, template_dir: Optional[str] = None):
        if template_dir is None:
            # Default: project_root/templates/reports/
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            template_dir = os.path.join(project_root, "templates", "reports")

        self._template_dir = template_dir
        self._env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape([]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        logger.info("NLG Generator initialized (templates: %s)", template_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_narrative(self, rca_data: Dict[str, Any]) -> str:
        """
        Generate a Markdown narrative from structured RCA data.

        Parameters
        ----------
        rca_data : dict
            Keys: incident_id, detected_at, summary, causal_chain,
                  evidence, root_cause, recommendations, prevention_summary
            Optional: confidence, anomaly_scores

        Returns
        -------
        str
            Markdown-formatted narrative report.
        """
        context = self._build_template_context(rca_data)

        try:
            template = self._env.get_template("rca_narrative.md.j2")
            return template.render(context)
        except Exception as exc:
            logger.error("Template rendering failed: %s", exc)
            # Fallback to basic rendering
            return self._fallback_narrative(rca_data)

    def generate_plaintext(self, rca_data: Dict[str, Any]) -> str:
        """
        Generate a plain-text narrative (Markdown stripped).
        FR-29 requires both Markdown and plain text output.
        """
        md = self.generate_narrative(rca_data)
        return self._strip_markdown(md)

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_template_context(self, rca_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw RCA data into template-ready context."""
        root_cause = rca_data.get("root_cause", "unknown")
        confidence = float(rca_data.get("confidence", 0.5))
        causal_chain = rca_data.get("causal_chain", [])
        evidence = rca_data.get("evidence", [])
        anomaly_scores = rca_data.get("anomaly_scores", {})

        # Max anomaly score for severity
        max_score = 0.0
        if anomaly_scores:
            max_score = max(anomaly_scores.values())
        elif evidence:
            max_score = max(e.get("score", 0) for e in evidence)

        # Build display-friendly causal chain
        enriched_chain = self._enrich_causal_chain(causal_chain)

        # Build display-friendly evidence
        enriched_evidence = self._enrich_evidence(evidence)

        # Generate prose summary
        summary = rca_data.get("summary") or _generate_prose_summary(
            root_cause, confidence, causal_chain, evidence, anomaly_scores
        )

        # Build prevention dict from prevention_summary or structured data
        prevention = self._build_prevention(rca_data)

        return {
            "incident_id": rca_data.get("incident_id", "UNKNOWN"),
            "detected_at": rca_data.get("detected_at", "N/A"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "root_cause": root_cause,
            "root_cause_display": _display_name(root_cause),
            "confidence": confidence,
            "confidence_label": _confidence_label(confidence),
            "severity_label": _severity_label(max_score),
            "summary": _polish_sentence(summary),
            "causal_chain": enriched_chain,
            "evidence": enriched_evidence,
            "recommendations": rca_data.get("recommendations", []),
            "prevention": prevention,
        }

    def _enrich_causal_chain(self, chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add display names, causal verbs, and lag formatting to chain."""
        enriched = []
        for i, step in enumerate(chain):
            verb = _CAUSAL_VERBS[i % len(_CAUSAL_VERBS)]
            enriched.append(
                {
                    "from": step.get("from", ""),
                    "to": step.get("to", ""),
                    "from_display": _display_name(step.get("from", "")),
                    "to_display": _display_name(step.get("to", "")),
                    "confidence": float(step.get("confidence", 0)),
                    "p_value": float(step.get("lag", step.get("p_value", 0))),
                    "lag_display": _lag_display(
                        float(step.get("lag", step.get("p_value", 0)))
                    ),
                    "causal_verb": verb,
                    "from_time": step.get("from_time"),
                    "to_time": step.get("to_time"),
                }
            )
        return enriched

    def _enrich_evidence(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add display names and ensure timestamps are present."""
        enriched = []
        for ev in evidence:
            enriched.append(
                {
                    "source": ev.get("source", ""),
                    "source_display": _display_name(ev.get("source", "")),
                    "detail": ev.get("detail", ""),
                    "score": float(ev.get("score", 0)),
                    "timestamp": ev.get("timestamp"),
                }
            )
        return enriched

    def _build_prevention(self, rca_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Build a structured prevention dict with 3 horizons.
        Handles both structured (dict) and flat (string) formats.
        """
        # Check if the pipeline already provides structured prevention
        prevention = rca_data.get("prevention", {})
        if isinstance(prevention, dict) and any(
            k in prevention for k in ("immediate", "short_term", "long_term")
        ):
            return prevention

        # Try to extract from prevention_summary string
        summary = rca_data.get("prevention_summary", "")
        if isinstance(summary, str) and summary:
            # Split on semicolons or newlines
            items = [s.strip() for s in re.split(r"[;\n]", summary) if s.strip()]
            # Distribute items across horizons
            n = len(items)
            return {
                "immediate": items[: max(1, n // 3)],
                "short_term": items[max(1, n // 3) : max(2, 2 * n // 3)],
                "long_term": items[max(2, 2 * n // 3) :],
            }

        return {"immediate": [], "short_term": [], "long_term": []}

    # ------------------------------------------------------------------
    # Plain-text conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_markdown(md: str) -> str:
        """Strip Markdown formatting to produce plain text."""
        text = md
        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        # Remove inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove horizontal rules
        text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
        # Remove blockquotes
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
        # Remove checkbox syntax
        text = re.sub(r"- \[[ x]\] ", "- ", text)
        # Clean up excess blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_narrative(rca_data: Dict[str, Any]) -> str:
        """Simple fallback if template rendering fails."""
        return (
            f"# RCA Report — {rca_data.get('incident_id', 'UNKNOWN')}\n\n"
            f"**Detected:** {rca_data.get('detected_at', 'N/A')}\n\n"
            f"## Summary\n{rca_data.get('summary', 'Analysis complete.')}\n\n"
            f"## Root Cause\n{rca_data.get('root_cause', 'Undetermined')}\n"
        )


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generator = NLGGenerator()

    dummy_data = {
        "incident_id": "INC-20260315-001",
        "detected_at": "2026-03-15 14:05:00",
        "root_cause": "db_migration_applied",
        "confidence": 0.87,
        "anomaly_scores": {"database": 0.91, "application": 0.78, "system": 0.12},
        "summary": (
            "A database schema migration deployed at 14:00 caused PostgreSQL "
            "to use an inefficient sequential scan instead of the composite index, "
            "slowing product queries from 50ms to 5000ms over 4 hours and "
            "exhausting the connection pool."
        ),
        "causal_chain": [
            {
                "from": "db_migration",
                "to": "query_latency",
                "confidence": 0.87,
                "lag": 0.5,
                "from_time": "14:00",
                "to_time": "14:30",
            },
            {
                "from": "query_latency",
                "to": "connection_pool",
                "confidence": 0.85,
                "lag": 2.25,
                "from_time": "14:30",
                "to_time": "16:45",
            },
        ],
        "evidence": [
            {
                "source": "database",
                "detail": "Applying migration: add_composite_index on product_variants",
                "score": 1.0,
                "timestamp": "14:00:12",
            },
            {
                "source": "database",
                "detail": "Query p95 latency exceeded 5000ms threshold",
                "score": 0.91,
                "timestamp": "14:32:05",
            },
            {
                "source": "application",
                "detail": "Connection pool exhausted: 100/100 connections active",
                "score": 0.78,
                "timestamp": "16:48:33",
            },
        ],
        "recommendations": [
            {
                "tier": "Tier 1",
                "description": "Flush query plan cache to reset planner statistics",
                "command": "SELECT pg_stat_reset();",
            },
            {
                "tier": "Tier 2",
                "description": "Roll back DB migration to previous revision",
            },
            {
                "tier": "Tier 3",
                "description": "Implement EXPLAIN ANALYZE benchmarks in CI/CD pipeline",
            },
        ],
        "prevention": {
            "immediate": [
                "Add index usage monitoring alert for all tables with >1M rows",
                "Increase connection pool from 100 to 200 as safety margin",
            ],
            "short_term": [
                "Implement pre-migration query performance benchmarks",
                "Enforce schema change review board with DBA sign-off",
            ],
            "long_term": [
                "Implement blue/green deployment strategy for DB migrations",
                "Adopt online schema change tools (gh-ost or pt-online-schema-change)",
            ],
        },
    }

    print("=" * 72)
    print("MARKDOWN OUTPUT")
    print("=" * 72)
    narrative = generator.generate_narrative(dummy_data)
    print(narrative)

    print("\n" + "=" * 72)
    print("PLAIN TEXT OUTPUT")
    print("=" * 72)
    plaintext = generator.generate_plaintext(dummy_data)
    print(plaintext)
