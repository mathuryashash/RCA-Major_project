"""
Module 7: Intelligent Remediation Engine

Implements FR-25 through FR-32 from the PRD:
  - FR-25: Safety Classification Engine (Tier 1/2/3)
  - FR-26: Immediate Command Generation (Jinja2 templates)
  - FR-27: Rollback Instruction Generator
  - FR-28: Guided Walkthrough Generator
  - FR-29: Long-Term Prevention Checklist
  - FR-30: Remediation Audit Log (in-memory, append-only)
  - FR-31: Confidence Gate (default 0.70)
  - FR-32: Human Confirmation Interface data (countdown, step checklist)

Design philosophy: "Never take a destructive or irreversible action autonomously."
Safety classification is deterministic (YAML rules, no ML) for 100% auditability.
"""

import os
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import yaml
from jinja2 import Template, Environment, FileSystemLoader, BaseLoader

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger(__name__)


class RemediationEngine:
    """
    Safety-tiered remediation engine.

    Loads rules from config/remediation_rules.yaml and safety policy from
    config/safety_rules.yaml. Generates fully rendered remediation plans
    with Tier 1 auto-actions, Tier 2 walkthroughs, Tier 3 advisories,
    and prevention checklists.
    """

    def __init__(
        self,
        rules_path: str = "config/remediation_rules.yaml",
        safety_path: str = "config/safety_rules.yaml",
        config_path: str = "config/config.yaml",
    ):
        self.rules_path = self._resolve_path(rules_path)
        self.safety_path = self._resolve_path(safety_path)
        self.config_path = self._resolve_path(config_path)

        self.rules = self._load_yaml(self.rules_path, default={})
        self.safety_config = self._load_yaml(self.safety_path, default={})
        self.app_config = self._load_yaml(self.config_path, default={})

        # Confidence gate threshold (FR-31)
        gate_cfg = self.safety_config.get("confidence_gate", {})
        self.confidence_threshold = gate_cfg.get("min_threshold", 0.70)

        # Auto-execution parameters
        tier1_cfg = self.safety_config.get("tier1_auto_execute", {})
        self.countdown_seconds = tier1_cfg.get("countdown_seconds", 30)

        # Jinja2 environment for rendering commands
        self._jinja_env = Environment(loader=BaseLoader())

        # In-memory audit log (FR-30) — append-only list
        self.audit_log: List[Dict[str, Any]] = []

        # Database configuration for TimescaleDB audit persistence
        self._db_config = self._load_db_config()
        self._db_available = self._check_db_connection()

        logger.info(
            "Remediation Engine initialized: %d rule sets, confidence gate=%.2f, DB audit=%s",
            len(self.rules),
            self.confidence_threshold,
            "enabled" if self._db_available else "disabled",
        )

    def _load_db_config(self) -> Dict[str, Any]:
        """Load TimescaleDB connection parameters from config.yaml or environment."""
        db_cfg = self.app_config.get("database", {})
        return {
            "host": os.environ.get("RCA_DB_HOST", db_cfg.get("host", "localhost")),
            "port": int(os.environ.get("RCA_DB_PORT", db_cfg.get("port", 5432))),
            "user": os.environ.get("RCA_DB_USER", db_cfg.get("user", "postgres")),
            "password": os.environ.get("RCA_DB_PASSWORD", db_cfg.get("password", "")),
            "dbname": os.environ.get("RCA_DB_NAME", db_cfg.get("dbname", "rca_system")),
        }

    def _check_db_connection(self) -> bool:
        """Check if TimescaleDB connection is available."""
        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not available — audit persistence disabled")
            return False
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception as exc:
            logger.warning(
                "TimescaleDB unavailable: %s — using in-memory fallback", exc
            )
            return False

    @contextmanager
    def _get_db_connection(self):
        """Get a database connection using context manager."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available")
        conn = psycopg2.connect(**self._db_config)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_audit_table(self) -> bool:
        """Create remediation_audit table if it doesn't exist."""
        if not self._db_available:
            return False
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS remediation_audit (
                            id              SERIAL PRIMARY KEY,
                            incident_id     VARCHAR(255) NOT NULL,
                            action_type     VARCHAR(255) NOT NULL,
                            command_executed TEXT,
                            executor        VARCHAR(255) NOT NULL DEFAULT 'system',
                            before_state    VARCHAR(500),
                            after_state     VARCHAR(500),
                            outcome         VARCHAR(50) NOT NULL DEFAULT 'pending',
                            timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_remediation_audit_incident 
                        ON remediation_audit (incident_id, timestamp DESC)
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_remediation_audit_executor 
                        ON remediation_audit (executor, timestamp DESC)
                    """)
                conn.commit()
            return True
        except Exception as exc:
            logger.warning("Failed to ensure audit table: %s", exc)
            return False

    def _save_audit_to_db(self, entry: Dict[str, Any]) -> bool:
        """Persist a single audit entry to TimescaleDB."""
        if not self._db_available:
            return False
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO remediation_audit 
                        (incident_id, action_type, command_executed, executor, 
                         before_state, after_state, outcome, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            entry["incident_id"],
                            entry["action_type"],
                            entry.get("command_executed"),
                            entry["executor"],
                            entry.get("before_state"),
                            entry.get("after_state"),
                            entry["outcome"],
                            datetime.fromisoformat(entry["timestamp"])
                            if isinstance(entry["timestamp"], str)
                            else entry["timestamp"],
                        ),
                    )
                conn.commit()
            return True
        except Exception as exc:
            logger.warning("DB write failed, using in-memory fallback: %s", exc)
            return False

    # ------------------------------------------------------------------
    # File loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve path relative to project root."""
        if os.path.isabs(path):
            return path
        base = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return os.path.join(base, path)

    @staticmethod
    def _load_yaml(path: str, default: Any = None) -> Any:
        """Load a YAML file, returning default if not found."""
        if not os.path.exists(path):
            logger.warning("YAML file not found: %s — using defaults", path)
            return default if default is not None else {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error("Error loading %s: %s", path, exc)
            return default if default is not None else {}

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render(self, template_str: str, context: Dict[str, Any]) -> str:
        """Render a Jinja2 template string with the given context."""
        if not template_str:
            return ""
        try:
            tmpl = Template(template_str)
            return tmpl.render(context)
        except Exception as exc:
            logger.debug("Template render error: %s", exc)
            return template_str  # Return raw string if rendering fails

    def _render_dict(
        self, data: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively render all string values in a dict."""
        rendered = {}
        for key, val in data.items():
            if isinstance(val, str):
                rendered[key] = self._render(val, context)
            elif isinstance(val, dict):
                rendered[key] = self._render_dict(val, context)
            elif isinstance(val, list):
                rendered[key] = [
                    self._render_dict(item, context)
                    if isinstance(item, dict)
                    else self._render(item, context)
                    if isinstance(item, str)
                    else item
                    for item in val
                ]
            else:
                rendered[key] = val
        return rendered

    # ------------------------------------------------------------------
    # FR-25: Safety Classification
    # ------------------------------------------------------------------

    def classify_safety_tier(self, action_name: str) -> str:
        """
        Classify an action into Tier 1, 2, or 3 based on deterministic rules.
        Returns 'tier1', 'tier2', or 'tier3'.
        """
        # Check Tier 1 patterns
        tier1_patterns = self.safety_config.get("tier1_auto_execute", {}).get(
            "action_patterns", []
        )
        for pattern_rule in tier1_patterns:
            pattern = pattern_rule.get("pattern", "")
            if self._matches_pattern(action_name, pattern):
                return "tier1"

        # Check Tier 2 patterns
        tier2_patterns = self.safety_config.get("tier2_guided", {}).get(
            "action_patterns", []
        )
        for pattern_rule in tier2_patterns:
            pattern = pattern_rule.get("pattern", "")
            if self._matches_pattern(action_name, pattern):
                return "tier2"

        # Default to Tier 3 (safest default)
        return "tier3"

    @staticmethod
    def _matches_pattern(action_name: str, pattern: str) -> bool:
        """Simple glob-style pattern matching (supports * wildcard)."""
        import fnmatch

        return fnmatch.fnmatch(action_name, pattern)

    # ------------------------------------------------------------------
    # FR-31: Confidence Gate
    # ------------------------------------------------------------------

    def check_confidence_gate(self, confidence: float) -> Dict[str, Any]:
        """
        Check if root cause confidence meets the threshold.

        Returns:
            dict with 'passed', 'confidence', 'threshold', and 'mode'
        """
        passed = confidence >= self.confidence_threshold
        return {
            "passed": passed,
            "confidence": round(confidence, 4),
            "threshold": self.confidence_threshold,
            "mode": "normal" if passed else "advisory_only",
            "message": (
                "Confidence meets threshold. All tiers enabled."
                if passed
                else f"Low confidence ({confidence:.2f} < {self.confidence_threshold}). "
                "Remediation presented as advisory only. Human confirmation required for Tier 1 actions."
            ),
        }

    # ------------------------------------------------------------------
    # FR-26 + FR-28: Full Remediation Plan Generation
    # ------------------------------------------------------------------

    def get_remediation_plan(
        self,
        root_cause: str,
        confidence: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete, structured remediation plan for a given root cause.

        Parameters
        ----------
        root_cause : str
            Key matching a rule set in remediation_rules.yaml
        confidence : float
            Root cause confidence score from the ranking module
        context : dict, optional
            Environment-specific variables for Jinja2 rendering
            (e.g. service, namespace, host, version, table)

        Returns
        -------
        dict with keys:
            root_cause, confidence_gate, description,
            tier1_auto_actions, tier2_walkthrough, tier3_advisory,
            prevention_checklist
        """
        ctx = context or {}
        gate = self.check_confidence_gate(confidence)

        if root_cause not in self.rules:
            return {
                "root_cause": root_cause,
                "confidence_gate": gate,
                "description": f"No remediation rules found for root cause: {root_cause}",
                "tier1_auto_actions": [],
                "tier2_walkthrough": {"total_steps": 0, "steps": []},
                "tier3_advisory": [],
                "prevention_checklist": {
                    "immediate": [],
                    "short_term": [],
                    "long_term": [],
                },
            }

        rule = self.rules[root_cause]

        # --- Tier 1: Auto-execute actions ---
        tier1_actions = self._build_tier1_actions(rule, gate, ctx)

        # --- Tier 2: Guided walkthrough ---
        tier2_walkthrough = self._build_tier2_walkthrough(rule, ctx)

        # --- Tier 3: Advisory ---
        tier3_advisory = self._build_tier3_advisory(rule, ctx)

        # --- Prevention checklist (FR-29) ---
        prevention = self._build_prevention_checklist(rule, ctx)

        return {
            "root_cause": root_cause,
            "confidence_gate": gate,
            "description": self._render(rule.get("description", ""), ctx),
            "tier1_auto_actions": tier1_actions,
            "tier2_walkthrough": tier2_walkthrough,
            "tier3_advisory": tier3_advisory,
            "prevention_checklist": prevention,
        }

    # ------------------------------------------------------------------
    # Tier builders
    # ------------------------------------------------------------------

    def _build_tier1_actions(
        self, rule: Dict, gate: Dict, ctx: Dict
    ) -> List[Dict[str, Any]]:
        """Build rendered Tier 1 auto-execute actions."""
        raw_actions = rule.get("tier1_auto", [])
        actions = []

        for action in raw_actions:
            rendered_cmd = self._render(action.get("command", ""), ctx)
            action_name = action.get("action", "unknown")

            entry = {
                "action": action_name,
                "description": self._render(action.get("description", ""), ctx),
                "command": rendered_cmd,
                "platform": action.get("platform", "unknown"),
                "est_seconds": action.get("est_seconds", 30),
                "reversible": action.get("reversible", True),
                "safety_tier": "Tier 1 - SAFE (Auto-Execute)",
                "auto_execute": gate[
                    "passed"
                ],  # Only auto-execute if confidence gate passed
                "countdown_seconds": self.countdown_seconds if gate["passed"] else None,
                "cancel_url": f"/remediate/cancel/{action_name}"
                if gate["passed"]
                else None,
            }
            actions.append(entry)

        return actions

    def _build_tier2_walkthrough(self, rule: Dict, ctx: Dict) -> Dict[str, Any]:
        """Build rendered Tier 2 guided walkthrough (FR-28)."""
        raw_steps = rule.get("tier2_guided", [])
        steps = []

        for step_data in raw_steps:
            rendered = {
                "step": step_data.get("step", len(steps) + 1),
                "title": self._render(step_data.get("title", ""), ctx),
                "command": self._render(step_data.get("command", ""), ctx),
                "expected_output": self._render(
                    step_data.get("expected_output", ""), ctx
                ),
                "verification": self._render(step_data.get("verification", ""), ctx),
                "rollback_step": self._render(step_data.get("rollback_step", ""), ctx),
                "est_minutes": step_data.get("est_minutes", 1),
                "safety_note": self._render(step_data.get("safety_note", ""), ctx),
            }
            steps.append(rendered)

        total_minutes = sum(s.get("est_minutes", 0) for s in steps)
        return {
            "total_steps": len(steps),
            "estimated_total_minutes": total_minutes,
            "steps": steps,
        }

    def _build_tier3_advisory(self, rule: Dict, ctx: Dict) -> List[Dict[str, Any]]:
        """Build rendered Tier 3 advisory recommendations."""
        raw_advisories = rule.get("tier3_advisory", [])
        advisories = []

        for adv in raw_advisories:
            advisories.append(
                {
                    "recommendation": self._render(adv.get("recommendation", ""), ctx),
                    "horizon": adv.get("horizon", "long_term"),
                    "owner": adv.get("owner", "Engineering"),
                    "priority": adv.get("priority", 0.5),
                    "effort": adv.get("effort", "Unknown"),
                    "safety_tier": "Tier 3 - ADVISORY (Inform Only)",
                }
            )

        return advisories

    def _build_prevention_checklist(
        self, rule: Dict, ctx: Dict
    ) -> Dict[str, List[str]]:
        """Build the prevention checklist by horizon (FR-29)."""
        prevention = rule.get("prevention", {})
        checklist = {}

        for horizon in ["immediate", "short_term", "long_term"]:
            items = prevention.get(horizon, [])
            checklist[horizon] = [
                self._render(item, ctx) if isinstance(item, str) else str(item)
                for item in items
            ]

        return checklist

    # ------------------------------------------------------------------
    # FR-30: Audit Log
    # ------------------------------------------------------------------

    def log_action(
        self,
        incident_id: str,
        action_type: str,
        command: str,
        executor: str = "system",
        before_state: Optional[str] = None,
        after_state: Optional[str] = None,
        outcome: str = "pending",
    ) -> Dict[str, Any]:
        """
        Append an entry to the immutable audit log.
        Persists to TimescaleDB with in-memory fallback.
        Exposed via GET /audit/{incident_id}.
        """
        entry = {
            "incident_id": incident_id,
            "action_type": action_type,
            "command_executed": command,
            "executor": executor,
            "before_state": before_state,
            "after_state": after_state,
            "timestamp": datetime.now(timezone.utc),
            "outcome": outcome,
        }

        # Always keep in-memory for immediate access
        self.audit_log.append(entry)

        # Persist to TimescaleDB with graceful degradation
        if not self._save_audit_to_db(entry):
            logger.warning(
                "Audit DB write failed for %s/%s — retained in-memory only",
                incident_id,
                action_type,
            )

        logger.info("Audit log entry: %s/%s — %s", incident_id, action_type, outcome)
        return entry

    def get_audit_log(self, incident_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all audit log entries for a given incident.
        Reads from TimescaleDB, falls back to in-memory.
        """
        if self._db_available:
            try:
                with self._get_db_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(
                            """
                            SELECT incident_id, action_type, command_executed,
                                   executor, before_state, after_state, outcome, timestamp
                            FROM remediation_audit
                            WHERE incident_id = %s
                            ORDER BY timestamp ASC
                        """,
                            (incident_id,),
                        )
                        rows = cur.fetchall()
                        if rows:
                            entries = []
                            for row in rows:
                                entry = dict(row)
                                if isinstance(entry.get("timestamp"), datetime):
                                    entry["timestamp"] = entry["timestamp"].isoformat()
                                entries.append(entry)
                            return entries
            except Exception as exc:
                logger.warning("DB read failed, using in-memory fallback: %s", exc)

        # Fallback to in-memory log
        entries = [
            entry for entry in self.audit_log if entry["incident_id"] == incident_id
        ]
        for entry in entries:
            if isinstance(entry.get("timestamp"), datetime):
                entry["timestamp"] = entry["timestamp"].isoformat()
        return entries

    # ------------------------------------------------------------------
    # Convenience: legacy-compatible interface
    # ------------------------------------------------------------------

    def get_simple_plan(self, root_cause: str, context: Optional[Dict] = None) -> Dict:
        """
        Backward-compatible interface matching the original API.
        Returns a simplified plan with 'actions' and 'prevention' keys.
        """
        full_plan = self.get_remediation_plan(
            root_cause, confidence=1.0, context=context
        )

        # Flatten into the old format
        actions = []
        for a in full_plan.get("tier1_auto_actions", []):
            actions.append(
                {
                    "tier": "Tier 1",
                    "description": a["description"],
                    "command": a["command"],
                }
            )
        for step in full_plan.get("tier2_walkthrough", {}).get("steps", []):
            actions.append(
                {
                    "tier": "Tier 2",
                    "description": step["title"],
                    "command": step["command"],
                }
            )
        for adv in full_plan.get("tier3_advisory", []):
            actions.append(
                {
                    "tier": "Tier 3",
                    "description": adv["recommendation"],
                    "command": "Advisory — see recommendation",
                }
            )

        # Flatten prevention
        prevention_items = []
        for horizon, items in full_plan.get("prevention_checklist", {}).items():
            prevention_items.extend(items)

        return {
            "root_cause": root_cause,
            "actions": actions,
            "prevention": "; ".join(prevention_items)
            if prevention_items
            else "No prevention details.",
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    engine = RemediationEngine()

    print("=" * 60)
    print("Case Study 1: DB Migration")
    print("=" * 60)
    plan = engine.get_remediation_plan(
        "db_migration_applied",
        confidence=0.87,
        context={"table": "product_variants", "migration_version": "v2.14.3"},
    )
    print(json.dumps(plan, indent=2, default=str))

    print("\n" + "=" * 60)
    print("Case Study 2: Memory Leak")
    print("=" * 60)
    plan = engine.get_remediation_plan(
        "memory_leak_detected",
        confidence=0.94,
        context={"service": "node-api", "namespace": "production", "revision": "13"},
    )
    print(json.dumps(plan, indent=2, default=str))

    print("\n" + "=" * 60)
    print("Confidence Gate: Low confidence")
    print("=" * 60)
    plan = engine.get_remediation_plan(
        "db_migration_applied",
        confidence=0.45,
    )
    print(f"Gate passed: {plan['confidence_gate']['passed']}")
    print(f"Mode: {plan['confidence_gate']['mode']}")
    print(f"Auto-execute enabled: {plan['tier1_auto_actions'][0]['auto_execute']}")

    print("\n" + "=" * 60)
    print("Unknown root cause")
    print("=" * 60)
    plan = engine.get_remediation_plan("unknown_failure", confidence=0.90)
    print(json.dumps(plan, indent=2, default=str))
