"""
Webhook signature validation for CI/CD event endpoints.
Supports GitHub (HMAC-SHA256) and GitLab (token) webhook signatures.
"""

import hmac
import hashlib
import logging
import os
from typing import Optional, Tuple

import yaml
from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


class WebhookValidator:
    """Validates webhook signatures from GitHub and GitLab."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self._config = self._load_config(config_path)
        self.github_secret = self._config.get("webhooks", {}).get("github_secret", "")
        self.gitlab_token = self._config.get("webhooks", {}).get("gitlab_token", "")
        env_override = os.environ.get("RCA_WEBHOOK_VALIDATE", "").lower()
        if env_override in ("false", "0", "no"):
            self.validate_signatures = False
        elif env_override in ("true", "1", "yes"):
            self.validate_signatures = True
        else:
            self.validate_signatures = self._config.get("webhooks", {}).get(
                "validate_signatures", True
            )

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            return {}

    def validate_github_signature(self, payload: bytes, signature_header: str) -> bool:
        """
        Validate GitHub webhook signature using HMAC-SHA256.

        Args:
            payload: Raw request body bytes
            signature_header: Value of X-Hub-Signature-256 header

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.github_secret:
            logger.warning("GitHub webhook secret not configured")
            return False

        if not signature_header:
            return False

        if not signature_header.startswith("sha256="):
            return False

        expected_signature = signature_header[7:]  # Remove 'sha256=' prefix

        computed_hmac = hmac.new(
            self.github_secret.encode("utf-8"), payload, hashlib.sha256
        )
        computed_signature = computed_hmac.hexdigest()

        return hmac.compare_digest(computed_signature, expected_signature)

    def validate_gitlab_token(self, token_header: str) -> bool:
        """
        Validate GitLab webhook token.

        Args:
            token_header: Value of X-Gitlab-Token header

        Returns:
            True if token is valid, False otherwise
        """
        if not self.gitlab_token:
            logger.warning("GitLab webhook token not configured")
            return False

        if not token_header:
            return False

        return hmac.compare_digest(token_header, self.gitlab_token)

    def validate(self, request: Request, payload: bytes) -> Tuple[bool, str]:
        """
        Validate webhook request from GitHub or GitLab.

        Args:
            request: FastAPI Request object
            payload: Raw request body bytes

        Returns:
            Tuple of (is_valid, provider) where provider is 'github', 'gitlab', or 'none'
        """
        if not self.validate_signatures:
            logger.info("Webhook signature validation is disabled in config")
            return True, "disabled"

        github_signature = request.headers.get("X-Hub-Signature-256")
        gitlab_token = request.headers.get("X-Gitlab-Token")

        if github_signature:
            is_valid = self.validate_github_signature(payload, github_signature)
            if is_valid:
                return True, "github"
            else:
                logger.warning("GitHub webhook signature validation failed")
                return False, "github"

        if gitlab_token:
            is_valid = self.validate_gitlab_token(gitlab_token)
            if is_valid:
                return True, "gitlab"
            else:
                logger.warning("GitLab webhook token validation failed")
                return False, "gitlab"

        logger.warning(
            "No webhook signature headers found (X-Hub-Signature-256 or X-Gitlab-Token)"
        )
        return False, "none"

    def require_valid_webhook(self, request: Request, payload: bytes) -> None:
        """
        Validate webhook and raise 401 if invalid.

        Args:
            request: FastAPI Request object
            payload: Raw request body bytes

        Raises:
            HTTPException: 401 Unauthorized if validation fails
        """
        is_valid, provider = self.validate(request, payload)

        if not is_valid:
            logger.warning(
                f"Webhook validation failed for provider: {provider}. "
                "Returning 401 Unauthorized."
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing webhook signature",
            )


_webhook_validator: Optional[WebhookValidator] = None


def get_webhook_validator() -> WebhookValidator:
    """Get singleton WebhookValidator instance."""
    global _webhook_validator
    if _webhook_validator is None:
        _webhook_validator = WebhookValidator()
    return _webhook_validator
