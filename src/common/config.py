import yaml
import os
import warnings
from pydantic import ValidationError as PydanticValidationError

from .config_models import (
    AppConfig,
    LogSource,
    LogWatchingConfig,
    MetricSourcesConfig,
    DatabaseConfig,
    RedisConfig,
    JWTConfig,
    AnomalyConfig,
    CausalConfig,
)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


DEPRECATED_KEYS = {
    "correlation_window_minutes": "Use causal_inference.correlation_window_minutes instead",
}


def _warn_deprecated(config: dict) -> None:
    """Check for deprecated keys and emit warnings."""
    for key, message in DEPRECATED_KEYS.items():
        if key in config:
            warnings.warn(
                f"Config key '{key}' is deprecated. {message}",
                DeprecationWarning,
                stacklevel=3,
            )
    if "causal_inference" in config:
        causal = config["causal_inference"]
        if "correlation_window_minutes" in causal:
            warnings.warn(
                "Config key 'causal_inference.correlation_window_minutes' is deprecated and will be removed. "
                "This field is no longer used in the causal inference pipeline.",
                DeprecationWarning,
                stacklevel=3,
            )


def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file, then applies environment variable
    overrides (for Docker networking where service names differ from localhost).

    Supported env vars:
        RCA_DB_HOST, RCA_DB_PORT, RCA_DB_USER, RCA_DB_PASSWORD, RCA_DB_NAME
        RCA_REDIS_HOST, RCA_REDIS_PORT
        RCA_PROMETHEUS_URL
    """
    if not os.path.exists(config_path):
        # Handle fallback if running from src or other subdirectories
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = os.path.join(base_dir, "config", "config.yaml")

    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ConfigValidationError(f"Error in configuration file: {exc}")

    if config is None:
        raise ConfigValidationError("Configuration file is empty")

    _warn_deprecated(config)

    # --- Check for full RCA config BEFORE applying env overrides ---
    # This determines whether we do strict validation
    rca_required_keys = {
        "log_sources",
        "database",
        "anomaly_detection",
        "causal_inference",
    }
    has_rca_keys = rca_required_keys.intersection(config.keys())

    # --- Environment variable overrides ---

    # Database (TimescaleDB)
    db = config.setdefault("database", {})
    if os.environ.get("RCA_DB_HOST"):
        db["host"] = os.environ["RCA_DB_HOST"]
    if os.environ.get("RCA_DB_PORT"):
        db["port"] = int(os.environ["RCA_DB_PORT"])
    if os.environ.get("RCA_DB_USER"):
        db["user"] = os.environ["RCA_DB_USER"]
    if os.environ.get("RCA_DB_PASSWORD"):
        db["password"] = os.environ["RCA_DB_PASSWORD"]
    if os.environ.get("RCA_DB_NAME"):
        db["dbname"] = os.environ["RCA_DB_NAME"]

    # Redis
    rd = config.setdefault("redis", {})
    if os.environ.get("RCA_REDIS_HOST"):
        rd["host"] = os.environ["RCA_REDIS_HOST"]
    if os.environ.get("RCA_REDIS_PORT"):
        rd["port"] = int(os.environ["RCA_REDIS_PORT"])

    # Prometheus
    ms = config.setdefault("metric_sources", {})
    if os.environ.get("RCA_PROMETHEUS_URL"):
        ms["prometheus_url"] = os.environ["RCA_PROMETHEUS_URL"]

    # --- Pydantic validation ---
    if has_rca_keys:
        # This looks like a full RCA config - validate it strictly
        try:
            validated_config = AppConfig.model_validate(config)
            return validated_config.model_dump()
        except PydanticValidationError as e:
            error_messages = []
            for error in e.errors():
                loc = ".".join(str(l) for l in error["loc"]) if error["loc"] else "root"
                msg = error["msg"]
                error_messages.append(f"  - {loc}: {msg}")
            error_summary = "\n".join(error_messages)
            raise ConfigValidationError(
                f"Configuration validation failed:\n{error_summary}\n"
                "Please fix your config.yaml file."
            ) from e
    else:
        # Partial/minimal config - return as-is without strict validation
        return config


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
